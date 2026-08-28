"""Active candidate-packed backend built from FlashSinkhorn primitives.

The upstream FlashSinkhorn 0.3 API solves one point-cloud problem per launch.
This module adds a local candidate axis to its fused shifted-potential LSE
update, keeps the upstream matrix-free plan-apply as release authority, and
physically compacts converged candidates.  A calibrated ``min_packed_width``
can hand the narrow tail back to the upstream single-candidate backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import importlib
import math
import time
from typing import Any, Literal, TypeAlias, cast

import torch

from .flash import (
    FlashSinkhornBackend,
    FlashSinkhornConfig,
    FlashSinkhornConvergenceError,
    FlashSinkhornResult,
    FlashSinkhornUnavailableError,
    _log_scalings_from_potentials,
    _validate_point_problem,
)


RetirementMode: TypeAlias = Literal[
    "fixed_width",
    "logical_mask",
    "physical_compaction",
]
_RETIREMENT_MODES = frozenset(
    {"fixed_width", "logical_mask", "physical_compaction"}
)


def _resolve_retirement_mode(
    *,
    retirement_mode: RetirementMode | str | None,
    active_compaction: bool | None,
) -> RetirementMode:
    """Resolve the explicit mode while retaining one unambiguous legacy alias."""

    if retirement_mode is not None and active_compaction is not None:
        raise ValueError(
            "retirement_mode and legacy active_compaction are mutually exclusive"
        )
    if active_compaction is not None:
        if not isinstance(active_compaction, bool):
            raise TypeError("active_compaction must be a bool or None")
        return "physical_compaction" if active_compaction else "fixed_width"
    if retirement_mode is None:
        # Preserve the historical solve_batch default.
        return "physical_compaction"
    if retirement_mode not in _RETIREMENT_MODES:
        raise ValueError(
            "retirement_mode must be one of fixed_width, logical_mask, "
            "physical_compaction"
        )
    return cast(RetirementMode, retirement_mode)


def _freeze_retired_potentials(
    next_value: torch.Tensor,
    previous_value: torch.Tensor,
    live_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore retired candidate rows without changing the physical shape."""

    if next_value.shape != previous_value.shape or next_value.ndim != 2:
        raise ValueError("logical-mask potentials must be matching 2-D tensors")
    if live_mask.shape != (next_value.shape[0],) or live_mask.dtype != torch.bool:
        raise ValueError("live_mask must be boolean with one entry per candidate")
    if live_mask.device != next_value.device or previous_value.device != next_value.device:
        raise ValueError("logical-mask tensors must share one device")
    torch.where(
        live_mask[:, None],
        next_value,
        previous_value,
        out=next_value,
    )
    return next_value


def _release_verification_candidates(
    screened_passed: list[int],
    active_original_ids: list[int],
    verified_first_passage: list[int | None],
    *,
    active_compaction: bool | None = None,
    retirement_mode: RetirementMode | str | None = None,
) -> list[int]:
    """Choose lanes that require upstream release verification at an audit."""

    mode = _resolve_retirement_mode(
        retirement_mode=retirement_mode,
        active_compaction=active_compaction,
    )
    if mode != "fixed_width":
        return list(screened_passed)
    # A static control does not release individual lanes.  Verifying early
    # screen passes would add diagnostic work that the active arm alone needs
    # for safe retirement and would therefore bias active/static timing.  The
    # static arm performs one upstream verification of every lane only at the
    # common final boundary.
    if len(screened_passed) == len(active_original_ids):
        return list(screened_passed)
    return []


def _load_packed_lse() -> Any:
    try:
        module = importlib.import_module("drainsinkhorn._flash_packed_triton")
        return getattr(module, "flashsinkhorn_lse_packed")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise FlashSinkhornUnavailableError(
            "packed FlashSinkhorn requires flash-sinkhorn>=0.3.3,<0.4, "
            "Triton>=3.1, and CUDA"
        ) from exc


@dataclass(frozen=True)
class FlashPackedConfig:
    epsilon: float
    marginal_tolerance: float = 1e-3
    max_iterations: int = 2_000
    check_every: int = 10
    cost_scale: float = 1.0
    allow_tf32: bool = False
    use_exp2: bool = True
    min_packed_width: int = 2
    block_m: int = 64
    block_n: int = 64
    block_k: int = 16
    num_warps: int = 4
    num_stages: int = 2
    batched_marginal_audit: bool = False
    row_only_batched_audit: bool = False
    initial_marginal_audit: bool = True
    reuse_packed_buffers: bool = False
    raise_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.marginal_tolerance) or self.marginal_tolerance <= 0:
            raise ValueError("marginal_tolerance must be finite and positive")
        if self.max_iterations < 1 or self.check_every < 1:
            raise ValueError("iteration and audit intervals must be positive")
        if not math.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("cost_scale must be finite and positive")
        if self.min_packed_width < 1:
            raise ValueError("min_packed_width must be positive")
        blocks = (self.block_m, self.block_n, self.block_k)
        if any(value < 1 or value & (value - 1) for value in blocks):
            raise ValueError("Triton block sizes must be positive powers of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1,2,4,8")
        if self.num_stages < 1:
            raise ValueError("num_stages must be positive")
        if self.row_only_batched_audit and not self.batched_marginal_audit:
            raise ValueError(
                "row_only_batched_audit requires batched_marginal_audit"
            )


@dataclass(frozen=True)
class FlashPackedWindowResult:
    results: tuple[FlashSinkhornResult, ...]
    verified_first_passage_iterations: tuple[int | None, ...]
    retirement_mode: RetirementMode
    physical_width_trace: tuple[int, ...]
    logical_live_width_trace: tuple[int, ...]
    active_width_trace: tuple[int, ...]
    physical_candidate_slots: int
    logical_live_candidate_slots: int
    frozen_but_computed_slots: int
    candidate_slots: int
    packed_iterations: int
    serial_tail_candidates: int
    compaction_events: int
    audit_events: int
    screen_pass_candidate_count: int
    correction_seconds: float
    audit_seconds: float
    compaction_seconds: float
    serial_tail_seconds: float
    release_verifier_candidate_count: int
    release_verified_candidate_count: int
    release_verifier_seconds: float
    logical_mask_applications: int
    logical_mask_seconds: float
    synchronization_seconds: float
    audit_iterations: tuple[int, ...]
    screen_positive_original_ids_trace: tuple[tuple[int, ...], ...]
    verifier_candidate_original_ids_trace: tuple[tuple[int, ...], ...]
    verified_release_original_ids_trace: tuple[tuple[int, ...], ...]
    duplicate_successful_release_verifications: int
    min_packed_width: int
    backend_name: str


@dataclass(frozen=True)
class FlashPackedFirstPassageProfile:
    """Timing-ineligible, full-width residual trajectory for one window.

    Traces are candidate-major and include the initialization at index zero.
    This object deliberately stays separate from ``FlashPackedWindowResult``:
    auditing every correction cycle synchronizes the device and must never be
    interpreted as production timing or production control flow.
    """

    row_residual_traces: tuple[tuple[float, ...], ...]
    column_residual_traces: tuple[tuple[float, ...], ...]
    max_residual_traces: tuple[tuple[float, ...], ...]
    first_passage_iterations: tuple[int | None, ...]
    recross_counts_after_first_passage: tuple[int, ...]
    post_first_passage_max_residuals: tuple[float | None, ...]
    profile_horizon: int
    full_width_trace: tuple[int, ...]
    candidate_slots: int
    timing_eligible: bool = False
    excluded_from_formal_timing: bool = True
    profile_kind: str = "full_width_upstream_two_sided_current_residual"
    batched_row_residual_traces: tuple[tuple[float, ...], ...] | None = None


def _residual_trajectory_statistics(
    max_residual_traces: Sequence[Sequence[float]],
    tolerance: float,
) -> tuple[tuple[int | None, ...], tuple[int, ...], tuple[float | None, ...]]:
    """Summarize first passage and later pass-to-fail recrossings.

    The trace index is the number of completed correction cycles, so an
    initialization that already passes has first-passage depth zero.  A
    recrossing is counted whenever a passing state is followed by a failing
    state after first passage.  No persistence assumption is made.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    first_passages: list[int | None] = []
    recross_counts: list[int] = []
    post_first_passage_maxima: list[float | None] = []
    for lane, raw_trace in enumerate(max_residual_traces):
        trace = [float(value) for value in raw_trace]
        if not trace:
            raise ValueError(f"residual trace for lane {lane} is empty")
        if any(not math.isfinite(value) or value < 0.0 for value in trace):
            raise ValueError(f"residual trace for lane {lane} is invalid")
        first = next(
            (index for index, value in enumerate(trace) if value <= tolerance),
            None,
        )
        first_passages.append(first)
        if first is None:
            recross_counts.append(0)
            post_first_passage_maxima.append(None)
            continue
        pass_flags = [value <= tolerance for value in trace[first:]]
        recross_counts.append(
            sum(
                previous and not current
                for previous, current in zip(pass_flags, pass_flags[1:])
            )
        )
        post_first_passage_maxima.append(max(trace[first:]))
    return (
        tuple(first_passages),
        tuple(recross_counts),
        tuple(post_first_passage_maxima),
    )


class FlashSinkhornPackedBackend:
    """Candidate-packed, single-device squared-Euclidean Sinkhorn backend."""

    distributed = False
    supports_generic_kernel = False
    supports_squared_euclidean = True
    supports_candidate_packing = True

    def __init__(self, config: FlashPackedConfig):
        self.config = config
        self._packed_lse: Any | None = None
        self._single = FlashSinkhornBackend(
            FlashSinkhornConfig(
                epsilon=config.epsilon,
                marginal_tolerance=config.marginal_tolerance,
                max_iterations=config.max_iterations,
                check_every=config.check_every,
                cost_scale=config.cost_scale,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=True,
                raise_on_nonconvergence=config.raise_on_nonconvergence,
            )
        )

    @property
    def version(self) -> str:
        return self._single.version

    @property
    def name(self) -> str:
        return f"flash-sinkhorn/{self.version}-active-packed-triton"

    def _ensure_packed_lse(self) -> Any:
        if self._packed_lse is None:
            self._packed_lse = _load_packed_lse()
        return self._packed_lse

    def _lse(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shifted: torch.Tensor,
        log_weights: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        function = self._ensure_packed_lse()
        return function(
            x,
            y,
            shifted,
            log_weights,
            self.config.epsilon,
            cost_scale=self.config.cost_scale,
            allow_tf32=self.config.allow_tf32,
            use_exp2=self.config.use_exp2,
            block_m=self.config.block_m,
            block_n=self.config.block_n,
            block_k=self.config.block_k,
            num_warps=self.config.num_warps,
            num_stages=self.config.num_stages,
            out=out,
        )

    @torch.no_grad()
    def shifted_step(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        log_a: torch.Tensor,
        log_targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
        *,
        scratch_f: torch.Tensor | None = None,
        scratch_g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one Gauss--Seidel update to a candidate-major active batch."""

        if ys.ndim != 3:
            raise ValueError("ys must have shape [width,m,d]")
        width, m, _dimension = ys.shape
        if f_hat.ndim != 2 or f_hat.shape[0] != width:
            raise ValueError("f_hat must have shape [width,n]")
        if g_hat.shape != (width, m) or log_targets.shape != (width, m):
            raise ValueError("g_hat/log_targets must have shape [width,m]")
        n = x.shape[-2]
        if log_a.shape == (n,):
            expanded_log_a = log_a.unsqueeze(0).expand(width, -1)
        elif log_a.shape == (width, n):
            expanded_log_a = log_a
        else:
            raise ValueError("log_a must have shape [n] or [width,n]")
        if self.config.reuse_packed_buffers:
            if scratch_f is None:
                scratch_f = torch.empty_like(f_hat)
            if scratch_g is None:
                scratch_g = torch.empty_like(g_hat)
            if scratch_f.shape != f_hat.shape or scratch_g.shape != g_hat.shape:
                raise ValueError("ping-pong scratch buffers have incompatible shapes")
            # Gauss--Seidel ordering is retained: G reads the newly written F
            # buffer, and only then is the pair returned for the next cycle.
            next_f = self._lse(x, ys, g_hat, log_targets, out=scratch_f)
            next_g = self._lse(ys, x, next_f, expanded_log_a, out=scratch_g)
        else:
            next_f = self._lse(x, ys, g_hat, log_targets)
            next_g = self._lse(ys, x, next_f, expanded_log_a)
        return next_f, next_g

    @torch.no_grad()
    def serial_shifted_step(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        log_a: torch.Tensor,
        log_targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference one-step loop through the upstream fused single kernel."""

        fused_lse, _ = self._single._ensure_functions()
        outputs_f: list[torch.Tensor] = []
        outputs_g: list[torch.Tensor] = []
        for index in range(ys.shape[0]):
            current_x = x if x.ndim == 2 else x[index]
            current_log_a = log_a if log_a.ndim == 1 else log_a[index]
            current_f = fused_lse(
                current_x,
                ys[index],
                g_hat[index],
                log_targets[index],
                self.config.epsilon,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=True,
            )
            current_g = fused_lse(
                ys[index],
                current_x,
                current_f,
                current_log_a,
                self.config.epsilon,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=True,
            )
            outputs_f.append(current_f)
            outputs_g.append(current_g)
        return torch.stack(outputs_f), torch.stack(outputs_g)

    @torch.no_grad()
    def _audit_serial(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        rows: list[float] = []
        columns: list[float] = []
        for index in range(ys.shape[0]):
            current_x = x if x.ndim == 2 else x[index]
            current_a = a if a.ndim == 1 else a[index]
            row, column = self._single._audit_shifted(
                current_x,
                ys[index],
                f_hat[index],
                g_hat[index],
                torch.log(current_a),
                torch.log(targets[index]),
                current_a,
                targets[index],
            )
            rows.append(row)
            columns.append(column)
        return rows, columns

    @torch.no_grad()
    def _audit_batched(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        """Audit every active lane with two candidate-batched LSE calls.

        For the represented plan,

        ``row = a * exp((f_hat - T_b(g_hat)) / epsilon)`` and
        ``column = b * exp((g_hat - T_a(f_hat)) / epsilon)``.

        ``T_b`` and ``T_a`` are the same shifted-potential log-sum-exp maps
        used by the solver.  Computing both maps over the candidate axis keeps
        the exact current-plan marginal semantics while replacing two
        upstream plan-apply launches per lane with two packed launches for the
        whole active set.  The returned Python lists induce one synchronization
        per side, rather than one per lane.
        """

        if ys.ndim != 3 or a.ndim != 2 or targets.ndim != 2:
            raise ValueError("batched audit expects candidate-major tensors")
        width = ys.shape[0]
        if (
            a.shape[0] != width
            or targets.shape[0] != width
            or f_hat.shape != a.shape
            or g_hat.shape != targets.shape
        ):
            raise ValueError("batched audit candidate shapes disagree")
        balanced_f = self._lse(x, ys, g_hat, torch.log(targets))
        inverse_epsilon = 1.0 / self.config.epsilon
        row_mass = a * torch.exp((f_hat - balanced_f) * inverse_epsilon)
        row_residual = torch.sum(torch.abs(row_mass - a), dim=1)
        if self.config.row_only_batched_audit:
            # After a Gauss--Seidel G update the column marginal is the
            # updated side by construction.  The zero here is only a screen
            # placeholder: every row-screen pass is still checked by the
            # upstream two-sided plan-apply verifier before retirement.
            return row_residual.cpu().tolist(), [0.0] * width
        balanced_g = self._lse(ys, x, f_hat, torch.log(a))
        column_mass = targets * torch.exp(
            (g_hat - balanced_g) * inverse_epsilon
        )
        column_residual = torch.sum(torch.abs(column_mass - targets), dim=1)
        return row_residual.cpu().tolist(), column_residual.cpu().tolist()

    @torch.no_grad()
    def _audit(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        if self.config.batched_marginal_audit:
            return self._audit_batched(x, ys, a, targets, f_hat, g_hat)
        return self._audit_serial(x, ys, a, targets, f_hat, g_hat)

    @torch.no_grad()
    def _verify_release_candidates(
        self,
        candidates: list[int],
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
        rows: list[float],
        columns: list[float],
    ) -> list[int]:
        """Confirm batched-audit passes with upstream plan-apply primitives.

        The batched LSE audit is a scheduling screen.  A lane may retire only
        after the upstream FlashSinkhorn plan-apply kernel independently
        recomputes both current marginals.  This keeps the optimization out of
        the release authority while paying the serial verifier only once near
        a lane's completion, rather than at every scheduled audit.
        """

        if not self.config.batched_marginal_audit:
            return candidates
        verified: list[int] = []
        for index in candidates:
            current_x = x if x.ndim == 2 else x[index]
            row, column = self._single._audit_shifted(
                current_x,
                ys[index],
                f_hat[index],
                g_hat[index],
                torch.log(a[index]),
                torch.log(targets[index]),
                a[index],
                targets[index],
            )
            rows[index] = row
            columns[index] = column
            if max(row, column) <= self.config.marginal_tolerance:
                verified.append(index)
        return verified

    @torch.no_grad()
    def project_target_potentials_fixed_source(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        source_f: torch.Tensor,
    ) -> torch.Tensor:
        """Project one source dual onto many targets without copying the source.

        ``x`` is one fixed source support shared by every candidate in ``ys``.
        The packed target-side c-transform therefore keeps a single physical
        copy of ``x`` and introduces only the candidate axis required by the
        target supports and shifted potentials.  The returned standard target
        potentials follow the project convention ``[n,width]``.  This helper
        covers the equal-cardinality stream contract used by the packed solver.
        """

        if x.ndim != 2 or ys.ndim != 3:
            raise ValueError("x must have shape [n,d] and ys [width,n,d]")
        width, target_n, dimension = ys.shape
        n = int(x.shape[0])
        if (
            target_n != n
            or x.shape[1] != dimension
            or width < 1
            or n < 1
            or dimension < 1
        ):
            raise ValueError(
                "fixed source and candidate targets must be non-empty with shape [n,d]"
            )
        if a.shape != (n,) or source_f.shape != (n,):
            raise ValueError("a and source_f must have shape [n]")
        if not x.is_cuda or any(value.device != x.device for value in (ys, a, source_f)):
            raise ValueError("all proposal tensors must share one CUDA device")
        if torch.any(a <= 0) or not bool(torch.isfinite(source_f).all()):
            raise ValueError("proposal requires positive masses and finite source_f")
        for target in ys:
            _validate_point_problem(x, target, a, a)

        x_work = x.float().contiguous()
        ys_work = ys.float().contiguous()
        a_work = a.float().contiguous()
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=1)
        beta = self.config.cost_scale * torch.sum(ys_work * ys_work, dim=2)
        f_hat = (
            source_f.float().contiguous() - alpha
        ).unsqueeze(0).expand(width, -1).contiguous()
        log_a = torch.log(a_work).unsqueeze(0).expand(width, -1).contiguous()
        g_hat = self._lse(ys_work, x_work, f_hat, log_a)
        g = (g_hat + beta).T.contiguous()
        if not bool(torch.isfinite(g).all()):
            raise FlashSinkhornConvergenceError(
                "fixed-source target c-transform produced a non-finite potential"
            )
        return g

    @torch.no_grad()
    def project_target_potentials_changing_source(
        self,
        x_candidates: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        source_f: torch.Tensor,
    ) -> torch.Tensor:
        """Project one anchor source dual onto every changing source/target pair.

        Each lane receives the same *standard* source potential ``source_f``
        from the already solved anchor.  Its source support can nevertheless
        differ, so the shifted potential is formed separately with that lane's
        squared-norm correction before the target-side c-transform.  This is
        the fixed-source proposal from the original packed executor extended
        to real changing-support streams; it does not select, subsample, or
        otherwise alter anchor points.

        ``x_candidates`` and ``ys`` have shape ``[width,n,d]`` and the return
        value has the project convention ``[n,width]`` for ``g_init``.
        """

        if x_candidates.ndim != 3 or ys.ndim != 3:
            raise ValueError("x_candidates and ys must have shape [width,n,d]")
        if x_candidates.shape != ys.shape:
            raise ValueError("changing source and target supports must have equal shapes")
        width, n, dimension = x_candidates.shape
        if width < 1 or n < 1 or dimension < 1:
            raise ValueError("candidate supports must be non-empty")
        if a.shape != (n,) or source_f.shape != (n,):
            raise ValueError("a and source_f must have shape [n]")
        if not x_candidates.is_cuda or any(
            value.device != x_candidates.device for value in (ys, a, source_f)
        ):
            raise ValueError("all proposal tensors must share one CUDA device")
        if torch.any(a <= 0) or not bool(torch.isfinite(source_f).all()):
            raise ValueError("proposal requires positive masses and finite source_f")
        for index in range(width):
            _validate_point_problem(x_candidates[index], ys[index], a, a)

        x_work = x_candidates.float().contiguous()
        ys_work = ys.float().contiguous()
        a_work = a.float().contiguous()
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=2)
        beta = self.config.cost_scale * torch.sum(ys_work * ys_work, dim=2)
        f_hat = source_f.float().contiguous().unsqueeze(0).expand(width, -1) - alpha
        log_a = torch.log(a_work).unsqueeze(0).expand(width, -1).contiguous()
        g_hat = self._lse(ys_work, x_work, f_hat.contiguous(), log_a)
        g = (g_hat + beta).T.contiguous()
        if not bool(torch.isfinite(g).all()):
            raise FlashSinkhornConvergenceError(
                "candidate target c-transform produced a non-finite potential"
            )
        return g

    @torch.no_grad()
    def project_two_sided_potentials_changing_source(
        self,
        anchor_y: torch.Tensor,
        x_candidates: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        anchor_b: torch.Tensor,
        anchor_g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both anchor dual sides onto a changing-support window.

        A direct index-wise copy of an anchor source potential is only a
        convention when source supports change.  This routine instead applies
        two ordinary log-domain c-transforms: first from the solved anchor
        target dual ``anchor_g`` to every current source support, then from
        each resulting current source dual to its current target support.
        It returns standard-potential ``(f_init, g_init)`` tensors in the
        project layout ``[n,width]``.  No support is selected, compressed, or
        replaced; subsequent Sinkhorn correction remains exact.
        """

        if x_candidates.ndim != 3 or ys.ndim != 3:
            raise ValueError("x_candidates and ys must have shape [width,n,d]")
        if x_candidates.shape != ys.shape:
            raise ValueError("changing source and target supports must have equal shapes")
        width, n, dimension = x_candidates.shape
        if anchor_y.shape != (n, dimension):
            raise ValueError("anchor_y must have shape [n,d] matching candidates")
        if any(value.shape != (n,) for value in (a, b, anchor_b, anchor_g)):
            raise ValueError("a, b, anchor_b, and anchor_g must have shape [n]")
        if width < 1 or n < 1 or dimension < 1:
            raise ValueError("candidate supports must be non-empty")
        tensors = (anchor_y, ys, a, b, anchor_b, anchor_g)
        if not x_candidates.is_cuda or any(
            value.device != x_candidates.device for value in tensors
        ):
            raise ValueError("all proposal tensors must share one CUDA device")
        if (
            torch.any(a <= 0)
            or torch.any(b <= 0)
            or torch.any(anchor_b <= 0)
            or not bool(torch.isfinite(anchor_g).all())
        ):
            raise ValueError("proposal requires positive masses and finite anchor_g")
        for index in range(width):
            _validate_point_problem(anchor_y, x_candidates[index], anchor_b, a)
            _validate_point_problem(x_candidates[index], ys[index], a, b)

        x_work = x_candidates.float().contiguous()
        ys_work = ys.float().contiguous()
        anchor_y_work = anchor_y.float().contiguous()
        a_work = a.float().contiguous()
        anchor_b_work = anchor_b.float().contiguous()
        log_a = torch.log(a_work).unsqueeze(0).expand(width, -1).contiguous()
        log_anchor_b = torch.log(anchor_b_work).unsqueeze(0).expand(
            width, -1
        ).contiguous()
        anchor_alpha = self.config.cost_scale * torch.sum(
            anchor_y_work * anchor_y_work, dim=1
        )
        anchor_g_hat = (
            anchor_g.float().contiguous() - anchor_alpha
        ).unsqueeze(0).expand(width, -1).contiguous()

        # f_hat is the shifted current-source dual for each candidate.  The
        # 2-D anchor support is intentionally shared by all candidate programs.
        f_hat = self._lse(x_work, anchor_y_work, anchor_g_hat, log_anchor_b)
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=2)
        beta = self.config.cost_scale * torch.sum(ys_work * ys_work, dim=2)
        g_hat = self._lse(ys_work, x_work, f_hat, log_a)
        f = (f_hat + alpha).T.contiguous()
        g = (g_hat + beta).T.contiguous()
        if not bool(torch.isfinite(f).all()) or not bool(torch.isfinite(g).all()):
            raise FlashSinkhornConvergenceError(
                "two-sided candidate c-transform produced a non-finite potential"
            )
        return f, g

    def _result(
        self,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        iterations: int,
        audits: int,
        row_residual: float,
        column_residual: float,
        converged: bool,
    ) -> FlashSinkhornResult:
        f = f_hat + alpha
        g = g_hat + beta
        log_u, log_v = _log_scalings_from_potentials(
            f, g, a, b, self.config.epsilon
        )
        return FlashSinkhornResult(
            f=f,
            g=g,
            log_u=log_u,
            log_v=log_v,
            u=torch.exp(log_u),
            v=torch.exp(log_v),
            n_iters=iterations,
            row_residual=row_residual,
            column_residual=column_residual,
            audits=audits,
            converged=converged,
            solver_name=self.name,
        )

    @torch.no_grad()
    def profile_first_passage_trajectory(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        *,
        horizon: int,
        f_init: torch.Tensor | None = None,
        g_init: torch.Tensor | None = None,
    ) -> FlashPackedFirstPassageProfile:
        """Replay a full-width path with a two-sided audit every cycle.

        This is a profile-only companion to ``solve_batch``.  It never
        compacts lanes and always audits with the upstream per-lane plan-apply
        primitive, even when the production screen is row-only.  The caller
        must choose ``horizon`` from a frozen production run (normally the
        static common final depth or ``max(ell_s)``) so that residual
        recrossings after first passage remain observable.

        The method intentionally does not return production results, run a
        consumer, or expose a timing claim.  Per-cycle audits synchronize the
        device and therefore make this path ineligible for formal timing.
        """

        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if ys.ndim != 3:
            raise ValueError("ys must have shape [width,m,d]")
        width, m, dimension = ys.shape
        if x.ndim == 2 and x.shape[1] == dimension:
            n = int(x.shape[0])
        elif x.ndim == 3 and x.shape[0] == width and x.shape[2] == dimension:
            n = int(x.shape[1])
        else:
            raise ValueError(
                "x must have shape [n,d] or [width,n,d] with the same d as ys"
            )
        if width < self.config.min_packed_width:
            raise ValueError(
                "profile replay requires width >= min_packed_width; set "
                "min_packed_width=1 for B=1 parity"
            )
        if targets.shape != (m, width):
            raise ValueError("targets must have shape [m,width]")
        if a.shape not in ((n,), (n, width)):
            raise ValueError("a must have shape [n] or [n,width]")
        if f_init is not None and f_init.shape != (n, width):
            raise ValueError("f_init must have shape [n,width]")
        if g_init is not None and g_init.shape != (m, width):
            raise ValueError("g_init must have shape [m,width]")
        if not x.is_cuda or not ys.is_cuda:
            raise ValueError("packed FlashSinkhorn profile requires CUDA point clouds")
        if any(value.device != x.device for value in (ys, a, targets)):
            raise ValueError("all packed profile tensors must share one device")
        if torch.any(a <= 0) or torch.any(targets <= 0):
            raise ValueError("FlashSinkhorn requires strictly positive marginals")

        def source_masses(index: int) -> torch.Tensor:
            return a if a.ndim == 1 else a[:, index]

        for index in range(width):
            _validate_point_problem(
                x if x.ndim == 2 else x[index],
                ys[index],
                source_masses(index),
                targets[:, index],
            )

        x_work = x.float().contiguous()
        ys_work = ys.float().contiguous()
        a_work = (
            a.float().unsqueeze(0).expand(width, -1).contiguous()
            if a.ndim == 1
            else a.T.float().contiguous()
        )
        targets_work = targets.T.float().contiguous()
        log_a = torch.log(a_work)
        log_targets = torch.log(targets_work)
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=-1)
        alpha_work = (
            alpha.unsqueeze(0).expand(width, -1).clone()
            if x_work.ndim == 2
            else alpha.contiguous()
        )
        beta_work = self.config.cost_scale * torch.sum(
            ys_work * ys_work, dim=2
        )
        f_hat = (
            -alpha_work.clone()
            if f_init is None
            else f_init.T.float().contiguous() - alpha_work
        )
        g_hat = (
            -beta_work.clone()
            if g_init is None
            else g_init.T.float().contiguous() - beta_work
        )
        scratch_f = (
            torch.empty_like(f_hat) if self.config.reuse_packed_buffers else None
        )
        scratch_g = (
            torch.empty_like(g_hat) if self.config.reuse_packed_buffers else None
        )
        row_traces: list[list[float]] = [[] for _ in range(width)]
        column_traces: list[list[float]] = [[] for _ in range(width)]
        batched_row_traces: list[list[float]] | None = (
            [[] for _ in range(width)]
            if self.config.batched_marginal_audit
            and self.config.row_only_batched_audit
            else None
        )

        def record_current_residual() -> None:
            batched_rows: list[float] | None = None
            if batched_row_traces is not None:
                batched_rows, _ = self._audit_batched(
                    x_work,
                    ys_work,
                    a_work,
                    targets_work,
                    f_hat,
                    g_hat,
                )
            rows, columns = self._audit_serial(
                x_work,
                ys_work,
                a_work,
                targets_work,
                f_hat,
                g_hat,
            )
            for lane, (row, column) in enumerate(zip(rows, columns)):
                row_traces[lane].append(float(row))
                column_traces[lane].append(float(column))
                if batched_row_traces is not None:
                    assert batched_rows is not None
                    batched_row_traces[lane].append(float(batched_rows[lane]))

        record_current_residual()
        for _iteration in range(horizon):
            f_hat, g_hat = self.shifted_step(
                x_work,
                ys_work,
                log_a,
                log_targets,
                f_hat,
                g_hat,
                scratch_f=scratch_f,
                scratch_g=scratch_g,
            )
            record_current_residual()

        max_traces = tuple(
            tuple(max(row, column) for row, column in zip(rows, columns))
            for rows, columns in zip(row_traces, column_traces)
        )
        first, recross, post_max = _residual_trajectory_statistics(
            max_traces,
            self.config.marginal_tolerance,
        )
        return FlashPackedFirstPassageProfile(
            row_residual_traces=tuple(tuple(values) for values in row_traces),
            column_residual_traces=tuple(
                tuple(values) for values in column_traces
            ),
            max_residual_traces=max_traces,
            first_passage_iterations=first,
            recross_counts_after_first_passage=recross,
            post_first_passage_max_residuals=post_max,
            profile_horizon=horizon,
            full_width_trace=(width,) * horizon,
            candidate_slots=width * horizon,
            batched_row_residual_traces=(
                tuple(tuple(values) for values in batched_row_traces)
                if batched_row_traces is not None else None
            ),
        )

    @torch.no_grad()
    def solve_batch(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        *,
        f_init: torch.Tensor | None = None,
        g_init: torch.Tensor | None = None,
        retirement_mode: RetirementMode | str | None = None,
        active_compaction: bool | None = None,
    ) -> FlashPackedWindowResult:
        """Solve a candidate window under fixed, logical, or physical retirement.

        ``targets`` uses the existing project convention ``[m,width]``;
        source marginals may be shared ``a[n]`` or candidate-specific
        ``a[n,width]``;
        initial standard potentials use ``[n,width]`` and ``[m,width]``.
        ``x`` may be either a shared source ``[n,d]`` or one source support
        per candidate ``[width,n,d]``.  The latter is required for real
        changing-support streams.  ``retirement_mode='fixed_width'`` retains
        and updates every lane until all candidates pass.  ``logical_mask``
        freezes verified lanes but retains the original physical width, while
        ``physical_compaction`` removes verified lanes.  The historical
        ``active_compaction`` keyword remains a mutually exclusive alias for
        the fixed-width/physical-compaction pair.
        """

        mode = _resolve_retirement_mode(
            retirement_mode=retirement_mode,
            active_compaction=active_compaction,
        )
        if mode == "logical_mask" and self.config.reuse_packed_buffers:
            raise ValueError(
                "logical_mask currently requires reuse_packed_buffers=False"
            )

        if ys.ndim != 3:
            raise ValueError("ys must have shape [width,m,d]")
        width, m, dimension = ys.shape
        if x.ndim == 2 and x.shape[1] == dimension:
            n = int(x.shape[0])
        elif x.ndim == 3 and x.shape[0] == width and x.shape[2] == dimension:
            n = int(x.shape[1])
        else:
            raise ValueError(
                "x must have shape [n,d] or [width,n,d] with the same d as ys"
            )
        if targets.shape != (m, width):
            raise ValueError("targets must have shape [m,width]")
        if a.shape not in ((n,), (n, width)):
            raise ValueError("a must have shape [n] or [n,width]")
        if f_init is not None and f_init.shape != (n, width):
            raise ValueError("f_init must have shape [n,width]")
        if g_init is not None and g_init.shape != (m, width):
            raise ValueError("g_init must have shape [m,width]")
        if not x.is_cuda or not ys.is_cuda:
            raise ValueError("packed FlashSinkhorn requires CUDA point clouds")
        if any(value.device != x.device for value in (ys, a, targets)):
            raise ValueError("all packed problem tensors must share one device")

        def source_masses(index: int) -> torch.Tensor:
            return a if a.ndim == 1 else a[:, index]

        # The disjoint warmup gate can prove that this width cannot amortize
        # packed staging/audit.  Route before creating packed state or running
        # even a batch-level device synchronization; the single backend below
        # remains the validation authority for every routed candidate.
        if width < self.config.min_packed_width and mode != "logical_mask":
            serial_results = tuple(
                self._single.solve(
                    x if x.ndim == 2 else x[index],
                    ys[index],
                    source_masses(index),
                    targets[:, index],
                    f_init=None if f_init is None else f_init[:, index],
                    g_init=None if g_init is None else g_init[:, index],
                )
                for index in range(width)
            )
            return FlashPackedWindowResult(
                results=tuple(
                    replace(
                        result,
                        solver_name=f"{self.name}+prefiltered-single",
                    )
                    for result in serial_results
                ),
                verified_first_passage_iterations=tuple(
                    int(result.n_iters) if result.converged else None
                    for result in serial_results
                ),
                retirement_mode=mode,
                physical_width_trace=(),
                logical_live_width_trace=(),
                active_width_trace=(),
                physical_candidate_slots=0,
                logical_live_candidate_slots=0,
                frozen_but_computed_slots=0,
                candidate_slots=0,
                packed_iterations=0,
                serial_tail_candidates=width,
                compaction_events=0,
                audit_events=sum(result.audits for result in serial_results),
                screen_pass_candidate_count=0,
                correction_seconds=0.0,
                audit_seconds=0.0,
                compaction_seconds=0.0,
                serial_tail_seconds=0.0,
                release_verifier_candidate_count=0,
                release_verified_candidate_count=0,
                release_verifier_seconds=0.0,
                logical_mask_applications=0,
                logical_mask_seconds=0.0,
                synchronization_seconds=0.0,
                audit_iterations=(),
                screen_positive_original_ids_trace=(),
                verifier_candidate_original_ids_trace=(),
                verified_release_original_ids_trace=(),
                duplicate_successful_release_verifications=0,
                min_packed_width=self.config.min_packed_width,
                backend_name=self.name,
            )
        if torch.any(a <= 0) or torch.any(targets <= 0):
            raise ValueError("FlashSinkhorn requires strictly positive marginals")
        for index in range(width):
            _validate_point_problem(
                x if x.ndim == 2 else x[index],
                ys[index],
                source_masses(index),
                targets[:, index],
            )

        x_work = x.float().contiguous()
        ys_active = ys.float().contiguous()
        a_work = (
            a.float().unsqueeze(0).expand(width, -1).contiguous()
            if a.ndim == 1
            else a.T.float().contiguous()
        )
        targets_active = targets.T.float().contiguous()
        log_a = torch.log(a_work)
        log_targets = torch.log(targets_active)
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=-1)
        alpha_active = (
            alpha.unsqueeze(0).expand(width, -1).clone()
            if x_work.ndim == 2
            else alpha.contiguous()
        )
        beta_active = self.config.cost_scale * torch.sum(
            ys_active * ys_active, dim=2
        )
        f_hat = (
            -alpha_active.clone()
            if f_init is None
            else f_init.T.float().contiguous() - alpha_active
        )
        g_hat = (
            -beta_active.clone()
            if g_init is None
            else g_init.T.float().contiguous() - beta_active
        )
        scratch_f = (
            torch.empty_like(f_hat) if self.config.reuse_packed_buffers else None
        )
        scratch_g = (
            torch.empty_like(g_hat) if self.config.reuse_packed_buffers else None
        )
        active_ids = torch.arange(width, device=x.device, dtype=torch.long)
        live_mask = torch.ones(width, device=x.device, dtype=torch.bool)
        live_flags = [True] * width
        logical_live_count = width
        outputs: list[FlashSinkhornResult | None] = [None] * width
        verified_first_passage: list[int | None] = [None] * width
        successful_verification_counts = [0] * width
        audit_counts = [0] * width
        width_trace: list[int] = []
        logical_live_width_trace: list[int] = []
        candidate_slots = 0
        logical_live_candidate_slots = 0
        frozen_but_computed_slots = 0
        packed_iterations = 0
        serial_tail_candidates = 0
        compactions = 0
        audit_events = 0
        screen_pass_candidate_count = 0
        correction_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        compaction_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        serial_tail_seconds = 0.0
        audit_seconds = 0.0
        release_verifier_candidate_count = 0
        release_verified_candidate_count = 0
        release_verifier_seconds = 0.0
        logical_mask_applications = 0
        logical_mask_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        audit_iterations: list[int] = []
        screen_positive_original_ids_trace: list[tuple[int, ...]] = []
        verifier_candidate_original_ids_trace: list[tuple[int, ...]] = []
        verified_release_original_ids_trace: list[tuple[int, ...]] = []
        duplicate_successful_release_verifications = 0

        if self.config.initial_marginal_audit:
            audit_started = time.perf_counter()
            rows, columns = self._audit(
                x_work, ys_active, a_work, targets_active, f_hat, g_hat
            )
            audit_seconds += time.perf_counter() - audit_started
            audit_events += 1
            for original in active_ids.tolist():
                audit_counts[original] += 1
        else:
            # Throughput route for proposals whose direct-accept probability
            # has already been established as negligible.  Infinity forces at
            # least one correction block; every eventual retirement still
            # passes the configured audit and upstream release verifier.
            rows = [math.inf] * width
            columns = [math.inf] * width
        fresh_audit = self.config.initial_marginal_audit

        iterations = 0
        while logical_live_count:
            if fresh_audit:
                screened_passed = [
                    index
                    for index, (row, column) in enumerate(zip(rows, columns))
                    if live_flags[index]
                    and max(row, column) <= self.config.marginal_tolerance
                ]
                audit_iterations.append(iterations)
                screen_positive_original_ids_trace.append(
                    tuple(int(active_ids[index]) for index in screened_passed)
                )
            else:
                screened_passed = []
            screen_pass_candidate_count += len(screened_passed)
            # A fixed-width window performs no early per-lane release checks.
            # Logical and physical retirement share the same per-lane verifier.
            verification_candidates = _release_verification_candidates(
                screened_passed,
                [int(value) for value in active_ids.tolist()],
                verified_first_passage,
                retirement_mode=mode,
            )
            if fresh_audit:
                verifier_candidate_original_ids_trace.append(
                    tuple(int(active_ids[index]) for index in verification_candidates)
                )
            if verification_candidates:
                verifier_started = time.perf_counter()
                passed = self._verify_release_candidates(
                    verification_candidates,
                    x_work,
                    ys_active,
                    a_work,
                    targets_active,
                    f_hat,
                    g_hat,
                    rows,
                    columns,
                )
                release_verifier_seconds += time.perf_counter() - verifier_started
            else:
                passed = []
            release_verifier_candidate_count += len(verification_candidates)
            release_verified_candidate_count += len(passed)
            for local in passed:
                original = int(active_ids[local])
                successful_verification_counts[original] += 1
                if successful_verification_counts[original] > 1:
                    duplicate_successful_release_verifications += 1
                if verified_first_passage[original] is None:
                    verified_first_passage[original] = iterations
            if self.config.batched_marginal_audit:
                for local in screened_passed:
                    audit_counts[int(active_ids[local])] += 1

            release_passed = (
                passed
                if mode != "fixed_width" or len(passed) == int(active_ids.numel())
                else []
            )
            if fresh_audit:
                verified_release_original_ids_trace.append(
                    tuple(int(active_ids[index]) for index in release_passed)
                )
            fresh_audit = False
            if release_passed:
                for local in release_passed:
                    original = int(active_ids[local])
                    outputs[original] = self._result(
                        f_hat[local],
                        g_hat[local],
                        alpha_active[local],
                        beta_active[local],
                        a_work[local],
                        targets_active[local],
                        iterations=iterations,
                        audits=audit_counts[original],
                        row_residual=rows[local],
                        column_residual=columns[local],
                        converged=True,
                    )
                if mode == "fixed_width":
                    logical_live_count = 0
                    active_ids = active_ids[:0]
                    break
                if mode == "logical_mask":
                    for local in release_passed:
                        live_flags[local] = False
                    live_mask[torch.tensor(release_passed, device=x.device)] = False
                    logical_live_count -= len(release_passed)
                    if logical_live_count == 0:
                        break
                else:
                    if len(release_passed) == int(active_ids.numel()):
                        logical_live_count = 0
                        active_ids = active_ids[:0]
                        break
                    compaction_start = torch.cuda.Event(enable_timing=True)
                    compaction_end = torch.cuda.Event(enable_timing=True)
                    compaction_start.record()
                    keep_mask = torch.ones(
                        active_ids.numel(), device=x.device, dtype=torch.bool
                    )
                    keep_mask[
                        torch.tensor(release_passed, device=x.device)
                    ] = False
                    active_ids = active_ids[keep_mask]
                    x_work = (
                        x_work
                        if x_work.ndim == 2
                        else x_work[keep_mask].contiguous()
                    )
                    ys_active = ys_active[keep_mask].contiguous()
                    targets_active = targets_active[keep_mask].contiguous()
                    a_work = a_work[keep_mask].contiguous()
                    log_a = log_a[keep_mask].contiguous()
                    log_targets = log_targets[keep_mask].contiguous()
                    alpha_active = alpha_active[keep_mask].contiguous()
                    beta_active = beta_active[keep_mask].contiguous()
                    f_hat = f_hat[keep_mask].contiguous()
                    g_hat = g_hat[keep_mask].contiguous()
                    live_mask = live_mask[keep_mask].contiguous()
                    live_flags = [True] * int(active_ids.numel())
                    logical_live_count = int(active_ids.numel())
                    if self.config.reuse_packed_buffers:
                        # A physical compaction changes the candidate-major
                        # shape; keep the ping-pong invariant by compacting
                        # the two scratch buffers exactly once per event.
                        assert scratch_f is not None and scratch_g is not None
                        scratch_f = scratch_f[keep_mask].contiguous()
                        scratch_g = scratch_g[keep_mask].contiguous()
                    rows = [
                        value
                        for index, value in enumerate(rows)
                        if index not in release_passed
                    ]
                    columns = [
                        value
                        for index, value in enumerate(columns)
                        if index not in release_passed
                    ]
                    compaction_end.record()
                    compaction_event_pairs.append((compaction_start, compaction_end))
                    compactions += 1

            if (
                mode == "physical_compaction"
                and active_ids.numel() < self.config.min_packed_width
            ):
                serial_tail_candidates += int(active_ids.numel())
                serial_tail_started = time.perf_counter()
                for local, original_tensor in enumerate(active_ids):
                    original = int(original_tensor)
                    single = self._single.solve(
                        x_work if x_work.ndim == 2 else x_work[local],
                        ys_active[local],
                        a_work[local],
                        targets_active[local],
                        f_init=f_hat[local] + alpha_active[local],
                        g_init=g_hat[local] + beta_active[local],
                    )
                    outputs[original] = replace(
                        single,
                        n_iters=iterations + single.n_iters,
                        audits=audit_counts[original] + single.audits,
                        solver_name=f"{self.name}+single-tail",
                    )
                    if single.converged:
                        verified_first_passage[original] = iterations + single.n_iters
                torch.cuda.synchronize(x.device)
                serial_tail_seconds += time.perf_counter() - serial_tail_started
                logical_live_count = 0
                active_ids = active_ids[:0]
                break

            if iterations >= self.config.max_iterations:
                break
            physical_width = int(active_ids.numel())
            width_trace.append(physical_width)
            logical_live_width_trace.append(logical_live_count)
            candidate_slots += physical_width
            logical_live_candidate_slots += logical_live_count
            frozen_but_computed_slots += physical_width - logical_live_count
            correction_start = torch.cuda.Event(enable_timing=True)
            correction_end = torch.cuda.Event(enable_timing=True)
            correction_start.record()
            previous_f_hat = f_hat
            previous_g_hat = g_hat
            next_f_hat, next_g_hat = self.shifted_step(
                x_work,
                ys_active,
                log_a,
                log_targets,
                f_hat,
                g_hat,
                scratch_f=scratch_f,
                scratch_g=scratch_g,
            )
            correction_end.record()
            correction_event_pairs.append((correction_start, correction_end))
            if mode == "logical_mask" and logical_live_count < physical_width:
                logical_mask_start = torch.cuda.Event(enable_timing=True)
                logical_mask_end = torch.cuda.Event(enable_timing=True)
                logical_mask_start.record()
                next_f_hat = _freeze_retired_potentials(
                    next_f_hat, previous_f_hat, live_mask
                )
                next_g_hat = _freeze_retired_potentials(
                    next_g_hat, previous_g_hat, live_mask
                )
                logical_mask_end.record()
                logical_mask_event_pairs.append(
                    (logical_mask_start, logical_mask_end)
                )
                logical_mask_applications += 1
            f_hat, g_hat = next_f_hat, next_g_hat
            iterations += 1
            packed_iterations += 1
            should_audit = (
                iterations % self.config.check_every == 0
                or iterations == self.config.max_iterations
            )
            if should_audit:
                audit_started = time.perf_counter()
                rows, columns = self._audit(
                    x_work, ys_active, a_work, targets_active, f_hat, g_hat
                )
                audit_seconds += time.perf_counter() - audit_started
                audit_events += 1
                fresh_audit = True
                for local, original in enumerate(active_ids.tolist()):
                    if live_flags[local]:
                        audit_counts[original] += 1

        if logical_live_count:
            if self.config.raise_on_nonconvergence:
                failed = [
                    int(active_ids[local])
                    for local, is_live in enumerate(live_flags)
                    if is_live
                ]
                raise FlashSinkhornConvergenceError(
                    "packed FlashSinkhorn marginal gate failed after "
                    f"{iterations} iterations for candidates {failed}"
                )
            audit_started = time.perf_counter()
            rows, columns = self._audit(
                x_work, ys_active, a_work, targets_active, f_hat, g_hat
            )
            audit_seconds += time.perf_counter() - audit_started
            for local, original_tensor in enumerate(active_ids):
                if not live_flags[local]:
                    continue
                original = int(original_tensor)
                outputs[original] = self._result(
                    f_hat[local],
                    g_hat[local],
                    alpha_active[local],
                    beta_active[local],
                    a_work[local],
                    targets_active[local],
                    iterations=iterations,
                    audits=audit_counts[original] + 1,
                    row_residual=rows[local],
                    column_residual=columns[local],
                    converged=False,
                )
        if any(value is None for value in outputs):
            raise RuntimeError("packed FlashSinkhorn lost a candidate result")
        # The endpoint runner synchronizes immediately after solve_batch.  Do
        # it here so CUDA-event telemetry is complete without introducing an
        # additional synchronization into the measured path.
        synchronization_started = time.perf_counter()
        torch.cuda.synchronize(x.device)
        synchronization_seconds = time.perf_counter() - synchronization_started
        correction_seconds = sum(
            start.elapsed_time(end) for start, end in correction_event_pairs
        ) / 1000.0
        compaction_seconds = sum(
            start.elapsed_time(end) for start, end in compaction_event_pairs
        ) / 1000.0
        logical_mask_seconds = sum(
            start.elapsed_time(end) for start, end in logical_mask_event_pairs
        ) / 1000.0
        return FlashPackedWindowResult(
            results=tuple(value for value in outputs if value is not None),
            verified_first_passage_iterations=tuple(verified_first_passage),
            retirement_mode=mode,
            physical_width_trace=tuple(width_trace),
            logical_live_width_trace=tuple(logical_live_width_trace),
            active_width_trace=tuple(width_trace),
            physical_candidate_slots=candidate_slots,
            logical_live_candidate_slots=logical_live_candidate_slots,
            frozen_but_computed_slots=frozen_but_computed_slots,
            candidate_slots=candidate_slots,
            packed_iterations=packed_iterations,
            serial_tail_candidates=serial_tail_candidates,
            compaction_events=compactions,
            audit_events=audit_events,
            screen_pass_candidate_count=screen_pass_candidate_count,
            correction_seconds=correction_seconds,
            audit_seconds=audit_seconds,
            compaction_seconds=compaction_seconds,
            serial_tail_seconds=serial_tail_seconds,
            release_verifier_candidate_count=release_verifier_candidate_count,
            release_verified_candidate_count=release_verified_candidate_count,
            release_verifier_seconds=release_verifier_seconds,
            logical_mask_applications=logical_mask_applications,
            logical_mask_seconds=logical_mask_seconds,
            synchronization_seconds=synchronization_seconds,
            audit_iterations=tuple(audit_iterations),
            screen_positive_original_ids_trace=tuple(
                screen_positive_original_ids_trace
            ),
            verifier_candidate_original_ids_trace=tuple(
                verifier_candidate_original_ids_trace
            ),
            verified_release_original_ids_trace=tuple(
                verified_release_original_ids_trace
            ),
            duplicate_successful_release_verifications=(
                duplicate_successful_release_verifications
            ),
            min_packed_width=self.config.min_packed_width,
            backend_name=self.name,
        )


__all__ = [
    "FlashPackedConfig",
    "FlashPackedFirstPassageProfile",
    "FlashPackedWindowResult",
    "FlashSinkhornPackedBackend",
    "RetirementMode",
]
