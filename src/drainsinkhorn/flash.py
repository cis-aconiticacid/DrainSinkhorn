"""Optional FlashSinkhorn point-cloud backend.

This module is intentionally an adapter around the external
``flash-sinkhorn`` package.  The package's fused Triton kernels work on point
clouds and squared-Euclidean costs in the log-potential domain; they do not
implement the public row-sharded ``K @ v`` contract in :mod:`.core`.

The adapter therefore exposes a truthful, higher-level solve API.  It is
single-device and point-cloud only.  ``solve_batch`` supports active-window
bookkeeping by solving only the requested candidates, but the current
upstream API has no candidate-batched kernel, so the candidates are launched
one at a time.  This is useful for correctness and composition experiments,
not yet a claim of packed FlashSinkhorn speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any

import torch

class FlashSinkhornUnavailableError(RuntimeError):
    """Raised when the optional FlashSinkhorn dependency cannot be loaded."""


class FlashSinkhornConvergenceError(RuntimeError):
    """Raised when measured marginals miss the configured tolerance."""


@dataclass(frozen=True)
class FlashSinkhornConfig:
    """Configuration for the measured fixed-epsilon FlashSinkhorn adapter.

    ``cost_scale`` means the cost is ``cost_scale * ||x-y||^2``.
    """

    epsilon: float
    marginal_tolerance: float = 1e-3
    max_iterations: int = 2_000
    check_every: int = 10
    cost_scale: float = 1.0
    allow_tf32: bool = False
    use_exp2: bool = True
    autotune: bool = True
    raise_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.marginal_tolerance) or self.marginal_tolerance <= 0:
            raise ValueError("marginal_tolerance must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.check_every < 1:
            raise ValueError("check_every must be positive")
        if not math.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("cost_scale must be finite and positive")


@dataclass(frozen=True)
class FlashSinkhornResult:
    """A solved point-cloud problem in both potential and scaling form.

    ``f`` and ``g`` are the upstream standard potentials.  ``u`` and ``v``
    use this project's convention ``diag(u) exp(-C/eps) diag(v)``.  The
    latter are gauge-fixed without changing the represented coupling.
    """

    f: torch.Tensor
    g: torch.Tensor
    log_u: torch.Tensor
    log_v: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    n_iters: int
    row_residual: float
    column_residual: float
    audits: int
    converged: bool
    solver_name: str


def _load_backend_functions() -> tuple[Any, Any, str]:
    """Load optional fused update/audit kernels without eager package import."""

    try:
        update_module = importlib.import_module(
            "flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid"
        )
        apply_module = importlib.import_module("flash_sinkhorn.kernels.apply_flash")
        fused_lse = getattr(update_module, "flashsinkhorn_lse_fused")
        apply_plan = getattr(apply_module, "apply_plan_vec_flashstyle")
        package = importlib.import_module("flash_sinkhorn")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise FlashSinkhornUnavailableError(
            "FlashSinkhorn is optional. Install it with "
            "`python3 -m pip install 'flash-sinkhorn>=0.3.3,<0.4'` "
            "on a CUDA environment (PyTorch >=2.5, Triton >=3.1)."
        ) from exc
    version = str(getattr(package, "__version__", "unknown"))
    return fused_lse, apply_plan, version


def _load_plan_matrix_function() -> Any:
    """Load the upstream matrix plan-apply kernel for multi-channel consumers."""

    try:
        module = importlib.import_module("flash_sinkhorn.kernels.apply_flash")
        return getattr(module, "apply_plan_mat_flashstyle")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise FlashSinkhornUnavailableError(
            "FlashSinkhorn matrix plan-apply requires "
            "flash-sinkhorn>=0.3.3,<0.4"
        ) from exc


def flashsinkhorn_available() -> bool:
    """Return whether the optional Python package can be imported."""

    try:
        _load_backend_functions()
    except FlashSinkhornUnavailableError:
        return False
    return True


def _validate_point_problem(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shapes [n, d] and [m, d]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    if a.ndim != 1 or a.shape[0] != x.shape[0]:
        raise ValueError("a must have shape [x.shape[0]]")
    if b.ndim != 1 or b.shape[0] != y.shape[0]:
        raise ValueError("b must have shape [y.shape[0]]")
    tensors = (x, y, a, b)
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("point clouds and marginals must use floating dtypes")
    if any(value.device != x.device for value in tensors):
        raise ValueError("point clouds and marginals must share a device")
    if torch.any(a <= 0) or torch.any(b <= 0):
        raise ValueError("FlashSinkhorn requires strictly positive marginals")
    source_mass = float(a.detach().sum())
    target_mass = float(b.detach().sum())
    if not math.isclose(source_mass, target_mass, rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("balanced OT requires source and target masses to match")


def _log_scalings_from_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert GeomLoss potentials to gauge-fixed log scalings."""

    # The upstream default convention is
    # P_ij = a_i b_j exp((f_i + g_j - C_ij) / epsilon).
    # Thus u=a*exp(f/epsilon), v=b*exp(g/epsilon).  Center log(v) and apply
    # the inverse factor to u so that the represented plan is unchanged,
    # without materializing an unnecessarily large unnormalized exp(g/eps).
    log_v = torch.log(b) + g / epsilon
    log_gauge = torch.mean(log_v)
    return torch.log(a) + f / epsilon + log_gauge, log_v - log_gauge


def _scalings_from_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert GeomLoss potentials to best-effort linear scalings."""

    log_u, log_v = _log_scalings_from_potentials(f, g, a, b, epsilon)
    return torch.exp(log_u), torch.exp(log_v)


class FlashSinkhornBackend:
    """Optional fused Triton backend for squared-Euclidean point-cloud OT.

    The upstream kernels are loaded lazily. This class exposes the
    single-device log-domain point-cloud path used by DrainSinkhorn. Use
    :meth:`solve` for one problem or :meth:`solve_batch` for a candidate list.
    """

    distributed = False
    supports_generic_kernel = False
    supports_squared_euclidean = True

    def __init__(self, config: FlashSinkhornConfig):
        self.config = config
        self._fused_lse: Any | None = None
        self._apply_plan: Any | None = None
        self._apply_plan_matrix: Any | None = None
        self._version: str | None = None

    @property
    def version(self) -> str:
        self._ensure_functions()
        assert self._version is not None
        return self._version

    @property
    def name(self) -> str:
        return f"flash-sinkhorn/{self.version}-measured-alternating"

    def _ensure_functions(self) -> tuple[Any, Any]:
        if self._fused_lse is None or self._apply_plan is None:
            self._fused_lse, self._apply_plan, self._version = (
                _load_backend_functions()
            )
        return self._fused_lse, self._apply_plan

    def _ensure_plan_matrix_function(self) -> Any:
        if self._apply_plan_matrix is None:
            self._apply_plan_matrix = _load_plan_matrix_function()
        return self._apply_plan_matrix

    def _audit_shifted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_hat: torch.Tensor,
        g_hat: torch.Tensor,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[float, float]:
        """Measure both marginals through the matrix-free plan-apply kernel."""

        _, apply_plan = self._ensure_functions()
        row = apply_plan(
            x,
            y,
            f_hat,
            g_hat,
            log_a,
            log_b,
            torch.ones_like(b),
            eps=self.config.epsilon,
            axis=1,
            cost_scale=self.config.cost_scale,
            allow_tf32=self.config.allow_tf32,
            use_exp2=self.config.use_exp2,
        )
        column = apply_plan(
            x,
            y,
            f_hat,
            g_hat,
            log_a,
            log_b,
            torch.ones_like(a),
            eps=self.config.epsilon,
            axis=0,
            cost_scale=self.config.cost_scale,
            allow_tf32=self.config.allow_tf32,
            use_exp2=self.config.use_exp2,
        )
        return (
            float(torch.sum(torch.abs(row - a))),
            float(torch.sum(torch.abs(column - b))),
        )

    @torch.no_grad()
    def apply_transport_to_target_values(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        result: FlashSinkhornResult,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a verified transport plan to target-side values.

        This is the matrix-free consumer interface ``P @ h``.  A vector
        ``target_values[m]`` returns ``[n]`` through the upstream plan-vector
        kernel.  When a value matrix has the same channel count as the point
        dimension, the upstream matrix plan-apply computes ``[n,d]`` in one
        call.  Other ``[m,k]`` matrices fall back to one vector call per
        channel because FlashSinkhorn 0.3.3 constrains its matrix kernel to
        ``k=d``.  No path materializes the ``n x m`` plan.
        """

        _validate_point_problem(x, y, a, b)
        if not x.is_cuda:
            raise ValueError("FlashSinkhornBackend requires CUDA point clouds")
        if not result.converged:
            raise ValueError("transport application requires a converged result")
        if result.f.shape != a.shape or result.g.shape != b.shape:
            raise ValueError("result potentials do not match the point problem")
        if result.f.device != x.device or result.g.device != x.device:
            raise ValueError("result potentials must share the point-cloud device")
        if target_values.ndim not in (1, 2) or target_values.shape[0] != y.shape[0]:
            raise ValueError("target_values must have shape [m] or [m,k]")
        if target_values.device != x.device or not target_values.is_floating_point():
            raise ValueError("target_values must be floating point on the point-cloud device")
        if not bool(torch.isfinite(target_values).all()):
            raise ValueError("target_values must be finite")

        _, apply_plan = self._ensure_functions()
        x_work = x.float().contiguous()
        y_work = y.float().contiguous()
        a_work = a.float().contiguous()
        b_work = b.float().contiguous()
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=1)
        beta = self.config.cost_scale * torch.sum(y_work * y_work, dim=1)
        f_hat = result.f.float().contiguous() - alpha
        g_hat = result.g.float().contiguous() - beta
        log_a = torch.log(a_work)
        log_b = torch.log(b_work)

        def apply_one(value: torch.Tensor) -> torch.Tensor:
            return apply_plan(
                x_work,
                y_work,
                f_hat,
                g_hat,
                log_a,
                log_b,
                value.float().contiguous(),
                eps=self.config.epsilon,
                axis=1,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
            )

        if target_values.ndim == 1:
            return apply_one(target_values)
        if target_values.shape[1] == 0:
            return torch.empty(
                (x.shape[0], 0), device=x.device, dtype=torch.float32
            )
        if target_values.shape[1] == x.shape[1]:
            apply_matrix = self._ensure_plan_matrix_function()
            return apply_matrix(
                x_work,
                y_work,
                f_hat,
                g_hat,
                log_a,
                log_b,
                target_values.float().contiguous(),
                eps=self.config.epsilon,
                axis=1,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=self.config.autotune,
            )
        return torch.stack(
            [apply_one(target_values[:, column]) for column in range(target_values.shape[1])],
            dim=1,
        )

    @torch.no_grad()
    def solve(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        f_init: torch.Tensor | None = None,
        g_init: torch.Tensor | None = None,
    ) -> FlashSinkhornResult:
        """Solve one balanced squared-Euclidean OT problem.

        FlashSinkhorn requires CUDA tensors.  The explicit check keeps a
        missing/CPU install from being mistaken for a valid backend result.
        ``f_init`` and ``g_init`` are standard upstream potentials, not the
        shifted internal representation.
        """

        _validate_point_problem(x, y, a, b)
        if not x.is_cuda:
            raise ValueError("FlashSinkhornBackend requires CUDA point clouds")
        if f_init is not None and (f_init.shape != a.shape or f_init.device != x.device):
            raise ValueError("f_init must have shape [n] on the point-cloud device")
        if g_init is not None and (g_init.shape != b.shape or g_init.device != x.device):
            raise ValueError("g_init must have shape [m] on the point-cloud device")

        fused_lse, _ = self._ensure_functions()
        x_work = x.float().contiguous()
        y_work = y.float().contiguous()
        a_work = a.float().contiguous()
        b_work = b.float().contiguous()
        log_a = torch.log(a_work)
        log_b = torch.log(b_work)
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=1)
        beta = self.config.cost_scale * torch.sum(y_work * y_work, dim=1)
        f_hat = -alpha if f_init is None else f_init.float().contiguous() - alpha
        g_hat = -beta if g_init is None else g_init.float().contiguous() - beta

        row_residual, column_residual = self._audit_shifted(
            x_work, y_work, f_hat, g_hat, log_a, log_b, a_work, b_work
        )
        audits = 1
        converged = max(row_residual, column_residual) <= self.config.marginal_tolerance
        n_iters = 0
        while not converged and n_iters < self.config.max_iterations:
            f_hat = fused_lse(
                x_work,
                y_work,
                g_hat,
                log_b,
                self.config.epsilon,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=self.config.autotune,
            )
            g_hat = fused_lse(
                y_work,
                x_work,
                f_hat,
                log_a,
                self.config.epsilon,
                cost_scale=self.config.cost_scale,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=self.config.autotune,
            )
            n_iters += 1
            should_audit = (
                n_iters % self.config.check_every == 0
                or n_iters == self.config.max_iterations
            )
            if should_audit:
                row_residual, column_residual = self._audit_shifted(
                    x_work,
                    y_work,
                    f_hat,
                    g_hat,
                    log_a,
                    log_b,
                    a_work,
                    b_work,
                )
                audits += 1
                converged = (
                    max(row_residual, column_residual)
                    <= self.config.marginal_tolerance
                )

        f = f_hat + alpha
        g = g_hat + beta
        if not converged and self.config.raise_on_nonconvergence:
            raise FlashSinkhornConvergenceError(
                "FlashSinkhorn measured marginal gate failed after "
                f"{n_iters} iterations: row_l1={row_residual:.6g}, "
                f"column_l1={column_residual:.6g}, "
                f"tolerance={self.config.marginal_tolerance:.6g}"
            )
        log_u, log_v = _log_scalings_from_potentials(
            f, g, a, b, self.config.epsilon
        )
        u, v = torch.exp(log_u), torch.exp(log_v)
        return FlashSinkhornResult(
            f=f,
            g=g,
            log_u=log_u,
            log_v=log_v,
            u=u,
            v=v,
            n_iters=int(n_iters),
            row_residual=row_residual,
            column_residual=column_residual,
            audits=audits,
            converged=converged,
            solver_name=self.name,
        )

    @torch.no_grad()
    def project_target_potential(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        source_f: torch.Tensor,
    ) -> torch.Tensor:
        """Project a causal source potential onto a new target support.

        The preceding certified solve supplies ``source_f`` on an unchanged
        source support.  One target-side log-domain c-transform then gives a
        valid, causal initial ``g`` for the new target.  This is deliberately
        separate from upstream's raw ``f_init``/``g_init`` interface: a raw
        target potential is tied to the *previous* target points and need not
        be a helpful initialization after their support changes.

        The returned tensor uses the same standard-potential convention as
        :meth:`solve`'s ``g_init``.  Calling code must charge this operation to
        its proposal/window time; this helper only constructs the proposal.
        """

        _validate_point_problem(x, y, a, b)
        if not x.is_cuda:
            raise ValueError("FlashSinkhornBackend requires CUDA point clouds")
        if (
            source_f.shape != a.shape
            or source_f.device != x.device
            or not source_f.is_floating_point()
            or not bool(torch.isfinite(source_f).all())
        ):
            raise ValueError("source_f must be finite with shape [n] on the point-cloud device")

        fused_lse, _ = self._ensure_functions()
        x_work = x.float().contiguous()
        y_work = y.float().contiguous()
        a_work = a.float().contiguous()
        alpha = self.config.cost_scale * torch.sum(x_work * x_work, dim=1)
        beta = self.config.cost_scale * torch.sum(y_work * y_work, dim=1)
        f_hat = source_f.float().contiguous() - alpha
        g_hat = fused_lse(
            y_work,
            x_work,
            f_hat,
            torch.log(a_work),
            self.config.epsilon,
            cost_scale=self.config.cost_scale,
            allow_tf32=self.config.allow_tf32,
            use_exp2=self.config.use_exp2,
            autotune=self.config.autotune,
        )
        g = g_hat + beta
        if not bool(torch.isfinite(g).all()):
            raise FlashSinkhornConvergenceError(
                "target c-transform produced a non-finite potential"
            )
        return g

    @torch.no_grad()
    def solve_batch(
        self,
        x: torch.Tensor,
        ys: torch.Tensor,
        a: torch.Tensor,
        targets: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        g_init: torch.Tensor | None = None,
    ) -> tuple[FlashSinkhornResult, ...]:
        """Solve selected candidates from a point-cloud window.

        ``ys`` has shape ``[width, m, d]`` and ``targets`` has shape
        ``[m, width]``.  Only ``indices`` are launched.  This is an explicit
        loop because the upstream 0.3 API exposes no candidate-batched fused
        solver.  It is the composition hook needed before a true packed
        FlashSinkhorn kernel is added upstream or locally.
        """

        if ys.ndim != 3:
            raise ValueError("ys must have shape [width, m, d]")
        width, m, _dimension = ys.shape
        if targets.shape != (m, width):
            raise ValueError("targets must have shape [m, width]")
        if indices is None:
            active = torch.arange(width, device=ys.device, dtype=torch.long)
        else:
            if indices.ndim != 1 or indices.dtype != torch.long:
                raise ValueError("indices must be a one-dimensional torch.long tensor")
            active = indices.to(device=ys.device)
            if active.numel() and (int(active.min()) < 0 or int(active.max()) >= width):
                raise IndexError("active candidate index outside ys")
        if g_init is not None and g_init.shape != (m, width):
            raise ValueError("g_init must have shape [m, width]")

        results: list[FlashSinkhornResult] = []
        for candidate in active.tolist():
            candidate_g = None if g_init is None else g_init[:, candidate]
            results.append(
                self.solve(
                    x,
                    ys[candidate],
                    a,
                    targets[:, candidate],
                    g_init=candidate_g,
                )
            )
        return tuple(results)


__all__ = [
    "FlashSinkhornBackend",
    "FlashSinkhornConvergenceError",
    "FlashSinkhornConfig",
    "FlashSinkhornResult",
    "FlashSinkhornUnavailableError",
    "flashsinkhorn_available",
]
