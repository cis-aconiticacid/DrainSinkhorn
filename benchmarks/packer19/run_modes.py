#!/usr/bin/env python3
"""Run the public Packer19 endpoint across DrainSinkhorn execution modes.

This is the compact public runner.  The paper's immutable two-host result and
fail-closed analyzer are preserved under ``results/c80_lm`` and ``analyze.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drainsinkhorn import (
    DrainSinkhorn,
    DrainSinkhornConfig,
    FlashSinkhornBackend,
    FlashSinkhornConfig,
)

MODES = ("official", "fixed_width", "logical_mask", "physical_compaction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_prepared(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("prepared NPZ and its .metadata.json sidecar are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("is_synthetic") is not False:
        raise ValueError("the Packer19 endpoint requires real prepared data")
    if metadata.get("output_npz_sha256") != sha256_file(path):
        raise ValueError("prepared NPZ hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"supports", "labels", "label_names"}
    if not required.issubset(arrays):
        raise ValueError(f"prepared NPZ is missing {sorted(required - arrays.keys())}")
    return arrays, metadata


def robust_cost_scale(source: torch.Tensor, target: torch.Tensor) -> float:
    sample_n = min(2048, source.shape[0], target.shape[0])
    squared = torch.cdist(
        source[:sample_n].double(), target[:sample_n].double()
    ).square()
    median = float(torch.median(squared))
    if not math.isfinite(median) or median <= 0:
        raise ValueError("invalid real-data squared-distance scale")
    return 1.0 / median


@torch.no_grad()
def consume(
    single: FlashSinkhornBackend,
    sources: torch.Tensor,
    targets: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    masses: torch.Tensor,
    results: tuple,
    label_count: int,
) -> dict[str, object]:
    transitions: list[torch.Tensor] = []
    for index, result in enumerate(results):
        target_one_hot = F.one_hot(
            target_labels[index].long(), num_classes=label_count
        ).float()
        transported = single.apply_transport_to_target_values(
            sources[index],
            targets[index],
            masses,
            masses,
            result,
            target_one_hot,
        )
        source_one_hot = F.one_hot(
            source_labels[index].long(), num_classes=label_count
        ).to(transported.dtype)
        transitions.append(source_one_hot.T @ transported)
    stacked = torch.stack(transitions)
    return {
        "sha256": sha256_tensor(stacked),
        "mass": stacked.sum(dim=(1, 2)).cpu().tolist(),
        "same_label_mass": torch.diagonal(
            stacked, dim1=1, dim2=2
        ).sum(dim=1).cpu().tolist(),
    }


@torch.no_grad()
def run_one(
    mode: str,
    packed: DrainSinkhorn,
    single: FlashSinkhornBackend,
    sources: torch.Tensor,
    targets: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    masses: torch.Tensor,
    label_count: int,
) -> dict[str, object]:
    torch.cuda.reset_peak_memory_stats(sources.device)
    started = time.perf_counter()
    if mode == "official":
        results = tuple(
            single.solve(sources[index], targets[index], masses, masses)
            for index in range(sources.shape[0])
        )
        physical_slots = None
        logical_slots = None
        frozen_slots = None
        widths: list[int] = []
    else:
        window = packed.solve_batch(
            sources,
            targets,
            masses,
            masses[:, None].expand(-1, sources.shape[0]).contiguous(),
            retirement_mode=mode,
        )
        results = window.results
        physical_slots = window.physical_candidate_slots
        logical_slots = window.logical_live_candidate_slots
        frozen_slots = window.frozen_but_computed_slots
        widths = list(window.physical_width_trace)
    consumer = consume(
        single,
        sources,
        targets,
        source_labels,
        target_labels,
        masses,
        results,
        label_count,
    )
    torch.cuda.synchronize(sources.device)
    elapsed = time.perf_counter() - started
    return {
        "mode": mode,
        "endpoint_seconds": elapsed,
        "all_converged": all(result.converged for result in results),
        "max_two_sided_residual": max(
            max(result.row_residual, result.column_residual) for result in results
        ),
        "candidate_iterations": [result.n_iters for result in results],
        "physical_candidate_slots": physical_slots,
        "logical_live_candidate_slots": logical_slots,
        "frozen_but_computed_slots": frozen_slots,
        "physical_width_trace": widths,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "consumer": consumer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--support-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--solver-tolerance", type=float, default=9.5e-4)
    parser.add_argument("--check-every", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=800)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    arrays, metadata = load_prepared(args.data.resolve())
    stop = args.support_offset + args.width + 1
    if args.support_offset < 0 or stop > arrays["supports"].shape[0]:
        raise SystemExit("requested temporal window is absent")
    supports = torch.from_numpy(
        np.ascontiguousarray(
            arrays["supports"][args.support_offset:stop], dtype=np.float32
        )
    ).cuda()
    labels = torch.from_numpy(
        np.ascontiguousarray(
            arrays["labels"][args.support_offset:stop], dtype=np.int64
        )
    ).cuda()
    sources, targets = supports[:-1].contiguous(), supports[1:].contiguous()
    source_labels, target_labels = labels[:-1], labels[1:]
    masses = torch.full(
        (sources.shape[1],), 1.0 / sources.shape[1], device=sources.device
    )
    cost_scale = robust_cost_scale(sources[0], targets[0])

    single = FlashSinkhornBackend(
        FlashSinkhornConfig(
            epsilon=args.epsilon,
            marginal_tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            check_every=args.check_every,
            cost_scale=cost_scale,
            allow_tf32=False,
            autotune=False,
        )
    )
    packed = DrainSinkhorn(
        DrainSinkhornConfig(
            epsilon=args.epsilon,
            marginal_tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            check_every=args.check_every,
            cost_scale=cost_scale,
            allow_tf32=False,
            min_packed_width=1,
            block_m=64,
            block_n=128,
            block_k=16,
            num_warps=4,
            num_stages=2,
            batched_marginal_audit=True,
            row_only_batched_audit=True,
            initial_marginal_audit=False,
            reuse_packed_buffers=False,
        )
    )

    # Compile every path before timing.
    for mode in MODES:
        run_one(
            mode,
            packed,
            single,
            sources,
            targets,
            source_labels,
            target_labels,
            masses,
            len(arrays["label_names"]),
        )

    rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
        for order_index, mode in enumerate(order):
            row = run_one(
                mode,
                packed,
                single,
                sources,
                targets,
                source_labels,
                target_labels,
                masses,
                len(arrays["label_names"]),
            )
            row.update({"repeat": repeat, "order_index": order_index})
            rows.append(row)

    payload = {
        "schema": "drainsinkhorn.public.packer19_modes.v1",
        "data_sha256": sha256_file(args.data.resolve()),
        "metadata_sha256": sha256_file(
            args.data.with_suffix(args.data.suffix + ".metadata.json").resolve()
        ),
        "prepared_is_synthetic": metadata.get("is_synthetic"),
        "config": vars(args) | {"data": str(args.data), "output": str(args.output)},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
