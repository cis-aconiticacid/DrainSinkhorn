#!/usr/bin/env python3
"""Prepare deterministic real Packer19 temporal supports for repeated EOT.

The script never synthesizes or duplicates observations.  It normalizes the
real count matrix, learns a shared sparse SVD embedding, and chooses the same
number of real cells from each requested embryo-time bin by a stable hash of
the published cell identifier.  The output is deliberately self-describing:
the source digest, selected cell-id digests, schema, timings, and every array
digest are recorded next to the compressed NumPy archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def stable_take(indices: np.ndarray, cell_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    scored: list[tuple[bytes, int]] = []
    for index in indices.tolist():
        key = f"{seed}:{cell_ids[index]}".encode("utf-8")
        scored.append((hashlib.blake2b(key, digest_size=16).digest(), int(index)))
    scored.sort()
    selected = np.asarray([index for _, index in scored[:n]], dtype=np.int64)
    if selected.size != n or np.unique(selected).size != n:
        raise RuntimeError("stable selection duplicated or lost real cells")
    return selected


def parse_bins(raw: str, available: list[int]) -> list[int]:
    if not raw:
        return available
    requested = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--bins contains duplicates")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"requested embryo bins are absent: {missing}")
    if requested != sorted(requested):
        raise ValueError("--bins must be in increasing temporal order")
    return requested


def stable_temporal_order(times: np.ndarray, cell_ids: np.ndarray, seed: int) -> np.ndarray:
    """Order real cells by time and use a stable hash only to break ties."""
    tie_breaks = np.empty(times.size, dtype="S16")
    for index, cell_id in enumerate(cell_ids.astype(str)):
        tie_breaks[index] = hashlib.blake2b(
            f"{seed}:{cell_id}".encode("utf-8"), digest_size=16
        ).digest()
    return np.lexsort((tie_breaks, times)).astype(np.int64)


def farthest_first_medoids(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Deterministic Gonzalez representatives and nearest-cell assignment."""
    if count <= 0 or count > points.shape[0]:
        raise ValueError("landmark count must lie in [1, support size]")
    points64 = np.asarray(points, dtype=np.float64)
    centre = np.mean(points64, axis=0)
    first = int(np.argmin(np.sum((points64 - centre) ** 2, axis=1)))
    medoids = np.empty(count, dtype=np.int64)
    medoids[0] = first
    minimum = np.sum((points64 - points64[first]) ** 2, axis=1)
    for position in range(1, count):
        selected = int(np.argmax(minimum))
        medoids[position] = selected
        candidate = np.sum((points64 - points64[selected]) ** 2, axis=1)
        minimum = np.minimum(minimum, candidate)
    if np.unique(medoids).size != count:
        raise RuntimeError("farthest-first selection produced duplicate representatives")
    distances = np.empty((points.shape[0], count), dtype=np.float32)
    block = 64
    for start in range(0, count, block):
        stop = min(start + block, count)
        delta = points[:, None, :] - points[medoids[start:stop]][None, :, :]
        distances[:, start:stop] = np.sum(delta * delta, axis=2, dtype=np.float32)
    assignment = np.argmin(distances, axis=1).astype(np.int32)
    nearest = distances[np.arange(points.shape[0]), assignment]
    cluster_counts = np.bincount(assignment, minlength=count)
    if np.any(cluster_counts == 0):
        raise RuntimeError("a selected medoid received no real cells")
    return medoids, assignment, {
        "mean_squared_radius": float(np.mean(nearest)),
        "p95_squared_radius": float(np.quantile(nearest, 0.95)),
        "max_squared_radius": float(np.max(nearest)),
        "minimum_cluster_size": int(np.min(cluster_counts)),
        "maximum_cluster_size": int(np.max(cluster_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--time-field", default="embryo.time.bin")
    parser.add_argument("--cell-id-field", default="cell")
    parser.add_argument("--label-field", default="cell.type")
    parser.add_argument("--support-mode", choices=("bins", "sliding"), default="bins")
    parser.add_argument("--bins", default="")
    parser.add_argument("--n", type=int, default=0, help="zero chooses the largest common multiple of world size")
    parser.add_argument("--width", type=int, default=0, help="candidate width for sliding supports")
    parser.add_argument("--stride", type=int, default=0, help="real cells advanced per sliding support")
    parser.add_argument("--start", type=int, default=0, help="offset into the stable temporal cell order")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--hvg", type=int, default=4000)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--landmarks", type=int, default=0)
    parser.add_argument("--target-sum", type=float, default=1.0e4)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    started = time.perf_counter()
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"missing source: {source}")
    observed_md5 = file_digest(source, "md5")
    if observed_md5 != args.expected_md5.lower():
        raise SystemExit(f"source MD5 mismatch: {observed_md5}")
    source_sha256 = file_digest(source)

    loaded_started = time.perf_counter()
    adata = ad.read_h5ad(source)
    load_seconds = time.perf_counter() - loaded_started
    for field in (args.time_field, args.cell_id_field, args.label_field):
        if field not in adata.obs:
            raise SystemExit(f"required obs field is absent: {field}")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise SystemExit("empty H5AD")
    counts = adata.X
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(np.asarray(counts))
    else:
        counts = counts.tocsr(copy=True)
    counts = counts.astype(np.float64, copy=False)
    if counts.nnz == 0 or not np.isfinite(counts.data).all() or np.min(counts.data) < 0:
        raise SystemExit("expression matrix is empty, negative, or non-finite")
    integer_fraction = float(np.mean(np.isclose(counts.data, np.round(counts.data))))
    if integer_fraction < 0.999:
        raise SystemExit(
            f"expected raw nonnegative counts, observed integer fraction {integer_fraction:.6f}"
        )

    preprocess_started = time.perf_counter()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(row_sums <= 0):
        raise SystemExit("zero-library cells are not admissible")
    normalized = counts.multiply((float(args.target_sum) / row_sums)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    hvg = min(int(args.hvg), adata.n_vars)
    if hvg < 2:
        raise SystemExit("at least two genes are required")
    hvg_indices = np.argsort(variance, kind="stable")[-hvg:]
    hvg_indices.sort()
    matrix = normalized[:, hvg_indices]
    components = min(int(args.components), matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 2:
        raise SystemExit("at least two embedding components are required")
    reducer = TruncatedSVD(
        n_components=components,
        n_iter=7,
        random_state=int(args.seed),
        algorithm="randomized",
    )
    features = reducer.fit_transform(matrix).astype(np.float32)
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_std = features.std(axis=0, dtype=np.float64)
    feature_std = np.maximum(feature_std, 1.0e-8)
    features = ((features - feature_mean) / feature_std).astype(np.float32)
    preprocess_seconds = time.perf_counter() - preprocess_started

    times = np.asarray(adata.obs[args.time_field], dtype=np.float64)
    available_bins = sorted(int(value) for value in np.unique(times))
    cell_ids = np.asarray(adata.obs[args.cell_id_field].astype(str))
    if np.unique(cell_ids).size != cell_ids.size:
        raise SystemExit("cell identifiers must be unique")
    labels_text = np.asarray(adata.obs[args.label_field].fillna("NA").astype(str))
    label_names = sorted(str(value) for value in np.unique(labels_text))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    labels_all = np.asarray([label_to_id[value] for value in labels_text], dtype=np.int32)

    selected_bins: list[int] | None
    counts_by_bin: list[int] | None
    support_time_summary: list[dict[str, float]]
    if args.support_mode == "bins":
        selected_bins = parse_bins(args.bins, available_bins)
        if len(selected_bins) < 3:
            raise SystemExit("at least three ordered time supports are required")
        indices_by_bin = [np.flatnonzero(times == value) for value in selected_bins]
        counts_by_bin = [int(values.size) for values in indices_by_bin]
        common = min(counts_by_bin)
        if args.n:
            n = int(args.n)
            if n > common:
                raise SystemExit(f"requested n={n} exceeds smallest real bin {common}")
        else:
            n = common - common % int(args.world_size)
        selected_indices = [
            stable_take(values, cell_ids, n, int(args.seed)) for values in indices_by_bin
        ]
    else:
        selected_bins = None
        counts_by_bin = None
        n = int(args.n)
        width = int(args.width)
        stride = int(args.stride)
        start = int(args.start)
        if n <= 0 or width <= 0 or stride <= 0 or start < 0:
            raise SystemExit("sliding mode requires positive --n/--width/--stride and nonnegative --start")
        support_count = width + 2
        stop = start + (support_count - 1) * stride + n
        if stop > adata.n_obs:
            raise SystemExit(
                f"sliding supports require ordered cells through {stop}, but only {adata.n_obs} exist"
            )
        order = stable_temporal_order(times, cell_ids, int(args.seed))
        selected_indices = [
            order[start + offset * stride : start + offset * stride + n]
            for offset in range(support_count)
        ]
    if n <= 0 or n % int(args.world_size):
        raise SystemExit("n must be a positive multiple of world size")
    support_time_summary = [
        {
            "min": float(np.min(times[values])),
            "median": float(np.median(times[values])),
            "max": float(np.max(times[values])),
        }
        for values in selected_indices
    ]
    adjacent_overlap = [
        float(
            np.intersect1d(selected_indices[left], selected_indices[left + 1]).size
            / n
        )
        for left in range(len(selected_indices) - 1)
    ]
    supports = np.stack([features[values] for values in selected_indices], axis=0)
    labels = np.stack([labels_all[values] for values in selected_indices], axis=0)
    selected_id_sha256 = [strings_digest(cell_ids[values]) for values in selected_indices]
    selected_index_sha256 = [array_digest(values) for values in selected_indices]
    if not np.isfinite(supports).all():
        raise SystemExit("non-finite prepared supports")

    landmark_indices: np.ndarray | None = None
    landmark_assignment: np.ndarray | None = None
    landmark_metadata: dict[str, Any] | None = None
    if int(args.landmarks) > 0:
        landmark_started = time.perf_counter()
        landmark_indices, landmark_assignment, coverage = farthest_first_medoids(
            supports[1], int(args.landmarks)
        )
        landmark_metadata = {
            "reference_support_index": 1,
            "count": int(args.landmarks),
            "selection": "deterministic Gonzalez farthest-first real medoids",
            "assignment": "nearest medoid in standardized real count-derived SVD space",
            "coverage": coverage,
            "seconds": time.perf_counter() - landmark_started,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = {
        "supports": supports,
        "labels": labels,
        "support_times": np.asarray(
            [entry["median"] for entry in support_time_summary], dtype=np.float64
        ),
        "label_names": np.asarray(label_names, dtype=str),
        "hvg_indices": hvg_indices.astype(np.int64),
        "feature_mean": feature_mean.astype(np.float64),
        "feature_std": feature_std.astype(np.float64),
        "explained_variance_ratio": reducer.explained_variance_ratio_.astype(np.float64),
    }
    if landmark_indices is not None and landmark_assignment is not None:
        output_arrays["reference_landmark_indices"] = landmark_indices
        output_arrays["reference_landmark_assignment"] = landmark_assignment
    np.savez_compressed(args.output, **output_arrays)
    output_sha256 = file_digest(args.output)
    metadata: dict[str, Any] = {
        "dataset": "Packer et al. 2019 C. elegans embryogenesis scRNA-seq",
        "source_record": "Zenodo 7496490",
        "is_synthetic": False,
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_md5": observed_md5,
        "source_sha256": source_sha256,
        "preparation_script_sha256": file_digest(Path(__file__).resolve()),
        "source_shape": [int(adata.n_obs), int(adata.n_vars)],
        "source_sparse_nnz": int(counts.nnz),
        "source_integer_fraction": integer_fraction,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "time_field": args.time_field,
        "support_mode": args.support_mode,
        "time_bins": selected_bins,
        "source_counts_by_time_bin": (
            dict(zip(map(str, selected_bins), counts_by_bin))
            if selected_bins is not None and counts_by_bin is not None
            else None
        ),
        "selected_real_cells_per_support": n,
        "support_count": len(selected_indices),
        "candidate_width": len(selected_indices) - 2,
        "support_time_summary": support_time_summary,
        "adjacent_real_cell_overlap_fraction": adjacent_overlap,
        "landmarks": landmark_metadata,
        "selected_cell_id_sha256": selected_id_sha256,
        "selected_index_sha256": selected_index_sha256,
        "cell_id_field": args.cell_id_field,
        "label_field": args.label_field,
        "label_names": label_names,
        "processing": {
            "library_size_target": float(args.target_sum),
            "transform": "log1p(library-size normalized raw counts)",
            "hvg_count": hvg,
            "hvg_selection": "largest sparse population variance; stable tie ordering",
            "embedding": "TruncatedSVD followed by global component standardization",
            "components": components,
            "svd_n_iter": 7,
            "seed": int(args.seed),
            "selection": (
                "smallest stable BLAKE2b(seed, published cell id) within each time bin; no replacement"
                if args.support_mode == "bins"
                else "contiguous slices of stable embryo-time order; no replacement within a support"
            ),
            "sliding_order": "embryo time with stable BLAKE2b cell-id tie break",
            "sliding_stride": int(args.stride) if args.support_mode == "sliding" else None,
            "sliding_start": int(args.start) if args.support_mode == "sliding" else None,
            "world_size_multiple": int(args.world_size),
        },
        "array_sha256": {
            "supports": array_digest(supports),
            "labels": array_digest(labels),
            "support_times": array_digest(
                np.asarray(
                    [entry["median"] for entry in support_time_summary], dtype=np.float64
                )
            ),
            "hvg_indices": array_digest(hvg_indices.astype(np.int64)),
            **(
                {
                    "reference_landmark_indices": array_digest(landmark_indices),
                    "reference_landmark_assignment": array_digest(landmark_assignment),
                }
                if landmark_indices is not None and landmark_assignment is not None
                else {}
            ),
        },
        "output_npz_sha256": output_sha256,
        "timing_seconds": {
            "load": load_seconds,
            "normalize_hvg_svd": preprocess_seconds,
            "total": time.perf_counter() - started,
        },
        "scope": (
            "balanced temporal EOT supports and real cell-type handoff; this is not "
            "a complete growth-aware unbalanced Waddington-OT pipeline"
        ),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
