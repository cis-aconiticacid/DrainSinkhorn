#!/usr/bin/env python3
"""Fail-closed two-host analysis for the C80-LM five-arm experiment.

The analyzer never edits a raw root.  It first proves the registered
O/W1/FW/LM/PC execution and LM/PC semantic parity, then computes the single
primary performance contrast T_LM/T_PC over 24 paired host/seed/GPU units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np


ARMS = ("official", "w1", "fw", "lm", "pc")
SEEDS = (1884470933, 55048408, 979986572)
GPUS = tuple(range(4))
REPEATS = tuple(range(3))
WIDTH = 8
PLAN_SCHEMA = "forgetot_c80_lm_five_arm_plan_v1"
RUN_SCHEMA = "forgetot_c80_lm_five_arm_run_v1"
CAMPAIGN_ID = "C80-LM"
METHODS = {
    "official": "official_cold",
    "w1": "local_cold_width_one",
    "fw": "local_cold_static",
    "lm": "local_cold_logical",
    "pc": "local_cold_active",
}
RETIREMENT_MODES = {
    "w1": "physical_compaction",
    "fw": "fixed_width",
    "lm": "logical_mask",
    "pc": "physical_compaction",
}
TRACE_KEYS = (
    "audit_iterations",
    "screen_positive_original_ids_trace",
    "verifier_candidate_original_ids_trace",
    "verified_release_original_ids_trace",
    "verified_first_passage_iterations",
    "logical_live_width_trace",
)
REQUIRED_NUMERIC_KEYS = (
    "ready_arrays_to_consumer_seconds",
    "correction_seconds",
    "audit_seconds",
    "release_verifier_seconds",
    "logical_mask_seconds",
    "compaction_seconds",
    "synchronization_seconds",
    "consumer_seconds",
    "unclassified_overhead_seconds",
    "gpu_energy_joules",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "physical_candidate_slots",
    "logical_live_candidate_slots",
    "frozen_but_computed_slots",
    "audit_events",
    "screen_pass_candidate_count",
    "release_verifier_candidate_count",
    "release_verified_candidate_count",
    "compaction_events",
    "logical_mask_applications",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"repeat row lacks required field {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def exact_int(row: dict[str, Any], key: str) -> int:
    value = finite(row, key)
    if value < 0 or value != int(value):
        raise ValueError(f"{key} is not a nonnegative integer")
    return int(value)


def verify_manifest(root: Path) -> int:
    """Verify every entry in the runner-created SHA256SUMS file."""

    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"missing SHA256SUMS: {root}")
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed SHA256SUMS row in {root}")
        path = root / match.group(2).removeprefix("./")
        if not path.is_file() or sha256_file(path) != match.group(1):
            raise ValueError(f"SHA256SUMS mismatch: {path}")
        count += 1
    if count == 0:
        raise ValueError(f"empty SHA256SUMS: {root}")
    return count


def run_metadata(root: Path) -> dict[str, str]:
    path = root / "RUN_METADATA"
    if not path.is_file():
        raise ValueError(f"missing RUN_METADATA: {root}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if values.get("schema") != RUN_SCHEMA or values.get("workload") != "packer19":
        raise ValueError(f"unexpected C80-LM run metadata: {root}")
    if values.get("mode") != "formal" or values.get("width") != str(WIDTH):
        raise ValueError(f"non-formal or wrong-width C80-LM root: {root}")
    if values.get("exit_code") != "0":
        raise ValueError(f"nonzero C80-LM root status: {root}")
    return values


def normalise_uuid(value: Any) -> str:
    text = str(value).strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def read_gpu_inventory(root: Path) -> dict[str, Any]:
    path = root / "gpus.csv"
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.reader(handle):
            if not raw or all(not item.strip() for item in raw):
                continue
            if len(raw) < 3:
                raise ValueError(f"malformed GPU inventory: {path}")
            rows.append({
                "index": int(raw[0].strip()),
                "uuid": normalise_uuid(raw[1]),
                "name": raw[2].strip(),
            })
    if len(rows) != 4 or {row["index"] for row in rows} != set(GPUS):
        raise ValueError(f"formal root does not contain exactly GPUs 0..3: {path}")
    if len({row["uuid"] for row in rows}) != 4:
        raise ValueError(f"GPU UUIDs are not unique within host: {path}")
    if any(row["name"] != "NVIDIA A100-SXM4-80GB" for row in rows):
        raise ValueError(f"formal inventory contains a non-A100-SXM4-80GB: {path}")
    rows.sort(key=lambda item: item["index"])
    return {"records": rows, "uuids": [row["uuid"] for row in rows]}


def parse_result_path(path: Path, root: Path) -> tuple[int, int, int, str]:
    parts = path.relative_to(root).parts
    if len(parts) != 6 or parts[0] != "formal" or parts[-1] != "result.json":
        raise ValueError(f"unexpected result path: {path}")
    seed_match = re.fullmatch(r"seed_(\d+)", parts[1])
    order_match = re.fullmatch(r"order_(\d+)_(\w+)", parts[2])
    gpu_match = re.fullmatch(r"gpu(\d+)", parts[3])
    if seed_match is None or order_match is None or gpu_match is None:
        raise ValueError(f"malformed result path: {path}")
    arm = order_match.group(2)
    if parts[4] != arm:
        raise ValueError(f"arm directory mismatch: {path}")
    return int(seed_match.group(1)), int(order_match.group(1)), int(gpu_match.group(1)), arm


def is_rotation(reference: Sequence[str], candidate: Sequence[str]) -> bool:
    if len(reference) != len(candidate):
        return False
    doubled = tuple(reference) + tuple(reference)
    return any(tuple(candidate) == doubled[index:index + len(reference)] for index in range(len(reference)))


def validate_plans(roots: Sequence[Path]) -> dict[int, dict[str, Any]]:
    reference: dict[int, dict[str, Any]] = {}
    reference_hashes: dict[int, str] = {}
    for root_index, root in enumerate(roots):
        for seed in SEEDS:
            path = root / "plans" / f"seed_{seed}.json"
            plan = read_json(path)
            plan_hash = sha256_file(path)
            if plan.get("schema") != PLAN_SCHEMA or plan.get("campaign_id") != CAMPAIGN_ID:
                raise ValueError(f"wrong C80-LM plan identity: {path}")
            if int(plan.get("paired_seed", -1)) != seed:
                raise ValueError(f"plan seed mismatch: {path}")
            if plan.get("seed_role") != "frozen_method_order_only":
                raise ValueError(f"plan seed has an unregistered role: {path}")
            if plan.get("stochastic_processes") != []:
                raise ValueError(f"formal plan introduces stochastic processes: {path}")
            if int(plan.get("width", -1)) != WIDTH or plan.get("candidate_order") != list(range(WIDTH)):
                raise ValueError(f"plan width/candidate order mismatch: {path}")
            order = plan.get("method_order")
            if not isinstance(order, list) or len(order) != 5 or set(order) != set(ARMS):
                raise ValueError(f"plan does not permute five registered arms: {path}")
            if plan.get("runner_methods") != METHODS:
                raise ValueError(f"plan runner method mapping mismatch: {path}")
            if root_index == 0:
                reference[seed] = plan
                reference_hashes[seed] = plan_hash
            elif plan != reference[seed] or plan_hash != reference_hashes[seed]:
                raise ValueError(f"host plans are not byte-identical for seed {seed}")
    orders = [reference[seed]["method_order"] for seed in SEEDS]
    if len({tuple(order) for order in orders}) != len(orders):
        raise ValueError("frozen method orders are not distinct")
    if not all(is_rotation(orders[0], order) for order in orders[1:]):
        raise ValueError("frozen method orders are not cyclically counterbalanced")
    return reference


def repeat_rows(document: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    repeats = document.get("repeats")
    if arm == "official":
        rows = repeats
    else:
        rows = repeats.get(METHODS[arm]) if isinstance(repeats, dict) else None
    if not isinstance(rows, list) or len(rows) != len(REPEATS):
        raise ValueError(f"{arm} must contain exactly three timing repeats")
    if [int(row.get("repeat_index", -1)) for row in rows] != list(REPEATS):
        raise ValueError(f"{arm} repeat indices are not 0,1,2")
    return rows


def validate_document(
    document: dict[str, Any], *, arm: str, seed: int, order_index: int, gpu: int,
    plan_hash: str, source_commit: str, data_hash: str, metadata_hash: str,
    flash_hash: str, inventory: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    config = document.get("config", {})
    formal = document.get("formal_plan", {})
    source = document.get("source", {})
    environment = document.get("environment", {})
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("campaign_arm") != arm:
        raise ValueError(f"result campaign identity mismatch: {arm}/{seed}/gpu{gpu}")
    if int(config.get("width", -1)) != WIDTH or int(config.get("support_offset", -1)) != 0:
        raise ValueError(f"result workload identity mismatch: {arm}/{seed}/gpu{gpu}")
    if config.get("expected_data_sha256") != data_hash or config.get("expected_metadata_sha256") != metadata_hash:
        raise ValueError(f"result data hashes mismatch: {arm}/{seed}/gpu{gpu}")
    if config.get("input_plan_schema") != PLAN_SCHEMA or config.get("expected_input_plan_sha256") != plan_hash:
        raise ValueError(f"result plan hash/schema mismatch: {arm}/{seed}/gpu{gpu}")
    if config.get("source_commit") != source_commit or source.get("commit") != source_commit:
        raise ValueError(f"result source commit mismatch: {arm}/{seed}/gpu{gpu}")
    if source.get("clean_checkout") is not True:
        raise ValueError(f"formal result came from a dirty checkout: {arm}/{seed}/gpu{gpu}")
    if formal.get("campaign_id") != CAMPAIGN_ID or formal.get("campaign_arm") != arm:
        raise ValueError(f"embedded formal plan identity mismatch: {arm}/{seed}/gpu{gpu}")
    if int(formal.get("paired_seed", -1)) != seed or int(formal.get("order_index", -1)) != order_index:
        raise ValueError(f"embedded seed/order mismatch: {arm}/{seed}/gpu{gpu}")
    if formal.get("input_plan_sha256") != plan_hash or formal.get("stochastic_processes") != []:
        raise ValueError(f"embedded plan hash/stochastic contract mismatch: {arm}/{seed}/gpu{gpu}")
    if int(config.get("physical_gpu_index", -1)) != gpu or int(environment.get("visible_device_count", -1)) != 1:
        raise ValueError(f"single-device process identity mismatch: {arm}/{seed}/gpu{gpu}")
    if environment.get("distributed_environment") not in ({}, None):
        raise ValueError(f"result inherited distributed state: {arm}/{seed}/gpu{gpu}")
    if normalise_uuid(environment.get("device_uuid")) != inventory["records"][gpu]["uuid"]:
        raise ValueError(f"result UUID mismatch: {arm}/{seed}/gpu{gpu}")
    if int(config.get("timing_repeats", -1)) != len(REPEATS):
        raise ValueError(f"result timing-repeat count mismatch: {arm}/{seed}/gpu{gpu}")
    if (
        float(config.get("epsilon", math.nan)) != 0.1
        or float(config.get("tolerance", math.nan)) != 1e-3
        or float(config.get("solver_tolerance", math.nan)) != 9.5e-4
        or int(config.get("check_every", -1)) != 4
        or config.get("allow_tf32") is not False
    ):
        raise ValueError(f"result numerical control contract mismatch: {arm}/{seed}/gpu{gpu}")
    if arm == "official":
        if config.get("method") != METHODS[arm]:
            raise ValueError(f"official method identity mismatch: {seed}/gpu{gpu}")
        distribution = source.get("flash_sinkhorn_distribution", {})
        observed_flash_hash = distribution.get("tree_sha256")
    else:
        if config.get("methods") != [METHODS[arm]]:
            raise ValueError(f"project method identity mismatch: {arm}/{seed}/gpu{gpu}")
        if (
            config.get("batched_marginal_audit") is not True
            or config.get("row_only_batched_audit") is not True
            or config.get("initial_marginal_audit") is not False
            or config.get("reuse_packed_buffers") is not False
        ):
            raise ValueError(f"project audit/buffer contract mismatch: {arm}/{seed}/gpu{gpu}")
        observed_flash_hash = environment.get("flash_sinkhorn", {}).get("tree_sha256")
    if observed_flash_hash != flash_hash or config.get("expected_flashsinkhorn_tree_sha256") != flash_hash:
        raise ValueError(f"result Flash tree identity mismatch: {arm}/{seed}/gpu{gpu}")
    hostname = environment.get("hostname", environment.get("host"))
    if not isinstance(hostname, str) or not hostname.strip():
        raise ValueError(f"missing hostname: {arm}/{seed}/gpu{gpu}")
    rows = repeat_rows(document, arm)
    for row in rows:
        if row.get("all_converged") is not True:
            raise ValueError(f"nonconverged repeat: {arm}/{seed}/gpu{gpu}")
        max_residual = max(finite(row, "max_row_l1"), finite(row, "max_column_l1"))
        if max_residual > 1e-3:
            raise ValueError(f"residual gate failed: {arm}/{seed}/gpu{gpu}")
        if arm != "official":
            if row.get("method") != METHODS[arm] or row.get("retirement_mode") != RETIREMENT_MODES[arm]:
                raise ValueError(f"repeat execution mode mismatch: {arm}/{seed}/gpu{gpu}")
            for key in REQUIRED_NUMERIC_KEYS:
                finite(row, key)
    return hostname.strip(), rows


def canonical_trace(value: Any, label: str) -> str:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def transition_matrices(row: dict[str, Any]) -> np.ndarray:
    consumer = row.get("post_solve_consumer", {})
    value = np.asarray(consumer.get("transition_matrices"), dtype=np.float64)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError("consumer transition matrices are missing or non-finite")
    return value


def gate_lm_pc(lm: dict[str, Any], pc: dict[str, Any]) -> dict[str, Any]:
    for key in TRACE_KEYS:
        if canonical_trace(lm.get(key), f"LM {key}") != canonical_trace(pc.get(key), f"PC {key}"):
            raise ValueError(f"LM/PC semantic trace mismatch: {key}")
    lm_width = [int(value) for value in lm.get("physical_width_trace", [])]
    pc_width = [int(value) for value in pc.get("physical_width_trace", [])]
    logical_width = [int(value) for value in lm["logical_live_width_trace"]]
    if not lm_width or any(value != WIDTH for value in lm_width):
        raise ValueError("LM physical width is not fixed at eight")
    if not pc_width or pc_width[0] != WIDTH or min(pc_width) >= WIDTH:
        raise ValueError("PC did not physically shrink below width eight")
    if len(lm_width) != len(logical_width) or len(pc_width) != len(logical_width):
        raise ValueError("physical/logical width traces have different horizons")
    if any(value < 0 or value > WIDTH for value in logical_width):
        raise ValueError("logical live width is outside [0,8]")
    lm_physical = exact_int(lm, "physical_candidate_slots")
    lm_logical = exact_int(lm, "logical_live_candidate_slots")
    lm_frozen = exact_int(lm, "frozen_but_computed_slots")
    pc_physical = exact_int(pc, "physical_candidate_slots")
    pc_logical = exact_int(pc, "logical_live_candidate_slots")
    if lm_physical != sum(lm_width) or pc_physical != sum(pc_width):
        raise ValueError("physical slot count does not equal physical width trace")
    if lm_logical != sum(logical_width) or pc_logical != sum(logical_width):
        raise ValueError("logical slot count does not equal logical-live trace")
    if lm_frozen != lm_physical - lm_logical:
        raise ValueError("LM frozen-but-computed slot identity failed")
    for name, row in (("LM", lm), ("PC", pc)):
        if exact_int(row, "duplicate_successful_release_verifications") != 0:
            raise ValueError(f"{name} contains duplicate successful releases")
        if exact_int(row, "release_verified_candidate_count") != WIDTH:
            raise ValueError(f"{name} did not release exactly eight candidates")
    expected = transition_matrices(lm)
    observed = transition_matrices(pc)
    if expected.shape != observed.shape:
        raise ValueError("LM/PC consumer tensor shapes differ")
    tv = float(np.max(0.5 * np.sum(np.abs(expected - observed), axis=(1, 2))))
    if tv > 0.002:
        raise ValueError(f"LM/PC consumer TV gate failed: {tv}")
    return {
        "consumer_total_variation": tv,
        "physical_slots_lm": lm_physical,
        "physical_slots_pc": pc_physical,
        "logical_slots": lm_logical,
        "frozen_but_computed_slots_lm": lm_frozen,
    }


def geometric_mean(values: Iterable[float]) -> float:
    sequence = [float(value) for value in values]
    if not sequence or any(not math.isfinite(value) or value <= 0 for value in sequence):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in sequence) / len(sequence))


def bootstrap_ci(values: Sequence[float], *, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    logs = np.log(np.asarray(values, dtype=np.float64))
    samples = rng.integers(0, len(logs), size=(draws, len(logs)))
    estimates = np.exp(np.mean(logs[samples], axis=1))
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def stratified_bootstrap_ci(
    values_by_host: Sequence[Sequence[float]], *, draws: int, seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    host_logs = [np.log(np.asarray(values, dtype=np.float64)) for values in values_by_host]
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means = []
        for values in host_logs:
            sample = rng.integers(0, len(values), size=len(values))
            means.append(float(np.mean(values[sample])))
        estimates[draw] = math.exp(float(np.mean(means)))
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def percentile95(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        directory = output.parent
        json_path = output
    else:
        directory = output
        json_path = directory / "analysis.json"
    directory.mkdir(parents=True, exist_ok=True)
    return directory, json_path


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to emit empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(roots: Sequence[Path], *, output: Path, bootstrap_draws: int, bootstrap_seed: int) -> dict[str, Any]:
    roots = [root.resolve() for root in roots]
    if len(roots) != 2 or roots[0] == roots[1]:
        raise ValueError("C80-LM formal analysis requires exactly two distinct roots")
    for root in roots:
        if (root / "STATUS").read_text(encoding="utf-8").strip() != "0" or not (root / "RUN_COMPLETE").is_file():
            raise ValueError(f"incomplete formal root: {root}")
        run_metadata(root)
    manifest_counts = [verify_manifest(root) for root in roots]
    inventories = [read_gpu_inventory(root) for root in roots]
    if set(inventories[0]["uuids"]) & set(inventories[1]["uuids"]):
        raise ValueError("formal roots do not represent disjoint physical GPU hosts")

    source_commits = [(root / "source_commit.txt").read_text(encoding="utf-8").strip() for root in roots]
    flash_hashes = [(root / "flash_tree_sha256.txt").read_text(encoding="utf-8").strip() for root in roots]
    if len(set(source_commits)) != 1 or not re.fullmatch(r"[0-9a-f]{40}", source_commits[0]):
        raise ValueError("formal roots have different or invalid source commits")
    if len(set(flash_hashes)) != 1 or not re.fullmatch(r"[0-9a-f]{64}", flash_hashes[0]):
        raise ValueError("formal roots have different or invalid Flash tree hashes")
    source_commit = source_commits[0]
    plans = validate_plans(roots)
    data_hashes: list[str] = []
    metadata_hashes: list[str] = []
    for root in roots:
        data = root / "inputs" / "packer19.npz"
        metadata = root / "inputs" / "packer19.npz.metadata.json"
        data_hashes.append(sha256_file(data))
        metadata_hashes.append(sha256_file(metadata))
    if len(set(data_hashes)) != 1 or len(set(metadata_hashes)) != 1:
        raise ValueError("formal roots do not contain identical frozen inputs")
    data_hash, metadata_hash = data_hashes[0], metadata_hashes[0]
    for seed, plan in plans.items():
        if plan.get("prepared_npz_sha256") != data_hash or plan.get("prepared_metadata_sha256") != metadata_hash:
            raise ValueError(f"plan {seed} is not bound to the embedded frozen input")
        if plan.get("source_commit") != source_commit:
            raise ValueError(f"plan {seed} is not bound to the executed source commit")

    documents: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    rows_by_key: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    hostnames: list[str] = []
    for host, root in enumerate(roots):
        paths = sorted(root.glob("formal/seed_*/order_*_*/gpu*/*/result.json"))
        if len(paths) != len(SEEDS) * len(ARMS) * len(GPUS):
            raise ValueError(f"expected 60 result documents in {root}, found {len(paths)}")
        seen_hostnames: set[str] = set()
        for path in paths:
            seed, order_index, gpu, arm = parse_result_path(path, root)
            if seed not in SEEDS or gpu not in GPUS or arm not in ARMS:
                raise ValueError(f"unregistered result path: {path}")
            plan = plans[seed]
            if plan["method_order"][order_index] != arm:
                raise ValueError(f"result path differs from frozen method order: {path}")
            plan_hash = sha256_file(root / "plans" / f"seed_{seed}.json")
            status_path = path.with_name("status.txt")
            if not status_path.is_file() or status_path.read_text(encoding="utf-8").strip() != "0":
                raise ValueError(f"nonzero or missing child status: {path}")
            document = read_json(path)
            hostname, rows = validate_document(
                document, arm=arm, seed=seed, order_index=order_index, gpu=gpu,
                plan_hash=plan_hash, source_commit=source_commit, data_hash=data_hash,
                metadata_hash=metadata_hash, flash_hash=flash_hashes[0],
                inventory=inventories[host],
            )
            seen_hostnames.add(hostname)
            key = (host, seed, gpu, arm)
            if key in documents:
                raise ValueError(f"duplicate result unit: {key}")
            documents[key] = document
            rows_by_key[key] = rows
        if len(seen_hostnames) != 1:
            raise ValueError(f"one raw root contains multiple hostnames: {root}")
        hostnames.append(next(iter(seen_hostnames)))
    if hostnames[0] == hostnames[1]:
        raise ValueError("formal roots report the same physical hostname")

    semantic_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    absolute_rows: list[dict[str, Any]] = []
    ratios_by_host: list[list[float]] = [[], []]
    for host in range(2):
        for seed in SEEDS:
            for gpu in GPUS:
                lm_rows = rows_by_key[(host, seed, gpu, "lm")]
                pc_rows = rows_by_key[(host, seed, gpu, "pc")]
                for repeat in REPEATS:
                    gate = gate_lm_pc(lm_rows[repeat], pc_rows[repeat])
                    semantic_rows.append({
                        "host_index": host, "hostname": hostnames[host], "seed": seed,
                        "gpu": gpu, "repeat": repeat, **gate,
                    })
                unit_times: dict[str, float] = {}
                for arm in ARMS:
                    rows = rows_by_key[(host, seed, gpu, arm)]
                    endpoint = [finite(row, "ready_arrays_to_consumer_seconds") for row in rows]
                    unit_times[arm] = median(endpoint)
                    record: dict[str, Any] = {
                        "host_index": host, "hostname": hostnames[host], "seed": seed,
                        "gpu": gpu, "arm": arm, "endpoint_seconds_median": unit_times[arm],
                    }
                    for key in REQUIRED_NUMERIC_KEYS:
                        if arm == "official" and key not in rows[0]:
                            record[f"{key}_median"] = ""
                        else:
                            record[f"{key}_median"] = median(finite(row, key) for row in rows)
                    absolute_rows.append(record)
                ratio = unit_times["lm"] / unit_times["pc"]
                ratios_by_host[host].append(ratio)
                unit_rows.append({
                    "host_index": host, "hostname": hostnames[host], "seed": seed,
                    "gpu": gpu, "lm_seconds": unit_times["lm"],
                    "pc_seconds": unit_times["pc"], "lm_over_pc": ratio,
                    "pc_win": int(ratio > 1.0),
                })

    all_ratios = ratios_by_host[0] + ratios_by_host[1]
    host_statistics = []
    for host, values in enumerate(ratios_by_host):
        host_statistics.append({
            "host_index": host,
            "hostname": hostnames[host],
            "paired_units": len(values),
            "geometric_mean_lm_over_pc": geometric_mean(values),
            "bootstrap_ci95": bootstrap_ci(values, draws=bootstrap_draws, seed=bootstrap_seed + host),
            "wins": sum(value > 1.0 for value in values),
            "paired_ratios": values,
        })
    host_gms = [entry["geometric_mean_lm_over_pc"] for entry in host_statistics]
    primary = {
        "definition": "T_LM / T_PC; values >1 mean physical compaction is faster",
        "paired_units": 24,
        "geometric_mean_lm_over_pc": math.sqrt(host_gms[0] * host_gms[1]),
        "equal_host_stratified_bootstrap_ci95": stratified_bootstrap_ci(
            ratios_by_host, draws=bootstrap_draws, seed=bootstrap_seed,
        ),
        "wins": sum(value > 1.0 for value in all_ratios),
        "host_statistics": host_statistics,
        "all_paired_ratios": all_ratios,
    }

    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        records = [row for row in absolute_rows if row["arm"] == arm]
        arm_summary[arm] = {
            "paired_units": len(records),
            "endpoint_seconds_median": median(float(row["endpoint_seconds_median"]) for row in records),
            "endpoint_seconds_p95": percentile95([float(row["endpoint_seconds_median"]) for row in records]),
        }
        for key in REQUIRED_NUMERIC_KEYS:
            values = [row[f"{key}_median"] for row in records if row[f"{key}_median"] != ""]
            arm_summary[arm][f"{key}_median"] = median(float(value) for value in values) if values else None

    directory, json_path = output_paths(output.resolve())
    paired_csv = directory / "paired_ratios.csv"
    absolute_csv = directory / "absolute_metrics.csv"
    semantic_csv = directory / "lm_pc_semantic_gates.csv"
    write_csv(paired_csv, unit_rows)
    write_csv(absolute_csv, absolute_rows)
    write_csv(semantic_csv, semantic_rows)
    payload = {
        "schema": "forgetot_c80_lm_five_arm_formal_analysis_v1",
        "status": "pass",
        "classification": "FORMAL_TWO_HOST_LOGICAL_MASK_VS_PHYSICAL_COMPACTION",
        "campaign_id": CAMPAIGN_ID,
        "run_roots": [str(root) for root in roots],
        "physical_hosts": hostnames,
        "gpu_inventories": inventories,
        "manifest_entry_counts": manifest_counts,
        "source_commit": source_commit,
        "flash_tree_sha256": flash_hashes[0],
        "data_sha256": data_hash,
        "metadata_sha256": metadata_hash,
        "plan_sha256": {
            str(seed): sha256_file(roots[0] / "plans" / f"seed_{seed}.json") for seed in SEEDS
        },
        "all_required_results": len(documents),
        "semantic_gate_records": len(semantic_rows),
        "maximum_consumer_total_variation": max(row["consumer_total_variation"] for row in semantic_rows),
        "primary_effect": primary,
        "arm_summary": arm_summary,
        "csv_outputs": [str(path) for path in (paired_csv, absolute_csv, semantic_csv)],
        "timing_repeat_is_not_data_seed": True,
        "claim_rule": "interpret T_LM/T_PC only after every LM/PC semantic and correctness gate passes",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory or explicit analysis.json path")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=80_195_001)
    args = parser.parse_args()
    if len(args.run_root) != 2:
        parser.error("exactly two --run-root arguments are required")
    if args.bootstrap_draws < 1_000:
        parser.error("bootstrap-draws must be at least 1000")
    payload = analyze(
        args.run_root, output=args.output, bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
