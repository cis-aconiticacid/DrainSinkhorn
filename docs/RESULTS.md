# Paper results

All speedups are `baseline time / proposed time`. Component ratios compare
matched paths. Deployment ratios include the whole registered implementation
path and are listed separately.

## C80-LM: logical masking versus physical compaction

- Workload: real Packer19 single-cell transition endpoint.
- Shape: `n=16384`, `W=8`, `d=64`.
- Hardware: two physical hosts, each with four A100-SXM4-80GB GPUs.
- Parallelism: eight independent complete-device workers; no NCCL and no
  sharding of one OT problem.
- Shared contract: cold initialization, `epsilon=0.1`, release tolerance
  `1e-3`, solver screen tolerance `9.5e-4`, `check_every=4`, strict FP32,
  row-only batched screen, upstream two-sided release verifier.

| Contrast | Equal-host geometric mean | 95% paired bootstrap CI | Wins |
|---|---:|---:|---:|
| fixed width / logical mask | 1.000069x | [0.999809, 1.000328] | neutral |
| logical mask / physical compaction | **1.224345x** | **[1.224016, 1.224682]** | **24/24** |
| public Flash serial / physical compaction | 14.192772x | [14.184883, 14.200847] | 24/24 |

Logical masking and physical compaction have the same 156 logical-live slots.
Logical masking executes 256 physical slots, including 100 frozen-but-computed
slots. Physical compaction executes 156 physical slots. Median endpoint energy
falls from 879.940 J to 688.133 J. Peak allocation rises from 80.81 MiB to
106.94 MiB, so this is a compute and energy result, not a memory-saving claim.

Machine-readable outputs:

- [`results/c80_lm/analysis.json`](../results/c80_lm/analysis.json)
- [`results/c80_lm/paired_ratios.csv`](../results/c80_lm/paired_ratios.csv)
- [`results/c80_lm/absolute_metrics.csv`](../results/c80_lm/absolute_metrics.csv)
- [`results/c80_lm/lm_pc_semantic_gates.csv`](../results/c80_lm/lm_pc_semantic_gates.csv)

## C66: MetroPT-3 predictive maintenance

Across two physical hosts, matched current static / current active compaction is
`1.744723x`, with host-stratified 95% CI `[1.743410, 1.746706]` and 6/6
host-seed wins. The complete active path is `4.074308x` faster than public
Flash serial. The deployment ratio includes the candidate-axis implementation;
the matched static/active ratio isolates elimination.

## M3: controlled depth heterogeneity

The same 32 real MetroPT problems and the same total correction work are only
regrouped. Static/active speedup rises from `1.1045x` under oracle homogeneous
grouping to `1.2172x` under deliberately heterogeneous grouping and
`1.2347x--1.2709x` under three frozen random layouts. This intervention connects
depth variation to the value of physical elimination.

## C63: ImageNet-32 feature-space OT-Flow-Matching

The complete active execution pipeline is `2.7857x` faster than public Flash
serial over 50k training updates. All frozen NFE 4/8/16 feature-space quality
checks pass. This endpoint uses real ImageNet-32 PCA500 features; it is not a
pixel-generation FID claim.

## C67: complete-device scale-out

Two physical hosts run eight independent A100 workers. Each worker receives
complete EOT windows and performs local DrainSinkhorn execution. There is no
cross-worker collective inside the Sinkhorn loop. Fixed-per-worker results are
reported for capacity/throughput deployment; fixed-total calibration is kept
separate.
