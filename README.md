# DrainSinkhorn

**Safe elimination for batched optimal transport.**

FlashSinkhorn makes one Sinkhorn update fast. DrainSinkhorn removes a second
source of waste: continuing to update batch lanes that have already converged.
It packs independent or related EOT problems along a candidate axis, screens
them with the updated-marginal identity, verifies every release with the
upstream two-sided residual contract, and physically compacts all remaining OT
state. The next kernel is narrower because finished lanes are gone.

```text
packed Sinkhorn cycle
        |
        v
batched one-sided screen
        |
        v
upstream two-sided verifier
        |
        v
verified lanes drain out  --->  narrower next kernel
```

## Features

- Candidate-axis fused Sinkhorn updates on top of FlashSinkhorn 0.3.3.
- A Sinkhorn-specific one-sided screen after the second scaling update.
- The upstream matrix-free two-sided verifier remains the release authority.
- Three matched execution modes: fixed width, logical masking, and physical
  compaction.
- Candidate-specific sources, targets, marginals, potentials, and scratch
  buffers compact together.
- Complete-device parallelism: different windows can run on independent GPUs
  without sharding one OT problem or adding a collective to every iteration.
- Machine-readable formal results and a fail-closed two-host analyzer.

## Install

DrainSinkhorn requires an NVIDIA CUDA environment with PyTorch 2.5+, Triton
3.1+, and FlashSinkhorn 0.3.3.post1.

```bash
git clone https://github.com/cis-aconiticacid/DrainSinkhorn.git
cd DrainSinkhorn
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest -m "not cuda"
pytest -m cuda                 # on a supported GPU host
```

## Quick start

```python
import torch
from drainsinkhorn import DrainSinkhorn, DrainSinkhornConfig

W, n, d = 8, 4096, 64
x = torch.randn(n, d, device="cuda")          # shared source [n,d]
ys = torch.randn(W, n, d, device="cuda")      # W targets [W,n,d]
a = torch.full((n,), 1 / n, device="cuda")
bs = torch.full((n, W), 1 / n, device="cuda")

solver = DrainSinkhorn(DrainSinkhornConfig(
    epsilon=0.1,
    marginal_tolerance=1e-3,
    max_iterations=800,
    check_every=4,
    min_packed_width=1,
    batched_marginal_audit=True,
    row_only_batched_audit=True,
    initial_marginal_audit=False,
))

result = solver.solve_batch(
    x, ys, a, bs,
    retirement_mode="physical_compaction",
)

print(result.verified_first_passage_iterations)
print(result.physical_width_trace)
print(result.physical_candidate_slots)
```

See [`examples/basic.py`](examples/basic.py) and the full [API reference](docs/API.md).

## Three execution modes

All three modes use the same packed update, audit schedule, row screen,
two-sided release verifier, tolerance, and initialization.

| Mode | What happens after a lane passes verification? | Physical kernel width |
|---|---|---|
| `fixed_width` | All lanes continue to the common final boundary | stays at `W` |
| `logical_mask` | Completed potentials are frozen | stays at `W` |
| `physical_compaction` | Every candidate-indexed tensor is compacted | decreases |

Logical masking is the direct control for physical elimination: it records the
same per-lane completion events but does not remove their compute slots.

## How it works

For one alternating Sinkhorn cycle,

```text
u <- a / (K v)
v <- b / (K^T u)
```

the second update makes `v * (K^T u) = b` in exact arithmetic. DrainSinkhorn
therefore evaluates the opposite marginal as a cheap candidate-batched screen.
A screen-positive lane is recomputed by FlashSinkhorn's full matrix-free
two-sided verifier and leaves only when

```text
max(||P 1 - a||_1, ||P^T 1 - b||_1) <= tolerance.
```

The screen controls verifier work; it never authorizes release. See the
[architecture note](docs/ARCHITECTURE.md) for the state machine and slot model.

## Results

The paper separates component ratios from full-deployment ratios. Every ratio
below is `baseline time / proposed time`.

### Matched component attribution

| Workload and comparison | Speedup | 95% CI | Paired units |
|---|---:|---:|---:|
| Packer19 batched audit / serial audit | 1.073x | [1.073, 1.074] | 24/24 wins |
| Packer19 one-sided screen / two-sided batched audit | 1.055x | [1.055, 1.056] | 24/24 wins |
| Packer19 logical mask / physical compaction | **1.224x** | **[1.2240, 1.2247]** | **24/24 wins** |
| MetroPT-3 static / active compaction | **1.745x** | [1.743, 1.747] | 6/6 host-seed wins |

In the Packer19 logical-mask control, physical slots fall from `256` to `156`
and endpoint energy falls from `879.94 J` to `688.13 J`.

### Complete deployments

| Endpoint | Complete deployment speedup | Scope |
|---|---:|---|
| Packer19 cell-transition consumer | 14.193x | public Flash serial / complete DrainSinkhorn path |
| MetroPT-3 predictive maintenance | 4.074x | public Flash serial / complete DrainSinkhorn path |
| ImageNet-32 PCA500 OT-Flow-Matching | 2.786x | 50k-update feature-space training pipeline |

The deployment ratios include the candidate-axis kernel and controller; they
are not relabeled as compaction-only gains. Full tables, scope, hashes, and
machine-readable outputs are in [`docs/RESULTS.md`](docs/RESULTS.md) and
[`results/`](results/).

## Reproduce

The exact C80-LM analysis output used by the paper is checked in under
[`results/c80_lm`](results/c80_lm). To analyze the original immutable raw roots:

```bash
python benchmarks/packer19/analyze.py \
  --run-root /path/to/host0/raw_formal \
  --run-root /path/to/host1/raw_formal \
  --output benchmark_outputs/c80_lm_analysis
```

To prepare the public Packer19 data and rerun the compact endpoint benchmark,
follow [`benchmarks/packer19/README.md`](benchmarks/packer19/README.md).

## Relationship to FlashSinkhorn

DrainSinkhorn is an outer execution layer, not a replacement inner kernel.
[FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn) streams one
EOT problem through fused Triton kernels. DrainSinkhorn adds a candidate program
axis and removes verified problems from subsequent work. Making one update
cheaper and eliminating unnecessary updates are complementary optimizations.

## Citation

```bibtex
@misc{drainsinkhorn2026,
  title        = {DrainSinkhorn: Safe Elimination for Batched Entropic Optimal Transport},
  year         = {2026},
  howpublished = {Software and accompanying manuscript}
}
```

## License

DrainSinkhorn is released under the MIT License. FlashSinkhorn remains an
independent dependency under its own MIT license; see [`NOTICE`](NOTICE).
