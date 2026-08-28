# DrainSinkhorn API

## Public entry points

```python
from drainsinkhorn import (
    DrainSinkhorn,
    DrainSinkhornConfig,
    DrainSinkhornResult,
)
```

The short names are aliases for `FlashSinkhornPackedBackend`,
`FlashPackedConfig`, and `FlashPackedWindowResult`.

## Configuration

```python
DrainSinkhornConfig(
    epsilon: float,
    marginal_tolerance: float = 1e-3,
    max_iterations: int = 2000,
    check_every: int = 10,
    cost_scale: float = 1.0,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    min_packed_width: int = 2,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    batched_marginal_audit: bool = False,
    row_only_batched_audit: bool = False,
    initial_marginal_audit: bool = True,
    reuse_packed_buffers: bool = False,
    raise_on_nonconvergence: bool = True,
)
```

`row_only_batched_audit=True` requires `batched_marginal_audit=True`. The
row-only value is a screen; the upstream two-sided plan-application verifier
still decides release.

## `solve_batch`

```python
result = solver.solve_batch(
    x,
    ys,
    a,
    targets,
    f_init=None,
    g_init=None,
    retirement_mode="physical_compaction",
)
```

### Inputs

| Name | Shape | Meaning |
|---|---|---|
| `x` | `[n,d]` or `[W,n,d]` | shared or candidate-specific source support |
| `ys` | `[W,m,d]` | candidate target supports |
| `a` | `[n]` or `[n,W]` | shared or candidate-specific source masses |
| `targets` | `[m,W]` | candidate target masses |
| `f_init` | `[n,W]` | optional standard source potentials |
| `g_init` | `[m,W]` | optional standard target potentials |

All tensors must be floating-point CUDA tensors on the same device. Marginals
must be strictly positive and each candidate's source and target masses must
have matching total mass.

### Retirement modes

- `fixed_width`: update all lanes until every lane passes at one audit.
- `logical_mask`: release and freeze passing lanes without changing physical
  tensor width.
- `physical_compaction`: release passing lanes and compact every
  candidate-indexed tensor.

### Result telemetry

`DrainSinkhornResult` contains:

- `results`: one verified `FlashSinkhornResult` per input candidate;
- `verified_first_passage_iterations`;
- `physical_width_trace` and `logical_live_width_trace`;
- `physical_candidate_slots`, `logical_live_candidate_slots`, and
  `frozen_but_computed_slots`;
- audit, verifier, compaction, logical-mask, tail, and synchronization timing;
- audit and release traces indexed by original candidate ID;
- `duplicate_successful_release_verifications`, which must remain zero in a
  valid production run.

## Single-problem backend

The package also exports `FlashSinkhornBackend` and `FlashSinkhornConfig` as a
width-one reference and as the release-verification authority. They wrap the
installed upstream FlashSinkhorn package without vendoring its code.
