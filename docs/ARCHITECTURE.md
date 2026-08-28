# Architecture

## The optimization boundary

FlashSinkhorn reduces the cost of one matrix-free Sinkhorn update. A batched
service has another resource: candidate slots. If candidate `s` first passes
the registered release contract at audited depth `ell_s`, then fixed-width
execution spends

```text
S_fixed = W * max_s ell_s
```

physical candidate-update slots. Logical masking records individual completion
but leaves the kernel shape unchanged, so

```text
S_mask = S_fixed.
```

Physical compaction spends

```text
S_drain = sum_s ell_s
```

up to audit-grid overshoot and any routed sequential tail. The wall-time gain
also depends on width-dependent kernel cost, audit and verifier work, compaction
cost, and synchronization.

## Numerical state machine

```text
ACTIVE
  | packed alternating scaling cycle
  v
SCREENED
  | screen negative --------------------------> ACTIVE
  | screen positive
  v
VERIFY
  | full two-sided residual fails ------------> ACTIVE
  | full two-sided residual passes
  v
RELEASED
  | write output under original candidate ID
  v
DRAINED
  | compact every candidate-indexed tensor
  v
narrower ACTIVE batch
```

A lane never returns after successful release. A failed verifier only returns
the lane to `ACTIVE`; this is why the implementation may invoke the verifier
more than once for a screen-positive-but-not-yet-valid lane while still
performing exactly one *successful* release per lane.

## Why the screen is Sinkhorn-specific

For

```text
u_next = a / (K v)
v_next = b / (K^T u_next),
```

the second assignment makes the column marginal current by construction in
exact arithmetic. The row marginal can still lag. DrainSinkhorn batches the row
residual as a cheap screen, then calls the upstream two-sided matrix-free plan
application on screen-positive lanes.

The screen is not a numerical shortcut to a different solution. It is a
control-flow shortcut to avoid full verification when a lane clearly cannot be
released.

## State that compacts

Physical elimination uses one keep mask for all candidate-indexed state:

- candidate-specific source and target supports;
- source and target marginals;
- standard and shifted dual potentials;
- squared-norm shifts and log weights;
- optional ping-pong scratch buffers;
- current residual and original-candidate index state.

Outputs are written by original candidate ID before the state is compacted.

## Scale-out

The primary multi-GPU design assigns complete candidate windows to independent
GPU workers. No problem is row-sharded and no cross-worker collective appears
inside a Sinkhorn iteration. A distributed queue can dispatch windows and
gather completed outputs. This is orthogonal to physical compaction within each
worker.

## Conceptual origin

The project began from the observation that positive fixed-point iterations,
including PageRank-like Perron--Frobenius workloads and Sinkhorn scaling, have
well-defined convergence but unequal realized depths. DrainSinkhorn turns that
observation into execution: convergence remains a per-problem numerical fact;
verified problems stop occupying shared parallel work.
