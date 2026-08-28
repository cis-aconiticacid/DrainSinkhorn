"""Minimal candidate-batched DrainSinkhorn solve."""

import torch

from drainsinkhorn import DrainSinkhorn, DrainSinkhornConfig

device = torch.device("cuda")
width, n, dimension = 8, 4096, 64

# A shared source and W independent target supports.  Candidate-specific
# sources [W,n,d] are also accepted.
x = torch.randn(n, dimension, device=device)
ys = torch.randn(width, n, dimension, device=device)
a = torch.full((n,), 1.0 / n, device=device)
bs = torch.full((n, width), 1.0 / n, device=device)

solver = DrainSinkhorn(
    DrainSinkhornConfig(
        epsilon=0.1,
        marginal_tolerance=1e-3,
        max_iterations=800,
        check_every=4,
        min_packed_width=1,
        batched_marginal_audit=True,
        row_only_batched_audit=True,
        initial_marginal_audit=False,
    )
)

window = solver.solve_batch(
    x,
    ys,
    a,
    bs,
    retirement_mode="physical_compaction",
)

print("depths:", window.verified_first_passage_iterations)
print("physical widths:", window.physical_width_trace)
print("candidate slots:", window.physical_candidate_slots)
print(
    "max residual:",
    max(max(result.row_residual, result.column_residual) for result in window.results),
)
