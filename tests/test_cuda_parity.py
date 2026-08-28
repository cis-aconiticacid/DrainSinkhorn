import pytest
import torch

from drainsinkhorn import (
    DrainSinkhorn,
    DrainSinkhornConfig,
    flashsinkhorn_available,
)

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_logical_mask_and_physical_compaction_share_release_contract() -> None:
    if not flashsinkhorn_available():
        pytest.skip("FlashSinkhorn is unavailable")

    torch.manual_seed(7)
    device = torch.device("cuda")
    width, n, dimension = 4, 128, 8
    x = torch.randn(width, n, dimension, device=device)
    ys = torch.randn(width, n, dimension, device=device)
    a = torch.full((n,), 1.0 / n, device=device)
    bs = torch.full((n, width), 1.0 / n, device=device)
    solver = DrainSinkhorn(
        DrainSinkhornConfig(
            epsilon=0.5,
            marginal_tolerance=1e-3,
            max_iterations=400,
            check_every=2,
            min_packed_width=1,
            batched_marginal_audit=True,
            row_only_batched_audit=True,
            initial_marginal_audit=False,
        )
    )

    logical = solver.solve_batch(x, ys, a, bs, retirement_mode="logical_mask")
    physical = solver.solve_batch(
        x, ys, a, bs, retirement_mode="physical_compaction"
    )

    for window in (logical, physical):
        assert all(result.converged for result in window.results)
        assert max(
            max(result.row_residual, result.column_residual)
            for result in window.results
        ) <= 1e-3
        assert window.duplicate_successful_release_verifications == 0

    assert physical.verified_first_passage_iterations == logical.verified_first_passage_iterations
    assert physical.physical_candidate_slots <= logical.physical_candidate_slots
