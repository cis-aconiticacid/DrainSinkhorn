import torch

import drainsinkhorn
from drainsinkhorn.flash_packed import (
    _freeze_retired_potentials,
    _release_verification_candidates,
    _resolve_retirement_mode,
)


def test_short_public_names_point_to_formal_backend() -> None:
    assert drainsinkhorn.DrainSinkhorn is drainsinkhorn.FlashSinkhornPackedBackend
    assert drainsinkhorn.DrainSinkhornConfig is drainsinkhorn.FlashPackedConfig
    assert drainsinkhorn.__version__ == "0.1.0"


def test_retirement_modes_are_explicit() -> None:
    assert _resolve_retirement_mode(
        retirement_mode="fixed_width", active_compaction=None
    ) == "fixed_width"
    assert _resolve_retirement_mode(
        retirement_mode="logical_mask", active_compaction=None
    ) == "logical_mask"
    assert _resolve_retirement_mode(
        retirement_mode="physical_compaction", active_compaction=None
    ) == "physical_compaction"


def test_fixed_width_waits_for_common_boundary() -> None:
    first_passage = [None, None, None]
    assert _release_verification_candidates(
        [0], [0, 1, 2], first_passage, retirement_mode="fixed_width"
    ) == []
    assert _release_verification_candidates(
        [0, 1, 2], [0, 1, 2], first_passage, retirement_mode="fixed_width"
    ) == [0, 1, 2]


def test_logical_and_physical_modes_verify_screen_positive_lanes() -> None:
    for mode in ("logical_mask", "physical_compaction"):
        assert _release_verification_candidates(
            [0, 2], [0, 1, 2], [None, None, None], retirement_mode=mode
        ) == [0, 2]


def test_logical_mask_freezes_retired_rows() -> None:
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    next_value = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    live = torch.tensor([True, False])
    observed = _freeze_retired_potentials(next_value, previous, live)
    assert torch.equal(observed, torch.tensor([[10.0, 20.0], [3.0, 4.0]]))
