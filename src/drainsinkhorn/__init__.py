"""DrainSinkhorn public API.

DrainSinkhorn adds candidate-axis execution, Sinkhorn-specific screening, and
verified physical elimination around a FlashSinkhorn inner backend.
"""

from .flash import (
    FlashSinkhornBackend,
    FlashSinkhornConfig,
    FlashSinkhornConvergenceError,
    FlashSinkhornResult,
    FlashSinkhornUnavailableError,
    flashsinkhorn_available,
)
from .flash_packed import (
    FlashPackedConfig,
    FlashPackedFirstPassageProfile,
    FlashPackedWindowResult,
    FlashSinkhornPackedBackend,
    RetirementMode,
)

__version__ = "0.1.0"

# The short names are the recommended public entry points.  The explicit
# Flash-prefixed names remain available for users who need to distinguish the
# outer executor from the upstream single-problem backend.
DrainSinkhorn = FlashSinkhornPackedBackend
DrainSinkhornConfig = FlashPackedConfig
DrainSinkhornResult = FlashPackedWindowResult

__all__ = [
    "DrainSinkhorn",
    "DrainSinkhornConfig",
    "DrainSinkhornResult",
    "FlashPackedConfig",
    "FlashPackedFirstPassageProfile",
    "FlashPackedWindowResult",
    "FlashSinkhornBackend",
    "FlashSinkhornConfig",
    "FlashSinkhornConvergenceError",
    "FlashSinkhornPackedBackend",
    "FlashSinkhornResult",
    "FlashSinkhornUnavailableError",
    "RetirementMode",
    "flashsinkhorn_available",
]
