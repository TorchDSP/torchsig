"""TorchSig transform modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import base_transforms as base_transforms
    from . import functional as functional
    from . import impairments as impairments
    from . import metadata_transforms as metadata_transforms
    from . import transforms as transforms

__all__ = ["base_transforms", "functional", "impairments", "metadata_transforms", "transforms"]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__}


def __getattr__(name: str) -> Any:
    """Load transform modules on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
