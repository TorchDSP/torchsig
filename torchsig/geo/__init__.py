"""TorchSig geolocation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import datasets as datasets
    from . import transforms as transforms
    from . import types as types
    from . import utils as utils

__all__ = ["datasets", "transforms", "types", "utils"]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__}


def __getattr__(name: str) -> Any:
    """Load geolocation modules on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
