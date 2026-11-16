"""TorchSig package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import datasets as datasets
    from . import geo as geo
    from . import signals as signals
    from . import transforms as transforms
    from . import utils as utils

__version__ = "2.2.0"
__all__ = ["__version__", "datasets", "geo", "signals", "transforms", "utils"]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__ if name != "__version__"}


def __getattr__(name: str) -> Any:
    """Load top-level packages on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
