"""TorchSig signal modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import builder as builder
    from . import signal_lists as signal_lists
    from . import signal_types as signal_types
    from . import signal_utils as signal_utils

__all__ = ["builder", "signal_lists", "signal_types", "signal_utils"]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__}


def __getattr__(name: str) -> Any:
    """Load signal modules on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
