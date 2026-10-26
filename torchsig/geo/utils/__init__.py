"""Geolocation utilities for RF signal processing.

This package provides utility functions for geographic coordinate systems
and RF propagation modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import coordinate_system as coordinate_system
    from . import file_handler as file_handler
    from . import propagation as propagation

__all__ = ["coordinate_system", "file_handler", "propagation"]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__}


def __getattr__(name: str) -> Any:
    """Load geolocation utility modules on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
