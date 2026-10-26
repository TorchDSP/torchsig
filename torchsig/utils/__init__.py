"""Utility modules for TorchSig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import abstractions as abstractions
    from . import coordinate_system as coordinate_system
    from . import data_loading as data_loading
    from . import dataset_summarizer as dataset_summarizer
    from . import defaults as defaults
    from . import dsp as dsp
    from . import experiment_config as experiment_config
    from . import file_handlers as file_handlers
    from . import metadata_logging as metadata_logging
    from . import printing as printing
    from . import random as random
    from . import signal_building as signal_building
    from . import verify as verify
    from . import writer as writer
    from . import yaml as yaml

__all__ = [
    "abstractions",
    "coordinate_system",
    "data_loading",
    "dataset_summarizer",
    "defaults",
    "dsp",
    "experiment_config",
    "file_handlers",
    "metadata_logging",
    "printing",
    "random",
    "signal_building",
    "verify",
    "writer",
    "yaml",
]
_EXPORTS = {name: (f"{__name__}.{name}", None) for name in __all__}


def __getattr__(name: str) -> Any:
    """Load utility modules on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
