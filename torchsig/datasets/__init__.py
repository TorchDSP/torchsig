"""TorchSig datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import datamodules as datamodules
    from . import dataset_utils as dataset_utils
    from . import datasets as datasets
    from . import structured as structured
    from .datamodules import TorchSigDataModule as TorchSigDataModule
    from .datasets import SafeTorchSigIterableDataset as SafeTorchSigIterableDataset
    from .datasets import TorchSigIterableDataset as TorchSigIterableDataset
    from .structured import StructuredHDF5Dataset as StructuredHDF5Dataset
    from .structured import materialize_structured_dataset as materialize_structured_dataset

_EXPORTS = {
    "datamodules": ("torchsig.datasets.datamodules", None),
    "dataset_utils": ("torchsig.datasets.dataset_utils", None),
    "datasets": ("torchsig.datasets.datasets", None),
    "structured": ("torchsig.datasets.structured", None),
    "SafeTorchSigIterableDataset": ("torchsig.datasets.datasets", "SafeTorchSigIterableDataset"),
    "TorchSigDataModule": ("torchsig.datasets.datamodules", "TorchSigDataModule"),
    "TorchSigIterableDataset": ("torchsig.datasets.datasets", "TorchSigIterableDataset"),
    "StructuredHDF5Dataset": ("torchsig.datasets.structured", "StructuredHDF5Dataset"),
    "materialize_structured_dataset": ("torchsig.datasets.structured", "materialize_structured_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load dataset modules and public classes on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
