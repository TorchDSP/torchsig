"""File-handler modules and public classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchsig._lazy import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from . import base_handler as base_handler
    from . import hdf5 as hdf5
    from . import homogeneous_hdf5 as homogeneous_hdf5
    from . import npy as npy
    from . import packed_hdf5 as packed_hdf5
    from . import structured_hdf5 as structured_hdf5
    from . import structured_schema as structured_schema
    from .audio_reader import AudioReader as AudioReader
    from .base_handler import BaseFileHandler as BaseFileHandler
    from .base_handler import FileReader as FileReader
    from .base_handler import FileWriter as FileWriter
    from .errors import CorruptDatasetError as CorruptDatasetError
    from .errors import FileHandlerError as FileHandlerError
    from .errors import FileHandlerIndexError as FileHandlerIndexError
    from .errors import IncompatibleOptionsError as IncompatibleOptionsError
    from .errors import IncompleteDatasetError as IncompleteDatasetError
    from .errors import UnsupportedSchemaError as UnsupportedSchemaError
    from .hdf5 import HDF5FileHandler as HDF5FileHandler
    from .hdf5 import HDF5Reader as HDF5Reader
    from .hdf5 import HDF5Writer as HDF5Writer
    from .homogeneous_hdf5 import HomogeneousHDF5Reader as HomogeneousHDF5Reader
    from .homogeneous_hdf5 import HomogeneousHDF5Writer as HomogeneousHDF5Writer
    from .metadata_reader import MetadataIndexError as MetadataIndexError
    from .metadata_reader import MetadataReader as MetadataReader
    from .metrics import FileHandlerMetrics as FileHandlerMetrics
    from .metrics import ValidationPolicy as ValidationPolicy
    from .npy import NPYReader as NPYReader
    from .ogg import OGGReader as OGGReader
    from .packed_hdf5 import PackedHDF5Reader as PackedHDF5Reader
    from .packed_hdf5 import PackedHDF5Writer as PackedHDF5Writer
    from .packed_npy import PackedNPYReader as PackedNPYReader
    from .packed_npy import PackedNPYWriter as PackedNPYWriter
    from .registry import BackendSpec as BackendSpec
    from .registry import FileHandlerRegistry as FileHandlerRegistry
    from .registry import file_handler_registry as file_handler_registry
    from .sigmf import SigMFReader as SigMFReader
    from .structured_hdf5 import StructuredHDF5Reader as StructuredHDF5Reader
    from .structured_hdf5 import StructuredHDF5Writer as StructuredHDF5Writer
    from .structured_schema import StructuredField as StructuredField
    from .structured_schema import StructuredNode as StructuredNode
    from .structured_schema import StructuredSampleSchema as StructuredSampleSchema
    from .structured_schema import infer_structured_sample_schema as infer_structured_sample_schema
    from .wav import WAVReader as WAVReader

_EXPORTS = {
    "base_handler": ("torchsig.utils.file_handlers.base_handler", None),
    "hdf5": ("torchsig.utils.file_handlers.hdf5", None),
    "homogeneous_hdf5": ("torchsig.utils.file_handlers.homogeneous_hdf5", None),
    "npy": ("torchsig.utils.file_handlers.npy", None),
    "packed_hdf5": ("torchsig.utils.file_handlers.packed_hdf5", None),
    "packed_npy": ("torchsig.utils.file_handlers.packed_npy", None),
    "structured_hdf5": ("torchsig.utils.file_handlers.structured_hdf5", None),
    "structured_schema": ("torchsig.utils.file_handlers.structured_schema", None),
    "BaseFileHandler": ("torchsig.utils.file_handlers.base_handler", "BaseFileHandler"),
    "AudioReader": ("torchsig.utils.file_handlers.audio_reader", "AudioReader"),
    "FileReader": ("torchsig.utils.file_handlers.base_handler", "FileReader"),
    "FileWriter": ("torchsig.utils.file_handlers.base_handler", "FileWriter"),
    "FileHandlerError": ("torchsig.utils.file_handlers.errors", "FileHandlerError"),
    "FileHandlerIndexError": ("torchsig.utils.file_handlers.errors", "FileHandlerIndexError"),
    "IncompleteDatasetError": ("torchsig.utils.file_handlers.errors", "IncompleteDatasetError"),
    "UnsupportedSchemaError": ("torchsig.utils.file_handlers.errors", "UnsupportedSchemaError"),
    "CorruptDatasetError": ("torchsig.utils.file_handlers.errors", "CorruptDatasetError"),
    "IncompatibleOptionsError": ("torchsig.utils.file_handlers.errors", "IncompatibleOptionsError"),
    "HDF5FileHandler": ("torchsig.utils.file_handlers.hdf5", "HDF5FileHandler"),
    "HDF5Reader": ("torchsig.utils.file_handlers.hdf5", "HDF5Reader"),
    "HDF5Writer": ("torchsig.utils.file_handlers.hdf5", "HDF5Writer"),
    "HomogeneousHDF5Reader": ("torchsig.utils.file_handlers.homogeneous_hdf5", "HomogeneousHDF5Reader"),
    "HomogeneousHDF5Writer": ("torchsig.utils.file_handlers.homogeneous_hdf5", "HomogeneousHDF5Writer"),
    "MetadataIndexError": ("torchsig.utils.file_handlers.metadata_reader", "MetadataIndexError"),
    "MetadataReader": ("torchsig.utils.file_handlers.metadata_reader", "MetadataReader"),
    "FileHandlerMetrics": ("torchsig.utils.file_handlers.metrics", "FileHandlerMetrics"),
    "ValidationPolicy": ("torchsig.utils.file_handlers.metrics", "ValidationPolicy"),
    "NPYReader": ("torchsig.utils.file_handlers.npy", "NPYReader"),
    "OGGReader": ("torchsig.utils.file_handlers.ogg", "OGGReader"),
    "PackedHDF5Reader": ("torchsig.utils.file_handlers.packed_hdf5", "PackedHDF5Reader"),
    "PackedHDF5Writer": ("torchsig.utils.file_handlers.packed_hdf5", "PackedHDF5Writer"),
    "PackedNPYReader": ("torchsig.utils.file_handlers.packed_npy", "PackedNPYReader"),
    "PackedNPYWriter": ("torchsig.utils.file_handlers.packed_npy", "PackedNPYWriter"),
    "BackendSpec": ("torchsig.utils.file_handlers.registry", "BackendSpec"),
    "FileHandlerRegistry": ("torchsig.utils.file_handlers.registry", "FileHandlerRegistry"),
    "file_handler_registry": ("torchsig.utils.file_handlers.registry", "file_handler_registry"),
    "SigMFReader": ("torchsig.utils.file_handlers.sigmf", "SigMFReader"),
    "StructuredHDF5Reader": ("torchsig.utils.file_handlers.structured_hdf5", "StructuredHDF5Reader"),
    "StructuredHDF5Writer": ("torchsig.utils.file_handlers.structured_hdf5", "StructuredHDF5Writer"),
    "StructuredField": ("torchsig.utils.file_handlers.structured_schema", "StructuredField"),
    "StructuredNode": ("torchsig.utils.file_handlers.structured_schema", "StructuredNode"),
    "StructuredSampleSchema": ("torchsig.utils.file_handlers.structured_schema", "StructuredSampleSchema"),
    "infer_structured_sample_schema": ("torchsig.utils.file_handlers.structured_schema", "infer_structured_sample_schema"),
    "WAVReader": ("torchsig.utils.file_handlers.wav", "WAVReader"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load file-handler modules and classes on first access."""
    return lazy_getattr(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    """List eagerly and lazily available attributes."""
    return lazy_dir(globals(), _EXPORTS)
