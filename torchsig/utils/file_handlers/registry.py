"""Central registry for TorchSIG storage backends and their capabilities."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from .base_handler import FileReader, FileWriter
from .hdf5 import HDF5Reader, HDF5Writer
from .homogeneous_hdf5 import HomogeneousHDF5Reader, HomogeneousHDF5Writer
from .npy import NPYReader
from .ogg import OGGReader
from .packed_hdf5 import PackedHDF5Reader, PackedHDF5Writer
from .packed_npy import PackedNPYReader, PackedNPYWriter
from .sigmf import SigMFReader
from .structured_hdf5 import StructuredHDF5Reader, StructuredHDF5Writer
from .wav import WAVReader

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["BackendSpec", "FileHandlerRegistry", "file_handler_registry"]


@dataclass(frozen=True)
class BackendSpec:
    """Describe one named storage backend and its supported behavior."""

    name: str
    reader: type[FileReader]
    writer: type[FileWriter] | None = None
    format_version: str = "1"
    capabilities: frozenset[str] = field(default_factory=frozenset)
    aliases: tuple[str, ...] = ()
    deprecated: bool = False

    def __post_init__(self) -> None:
        """Validate immutable backend metadata."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("backend name must be a non-empty string")
        if not inspect.isclass(self.reader) or not issubclass(self.reader, FileReader):
            raise TypeError("backend reader must be a FileReader class")
        if self.writer is not None and (not inspect.isclass(self.writer) or not issubclass(self.writer, FileWriter)):
            raise TypeError("backend writer must be a FileWriter class")
        if self.name in self.aliases or len(self.aliases) != len(set(self.aliases)):
            raise ValueError(f"backend {self.name!r} contains duplicate aliases")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def accepted_writer_options(self) -> frozenset[str]:
        """Return explicit keyword options accepted by the writer constructor."""
        if self.writer is None:
            return frozenset()
        return frozenset(
            name for name, parameter in inspect.signature(self.writer).parameters.items() if name != "root" and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )

    def supports(self, capability: str) -> bool:
        """Return whether the backend advertises ``capability``."""
        return capability in self.capabilities


class FileHandlerRegistry:
    """Register and resolve file-handler backend specifications."""

    def __init__(self, specs: Iterable[BackendSpec] = ()) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._names: dict[str, str] = {}
        self._writers: dict[type[FileWriter], BackendSpec] = {}
        for spec in specs:
            self.register(spec)

    @property
    def specs(self) -> Mapping[str, BackendSpec]:
        """Return registered canonical names and specifications."""
        return MappingProxyType(self._specs)

    def register(self, spec: BackendSpec) -> None:
        """Register ``spec`` and reject ambiguous names or writer mappings."""
        names = (spec.name, *spec.aliases)
        duplicates = sorted(name for name in names if name in self._names)
        if duplicates:
            raise ValueError(f"file-handler names already registered: {duplicates}")
        if spec.writer is not None and spec.writer in self._writers:
            raise ValueError(f"writer {spec.writer.__name__} is already registered")
        self._specs[spec.name] = spec
        for name in names:
            self._names[name] = spec.name
        if spec.writer is not None:
            self._writers[spec.writer] = spec

    def resolve(self, name: str) -> BackendSpec:
        """Resolve a canonical name or alias."""
        try:
            return self._specs[self._names[name]]
        except KeyError as error:
            raise ValueError(f"unknown file-handler backend {name!r}; available: {sorted(self._specs)}") from error

    def for_writer(self, writer: type[FileWriter]) -> BackendSpec | None:
        """Return the specification registered for ``writer``, if any."""
        return self._writers.get(writer)

    def reader_for_writer(self, writer: type[FileWriter]) -> type[FileReader] | None:
        """Return the registered reader paired with ``writer``."""
        spec = self.for_writer(writer)
        return None if spec is None else spec.reader

    def require(self, name: str, *capabilities: str) -> BackendSpec:
        """Resolve a backend and require all requested capabilities."""
        spec = self.resolve(name)
        missing = sorted(set(capabilities).difference(spec.capabilities))
        if missing:
            raise ValueError(f"backend {spec.name!r} does not support capabilities: {missing}")
        return spec


def _builtin_registry() -> FileHandlerRegistry:
    return FileHandlerRegistry(
        [
            BackendSpec("hdf5", HDF5Reader, HDF5Writer, capabilities=frozenset({"compression", "multiprocessing_safe"}), aliases=("legacy_hdf5",)),
            BackendSpec("packed_hdf5", PackedHDF5Reader, PackedHDF5Writer, capabilities=frozenset({"variable_shape", "mixed_dtype", "components", "compression"})),
            BackendSpec("homogeneous_hdf5", HomogeneousHDF5Reader, HomogeneousHDF5Writer, capabilities=frozenset({"batch_read", "compression", "multiprocessing_safe"})),
            BackendSpec("packed_npy", PackedNPYReader, PackedNPYWriter, capabilities=frozenset({"variable_shape", "mixed_dtype", "memory_mapping", "multiprocessing_safe"})),
            BackendSpec(
                "structured_hdf5",
                StructuredHDF5Reader,
                StructuredHDF5Writer,
                capabilities=frozenset({"variable_shape", "mixed_dtype", "components", "batch_read", "compression", "multiprocessing_safe"}),
            ),
            BackendSpec("npy", NPYReader, capabilities=frozenset({"memory_mapping"})),
            BackendSpec("wav", WAVReader, capabilities=frozenset({"variable_shape", "multiprocessing_safe"})),
            BackendSpec("ogg", OGGReader, capabilities=frozenset({"variable_shape", "multiprocessing_safe"})),
            BackendSpec("sigmf", SigMFReader, capabilities=frozenset({"variable_shape"})),
        ]
    )


file_handler_registry = _builtin_registry()
