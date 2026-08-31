"""Packed HDF5 reader and writer for TorchSig signals.

Unlike the current one-dataset-per-signal layout, this format stores signal
sample arrays, record descriptors, component links, and metadata in a small
set of appendable datasets. Signal data is not restricted to one-dimensional
complex IQ: any non-object NumPy-compatible array is flattened for storage and
restored to its original shape and dtype when read. This includes real or
complex multidimensional data such as spectrograms.

The frozen on-disk format identifier is ``torchsig-packed`` and its current
schema version is ``1.0``. It is not compatible with the legacy
object-per-record layout in :mod:`torchsig.utils.file_handlers.hdf5`.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.dsp import torchsig_cache_version
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.hdf5_schema import (
    PackedHDF5Schema,
    default_packed_schema,
    read_schema,
    write_schema,
)
from torchsig.utils.file_handlers.metadata_codec import (
    decode_metadata,
    encode_metadata,
)

__all__ = ["PackedHDF5Reader", "PackedHDF5Writer"]

_RECORD_DTYPE = np.dtype(
    [
        ("data_offset", np.uint64),
        ("data_length", np.uint64),
        ("dtype_id", np.uint32),
        ("shape_offset", np.uint64),
        ("shape_count", np.uint16),
        ("component_offset", np.uint64),
        ("component_count", np.uint32),
        ("parent_id", np.uint64),
    ]
)
_PARENT_DTYPE = np.dtype([("parent_id", np.uint64)])
_VALIDATION_CHUNK_SIZE = 65_536


class _WriterState(Enum):
    """Lifecycle state of a packed HDF5 writer."""

    NEW = auto()
    OPEN = auto()
    FAILED = auto()
    CLOSED = auto()


@dataclass
class _PreparedBatch:
    """Validated in-memory representation of one append operation."""

    top_ids: np.ndarray
    records: np.ndarray
    metadata: list[str]
    shapes: np.ndarray
    components: np.ndarray
    arrays_by_dtype: dict[int, np.ndarray]
    new_dtypes: list[tuple[int, np.dtype]]
    new_parents: list[tuple[int, int, str]]
    dtype_ids: dict[str, int]
    parent_ids: dict[tuple[str, int], int]


def _append(dataset: h5py.Dataset, values: Any) -> int:
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values
    return start


def _validate_declared_datasets(file: h5py.File, schema: PackedHDF5Schema) -> None:
    """Ensure required logical specifications and physical paths exist."""
    required = set(default_packed_schema().datasets)
    missing_specs = required - set(schema.datasets)
    if missing_specs:
        raise ValueError(f"Packed HDF5 schema is missing dataset specifications: {sorted(missing_specs)}")
    missing_paths = [item.path for item in schema.datasets.values() if item.path not in file]
    if missing_paths:
        raise ValueError(f"Packed HDF5 file is missing declared paths: {missing_paths}")


def _validate_physical_schema(file: h5py.File, schema: PackedHDF5Schema) -> None:
    """Ensure declared objects have compatible HDF5 types and shapes."""
    paths: dict[str, str] = {}
    for logical_name, specification in schema.datasets.items():
        canonical_path = file[specification.path].name
        if canonical_path in paths:
            raise ValueError(f"Invalid packed HDF5 schema: logical datasets {paths[canonical_path]!r} and {logical_name!r} share path {canonical_path!r}")
        paths[canonical_path] = logical_name

    data_group = file[schema.datasets["data"].path]
    if not isinstance(data_group, h5py.Group):
        raise ValueError(  # noqa: TRY004
            "Invalid packed HDF5 file: data path must be a group"
        )

    for logical_name in ("index", "shapes", "components"):
        dataset = file[schema.datasets[logical_name].path]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
            raise ValueError(f"Invalid packed HDF5 file: {logical_name} must be a one-dimensional dataset")
        if dataset.dtype != np.dtype(np.uint64):
            raise ValueError(f"Invalid packed HDF5 file: {logical_name} must have dtype uint64")

    for logical_name in ("dtypes", "metadata", "parent_metadata"):
        dataset = file[schema.datasets[logical_name].path]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
            raise ValueError(f"Invalid packed HDF5 file: {logical_name} must be a one-dimensional dataset")
        string_info = h5py.check_string_dtype(dataset.dtype)
        if string_info is None or string_info.encoding != "utf-8":
            raise ValueError(f"Invalid packed HDF5 file: {logical_name} must contain UTF-8 strings")

    for logical_name, expected_dtype in (
        ("records", _RECORD_DTYPE),
        ("parent_records", _PARENT_DTYPE),
    ):
        specification = schema.datasets[logical_name]
        dataset = file[specification.path]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
            raise ValueError(f"Invalid packed HDF5 file: {logical_name} must be a one-dimensional dataset")
        if specification.fields is None:
            raise ValueError(f"Invalid packed HDF5 schema: missing {logical_name} field mappings")
        for logical_field, expected_field_dtype in expected_dtype.fields.items():
            physical_field = specification.fields.get(logical_field)
            actual_fields = dataset.dtype.fields or {}
            if physical_field not in actual_fields:
                raise ValueError(f"Invalid packed HDF5 file: {logical_name} is missing field {physical_field!r}")
            actual_field_dtype = actual_fields[physical_field][0]
            if actual_field_dtype != expected_field_dtype[0]:
                raise ValueError(f"Invalid packed HDF5 file: {logical_name} field {physical_field!r} has incompatible dtype")

    no_parent = schema.sentinels.get("no_parent")
    if no_parent is None or not 0 <= no_parent <= np.iinfo(np.uint64).max:
        raise ValueError("Invalid packed HDF5 schema: no_parent sentinel does not fit uint64")

    for name, stream in data_group.items():
        if not isinstance(stream, h5py.Dataset) or stream.ndim != 1:
            raise ValueError(f"Invalid packed HDF5 file: data stream {name!r} must be a one-dimensional dataset")


def _validate_complete_file(file: h5py.File) -> None:
    """Reject files which were not finalized by a successful writer."""
    if "complete" not in file.attrs:
        raise ValueError("Invalid packed HDF5 file: missing completeness marker")
    if not bool(file.attrs["complete"]):
        raise ValueError("Packed HDF5 file is incomplete")


def _validate_acyclic_links(links: list[list[int]], *, relationship: str) -> None:
    """Reject cycles in a table of record relationships."""
    unvisited = 0
    visiting = 1
    visited = 2
    states = np.zeros(len(links), dtype=np.uint8)

    for record_id in range(len(links)):
        if states[record_id] != unvisited:
            continue
        stack = [(record_id, False)]
        while stack:
            linked_id, exiting = stack.pop()
            if exiting:
                states[linked_id] = visited
                continue
            if states[linked_id] == visiting:
                raise ValueError(f"Invalid packed HDF5 file: {relationship} cycle at record {linked_id}")
            if states[linked_id] == visited:
                continue
            states[linked_id] = visiting
            stack.append((linked_id, True))
            stack.extend((child_id, False) for child_id in reversed(links[linked_id]))


class PackedHDF5Writer(FileWriter):
    """Write signal arrays into a small set of appendable HDF5 datasets.

    Each :class:`~torchsig.signals.signal_types.Signal` data array is flattened
    into a stream shared by arrays of the same NumPy dtype. Its original shape
    is stored separately and reconstructed by :class:`PackedHDF5Reader`.
    Batches may therefore mix dtypes and shapes, including one-dimensional
    complex IQ and multidimensional real or complex spectrogram data. NumPy
    object arrays are not supported.

    Component counts, component shapes, and component dtypes may vary.
    Hierarchical parent relationships are preserved on disk and reconstructed.
    """

    def __init__(
        self,
        root,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 10 * 1024 * 1024,
        max_batches_in_memory: int = 4,
    ) -> None:
        """Initialize a packed HDF5 writer.

        Args:
            root: Directory in which ``data.h5`` is created.
            compression: HDF5 compression filter name, or ``None``.
            compression_opts: Options passed to the selected compression
                filter.
            shuffle: Whether to apply the HDF5 shuffle filter.
            fletcher32: Whether to apply the Fletcher32 checksum filter.
            chunk_cache_size: HDF5 raw chunk cache size in bytes.
            max_batches_in_memory: Maximum out-of-order batches retained while
                waiting for a contiguous batch-index prefix.
        """
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_cache_size = chunk_cache_size
        if not isinstance(max_batches_in_memory, int) or isinstance(max_batches_in_memory, bool):
            raise TypeError("Packed HDF5 max_batches_in_memory must be an integer")
        if max_batches_in_memory < 1:
            raise ValueError("Packed HDF5 max_batches_in_memory must be positive")
        self.max_batches_in_memory = max_batches_in_memory
        self._file: h5py.File | None = None
        self._data: dict[int, h5py.Dataset] = {}
        self._dtype_ids: dict[str, int] = {}
        self._batch_buffer: dict[int, list[Signal]] = {}
        self._next_batch_idx = 0
        self._parent_ids: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()
        self._write_failed = False
        self._state = _WriterState.NEW
        self.schema = default_packed_schema()

    def setup(self) -> None:
        """Create a new packed file and transition the writer to open."""
        if self._state is not _WriterState.NEW:
            raise RuntimeError(f"Packed HDF5 writer setup is only valid for a new writer; current state is {self._state.name.lower()}")
        try:
            super().setup()
        except Exception:
            self._write_failed = True
            self._state = _WriterState.FAILED
            if self._file is not None:
                self._file.close()
                self._file = None
            raise
        self._state = _WriterState.OPEN

    def _setup(self) -> None:
        self._file = h5py.File(
            self.datapath,
            "w",
            libver="latest",
            rdcc_nbytes=self.chunk_cache_size,
            rdcc_w0=0.75,
        )
        self._file.attrs["torchsig_version"] = torchsig_cache_version()
        self._file.attrs["format"] = self.schema.format
        self._file.attrs["compression"] = self.compression or "none"
        self._file.attrs["complete"] = False
        write_schema(self._file, self.schema)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        spec = self.schema.datasets
        self._data_group = self._file.create_group(spec["data"].path)
        self._dtypes = self._file.create_dataset(
            spec["dtypes"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._records = self._file.create_dataset(
            spec["records"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=_RECORD_DTYPE,
            chunks=True,
        )
        self._metadata = self._file.create_dataset(
            spec["metadata"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._components = self._file.create_dataset(
            spec["components"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._shapes = self._file.create_dataset(
            spec["shapes"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._index = self._file.create_dataset(
            spec["index"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._parent_records = self._file.create_dataset(
            spec["parent_records"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=_PARENT_DTYPE,
            chunks=True,
        )
        self._parent_metadata = self._file.create_dataset(
            spec["parent_metadata"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._parent_ids.clear()
        self._dtype_ids.clear()
        self._data.clear()
        self._batch_buffer.clear()
        self._next_batch_idx = 0
        self._write_failed = False

    def _data_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"chunks": True}
        if self.compression is not None:
            kwargs["compression"] = self.compression
            if self.compression != "lzf" and self.compression_opts is not None:
                kwargs["compression_opts"] = self.compression_opts
        if self.shuffle:
            kwargs["shuffle"] = True
        if self.fletcher32:
            kwargs["fletcher32"] = True
        return kwargs

    def _prepare_batch(self, signals: list[Signal]) -> _PreparedBatch | None:
        """Validate and encode a batch without modifying the HDF5 file."""
        flattened: list[Signal] = []
        component_ids: list[list[int]] = []

        def add_signal(signal: Signal) -> int:
            active_components: set[int] = set()
            root_id = len(self._records) + len(flattened)
            stack: list[tuple[Signal, list[int] | None, bool]] = [(signal, None, False)]
            while stack:
                current, destination, exiting = stack.pop()
                current_identity = id(current)
                if exiting:
                    active_components.remove(current_identity)
                    continue
                if not isinstance(current, Signal):
                    raise TypeError("Packed HDF5 batches must contain Signal instances")
                if current_identity in active_components:
                    raise ValueError("Packed HDF5 component signal cycle detected")
                active_components.add(current_identity)
                record_id = len(self._records) + len(flattened)
                flattened.append(current)
                component_ids.append([])
                if destination is not None:
                    destination.append(record_id)
                children = component_ids[-1]
                stack.append((current, None, True))
                stack.extend((item, children, False) for item in reversed(current.component_signals))
            return root_id

        top_ids = [add_signal(signal) for signal in signals]
        if not flattened:
            return None

        dtype_ids = self._dtype_ids.copy()
        parent_ids = self._parent_ids.copy()
        new_dtypes: list[tuple[int, np.dtype]] = []
        new_parents: list[tuple[int, int, str]] = []

        def prepare_dtype(dtype: np.dtype) -> int:
            if dtype.hasobject:
                raise TypeError("Packed HDF5 does not support object signal dtypes")
            dtype_string = dtype.str
            if dtype_string not in dtype_ids:
                dtype_id = len(dtype_ids)
                if dtype_id > np.iinfo(np.uint32).max:
                    raise OverflowError("Packed HDF5 has too many signal dtypes")
                dtype_ids[dtype_string] = dtype_id
                new_dtypes.append((dtype_id, dtype))
            return dtype_ids[dtype_string]

        def prepare_parent(
            parent: HierarchicalMetadataObject | None,
        ) -> int:
            if parent is None:
                return self.schema.sentinels["no_parent"]
            chain = []
            active_parents: set[int] = set()
            current = parent
            while current is not None:
                current_identity = id(current)
                if current_identity in active_parents:
                    raise ValueError("Packed HDF5 parent metadata cycle detected")
                active_parents.add(current_identity)
                chain.append(current)
                current = current.parent

            ancestor_id = self.schema.sentinels["no_parent"]
            for current in reversed(chain):
                encoded_metadata = encode_metadata(current)
                parent_key = (encoded_metadata, ancestor_id)
                if parent_key not in parent_ids:
                    parent_id = len(parent_ids)
                    parent_ids[parent_key] = parent_id
                    new_parents.append((parent_id, ancestor_id, encoded_metadata))
                ancestor_id = parent_ids[parent_key]
            return ancestor_id

        arrays_by_dtype: dict[int, list[np.ndarray]] = {}
        offsets_by_dtype: dict[int, int] = {}
        records = np.empty(len(flattened), dtype=_RECORD_DTYPE)
        metadata: list[str] = []
        links: list[int] = []
        shapes: list[int] = []
        component_offset = len(self._components)
        shape_offset = len(self._shapes)
        for idx, signal in enumerate(flattened):
            array = np.asarray(signal.data)
            if array.ndim > np.iinfo(np.uint16).max:
                raise OverflowError("Packed HDF5 signal has too many dimensions")
            children = component_ids[idx]
            if len(children) > np.iinfo(np.uint32).max:
                raise OverflowError("Packed HDF5 signal has too many component signals")
            encoded_metadata = encode_metadata(signal)
            parent_id = prepare_parent(signal.parent)
            dtype_id = prepare_dtype(array.dtype)
            if dtype_id not in offsets_by_dtype:
                offsets_by_dtype[dtype_id] = len(self._data[dtype_id]) if dtype_id in self._data else 0
                arrays_by_dtype[dtype_id] = []
            data_offset = offsets_by_dtype[dtype_id]
            arrays_by_dtype[dtype_id].append(array.reshape(-1))
            records[idx] = (
                data_offset,
                array.size,
                dtype_id,
                shape_offset + len(shapes),
                array.ndim,
                component_offset + len(links),
                len(children),
                parent_id,
            )
            offsets_by_dtype[dtype_id] += array.size
            shapes.extend(array.shape)
            links.extend(children)
            metadata.append(encoded_metadata)

        return _PreparedBatch(
            top_ids=np.asarray(top_ids, dtype=np.uint64),
            records=records,
            metadata=metadata,
            shapes=np.asarray(shapes, dtype=np.uint64),
            components=np.asarray(links, dtype=np.uint64),
            arrays_by_dtype={dtype_id: np.concatenate(arrays) for dtype_id, arrays in arrays_by_dtype.items()},
            new_dtypes=new_dtypes,
            new_parents=new_parents,
            dtype_ids=dtype_ids,
            parent_ids=parent_ids,
        )

    def _commit_batch(self, batch: _PreparedBatch) -> None:
        """Append an already validated batch to the open HDF5 file."""
        datasets = {
            "dtypes": self._dtypes,
            "records": self._records,
            "metadata": self._metadata,
            "shapes": self._shapes,
            "components": self._components,
            "index": self._index,
            "parent_records": self._parent_records,
            "parent_metadata": self._parent_metadata,
        }
        dataset_lengths = {name: len(dataset) for name, dataset in datasets.items()}
        data_lengths = {dtype_id: len(dataset) for dtype_id, dataset in self._data.items()}
        existing_data_ids = set(self._data)
        dtype_ids = self._dtype_ids.copy()
        parent_ids = self._parent_ids.copy()

        try:
            for dtype_id, dtype in batch.new_dtypes:
                _append(self._dtypes, [dtype.str])
                self._data[dtype_id] = self._data_group.create_dataset(
                    str(dtype_id),
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    **self._data_kwargs(),
                )
            self._dtype_ids = batch.dtype_ids

            for _, ancestor_id, encoded_metadata in batch.new_parents:
                _append(
                    self._parent_records,
                    np.array([(ancestor_id,)], dtype=_PARENT_DTYPE),
                )
                _append(self._parent_metadata, [encoded_metadata])
            self._parent_ids = batch.parent_ids

            for dtype_id, array in batch.arrays_by_dtype.items():
                _append(self._data[dtype_id], array)
            _append(self._records, batch.records)
            _append(self._metadata, batch.metadata)
            if len(batch.shapes):
                _append(self._shapes, batch.shapes)
            if len(batch.components):
                _append(self._components, batch.components)
            _append(self._index, batch.top_ids)
        except Exception as error:
            self._write_failed = True
            self._state = _WriterState.FAILED
            rollback_errors = []

            def attempt_rollback(target: str, operation: Callable[[], None]) -> None:
                try:
                    operation()
                except Exception as rollback_error:  # noqa: BLE001  # pragma: no cover
                    rollback_errors.append(f"{target}: {rollback_error}")

            for name, dataset in datasets.items():
                attempt_rollback(
                    name,
                    lambda dataset=dataset, length=dataset_lengths[name]: dataset.resize(length, axis=0),
                )
            for dtype_id in existing_data_ids:
                attempt_rollback(
                    f"data/{dtype_id}",
                    lambda dtype_id=dtype_id: self._data[dtype_id].resize(data_lengths[dtype_id], axis=0),
                )
            for dtype_id in set(self._data) - existing_data_ids:
                attempt_rollback(
                    f"data/{dtype_id}",
                    lambda dtype_id=dtype_id: self._data_group.__delitem__(str(dtype_id)),
                )
            self._data = {dtype_id: self._data[dtype_id] for dtype_id in existing_data_ids}
            self._dtype_ids = dtype_ids
            self._parent_ids = parent_ids
            if rollback_errors:
                error.add_note("Packed HDF5 rollback errors: " + "; ".join(rollback_errors))
            raise

    def _write_batch(self, signals: list[Signal]) -> None:
        batch = self._prepare_batch(signals)
        if batch is not None:
            self._commit_batch(batch)

    def _flush_buffer(self, *, final: bool = False) -> None:
        with self._lock:
            while self._next_batch_idx in self._batch_buffer:
                signals = self._batch_buffer.pop(self._next_batch_idx)
                self._write_batch(signals)
                self._next_batch_idx += 1
            if final and self._batch_buffer:
                pending = sorted(self._batch_buffer)
                raise ValueError(f"Cannot finalize packed HDF5 file: missing batch index {self._next_batch_idx}; pending batch indices: {pending}")
            if self._file is not None:
                self._file.flush()

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Buffer a uniquely indexed batch and write each contiguous prefix.

        Batch indices must be non-negative and form a contiguous sequence
        beginning at zero. Batches may arrive out of order, but a batch is not
        committed until every preceding batch has arrived. At most
        ``max_batches_in_memory`` out-of-order batches may be buffered; the
        next expected batch is always accepted because it can drain the buffer.
        """
        if self._state is _WriterState.FAILED:
            raise RuntimeError("Packed HDF5 writer cannot continue after a failed write")
        if self._state is not _WriterState.OPEN:
            raise RuntimeError(f"Packed HDF5 writer is not open; current state is {self._state.name.lower()}")
        if not isinstance(batch_idx, int) or isinstance(batch_idx, bool):
            raise TypeError("Packed HDF5 batch index must be an integer")
        if batch_idx < 0:
            raise ValueError("Packed HDF5 batch index must be non-negative")
        with self._lock:
            if batch_idx < self._next_batch_idx or batch_idx in self._batch_buffer:
                raise ValueError(f"Duplicate packed HDF5 batch index: {batch_idx}")
            if batch_idx != self._next_batch_idx and len(self._batch_buffer) >= self.max_batches_in_memory:
                raise BufferError(f"Packed HDF5 out-of-order batch buffer is full; expected batch index {self._next_batch_idx}")
            self._batch_buffer[batch_idx] = list(data)
            should_flush = len(self._batch_buffer) >= self.max_batches_in_memory
        if should_flush:
            try:
                self._flush_buffer()
            except Exception:
                self._write_failed = True
                self._state = _WriterState.FAILED
                raise

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        if self._state is not _WriterState.OPEN:
            raise RuntimeError(f"Packed HDF5 writer length is only available while open; current state is {self._state.name.lower()}")
        return len(self._index)

    def teardown(self) -> None:
        """Flush pending batches and close the packed file."""
        if self._state is _WriterState.CLOSED:
            return
        if self._file is None:
            self._state = _WriterState.CLOSED
            return
        try:
            if self._state is _WriterState.OPEN:
                self._flush_buffer(final=True)
                self._file.attrs["complete"] = True
                self._file.flush()
        except Exception:
            self._write_failed = True
            self._state = _WriterState.FAILED
            raise
        finally:
            self._file.close()
            self._file = None
            self._data.clear()
            self._state = _WriterState.CLOSED

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file while preserving an incomplete marker on failure."""
        if exc_type is not None:
            self._write_failed = True
            self._state = _WriterState.FAILED
            if self._file is not None:
                self._file.close()
                self._file = None
                self._data.clear()
            self._state = _WriterState.CLOSED
            return False
        self.teardown()
        return False


class PackedHDF5Reader(FileReader):
    """Read signals from the packed HDF5 schema.

    Top-level and component arrays are reconstructed with their original NumPy
    dtype and shape. This supports IQ, wideband arrays, spectrograms, and other
    non-object NumPy-compatible arrays. Parent metadata hierarchy and shared
    parent relationships are reconstructed. Python object identity is not
    guaranteed across separate calls to :meth:`read`.
    """

    def __init__(self, root) -> None:
        """Initialize a lazy reader for ``root/data.h5``."""
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file: h5py.File | None = None
        self._len_cache: int | None = None
        self._parent_cache: dict[int, tuple[dict[str, Any], int]] = {}
        self._metadata_cache: dict[int, dict[str, Any]] = {}
        self._locking = False
        self.schema: PackedHDF5Schema | None = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.datapath, "r", locking=self._locking)
            try:
                self.schema = read_schema(self._file)
                _validate_complete_file(self._file)
                spec = self.schema.datasets
                _validate_declared_datasets(self._file, self.schema)
                _validate_physical_schema(self._file, self.schema)
                self._records = self._file[spec["records"].path]
                self._metadata = self._file[spec["metadata"].path]
                self._components = self._file[spec["components"].path]
                self._shapes = self._file[spec["shapes"].path]
                self._index = self._file[spec["index"].path]
                self._data = self._file[spec["data"].path]
                self._data_streams = {int(dtype_id): self._data[dtype_id] for dtype_id in self._data}
                self._dtypes = self._file[spec["dtypes"].path]
                self._parent_records = self._file[spec["parent_records"].path]
                self._parent_metadata = self._file[spec["parent_metadata"].path]
                self._record_fields = spec["records"].fields
                self._parent_fields = spec["parent_records"].fields
                self._no_parent = self.schema.sentinels["no_parent"]
                self._validate_integrity()
            except Exception:
                self._file.close()
                self._file = None
                raise

    def _validate_integrity(self) -> None:
        """Validate cross-dataset references before serving any records."""
        if len(self._records) != len(self._metadata):
            raise ValueError("Invalid packed HDF5 file: records and metadata lengths differ")
        if len(self._parent_records) != len(self._parent_metadata):
            raise ValueError("Invalid packed HDF5 file: parent record and metadata lengths differ")

        required_record_fields = set(_RECORD_DTYPE.names or ())
        if self._record_fields is None or required_record_fields - set(self._record_fields):
            raise ValueError("Invalid packed HDF5 schema: missing record field mappings")
        required_parent_fields = set(_PARENT_DTYPE.names or ())
        if self._parent_fields is None or required_parent_fields - set(self._parent_fields):
            raise ValueError("Invalid packed HDF5 schema: missing parent field mappings")
        missing_record_fields = set(self._record_fields.values()) - set(self._records.dtype.names or ())
        if missing_record_fields:
            raise ValueError(f"Invalid packed HDF5 file: record dataset is missing fields {sorted(missing_record_fields)}")
        missing_parent_fields = set(self._parent_fields.values()) - set(self._parent_records.dtype.names or ())
        if missing_parent_fields:
            raise ValueError(f"Invalid packed HDF5 file: parent record dataset is missing fields {sorted(missing_parent_fields)}")

        dtype_count = len(self._dtypes)
        expected_dtype_ids = set(range(dtype_count))
        if set(self._data_streams) != expected_dtype_ids:
            raise ValueError("Invalid packed HDF5 file: data stream IDs do not match the dtype table")
        for dtype_id, stream in self._data_streams.items():
            dtype_value = self._dtypes[dtype_id]
            if isinstance(dtype_value, bytes):
                dtype_value = dtype_value.decode("utf-8")
            try:
                declared_dtype = np.dtype(dtype_value)
            except TypeError as error:
                raise ValueError(f"Invalid packed HDF5 file: invalid dtype descriptor at index {dtype_id}") from error
            if stream.dtype != declared_dtype:
                raise ValueError(f"Invalid packed HDF5 file: data stream dtype mismatch at index {dtype_id}")

        record_count = len(self._records)
        parent_count = len(self._parent_records)
        component_links: list[list[int]] = []
        fields = self._record_fields
        for chunk_start in range(0, record_count, _VALIDATION_CHUNK_SIZE):
            chunk_stop = min(chunk_start + _VALIDATION_CHUNK_SIZE, record_count)
            records = self._records[chunk_start:chunk_stop]
            dtype_ids = records[fields["dtype_id"]]
            invalid = np.flatnonzero(dtype_ids >= dtype_count)
            if len(invalid):
                offset = int(invalid[0])
                record_id = chunk_start + offset
                raise ValueError(f"Invalid packed HDF5 file: invalid dtype ID {int(dtype_ids[offset])} in record {record_id}")

            data_starts = records[fields["data_offset"]]
            data_lengths = records[fields["data_length"]]
            for dtype_id in np.unique(dtype_ids):
                selected = np.flatnonzero(dtype_ids == dtype_id)
                stream_length = len(self._data_streams[int(dtype_id)])
                starts = data_starts[selected]
                lengths = data_lengths[selected]
                out_of_bounds = starts > stream_length
                in_bounds = ~out_of_bounds
                out_of_bounds[in_bounds] = lengths[in_bounds] > stream_length - starts[in_bounds]
                invalid = np.flatnonzero(out_of_bounds)
                if len(invalid):
                    record_id = chunk_start + int(selected[int(invalid[0])])
                    raise ValueError(f"Invalid packed HDF5 file: data slice out of bounds in record {record_id}")

            shape_starts = records[fields["shape_offset"]]
            shape_counts = records[fields["shape_count"]]
            shape_out_of_bounds = shape_starts > len(self._shapes)
            shape_in_bounds = ~shape_out_of_bounds
            shape_out_of_bounds[shape_in_bounds] = shape_counts[shape_in_bounds] > len(self._shapes) - shape_starts[shape_in_bounds]
            invalid = np.flatnonzero(shape_out_of_bounds)
            if len(invalid):
                record_id = chunk_start + int(invalid[0])
                raise ValueError(f"Invalid packed HDF5 file: shape slice out of bounds in record {record_id}")
            if len(records):
                shape_data_start = int(np.min(shape_starts))
                shape_data_stop = int(np.max(shape_starts + shape_counts))
                shape_values = self._shapes[shape_data_start:shape_data_stop]
                for offset, (shape_start, shape_count, data_length) in enumerate(zip(shape_starts, shape_counts, data_lengths, strict=True)):
                    relative_start = int(shape_start) - shape_data_start
                    relative_stop = relative_start + int(shape_count)
                    if int(np.prod(shape_values[relative_start:relative_stop], dtype=np.uint64)) != int(data_length):
                        record_id = chunk_start + offset
                        raise ValueError(f"Invalid packed HDF5 file: shape does not match data length in record {record_id}")

            component_starts = records[fields["component_offset"]]
            component_counts = records[fields["component_count"]]
            component_out_of_bounds = component_starts > len(self._components)
            component_in_bounds = ~component_out_of_bounds
            component_out_of_bounds[component_in_bounds] = component_counts[component_in_bounds] > len(self._components) - component_starts[component_in_bounds]
            invalid = np.flatnonzero(component_out_of_bounds)
            if len(invalid):
                record_id = chunk_start + int(invalid[0])
                raise ValueError(f"Invalid packed HDF5 file: component slice out of bounds in record {record_id}")
            if np.any(component_counts):
                component_data_start = int(np.min(component_starts))
                component_data_stop = int(np.max(component_starts + component_counts))
                component_values = self._components[component_data_start:component_data_stop]
                for offset, (component_start, component_count) in enumerate(zip(component_starts, component_counts, strict=True)):
                    relative_start = int(component_start) - component_data_start
                    relative_stop = relative_start + int(component_count)
                    children = component_values[relative_start:relative_stop]
                    if np.any(children >= record_count):
                        record_id = chunk_start + offset
                        raise ValueError(f"Invalid packed HDF5 file: invalid component record ID in record {record_id}")
                    component_links.append([int(value) for value in children])
            else:
                component_links.extend([] for _ in range(len(records)))

            parent_ids = records[fields["parent_id"]]
            invalid = np.flatnonzero((parent_ids != self._no_parent) & (parent_ids >= parent_count))
            if len(invalid):
                offset = int(invalid[0])
                record_id = chunk_start + offset
                raise ValueError(f"Invalid packed HDF5 file: invalid parent ID {int(parent_ids[offset])} in record {record_id}")

        for chunk_start in range(0, len(self._index), _VALIDATION_CHUNK_SIZE):
            top_ids = self._index[chunk_start : chunk_start + _VALIDATION_CHUNK_SIZE]
            if np.any(top_ids >= record_count):
                raise ValueError("Invalid packed HDF5 file: invalid top-level record ID")
        _validate_acyclic_links(component_links, relationship="component relationship")

        parent_field = self._parent_fields["parent_id"]
        parent_values = self._parent_records[:][parent_field]
        invalid = np.flatnonzero((parent_values != self._no_parent) & (parent_values >= parent_count))
        if len(invalid):
            parent_id = int(invalid[0])
            ancestor_id = int(parent_values[parent_id])
            raise ValueError(f"Invalid packed HDF5 file: invalid ancestor ID {ancestor_id} in parent record {parent_id}")
        parent_links: list[list[int]] = []
        for ancestor_value in parent_values:
            ancestor_id = int(ancestor_value)
            if ancestor_id == self._no_parent:
                parent_links.append([])
            else:
                parent_links.append([ancestor_id])
        _validate_acyclic_links(parent_links, relationship="parent relationship")

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        self._ensure_open()
        if self._len_cache is None:
            self._len_cache = len(self._index)
        return self._len_cache

    def _build_parent(self, parent_id: int) -> HierarchicalMetadataObject | None:
        if parent_id == self._no_parent:
            return None
        chain = []
        while parent_id != self._no_parent:
            try:
                metadata, ancestor_id = self._parent_cache[parent_id]
            except KeyError:
                record = self._parent_records[parent_id]
                ancestor_id = int(record[self._parent_fields["parent_id"]])
                metadata = decode_metadata(self._parent_metadata[parent_id])
                self._parent_cache[parent_id] = (metadata, ancestor_id)
            chain.append(metadata)
            parent_id = ancestor_id

        parent = None
        for metadata in reversed(chain):
            current = HierarchicalMetadataObject(metadata=deepcopy(metadata))
            if parent is not None:
                current.add_parent(parent, register=False)
            parent = current
        return parent

    def _read_record(self, record_id: int) -> Signal:
        stack: list[tuple[int, np.void | None, int]] = [(record_id, None, 0)]
        built: list[Signal] = []
        while stack:
            current_id, record, component_count = stack.pop()
            if record is None:
                record = self._records[current_id]
                fields = self._record_fields
                component_start = int(record[fields["component_offset"]])
                component_count = int(record[fields["component_count"]])
                component_ids = self._components[component_start : component_start + component_count]
                stack.append((current_id, record, component_count))
                stack.extend((int(component_id), None, 0) for component_id in reversed(component_ids))
                continue

            if component_count:
                component_signals = built[-component_count:]
                del built[-component_count:]
            else:
                component_signals = []
            built.append(self._build_record_signal(current_id, record, component_signals))
        return built[0]

    def _build_record_signal(
        self,
        record_id: int,
        record: np.void,
        component_signals: list[Signal],
    ) -> Signal:
        """Construct one signal after its component signals are available."""
        fields = self._record_fields
        data_start = int(record[fields["data_offset"]])
        data_stop = data_start + int(record[fields["data_length"]])
        dtype_id = int(record[fields["dtype_id"]])
        shape_start = int(record[fields["shape_offset"]])
        shape_stop = shape_start + int(record[fields["shape_count"]])
        shape = tuple(int(value) for value in self._shapes[shape_start:shape_stop])
        try:
            metadata = self._metadata_cache[record_id]
        except KeyError:
            metadata = decode_metadata(self._metadata[record_id])
            self._metadata_cache[record_id] = metadata
        signal = Signal(
            data=self._data_streams[dtype_id][data_start:data_stop].reshape(shape),
            component_signals=component_signals,
            metadata=deepcopy(metadata),
        )
        parent = self._build_parent(int(record[fields["parent_id"]]))
        if parent is not None:
            signal.add_parent(parent, register=False)
        return signal

    def read(self, idx: int) -> Signal:
        """Read a top-level signal by dataset index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Packed HDF5 sample index out of range: {idx}")
        return self._read_record(int(self._index[idx]))

    def teardown(self) -> None:
        """Close the packed file and clear cached parent metadata."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._len_cache = None
        self.schema = None
        self._parent_cache.clear()
        self._metadata_cache.clear()
