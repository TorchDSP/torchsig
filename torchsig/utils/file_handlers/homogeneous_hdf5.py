"""HDF5 format with homogeneous top-level signal arrays.

Top-level arrays share one fixed dtype and shape and are stored in a native
``(num_signals, *signal_shape)`` dataset. Component signals remain ragged:
each sample indexes a variable-length range of component records whose sample
arrays are stored in flattened dtype streams.

Parent metadata is flattened into each signal's own metadata during writing;
parent object identity and hierarchy are therefore not reconstructed. The
format does not support nested component signals. It is separate from the
versioned packed HDF5 schema. Its frozen format identifier is
``torchsig-homogeneous`` and its current schema version is ``1``.
"""

from __future__ import annotations

import os
from math import prod
from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.metadata_codec import (
    decode_metadata,
    encode_metadata,
)

__all__ = ["HomogeneousHDF5Reader", "HomogeneousHDF5Writer"]

_FORMAT = "torchsig-homogeneous"
_SCHEMA_VERSION = 1
_TARGET_CHUNK_BYTES = 1024**2
_LARGE_COMPRESSED_SAMPLE_BYTES = 64 * 1024
_VALIDATION_CHUNK_RECORDS = 65_536
_VALIDATION_CACHE_SIZE = 128
_VALIDATED_FILES: dict[str, tuple[int, int, int, int]] = {}
_COMPONENT_DTYPE = np.dtype(
    [
        ("data_offset", np.uint64),
        ("data_length", np.uint64),
        ("dtype_id", np.uint32),
        ("shape_offset", np.uint64),
        ("shape_count", np.uint16),
    ]
)


def _append(dataset: h5py.Dataset, values: Any) -> int:
    """Append values to a one-dimensional extensible dataset."""
    start = len(dataset)
    if len(values) == 0:
        return start
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values
    return start


def _encode_flat_metadata(signal: Signal) -> str:
    chain = []
    active: set[int] = set()
    current: HierarchicalMetadataObject | None = signal
    while current is not None:
        identity = id(current)
        if identity in active:
            raise ValueError("Homogeneous HDF5 parent metadata cycle detected")
        active.add(identity)
        chain.append(current)
        current = current.parent
    values = {}
    for item in reversed(chain):
        values.update({key: item[key] for key in item.keys()})  # noqa: SIM118
    metadata = HierarchicalMetadataObject(
        metadata=values,
    )
    return encode_metadata(metadata)


class HomogeneousHDF5Writer(FileWriter):
    """Write fixed-shape top-level arrays with ragged component signals.

    Every top-level signal must have the same NumPy dtype and shape. Component
    counts, component shapes, and component dtypes may vary by sample. The
    effective metadata inherited from parent objects is flattened into each
    stored signal; parent hierarchy and identity are not retained. The
    ``chunk_samples`` argument is an upper bound; large observations use fewer
    samples per chunk to keep top-level chunks at or below roughly one MiB.
    Compressed observations of at least 64 KiB use one sample per chunk to
    avoid decompressing unrelated samples during random reads.
    """

    def __init__(
        self,
        root,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_samples: int = 32,
    ) -> None:
        """Initialize a homogeneous HDF5 writer.

        Args:
            root: Directory in which ``data.h5`` is created.
            compression: HDF5 compression filter name, or ``None``.
            compression_opts: Options passed to the selected compression
                filter.
            shuffle: Whether to apply the HDF5 shuffle filter.
            fletcher32: Whether to apply the Fletcher32 checksum filter.
            chunk_samples: Upper bound on top-level samples per HDF5 chunk.
        """
        super().__init__(root=root)
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")
        self.datapath = self.root / "data.h5"
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_samples = chunk_samples
        self._file: h5py.File | None = None
        self._data: h5py.Dataset | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self._next_batch_idx = 0
        self._component_data: dict[int, h5py.Dataset] = {}
        self._component_dtype_ids: dict[str, int] = {}
        self._failed = False

    def setup(self) -> None:
        """Open a new file, resetting state from any completed prior use."""
        if self._file is not None:
            raise RuntimeError("Homogeneous HDF5 writer is already open")
        super().setup()

    def _filter_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.compression is not None:
            kwargs["compression"] = self.compression
            if self.compression != "lzf" and self.compression_opts is not None:
                kwargs["compression_opts"] = self.compression_opts
        if self.shuffle:
            kwargs["shuffle"] = True
        if self.fletcher32:
            kwargs["fletcher32"] = True
        return kwargs

    def _setup(self) -> None:
        self._data = None
        self._shape = None
        self._dtype = None
        self._next_batch_idx = 0
        self._component_data.clear()
        self._component_dtype_ids.clear()
        self._failed = False
        self._file = h5py.File(self.datapath, "w", libver="latest")
        self._file.attrs["format"] = _FORMAT
        self._file.attrs["schema_version"] = _SCHEMA_VERSION
        self._file.attrs["complete"] = False
        self._file.attrs["compression"] = self.compression or "none"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._metadata = self._file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_offsets = self._file.create_dataset(
            "component_offsets",
            data=np.array([0], dtype=np.uint64),
            maxshape=(None,),
            chunks=True,
        )
        self._components = self._file.create_dataset(
            "components",
            shape=(0,),
            maxshape=(None,),
            dtype=_COMPONENT_DTYPE,
            chunks=True,
        )
        self._component_metadata = self._file.create_dataset(
            "component_metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_shapes = self._file.create_dataset(
            "component_shapes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._component_dtypes = self._file.create_dataset(
            "component_dtypes",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_data_group = self._file.create_group("component_data")

    def _validate_signal(self, signal: Signal) -> np.ndarray:
        if not isinstance(signal, Signal):
            raise TypeError("Homogeneous HDF5 batches must contain Signal instances")
        array = np.asarray(signal.data)
        if array.dtype.hasobject:
            raise TypeError("Homogeneous HDF5 does not support object arrays")
        for component in signal.component_signals:
            if component.component_signals:
                raise ValueError("Homogeneous HDF5 does not support nested components")
            component_array = np.asarray(component.data)
            if component_array.dtype.hasobject:
                raise TypeError("Homogeneous HDF5 does not support object arrays")
        return array

    def _samples_per_chunk(self, array: np.ndarray) -> int:
        if self.compression is not None and array.nbytes >= _LARGE_COMPRESSED_SAMPLE_BYTES:
            return 1
        return max(
            1,
            min(
                self.chunk_samples,
                _TARGET_CHUNK_BYTES // max(array.nbytes, 1),
            ),
        )

    def _create_data(self, array: np.ndarray) -> None:
        self._shape = array.shape
        self._dtype = array.dtype
        chunk_shape = (self._samples_per_chunk(array), *array.shape)
        self._data = self._file.create_dataset(
            "data",
            shape=(0, *array.shape),
            maxshape=(None, *array.shape),
            dtype=array.dtype,
            chunks=chunk_shape,
            **self._filter_kwargs(),
        )

    def _validate_homogeneity(self, arrays: list[np.ndarray]) -> None:
        for array in arrays:
            if array.shape != self._shape or array.dtype != self._dtype:
                raise ValueError("Homogeneous HDF5 top-level arrays must share one dtype and shape")

    def _component_dtype_id(self, dtype: np.dtype) -> int:
        key = dtype.str
        try:
            return self._component_dtype_ids[key]
        except KeyError:
            dtype_id = len(self._component_dtype_ids)
            self._component_dtype_ids[key] = dtype_id
            _append(self._component_dtypes, [key])
            self._component_data[dtype_id] = self._component_data_group.create_dataset(
                str(dtype_id),
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=True,
                **self._filter_kwargs(),
            )
            return dtype_id

    def _append_components(self, signals: list[Signal]) -> None:
        component_total = len(self._components)
        shape_offset = len(self._component_shapes)
        data_offsets: dict[int, int] = {}
        data_by_dtype: dict[int, list[np.ndarray]] = {}
        records = []
        metadata = []
        shapes = []
        signal_offsets = []

        for signal in signals:
            for component in signal.component_signals:
                array = np.asarray(component.data)
                dtype_id = self._component_dtype_id(array.dtype)
                if dtype_id not in data_offsets:
                    data_offsets[dtype_id] = len(self._component_data[dtype_id])
                    data_by_dtype[dtype_id] = []
                records.append(
                    (
                        data_offsets[dtype_id],
                        array.size,
                        dtype_id,
                        shape_offset,
                        array.ndim,
                    )
                )
                metadata.append(_encode_flat_metadata(component))
                shapes.extend(array.shape)
                data_by_dtype[dtype_id].append(array.reshape(-1))
                data_offsets[dtype_id] += array.size
                shape_offset += array.ndim
                component_total += 1
            signal_offsets.append(component_total)

        for dtype_id, arrays in data_by_dtype.items():
            _append(self._component_data[dtype_id], np.concatenate(arrays))
        _append(
            self._component_shapes,
            np.asarray(shapes, dtype=np.uint64),
        )
        _append(
            self._components,
            np.asarray(records, dtype=_COMPONENT_DTYPE),
        )
        _append(self._component_metadata, metadata)
        _append(
            self._component_offsets,
            np.asarray(signal_offsets, dtype=np.uint64),
        )

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Append one sequentially indexed batch."""
        if self._file is None:
            raise RuntimeError("Homogeneous HDF5 writer is not open")
        if self._failed:
            raise RuntimeError("Homogeneous HDF5 writer cannot continue after failure")
        if not isinstance(batch_idx, int) or isinstance(batch_idx, bool):
            raise TypeError("Homogeneous HDF5 batch index must be an integer")
        if batch_idx < 0:
            raise ValueError("Homogeneous HDF5 batch index must be non-negative")
        if batch_idx != self._next_batch_idx:
            raise ValueError(f"Homogeneous HDF5 requires sequential batch indices; expected {self._next_batch_idx}, got {batch_idx}")
        if not data:
            raise ValueError("Homogeneous HDF5 batches must not be empty")
        try:
            arrays = [self._validate_signal(signal) for signal in data]
            if arrays and self._data is None:
                self._create_data(arrays[0])
            self._validate_homogeneity(arrays)

            if arrays:
                start = len(self._data)
                self._data.resize(start + len(arrays), axis=0)
                self._data[start:] = np.stack(arrays)
            _append(
                self._metadata,
                [_encode_flat_metadata(signal) for signal in data],
            )
            self._append_components(data)
            self._next_batch_idx += 1
        except Exception:
            self._failed = True
            raise

    def __len__(self) -> int:
        """Return the number of stored top-level signals."""
        if self._file is None:
            raise RuntimeError("Homogeneous HDF5 writer is not open")
        return len(self._metadata)

    def teardown(self) -> None:
        """Finalize and close the homogeneous HDF5 file."""
        if self._file is None:
            return
        try:
            if not self._failed:
                if self._data is None:
                    self._failed = True
                    raise ValueError("Homogeneous HDF5 cannot finalize an empty dataset")
                self._file.attrs["complete"] = True
                self._file.flush()
        finally:
            self._file.close()
            self._file = None

    def __exit__(self, exc_type, exc_value, traceback):
        """Close while leaving failed files incomplete."""
        if exc_type is not None:
            self._failed = True
        self.teardown()
        return False


class HomogeneousHDF5Reader(FileReader):
    """Read fixed-shape top-level arrays and ragged component signals.

    :meth:`read` and :meth:`read_signals_batch` reconstruct complete
    :class:`~torchsig.signals.signal_types.Signal` objects. :meth:`read_batch`
    is the lower-overhead path for contiguous top-level arrays and deliberately
    omits metadata and component signals.
    """

    def __init__(self, root) -> None:
        """Initialize a process-safe lazy reader for ``root/data.h5``."""
        super().__init__(root=root)
        self.datapath = self.root / "data.h5"
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._file is not None and self._pid != pid:
            self._file.close()
            self._file = None
        if self._file is not None:
            return
        self._file = h5py.File(self.datapath, "r", locking=False)
        self._pid = pid
        try:
            self._validate_file()
            self._data = self._file["data"]
            self._metadata = self._file["metadata"]
            self._component_offsets = self._file["component_offsets"]
            self._components = self._file["components"]
            self._component_metadata = self._file["component_metadata"]
            self._component_shapes = self._file["component_shapes"]
            self._component_data = self._file["component_data"]
        except Exception:
            self._file.close()
            self._file = None
            self._pid = None
            raise

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without process-local HDF5 handles."""
        state = self.__dict__.copy()
        for name in (
            "_file",
            "_data",
            "_metadata",
            "_component_offsets",
            "_components",
            "_component_metadata",
            "_component_shapes",
            "_component_data",
        ):
            state.pop(name, None)
        state["_file"] = None
        state["_pid"] = None
        return state

    def _validate_file(self) -> None:
        if self._file.attrs.get("format") != _FORMAT:
            raise ValueError("Not a homogeneous TorchSig HDF5 file")
        if self._file.attrs.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported homogeneous HDF5 schema version: {self._file.attrs.get('schema_version')!r}")
        if not bool(self._file.attrs.get("complete", False)):
            raise ValueError("Homogeneous HDF5 file is incomplete")
        cache_key = str(self.datapath)
        fingerprint = self._fingerprint()
        if _VALIDATED_FILES.get(cache_key) == fingerprint:
            return
        self._validate_structure()
        self._validate_integrity()
        if cache_key not in _VALIDATED_FILES and len(_VALIDATED_FILES) >= _VALIDATION_CACHE_SIZE:
            del _VALIDATED_FILES[next(iter(_VALIDATED_FILES))]
        _VALIDATED_FILES[cache_key] = self._fingerprint()

    def _fingerprint(self) -> tuple[int, int, int, int]:
        stat = self.datapath.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _dataset(self, name: str, ndim: int | None = 1) -> h5py.Dataset:
        if name not in self._file or not isinstance(
            self._file[name],
            h5py.Dataset,
        ):
            raise ValueError(f"Homogeneous HDF5 file is missing required dataset: {name}")
        dataset = self._file[name]
        if ndim is not None and dataset.ndim != ndim:
            raise ValueError(f"Homogeneous HDF5 dataset {name!r} must have rank {ndim}")
        return dataset

    def _validate_structure(self) -> None:
        data = self._dataset("data", ndim=None)
        if data.ndim < 1:
            raise ValueError("Homogeneous HDF5 dataset 'data' must have rank at least 1")
        metadata = self._dataset("metadata")
        component_offsets = self._dataset("component_offsets")
        components = self._dataset("components")
        component_metadata = self._dataset("component_metadata")
        component_shapes = self._dataset("component_shapes")
        component_dtypes = self._dataset("component_dtypes")
        if "component_data" not in self._file:
            raise ValueError("Homogeneous HDF5 file is missing required group: component_data")
        if not isinstance(self._file["component_data"], h5py.Group):
            raise TypeError("Homogeneous HDF5 object 'component_data' must be a group")
        if h5py.check_string_dtype(metadata.dtype) is None:
            raise ValueError("Homogeneous HDF5 dataset 'metadata' must contain strings")
        if h5py.check_string_dtype(component_metadata.dtype) is None:
            raise ValueError("Homogeneous HDF5 dataset 'component_metadata' must contain strings")
        if h5py.check_string_dtype(component_dtypes.dtype) is None:
            raise ValueError("Homogeneous HDF5 dataset 'component_dtypes' must contain strings")
        if component_offsets.dtype != np.dtype(np.uint64):
            raise ValueError("Homogeneous HDF5 dataset 'component_offsets' must use uint64")
        if component_shapes.dtype != np.dtype(np.uint64):
            raise ValueError("Homogeneous HDF5 dataset 'component_shapes' must use uint64")
        if components.dtype != _COMPONENT_DTYPE:
            raise ValueError("Homogeneous HDF5 dataset 'components' has an invalid dtype")

    def _component_streams(self) -> dict[int, h5py.Dataset]:
        dtype_values = self._file["component_dtypes"][:]
        group = self._file["component_data"]
        streams = {}
        for dtype_id, encoded_dtype in enumerate(dtype_values):
            name = str(dtype_id)
            if name not in group or not isinstance(group[name], h5py.Dataset):
                raise ValueError(f"Homogeneous HDF5 component data stream is missing for dtype ID {dtype_id}")
            try:
                dtype = np.dtype(encoded_dtype.decode() if isinstance(encoded_dtype, bytes) else encoded_dtype)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Homogeneous HDF5 component dtype ID {dtype_id} is invalid") from error
            stream = group[name]
            if stream.ndim != 1 or stream.dtype != dtype:
                raise ValueError(f"Homogeneous HDF5 component data stream does not match dtype ID {dtype_id}")
            streams[dtype_id] = stream
        return streams

    def _validate_offsets(
        self,
        offsets: h5py.Dataset,
        signal_count: int,
        component_count: int,
    ) -> None:
        if len(offsets) != signal_count + 1:
            raise ValueError("Homogeneous HDF5 component offset count does not match signal count")
        previous: int | None = None
        for start in range(0, len(offsets), _VALIDATION_CHUNK_RECORDS):
            values = offsets[start : start + _VALIDATION_CHUNK_RECORDS]
            if len(values) == 0 or (previous is None and int(values[0]) != 0) or (previous is not None and int(values[0]) < previous) or np.any(values[1:] < values[:-1]):
                raise ValueError("Homogeneous HDF5 component offsets are invalid")
            previous = int(values[-1])
        if previous != component_count:
            raise ValueError("Homogeneous HDF5 component offsets are invalid")

    def _validate_component_records(
        self,
        components: h5py.Dataset,
        shapes: h5py.Dataset,
        streams: dict[int, h5py.Dataset],
    ) -> None:
        shape_position = 0
        data_positions = dict.fromkeys(streams, 0)
        for start in range(0, len(components), _VALIDATION_CHUNK_RECORDS):
            records = components[start : start + _VALIDATION_CHUNK_RECORDS]
            chunk_shape_start = shape_position
            for record in records:
                shape_offset = int(record["shape_offset"])
                shape_count = int(record["shape_count"])
                if shape_offset != shape_position:
                    raise ValueError("Homogeneous HDF5 component shape offsets are invalid")
                shape_position += shape_count
            if shape_position > len(shapes):
                raise ValueError("Homogeneous HDF5 component shape range is out of bounds")
            shape_values = shapes[chunk_shape_start:shape_position]
            local_shape_position = 0
            for record in records:
                dtype_id = int(record["dtype_id"])
                if dtype_id not in streams:
                    raise ValueError(f"Homogeneous HDF5 component references invalid dtype ID {dtype_id}")
                data_offset = int(record["data_offset"])
                data_length = int(record["data_length"])
                if data_offset + data_length > len(streams[dtype_id]):
                    raise ValueError("Homogeneous HDF5 component data range is out of bounds")
                if data_offset != data_positions[dtype_id]:
                    raise ValueError("Homogeneous HDF5 component data offsets are invalid")
                data_positions[dtype_id] += data_length
                shape_count = int(record["shape_count"])
                shape_stop = local_shape_position + shape_count
                shape = (int(value) for value in shape_values[local_shape_position:shape_stop])
                if prod(shape) != data_length:
                    raise ValueError("Homogeneous HDF5 component shape does not match its data length")
                local_shape_position = shape_stop
        if shape_position != len(shapes):
            raise ValueError("Homogeneous HDF5 component shape offsets are invalid")
        if any(data_positions[dtype_id] != len(stream) for dtype_id, stream in streams.items()):
            raise ValueError("Homogeneous HDF5 component data offsets are invalid")

    def _validate_integrity(self) -> None:
        data = self._file["data"]
        metadata = self._file["metadata"]
        offsets = self._file["component_offsets"]
        components = self._file["components"]
        component_metadata = self._file["component_metadata"]
        shapes = self._file["component_shapes"]
        if len(data) != len(metadata):
            raise ValueError("Homogeneous HDF5 top-level data and metadata lengths differ")
        self._validate_offsets(offsets, len(data), len(components))
        if len(component_metadata) != len(components):
            raise ValueError("Homogeneous HDF5 component metadata length does not match records")

        streams = self._component_streams()
        self._validate_component_records(components, shapes, streams)

    def __len__(self) -> int:
        """Return the number of stored top-level signals."""
        self._ensure_open()
        return len(self._metadata)

    def _read_components(self, start: int, stop: int) -> list[Signal]:
        if start == stop:
            return []

        records = self._components[start:stop]
        metadata = self._component_metadata[start:stop]
        shape_start = int(records[0]["shape_offset"])
        shape_stop = max(int(record["shape_offset"]) + int(record["shape_count"]) for record in records)
        shapes = self._component_shapes[shape_start:shape_stop]

        data_ranges: dict[int, tuple[int, int]] = {}
        for record in records:
            dtype_id = int(record["dtype_id"])
            data_start = int(record["data_offset"])
            data_stop = data_start + int(record["data_length"])
            if dtype_id in data_ranges:
                range_start, range_stop = data_ranges[dtype_id]
                data_ranges[dtype_id] = (
                    min(range_start, data_start),
                    max(range_stop, data_stop),
                )
            else:
                data_ranges[dtype_id] = (data_start, data_stop)
        data_by_dtype = {dtype_id: self._component_data[str(dtype_id)][range_start:range_stop] for dtype_id, (range_start, range_stop) in data_ranges.items()}

        components = []
        for record, component_metadata in zip(records, metadata, strict=True):
            dtype_id = int(record["dtype_id"])
            data_start = int(record["data_offset"])
            data_length = int(record["data_length"])
            range_start = data_ranges[dtype_id][0]
            local_start = data_start - range_start
            component_shape_start = int(record["shape_offset"]) - shape_start
            component_shape_stop = component_shape_start + int(record["shape_count"])
            shape = tuple(int(value) for value in shapes[component_shape_start:component_shape_stop])
            components.append(
                Signal(
                    data=data_by_dtype[dtype_id][local_start : local_start + data_length].reshape(shape),
                    metadata=decode_metadata(component_metadata),
                )
            )
        return components

    def read(self, idx: int) -> Signal:
        """Read one complete signal by top-level dataset index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Homogeneous HDF5 sample index out of range: {idx}")
        component_start = int(self._component_offsets[idx])
        component_stop = int(self._component_offsets[idx + 1])
        return Signal(
            data=self._data[idx],
            component_signals=self._read_components(component_start, component_stop),
            metadata=decode_metadata(self._metadata[idx]),
        )

    def read_signals_batch(self, start: int, stop: int) -> list[Signal]:
        """Read complete signals in the half-open range ``[start, stop)``.

        The top-level arrays, metadata records, component offsets, and complete
        component range are each fetched with one contiguous HDF5 read.
        """
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(f"Homogeneous HDF5 batch range out of bounds: [{start}, {stop})")
        if start == stop:
            return []

        data = self._data[start:stop]
        metadata = self._metadata[start:stop]
        offsets = self._component_offsets[start : stop + 1]
        component_start = int(offsets[0])
        components = self._read_components(component_start, int(offsets[-1]))

        signals = []
        for batch_idx, (signal_data, signal_metadata) in enumerate(zip(data, metadata, strict=True)):
            local_start = int(offsets[batch_idx]) - component_start
            local_stop = int(offsets[batch_idx + 1]) - component_start
            signals.append(
                Signal(
                    data=signal_data,
                    component_signals=components[local_start:local_stop],
                    metadata=decode_metadata(signal_metadata),
                )
            )
        return signals

    def read_batch(self, start: int, stop: int) -> np.ndarray:
        """Read top-level arrays in ``[start, stop)``.

        This optimized method returns only a NumPy array. It does not decode
        signal metadata or reconstruct component signals.
        """
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(f"Homogeneous HDF5 batch range out of bounds: [{start}, {stop})")
        return self._data[start:stop]

    def teardown(self) -> None:
        """Close the homogeneous HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._pid = None
