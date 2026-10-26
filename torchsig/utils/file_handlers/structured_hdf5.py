"""HDF5 storage for fixed-schema structured homogeneous samples."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.structured_schema import (
    STRUCTURED_FORMAT,
    STRUCTURED_SCHEMA_MAJOR,
    STRUCTURED_SCHEMA_MINOR,
    StructuredNode,
    StructuredSampleSchema,
    infer_structured_sample_schema,
)

__all__ = ["StructuredHDF5Reader", "StructuredHDF5Writer"]

_SCHEMA_PATH = "schema"
_TARGET_CHUNK_BYTES = 1024**2


def _reconstruct(node: StructuredNode, leaves: Sequence[Any]) -> Any:
    if node.kind == "leaf":
        return leaves[node.field_index]  # type: ignore[index]
    values = [(key, _reconstruct(child, leaves)) for key, child in node.children]
    if node.kind == "mapping":
        return dict(values)
    items = [value for _, value in values]
    return tuple(items) if node.kind == "tuple" else items


class StructuredHDF5Writer(FileWriter):
    """Write structured samples with one extensible dataset per schema leaf."""

    def __init__(
        self,
        root,
        *,
        schema: StructuredSampleSchema | None = None,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_samples: int = 32,
    ) -> None:
        """Initialize a writer that creates ``root/data.h5``.

        An explicit ``schema`` allows a valid empty dataset to be created.
        Otherwise the schema is inferred from the first sample written.
        """
        super().__init__(root=root)
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")
        self.datapath = self.root / "data.h5"
        self.initial_schema = schema
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_samples = chunk_samples
        self.schema: StructuredSampleSchema | None = None
        self._file: h5py.File | None = None
        self._datasets: list[h5py.Dataset] = []
        self._next_batch_idx = 0
        self._length = 0
        self._failed = False

    def setup(self) -> None:
        """Open a new output file and reset state from prior use."""
        if self._file is not None:
            raise RuntimeError("Structured HDF5 writer is already open")
        super().setup()

    def _setup(self) -> None:
        self.schema = None
        self._datasets = []
        self._next_batch_idx = 0
        self._length = 0
        self._failed = False
        self._file = h5py.File(self.datapath, "w", libver="latest")
        self._file.attrs["format"] = STRUCTURED_FORMAT
        self._file.attrs["schema_major"] = STRUCTURED_SCHEMA_MAJOR
        self._file.attrs["schema_minor"] = STRUCTURED_SCHEMA_MINOR
        self._file.attrs["length"] = 0
        self._file.attrs["complete"] = False
        if self.initial_schema is not None:
            self._create_storage(self.initial_schema)

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

    def _create_storage(self, schema: StructuredSampleSchema) -> None:
        if self._file is None:
            raise RuntimeError("Structured HDF5 writer is not open")
        self.schema = StructuredSampleSchema.from_dict(schema.to_dict())
        payload = json.dumps(self.schema.to_dict(), separators=(",", ":"), sort_keys=True)
        self._file.create_dataset(_SCHEMA_PATH, data=payload, dtype=h5py.string_dtype("utf-8"))
        self._file.create_group("fields")
        for field in self.schema.fields:
            dtype = np.dtype(field.dtype)
            sample_bytes = max(dtype.itemsize * int(np.prod(field.shape, dtype=np.int64)), 1)
            samples_per_chunk = max(1, min(self.chunk_samples, _TARGET_CHUNK_BYTES // sample_bytes))
            chunk_shape = (samples_per_chunk, *(max(size, 1) for size in field.shape))
            maxshape = (None, *(None if size == 0 else size for size in field.shape))
            self._datasets.append(
                self._file.create_dataset(
                    field.dataset_path,
                    shape=(0, *field.shape),
                    maxshape=maxshape,
                    dtype=dtype,
                    chunks=chunk_shape,
                    **self._filter_kwargs(),
                )
            )

    def write(self, batch_idx: int, data: Sequence[Any]) -> None:
        """Validate and append one non-empty, sequentially indexed batch."""
        if self._file is None:
            raise RuntimeError("Structured HDF5 writer is not open")
        if self._failed:
            raise RuntimeError("Structured HDF5 writer cannot continue after failure")
        if not isinstance(batch_idx, int) or isinstance(batch_idx, bool):
            raise TypeError("Structured HDF5 batch index must be an integer")
        if batch_idx < 0:
            raise ValueError("Structured HDF5 batch index must be non-negative")
        if batch_idx != self._next_batch_idx:
            raise ValueError(f"Structured HDF5 requires sequential batch indices; expected {self._next_batch_idx}, got {batch_idx}")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError("Structured HDF5 data must be a sequence of samples")
        if not data:
            raise ValueError("Structured HDF5 batches must not be empty")
        try:
            if self.schema is None:
                self._create_storage(infer_structured_sample_schema(data[0]))
            leaves = [self.schema.validate(sample) for sample in data]
            start = self._length
            stop = start + len(data)
            for field_index, dataset in enumerate(self._datasets):
                dataset.resize(stop, axis=0)
                dataset[start:stop] = np.stack([sample[field_index] for sample in leaves])
            self._length = stop
            self._file.attrs["length"] = self._length
            self._next_batch_idx += 1
        except Exception:
            self._failed = True
            raise

    def __len__(self) -> int:
        """Return the number of samples appended so far."""
        if self._file is None:
            raise RuntimeError("Structured HDF5 writer is not open")
        return self._length

    def teardown(self) -> None:
        """Mark a successful output complete, flush it, and close it."""
        if self._file is None:
            return
        try:
            if not self._failed:
                if self.schema is None:
                    self._failed = True
                    raise ValueError("Structured HDF5 cannot finalize without a schema")
                self._file.attrs["complete"] = True
                self._file.flush()
        finally:
            self._file.close()
            self._file = None
            self._datasets = []

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file while retaining an incomplete marker on failure."""
        if exc_type is not None:
            self._failed = True
        self.teardown()
        return False


class StructuredHDF5Reader(FileReader):
    """Read structured homogeneous HDF5 samples with process-local handles."""

    def __init__(self, root) -> None:
        """Initialize a process-safe lazy reader for ``root/data.h5``."""
        super().__init__(root=root)
        self.datapath = self.root / "data.h5"
        self._file: h5py.File | None = None
        self._pid: int | None = None
        self._datasets: list[h5py.Dataset] = []

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._file is not None and self._pid != pid:
            self._file.close()
            self._file = None
            self._datasets = []
        if self._file is not None:
            return
        self._file = h5py.File(self.datapath, "r", locking=False)
        self._pid = pid
        try:
            self.schema, self._length, self._datasets = self._validate_file()
        except Exception:
            self._file.close()
            self._file = None
            self._pid = None
            self._datasets = []
            raise

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without process-local HDF5 objects."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        state["_datasets"] = []
        return state

    def _validate_file(self) -> tuple[StructuredSampleSchema, int, list[h5py.Dataset]]:
        if self._file.attrs.get("format") != STRUCTURED_FORMAT:
            raise ValueError("Not a structured homogeneous TorchSig HDF5 file")
        if self._file.attrs.get("schema_major") != STRUCTURED_SCHEMA_MAJOR:
            raise ValueError(f"Unsupported structured HDF5 schema major version: {self._file.attrs.get('schema_major')!r}")
        if self._file.attrs.get("schema_minor") != STRUCTURED_SCHEMA_MINOR:
            raise ValueError(f"Unsupported structured HDF5 schema minor version: {self._file.attrs.get('schema_minor')!r}")
        if "complete" not in self._file.attrs or not bool(self._file.attrs["complete"]):
            raise ValueError("Structured HDF5 file is incomplete")
        if _SCHEMA_PATH not in self._file or not isinstance(self._file[_SCHEMA_PATH], h5py.Dataset):
            raise ValueError("Structured HDF5 file is missing required schema dataset")
        try:
            payload = self._file[_SCHEMA_PATH].asstr()[()]
            schema = StructuredSampleSchema.from_dict(json.loads(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Structured HDF5 schema dataset is invalid") from error
        if schema.format != self._file.attrs["format"] or schema.schema_major != self._file.attrs["schema_major"] or schema.schema_minor != self._file.attrs["schema_minor"]:
            raise ValueError("Structured HDF5 schema does not match file attributes")
        length = self._file.attrs.get("length")
        if not isinstance(length, (int, np.integer)) or isinstance(length, (bool, np.bool_)) or length < 0:
            raise ValueError("Structured HDF5 file has an invalid declared length")
        datasets = []
        for field in schema.fields:
            path = field.dataset_path
            if path not in self._file or not isinstance(self._file[path], h5py.Dataset):
                raise ValueError(f"Structured HDF5 file is missing required field dataset: {path}")
            dataset = self._file[path]
            expected_shape = (int(length), *field.shape)
            if dataset.shape != expected_shape:
                raise ValueError(f"Structured HDF5 field {path} has shape {dataset.shape}; expected {expected_shape}")
            if dataset.dtype != np.dtype(field.dtype):
                raise ValueError(f"Structured HDF5 field {path} has dtype {dataset.dtype}; expected {np.dtype(field.dtype)}")
            datasets.append(dataset)
        return schema, int(length), datasets

    def __len__(self) -> int:
        """Return the declared and validated sample count."""
        self._ensure_open()
        return self._length

    def read(self, idx: int) -> Any:
        """Read and reconstruct one sample by non-negative index."""
        self._ensure_open()
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise TypeError("Structured HDF5 sample index must be an integer")
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Structured HDF5 sample index out of range: {idx}")
        return _reconstruct(self.schema.root, [dataset[idx] for dataset in self._datasets])

    def read_batch(self, start: int, stop: int) -> list[Any]:
        """Read samples in ``[start, stop)`` using one slice per field."""
        self._ensure_open()
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(stop, int) or isinstance(stop, bool):
            raise TypeError("Structured HDF5 batch bounds must be integers")
        if start < 0 or stop < start or stop > self._length:
            raise IndexError(f"Structured HDF5 batch range out of bounds: [{start}, {stop})")
        leaves = [dataset[start:stop] for dataset in self._datasets]
        return [_reconstruct(self.schema.root, [field_values[index] for field_values in leaves]) for index in range(stop - start)]

    def read_indices(self, indices: Sequence[int]) -> list[Any]:
        """Read an index batch efficiently and restore its requested order.

        Indices are validated, sorted, and deduplicated before accessing each
        schema leaf. Adjacent runs are coalesced when doing so materially
        reduces the operation count. Sparse requests and fragmented
        single-leaf samples retain direct reads because h5py point selection,
        sorting, and batch copying are slower for those cases. Duplicate
        indices are reconstructed in their original positions.
        """
        self._ensure_open()
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise TypeError("Structured HDF5 indices must be a sequence of integers")
        if not indices:
            return []
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("Structured HDF5 sample indices must be integers")
            if index < 0 or index >= self._length:
                raise IndexError(f"Structured HDF5 sample index out of range: {index}")

        if len(self._datasets) == 1:
            dataset = self._datasets[0]
            if all(index == indices[0] + offset for offset, index in enumerate(indices)):
                values = dataset[indices[0] : indices[-1] + 1]
                return [_reconstruct(self.schema.root, [value]) for value in values]
            return [_reconstruct(self.schema.root, [dataset[index]]) for index in indices]
        unique_indices, inverse = np.unique(np.asarray(indices, dtype=np.intp), return_inverse=True)
        breaks = np.flatnonzero(np.diff(unique_indices) != 1) + 1
        runs = np.split(unique_indices, breaks)
        if len(runs) * 2 > len(unique_indices):
            return [_reconstruct(self.schema.root, [dataset[index] for dataset in self._datasets]) for index in indices]
        leaves = [np.concatenate([dataset[int(run[0]) : int(run[-1]) + 1] for run in runs], axis=0) for dataset in self._datasets]
        return [_reconstruct(self.schema.root, [field_values[position] for field_values in leaves]) for position in inverse]

    def teardown(self) -> None:
        """Close the process-local HDF5 file and field handles."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._pid = None
            self._datasets = []

    close = teardown

    def __enter__(self):
        """Return this open reader."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this reader when leaving its context."""
        self.teardown()
        return False
