"""Memory-mappable variable-length NumPy storage for TorchSig signals.

The format stores all sample values in one uncompressed ``data.npy`` array and
uses ``records.npy`` plus ``shapes.npy`` as a versioned descriptor table.
Metadata is associated by record index in ``metadata.json``.  Version 1 uses a
single dataset-wide dtype, preserving its explicit NumPy byte-order marker.
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.metadata_codec import decode_metadata, encode_metadata

__all__ = ["PackedNPYReader", "PackedNPYWriter"]

FORMAT_NAME = "torchsig-packed-npy"
FORMAT_VERSION = 1
_RECORD_DTYPE = np.dtype(
    [
        ("record_index", "<u8"),
        ("sample_offset", "<u8"),
        ("sample_count", "<u8"),
        ("shape_offset", "<u8"),
        ("shape_rank", "<u4"),
        ("metadata_index", "<u8"),
    ]
)


class PackedNPYWriter(FileWriter):
    """Write variable-length arrays to the packed NPY format.

    All records in one dataset must have the same non-object dtype. Shapes and
    lengths may differ. Batches can arrive out of order but must form a unique,
    contiguous sequence beginning at zero when the writer closes.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(root=str(root))
        self._lock = threading.Lock()
        self._pending: dict[int, list[Signal]] = {}
        self._next_batch = 0
        self._dtype: np.dtype[Any] | None = None
        self._records: list[tuple[int, int, int, int, int, int]] = []
        self._shapes: list[int] = []
        self._metadata: list[str] = []
        self._sample_count = 0
        self._raw_file: Any = None
        self._is_open = False

    @property
    def _raw_path(self) -> Path:
        return self.root / ".data.raw"

    def _setup(self) -> None:
        self._pending.clear()
        self._next_batch = 0
        self._dtype = None
        self._records.clear()
        self._shapes.clear()
        self._metadata.clear()
        self._sample_count = 0
        self._raw_file = self._raw_path.open("wb")
        self._is_open = True

    def _append_batch(self, signals: list[Signal]) -> None:
        prepared: list[tuple[np.ndarray[Any, Any], str]] = []
        expected_dtype = self._dtype
        for signal in signals:
            if signal.component_signals:
                raise ValueError("Packed NPY does not support component signals")
            array = np.asarray(signal.data)
            if array.dtype.hasobject:
                raise TypeError("Packed NPY does not support object arrays")
            if expected_dtype is None:
                expected_dtype = array.dtype
            if array.dtype != expected_dtype:
                raise TypeError(f"Packed NPY requires one dtype; expected {expected_dtype.str}, got {array.dtype.str}")
            prepared.append((np.ascontiguousarray(array), encode_metadata(signal)))

        self._dtype = expected_dtype
        for array, metadata in prepared:
            flat = array.reshape(-1)
            shape_offset = len(self._shapes)
            record_index = len(self._records)
            self._shapes.extend(array.shape)
            self._records.append(
                (
                    record_index,
                    self._sample_count,
                    flat.size,
                    shape_offset,
                    array.ndim,
                    record_index,
                )
            )
            self._metadata.append(metadata)
            flat.tofile(self._raw_file)
            self._sample_count += flat.size

    def _flush(self, *, final: bool = False) -> None:
        while self._next_batch in self._pending:
            self._append_batch(self._pending.pop(self._next_batch))
            self._next_batch += 1
        if final and self._pending:
            raise ValueError(f"Cannot finalize packed NPY dataset: missing batch index {self._next_batch}")

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Write one uniquely indexed batch."""
        if not self._is_open:
            raise RuntimeError("Packed NPY writer is not open")
        if not isinstance(batch_idx, int) or isinstance(batch_idx, bool) or batch_idx < 0:
            raise ValueError("Packed NPY batch index must be a non-negative integer")
        with self._lock:
            if batch_idx < self._next_batch or batch_idx in self._pending:
                raise ValueError(f"Duplicate packed NPY batch index: {batch_idx}")
            self._pending[batch_idx] = list(data)
            self._flush()

    def __len__(self) -> int:
        """Return the number of records already committed."""
        return len(self._records)

    def teardown(self) -> None:
        """Finalize NPY arrays and the format manifest."""
        if not self._is_open:
            return
        try:
            with self._lock:
                self._flush(final=True)
                self._raw_file.close()
                self._raw_file = None
                if self._dtype is None:
                    raise ValueError("Cannot create an empty packed NPY dataset because its dtype is unknown")
                data = np.lib.format.open_memmap(
                    self.root / "data.npy",
                    mode="w+",
                    dtype=self._dtype,
                    shape=(self._sample_count,),
                )
                if self._sample_count:
                    raw = np.memmap(self._raw_path, mode="r", dtype=self._dtype, shape=(self._sample_count,))
                    data[:] = raw
                    del raw
                data.flush()
                del data
                np.save(self.root / "records.npy", np.asarray(self._records, dtype=_RECORD_DTYPE), allow_pickle=False)
                np.save(self.root / "shapes.npy", np.asarray(self._shapes, dtype=np.uint64), allow_pickle=False)
                (self.root / "metadata.json").write_text(json.dumps(self._metadata, separators=(",", ":")), encoding="utf-8")
                manifest = {
                    "format": FORMAT_NAME,
                    "version": FORMAT_VERSION,
                    "record_count": len(self._records),
                    "dtype": self._dtype.str,
                    "byte_order": self._dtype.byteorder,
                }
                (self.root / "packed_npy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                self._raw_path.unlink()
        finally:
            if self._raw_file is not None:
                self._raw_file.close()
                self._raw_file = None
            self._is_open = False

    def __exit__(self, exc_type, exc_value, traceback):
        """Discard staging data when the surrounding write fails."""
        if exc_type is not None:
            if self._raw_file is not None:
                self._raw_file.close()
                self._raw_file = None
            if self._raw_path.exists():
                self._raw_path.unlink()
            self._is_open = False
            return False
        self.teardown()
        return False


class PackedNPYReader(FileReader):
    """Read validated variable-length records without loading the data array."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root=str(root))
        self._data: np.memmap[Any, Any] | None = None
        self._data_pid: int | None = None
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        manifest_path = self.root / "packed_npy.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read packed NPY manifest {manifest_path}: {exc}") from exc
        if manifest.get("format") != FORMAT_NAME or manifest.get("version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported packed NPY format or version in {manifest_path}")
        try:
            self.dtype = np.dtype(manifest["dtype"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Packed NPY manifest has an invalid dtype") from exc
        if self.dtype.hasobject:
            raise ValueError("Packed NPY object dtype is not supported")

        self.records = np.load(self.root / "records.npy", mmap_mode="r", allow_pickle=False)
        self.shapes = np.load(self.root / "shapes.npy", mmap_mode="r", allow_pickle=False)
        if self.records.dtype != _RECORD_DTYPE:
            raise ValueError(f"Invalid packed NPY record dtype: {self.records.dtype}")
        try:
            raw_metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read packed NPY metadata: {exc}") from exc
        if not isinstance(raw_metadata, list):
            raise TypeError("Packed NPY metadata must be a list")
        self._metadata = raw_metadata

        data = np.load(self.root / "data.npy", mmap_mode="r", allow_pickle=False)
        if data.ndim != 1 or data.dtype != self.dtype:
            raise ValueError("Packed NPY data array does not match its declared one-dimensional dtype")
        data_length = len(data)
        del data
        if manifest.get("record_count") != len(self.records):
            raise ValueError("Packed NPY manifest record count does not match records.npy")
        self._validate_records(data_length)

    def _validate_records(self, data_length: int) -> None:
        previous_end = 0
        for position, record in enumerate(self.records):
            record_index = int(record["record_index"])
            offset = int(record["sample_offset"])
            count = int(record["sample_count"])
            shape_offset = int(record["shape_offset"])
            shape_rank = int(record["shape_rank"])
            metadata_index = int(record["metadata_index"])
            if record_index != position:
                raise ValueError(f"Invalid packed NPY record {position}: record_index is {record_index}")
            if offset < previous_end:
                raise ValueError(f"Invalid packed NPY record {position}: sample range overlaps the previous record")
            if offset > data_length or count > data_length - offset:
                raise ValueError(f"Invalid packed NPY record {position}: sample range is out of bounds")
            if shape_offset > len(self.shapes) or shape_rank > len(self.shapes) - shape_offset:
                raise ValueError(f"Invalid packed NPY record {position}: shape range is out of bounds")
            shape = self.shapes[shape_offset : shape_offset + shape_rank]
            if int(np.prod(shape, dtype=np.uint64)) != count:
                raise ValueError(f"Invalid packed NPY record {position}: shape does not match sample count")
            if metadata_index >= len(self._metadata):
                raise ValueError(f"Invalid packed NPY record {position}: metadata index is out of bounds")
            previous_end = offset + count

    def _ensure_data(self) -> np.memmap[Any, Any]:
        pid = os.getpid()
        if self._data is None or self._data_pid != pid:
            self._data = np.load(self.root / "data.npy", mmap_mode="r", allow_pickle=False)
            self._data_pid = pid
        return self._data

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local mappings when pickled for spawned workers."""
        state = self.__dict__.copy()
        state["_data"] = None
        state["_data_pid"] = None
        state["records"] = None
        state["shapes"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reopen descriptors in a spawned worker process."""
        self.__dict__.update(state)
        self._load_and_validate()

    def read(self, idx: int) -> Signal:
        """Read one record by index as a memory-mapped array view."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Packed NPY sample index out of range: {idx}")
        record = self.records[idx]
        start = int(record["sample_offset"])
        stop = start + int(record["sample_count"])
        shape_start = int(record["shape_offset"])
        shape_stop = shape_start + int(record["shape_rank"])
        shape = tuple(int(value) for value in self.shapes[shape_start:shape_stop])
        metadata_index = int(record["metadata_index"])
        metadata = decode_metadata(self._metadata[metadata_index])
        return Signal(data=self._ensure_data()[start:stop].reshape(shape), component_signals=[], metadata=deepcopy(metadata))

    def __len__(self) -> int:
        """Return the number of packed records."""
        return len(self.records)

    def teardown(self) -> None:
        """Release the current process's memory map."""
        self._data = None
        self._data_pid = None
