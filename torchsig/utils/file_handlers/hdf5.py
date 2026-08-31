"""Legacy object-per-record HDF5 handlers for TorchSig datasets.

This module preserves TorchSig's original HDF5 layout for existing datasets.
For newly created datasets, use
:class:`~torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Writer` when
top-level array shapes or dtypes may vary, or
:class:`~torchsig.utils.file_handlers.homogeneous_hdf5.HomogeneousHDF5Writer`
when they are uniform.
"""

from __future__ import annotations

# Built-In
import threading
from typing import Any

import h5py

# Third Party
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject

# TorchSig
from torchsig.utils.dsp import torchsig_cache_version
from torchsig.utils.file_handlers.base_handler import BaseFileHandler, FileReader, FileWriter

__all__ = [
    "HDF5FileHandler",
    "HDF5Reader",
    "HDF5Writer",
    "fill_object_metadata_from_group_and_id",
    "handle_bytes_as_string",
    "load_signal_from_group_by_id",
    "load_signal_from_group_by_index",
    "load_value_from_group",
    "populate_hdf5_group_with_component_signals",
    "populate_hdf5_group_with_metadata",
    "populate_hdf5_group_with_signal",
    "populate_hdf5_group_with_signal_data",
    "populate_hdf5_group_with_signals",
]


def _hdf5_key(obj) -> str:
    """Return the HDF5 group key to use for *obj*.

    Ephemeral objects (generated Signal instances) receive a short sequential
    integer key that is stamped onto them by ``HDF5Writer._assign_hdf5_keys``
    immediately before writing. Persistent objects (generators, datasets)
    that are never garbage-collected within a write session fall back to
    ``str(id(obj))``, which is stable for the lifetime of the writer.

    Using a counter for signals avoids the id()-reuse hazard that arises when
    CPython recycles the memory address of a freed signal and a later signal
    lands at the same address, causing the "already exists" guard to skip the
    write silently.
    """
    try:
        return obj._hdf5_key
    except AttributeError:
        return str(id(obj))


def populate_hdf5_group_with_metadata(group, metadata_obj) -> bool:
    """Store an object's local metadata and recursively store its parents.

    Returns ``True`` when a metadata group is created for ``metadata_obj`` and
    ``False`` when its key already exists.
    """
    key = _hdf5_key(metadata_obj)
    # Persistent objects (generators, datasets) share a stable key across
    # many signal writes and must only be written once; skip if present.
    # Signal objects always get a unique counter key so this check is a no-op
    # for them, but keeping it here is harmless
    if key in group:
        return False
    metadata_group = group.create_group(key)
    for k in metadata_obj.keys():
        if not metadata_obj[k] == None:
            metadata_group.create_dataset(k, data=metadata_obj[k])
    if not metadata_obj.parent == None:
        try:
            metadata_group.create_dataset("parent_metadata_id", data=_hdf5_key(metadata_obj.parent))
            populate_hdf5_group_with_metadata(group, metadata_obj.parent)
        except ValueError:
            print("hdf5: metadata_group create dataset ValueError")
    return True


def populate_hdf5_group_with_signal_data(group, signal, dataset_kwargs=None):
    """Store one signal array under its object key.

    ``dataset_kwargs`` are forwarded to :meth:`h5py.Group.create_dataset`.
    Returns ``False`` when the signal key is already present.
    """
    key = _hdf5_key(signal)
    # Signal keys are unique counters so this check only fires for persistent
    # objects that reuse id()-based keys; skip them if already written
    if key in group:
        return False
    try:
        group.create_dataset(key, data=signal.data, **(dataset_kwargs or {}))
    except ValueError:
        print("hdf5: signal data create dataset ValueError")
    return True


def populate_hdf5_group_with_component_signals(group, signal):
    """Populates the HDF5 group with component signals.

    Args:
        group: The HDF5 group to populate.
        signal: The signal whose component signals should be added.

    Returns:
        bool: True if component signals were added, False otherwise.
    """
    if len(signal.component_signals) > 0:
        try:
            group.create_dataset(
                _hdf5_key(signal),
                data=[_hdf5_key(cs) for cs in signal.component_signals],
            )
        except ValueError:
            print("hdf5: component signals create dataset ValueError")
        return True
    return False


def _populate_hdf5_group_with_signal(group, signal, data_dataset_kwargs=None):
    """Internal helper to populate HDF5 group with signal data.

    Args:
        group: The HDF5 group to populate.
        signal: The signal to add to the group.
        data_dataset_kwargs: Optional keyword arguments for dataset creation.
    """
    populate_hdf5_group_with_metadata(group["metadata"], signal)
    populate_hdf5_group_with_signal_data(group["data"], signal, dataset_kwargs=data_dataset_kwargs)
    populate_hdf5_group_with_component_signals(group["component_signals"], signal)
    for component_signal in signal.component_signals:
        _populate_hdf5_group_with_signal(group, component_signal, data_dataset_kwargs=data_dataset_kwargs)


def populate_hdf5_group_with_signal(group, signal, index=True, data_dataset_kwargs=None):
    """Populates an HDF5 group with a signal and optionally indexes it.

    Args:
        group: The HDF5 group to populate.
        signal: The signal to add to the group.
        index: Whether to index the signal.
        data_dataset_kwargs: Optional keyword arguments for dataset creation.
    """
    _populate_hdf5_group_with_signal(group, signal, data_dataset_kwargs=data_dataset_kwargs)
    if index:
        group["index"].create_dataset(str(len(group["index"])), data=_hdf5_key(signal))  # keep track of this index in a dataset


def populate_hdf5_group_with_signals(group, signals, index=True, data_dataset_kwargs=None):
    """Populates an HDF5 group with multiple signals and optionally indexes them.

    Args:
        group: The HDF5 group to populate.
        signals: The signals to add to the group.
        index: Whether to index the signals.
        data_dataset_kwargs: Optional keyword arguments for dataset creation.
    """
    for signal in signals:
        populate_hdf5_group_with_signal(group, signal, index=index, data_dataset_kwargs=data_dataset_kwargs)


class HDF5Writer(FileWriter):
    """Write Signals using TorchSig's legacy object-per-record HDF5 layout.

    Each signal array, metadata object, and component relationship is stored
    as a separate HDF5 object. Batches are buffered and committed in batch
    index order.
    """

    def __init__(
        self,
        root,
        compression: str = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 1024 * 1024 * 10,  # 10MB cache
        max_batches_in_memory: int = 4,
    ):
        """Initialize the legacy HDF5 writer.

        Args:
            root: Directory in which ``data.h5`` is created.
            compression: HDF5 compression filter name, or ``None``.
            compression_opts: Options passed to the selected compression
                filter.
            shuffle: Whether to apply the HDF5 shuffle filter.
            fletcher32: Whether to apply the Fletcher32 checksum filter.
            chunk_cache_size: HDF5 raw chunk cache size in bytes.
            max_batches_in_memory: Number of batches buffered before a flush.
        """
        # compression
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_cache_size = chunk_cache_size
        self.max_batches_in_memory = max_batches_in_memory

        # Internal state
        self._file = None
        self._data_group = None
        self._batch_buffer: list[tuple[int, Any]] = []
        # Monotonically-increasing counter used to stamp each Signal with a
        # unique short string key (_hdf5_key attribute) before it is written.
        self._key_counter: int = 0

        self._current_sample_index = 0
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")  # fixed data file name
        # Thread safety
        self._lock = threading.Lock()

    def _data_dataset_kwargs(self) -> dict[str, Any]:
        """Handle HDF5 dataset kwarg tracking and implementation."""
        kwargs: dict[str, Any] = {}

        if self.compression is not None:
            kwargs["compression"] = self.compression
            if self.compression != "lzf" and self.compression_opts is not None:
                kwargs["compression_opts"] = self.compression_opts

        if self.shuffle:
            kwargs["shuffle"] = True
        if self.fletcher32:
            kwargs["fletcher32"] = True

        if kwargs:
            kwargs["chunks"] = True  # explicit, testable, and required for filters

        return kwargs

    def _setup(self) -> None:
        """Set up HDF5 file and initial structure."""
        # Create HDF5 file with optimized settings
        self._file = h5py.File(
            self.datapath,
            "w",
            libver="latest",  # Use latest HDF5 format for better performance
            swmr=False,  # Single writer mode for dataset creation
            rdcc_nbytes=self.chunk_cache_size,  # Chunk cache size
            rdcc_w0=0.75,  # Chunk cache policy
        )

        # Set global attributes
        self._file.attrs["torchsig_version"] = torchsig_cache_version()
        self._file.attrs["compression"] = self.compression or "none"
        self._file.attrs["created_by"] = "TorchSig HDF5FileHandler"
        self._file.create_group("data")
        self._file.create_group("metadata")
        self._file.create_group("index")
        self._file.create_group("component_signals")

    def teardown(self) -> None:
        """Clean up resources and close HDF5 file."""
        # Flush any remaining data if buffer exists
        if hasattr(self, "_batch_buffer") and self._batch_buffer:
            self._flush_buffer()
        # Close file
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass  # File might already be closed
            del self._file

    def _assign_hdf5_keys(self, signal) -> None:
        """Stamp *signal* and all its component signals with a unique ``_hdf5_key``.

        Called immediately before each batch is written so that every Signal
        object receives a short, monotonically-increasing string key.  The key
        is used by the module-level populate helpers instead of ``str(id(signal))``,
        making the HDF5 layout independent of CPython memory addresses and
        allowing signals to be garbage-collected as soon as they leave scope.
        """
        signal._hdf5_key = str(self._key_counter)
        self._key_counter += 1
        for cs in signal.component_signals:
            self._assign_hdf5_keys(cs)

    def _write_batch_to_hdf5(self, data) -> None:
        """Write a batch of signals to the open file.

        Args:
            data: Signals to write to the HDF5 file.
        """
        # Assign stable write keys before touching HDF5 so the populate
        # helpers never fall back to id()-based keys for signal objects.
        for signal in data:
            self._assign_hdf5_keys(signal)

        populate_hdf5_group_with_signals(
            self._file,
            data,
            data_dataset_kwargs=self._data_dataset_kwargs(),
        )

    def _flush_buffer(self) -> None:
        """Flush buffered batches to HDF5 file."""
        if not self._batch_buffer:
            return

        # Ensure file is open for writing
        if not self._file:
            self._setup()

        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()

        with self._lock:
            # Sort buffer by batch index to maintain order
            self._batch_buffer.sort(key=lambda x: x[0])

            # Process all batches in buffer
            for batch_idx, data in self._batch_buffer:
                self._write_batch_to_hdf5(data)

            # Clear buffer
            self._batch_buffer.clear()

            # Force flush to disk
            if self._file:
                self._file.flush()

    def write(self, batch_idx: int, data) -> None:
        """Write a batch of data to HDF5 file.

        Args:
            batch_idx (int): Index of the batch being written.
            data (Any): Signal data to write.
        """
        # Add to buffer
        with self._lock:
            self._batch_buffer.append((batch_idx, data))
            should_flush = len(self._batch_buffer) >= self.max_batches_in_memory

        # Flush buffer if it is getting too large
        if should_flush:
            self._flush_buffer()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self._file["index"])


def handle_bytes_as_string(bts):
    """Converts bytes to string if needed.

    Args:
        bts: The value to potentially convert.

    Returns:
        The converted value.
    """
    if isinstance(bts, bytes):
        return str(bts.decode())
    if isinstance(bts, np.ndarray) and bts.dtype == np.dtype("O"):
        return bts.astype(np.str_)
    return bts


def load_value_from_group(group, key):
    """Loads a value from an HDF5 group.

    Args:
        group: The HDF5 group to load from.
        key: The key of the value to load.

    Returns:
        The loaded value.
    """
    return handle_bytes_as_string(group[key][()])


def fill_object_metadata_from_group_and_id(obj, group, id_str):
    """Fills an object's metadata from an HDF5 group.

    Args:
        obj: The object to fill metadata for.
        group: The HDF5 group to load from.
        id_str: The ID string of the metadata to load.

    Returns:
        The object with filled metadata.
    """
    for key in group["metadata"][id_str].keys():
        if not key == "parent_metadata_id":
            obj[key] = load_value_from_group(group["metadata"][id_str], key)
    try:
        parent_id = load_value_from_group(group["metadata"][id_str], "parent_metadata_id")
        metadata_obj = fill_object_metadata_from_group_and_id(HierarchicalMetadataObject(), group, parent_id)
        obj.add_parent(metadata_obj)
    except:
        pass  # we have no parent set; do nothing
    return obj


def load_signal_from_group_by_id(group, id_str):
    """Loads a signal from an HDF5 group by its ID.

    Args:
        group: The HDF5 group to load from.
        id_str: The ID string of the signal to load.

    Returns:
        Signal: The loaded signal.
    """
    component_signals = []
    try:
        component_signals = [load_signal_from_group_by_id(group, temp_id) for temp_id in load_value_from_group(group["component_signals"], id_str)]
    except:
        pass
    signal = Signal(
        data=load_value_from_group(group["data"], id_str),
        component_signals=component_signals,
    )
    signal = fill_object_metadata_from_group_and_id(signal, group, id_str)
    return signal


def load_signal_from_group_by_index(group, ind):
    """Loads a signal from an HDF5 group by its index.

    Args:
        group: The HDF5 group to load from.
        ind: The index of the signal to load.

    Returns:
        Signal: The loaded signal.
    """
    id_str = load_value_from_group(group["index"], str(ind))
    return load_signal_from_group_by_id(group, id_str)


class HDF5Reader(FileReader):
    """Read Signals from TorchSig's legacy object-per-record HDF5 layout."""

    def __init__(self, root) -> None:
        """Initializes the HDF5Reader.

        Args:
            root (str): The root directory containing the HDF5 file.
        """
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file = None
        self._len_cache = None
        self._locking = False  # do not lock data file

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self._len_cache is None:
            # Open ONLY to read metadata, then close immediately
            with h5py.File(self.datapath, "r", locking=self._locking) as f:
                self._len_cache = len(f["index"])
        return self._len_cache

    def read(self, idx: int) -> Signal:
        """Read one signal, including metadata and component signals.

        Args:
            idx (int): The index of the sample to read.

        Returns:
            Signal: The sample as a Signal object.
        """
        self._ensure_open()
        return load_signal_from_group_by_index(self._file, idx)

    def teardown(self) -> None:
        """Closes the HDF5 file handle."""
        if self._file:
            self._file.close()
            self._file = None

    def _ensure_open(self) -> None:
        """Ensures that the HDF5 file is open for reading. Note that the file
        is opened lazily to mitigate issues with multiprocessing and worker
        initialization in PyTorch DataLoaders.
        """
        if self._file is None:
            self._file = h5py.File(self.datapath, "r", locking=self._locking)


class HDF5FileHandler(BaseFileHandler):
    """Create a legacy :class:`HDF5Reader` or :class:`HDF5Writer`."""

    reader_class: FileReader = HDF5Reader
    writer_class: FileWriter = HDF5Writer

    @staticmethod
    def create_handler(mode: str, root: str, **kwargs) -> HDF5Writer | HDF5Reader:
        """Creates an instance of HDF5Reader or HDF5Writer based on the mode.

        Args:
            mode (str): The mode, either "r" for read or "w" for write.
            root (str): The root directory for the file handler.
            **kwargs: Additional arguments for the file handler.

        Returns:
            HDF5Writer | HDF5Reader: The created file handler.

        Raises:
            ValueError: If the mode is invalid.
        """
        if mode == "r":
            return HDF5FileHandler.reader_class(root, **kwargs)
        if mode == "w":
            return HDF5FileHandler.writer_class(root, **kwargs)
        raise ValueError(f"Invalid File Handler mode: {mode}")
