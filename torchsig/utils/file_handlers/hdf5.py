"""HDF5 File Handler for TorchSig datasets.

High-performance HDF5 storage with optimized compression and chunking.
"""

from __future__ import annotations

# Built-In
import threading
from typing import Any

import h5py

# Third Party
import numpy as np

from torchsig import __version__ as torchsig_version

# TorchSig
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers import BaseFileHandler, FileReader, FileWriter


def _load_metadata_lazy(*args, **kwargs):
    """Deferred to avoid touching torchsig.datasets at import time."""
    from torchsig.datasets.dataset_metadata import load_dataset_metadata

    return load_dataset_metadata(*args, **kwargs)




def populate_hdf5_group_with_metadata(group, metadata_obj):
    """Makes sure this and all parent metadata objects are represented in the hdf5 group (returns true iff a new group was added)"""
    id_string = str(id(metadata_obj))
    try:
        # if there is already metadata awith this id, do nothing
        temp = group[id_string]
        return False
    except KeyError:
        # there is not already metadata with this id
        metadata_group = group.create_group(id_string)
        for key in metadata_obj.keys():
            if not metadata_obj[key] == None:
                metadata_group.create_dataset(key, data=metadata_obj[key])
        if not metadata_obj.parent == None:
            try:
                metadata_group.create_dataset(
                    "parent_metadata_id", data=str(id(metadata_obj.parent))
                )
                populate_hdf5_group_with_metadata(group, metadata_obj.parent)
            except ValueError:
                pass
        return True


def populate_hdf5_group_with_signal_data(group, signal):
    """Makes sure this and all parent metadata objects are represented in the hdf5 group (returns true iff a new group was added)"""
    id_string = str(id(signal))
    try:
        # if there is already data awith this id, do nothing
        temp = group[id_string]
        return False
    except KeyError:
        # there is not already data with this id
        try:
            group.create_dataset(id_string, data=signal.data)
        except ValueError:
            pass
        return True


def populate_hdf5_group_with_component_signals(group, signal):
    if len(signal.component_signals) > 0:
        try:
            group.create_dataset(
                str(id(signal)),
                data=[
                    str(id(component_signal))
                    for component_signal in signal.component_signals
                ],
            )
        except ValueError:
            pass
        return True
    return False


def _populate_hdf5_group_with_signal(group, signal):
    populate_hdf5_group_with_metadata(group["metadata"], signal)
    populate_hdf5_group_with_signal_data(group["data"], signal)
    populate_hdf5_group_with_component_signals(group["component_signals"], signal)
    for component_signal in signal.component_signals:
        _populate_hdf5_group_with_signal(group, component_signal)


def populate_hdf5_group_with_signal(group, signal, index=True):
    _populate_hdf5_group_with_signal(group, signal)
    if index:
        group["index"].create_dataset(
            str(len(group["index"])), data=str(id(signal))
        )  # keep track of this index in a dataset


def populate_hdf5_group_with_signals(group, signals, index=True):
    for signal in signals:
        populate_hdf5_group_with_signal(group, signal, index=index)


class HDF5Writer(FileWriter):
    """Handles writing Signal data to HDF5 files with specified compression and buffering."""

    def __init__(
        self,
        root,
        compression: str = "gzip",
        compression_opts: int = 6,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 1024 * 1024 * 10,  # 10MB cache
        max_batches_in_memory: int = 4,
    ):
        """Initializes the HDF5FileHandler.

        Args:
            root (str): Where to write dataset on disk.
            compression (str, optional): Compression algorithm ('gzip', 'szip', 'lzf'). Defaults to 'gzip'.
            compression_opts (int, optional): Compression level (0-9 for gzip). Defaults to 6.
            shuffle (bool, optional): Enable shuffle filter for better compression. Defaults to True.
            fletcher32 (bool, optional): Enable Fletcher32 checksum filter. Defaults to True.
            chunk_cache_size (int, optional): HDF5 chunk cache size in bytes. Defaults to 10MB.
            max_batches_in_memory (int, optional): Maximum batches to keep in memory before flushing. Defaults to 4.
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

        self._current_sample_index = 0
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        # Thread safety
        self._lock = threading.Lock()

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
        self._file.attrs["torchsig_version"] = torchsig_version
        self._file.attrs["compression"] = self.compression
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

    def _write_batch_to_hdf5(self, data) -> None:
        """Writes a batch of signals (as List[Signal]) to the file.

        Args:
            data (List[Signal]): The list of signals to write to the HDF5 file.
        """
        populate_hdf5_group_with_signals(self._file, data)

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
                # breakpoint()
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
        self._batch_buffer.append((batch_idx, data))

        # Flush buffer if it's getting too large
        if len(self._batch_buffer) >= self.max_batches_in_memory:
            self._flush_buffer()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self._file["index"])


def handle_bytes_as_string(bts):
    if isinstance(bts, bytes):
        return str(bts.decode())
    if isinstance(bts, np.ndarray):
        if bts.dtype == np.dtype("O"):
            return bts.astype(np.str_)
    return bts


def load_value_from_group(group, key):
    return handle_bytes_as_string(group[key][()])


def fill_object_metadata_from_group_and_id(obj, group, id_str):
    for key in group["metadata"][id_str].keys():
        if not key == "parent_metadata_id":
            obj[key] = load_value_from_group(group["metadata"][id_str], key)
    try:
        parent_id = load_value_from_group(
            group["metadata"][id_str], "parent_metadata_id"
        )
        metadata_obj = fill_object_metadata_from_group_and_id(
            HierarchicalMetadataObject(), group, parent_id
        )
        obj.add_parent(metadata_obj)
    except:
        pass  # we have no parent set; do nothing
    return obj


def load_signal_from_group_by_id(group, id_str):
    component_signals = []
    try:
        component_signals = [
            load_signal_from_group_by_id(group, temp_id)
            for temp_id in load_value_from_group(group["component_signals"], id_str)
        ]
    except:
        pass
    signal = Signal(
        data=load_value_from_group(group["data"], id_str),
        component_signals=component_signals,
    )
    signal = fill_object_metadata_from_group_and_id(signal, group, id_str)
    return signal


def load_signal_from_group_by_index(group, ind):
    id_str = load_value_from_group(group["index"], str(ind))
    return load_signal_from_group_by_id(group, id_str)


class HDF5Reader(FileReader):
    """Handles reading Signal data from HDF5 files."""

    def __init__(self, root) -> None:
        """Initializes the HDF5Reader.

        Args:
            root (str): The root directory containing the HDF5 file.
        """
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file = h5py.File(self.datapath, "r")

    def read(self, idx: int) -> Signal:
        """Reads a single sample and its corresponding targets from the HDF5 file.

        Args:
            idx (int): The index of the sample to read.

        Returns:
            Signal: The sample as a Signal object.
        """
        return load_signal_from_group_by_index(self._file, idx)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self._file["index"])

    def teardown(self) -> None:
        """Closes the HDF5 file handle."""
        if self._file:
            self._file.close()
            self._file = None


class HDF5FileHandler(BaseFileHandler):
    """HDF5FileHandler creates a reader or writer for HDF5 files."""

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
