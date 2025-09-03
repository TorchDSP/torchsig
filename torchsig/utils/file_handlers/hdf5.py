""" HDF5 File Handler for TorchSig datasets.

High-performance HDF5 storage with optimized compression and chunking.
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_metadata import load_dataset_metadata
from torchsig.signals.signal_types import (Signal, SignalMetadata)
from torchsig import __version__ as torchsig_version
from torchsig.utils.file_handlers import (
    FileWriter,
    FileReader,
    BaseFileHandler
)

# Third Party
import numpy as np
import h5py

# Built-In
import threading



def populate_hdf5_group_with_signal(group, sig):
    """Inserts a Signal object's data and metadata into the HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to add the Signal to.
        sig (Signal): The Signal object.
    """
    group.create_dataset("data", data=sig.data)
    metadata_group = group.create_group("metadata")
    metadatas = sig.get_full_metadata()
    counter = -1
    for metadata in metadatas:
        counter += 1
        if metadata is not None:
            index_group = metadata_group.create_group(str(counter))
            metadata_dict = metadata.to_dict()
            for key in metadata_dict.keys():
                if metadata_dict[key] is not None:
                    if isinstance(metadata_dict[key], str) or np.isscalar(metadata_dict[key]):
                        index_group.create_dataset(key, data=metadata_dict[key])
                    else:
                        try:
                            index_group.create_dataset(key, data=np.array(metadata_dict[key]))
                        except:
                            index_group.create_dataset(key, data=metadata_dict[key])

def populate_hdf5_group_with_signals(group, sigs):
    """Inserts a list of Signal objects into the HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to add the Signals to.
        sigs (List[Signal]): The list of Signal objects.
    """
    for i in range(len(sigs)):
        signal_group = group.create_group(str(len(group)))
        populate_hdf5_group_with_signal(signal_group, sigs[i])

class HDF5Writer(FileWriter):
    """Handles writing Signal data to HDF5 files with specified compression and buffering."""

    def __init__(  # noqa: D107
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
        self._batch_buffer: List[Tuple[int, Any]] = []
        
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
            'w',
            libver='latest',  # Use latest HDF5 format for better performance
            swmr=False,  # Single writer mode for dataset creation
            rdcc_nbytes=self.chunk_cache_size,  # Chunk cache size
            rdcc_w0=0.75,  # Chunk cache policy
        )
        
        # Set global attributes
        self._file.attrs['torchsig_version'] = torchsig_version
        self._file.attrs['compression'] = self.compression
        self._file.attrs['created_by'] = 'TorchSig HDF5FileHandler'

    def teardown(self) -> None:
        """Clean up resources and close HDF5 file."""
        # Flush any remaining data if buffer exists
        if hasattr(self, '_batch_buffer') and self._batch_buffer:
            self._flush_buffer()
        # Close file
        if hasattr(self, '_file') and self._file is not None:
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
            
        if not hasattr(self, '_lock'):
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
        return len(self._file)

def pop_nullable_dict_field(dic, field) -> Any:
    """Removes a field from a dictionary and returns its value, handling byte strings.

    Args:
        dic (dict): The dictionary to pop the field from.
        field (Any): The field to pop.

    Returns:
        Any: The value associated with the field, or None if the field is not present.
    """
    if field in dic.keys():
        value = dic[field]
        del dic[field]
        return handle_bytes_as_string(value)
    return None

def handle_bytes_as_string(bts) -> Any:
    """Converts byte strings to standard strings.

    Args:
        bts (Any): The input, which can be a byte string.

    Returns:
        str: The standard string if input was a byte string, otherwise returns the input as is.
    """
    if isinstance(bts, bytes):
        return str(bts.decode())
    return bts

def dict_to_signal_metadata(dic) -> SignalMetadata:
    """Converts a dictionary to a SignalMetadata object.

    Args:
        dic (dict): The dictionary to convert.

    Returns:
        SignalMetadata: The generated SignalMetadata object.
    """
    dataset_metadata = pop_nullable_dict_field(dic, "dataset_metadata")
    center_freq = pop_nullable_dict_field(dic, "center_freq")
    bandwidth = pop_nullable_dict_field(dic, "bandwidth")
    start_in_samples = pop_nullable_dict_field(dic, "start_in_samples")
    duration_in_samples = pop_nullable_dict_field(dic, "duration_in_samples")
    snr_db = pop_nullable_dict_field(dic, "snr_db")
    class_name = pop_nullable_dict_field(dic, "class_name")
    class_index = pop_nullable_dict_field(dic, "class_index")

    new_metadata = SignalMetadata(
        dataset_metadata=dataset_metadata,
        center_freq=center_freq,
        bandwidth=bandwidth,
        start_in_samples=start_in_samples,
        duration_in_samples=duration_in_samples,
        snr_db=snr_db,
        class_name=class_name,
        class_index=class_index,
    )
    for field in dic.keys():
        setattr(new_metadata, field, handle_bytes_as_string(dic[field]))
    return new_metadata

def hdf5_group_to_dict(group) -> dict:
    """Converts an HDF5 group to a dictionary.

    Args:
        group (h5py.Group): The HDF5 group to convert.

    Returns:
        dict: The resulting dictionary.
    """
    new_dict = {}
    for key in group.keys():
        new_dict[key] = group[key][()]
    return new_dict

def hdf5_group_to_signal(group) -> Signal:
    """Converts an HDF5 group to a Signal object.

    Args:
        group (h5py.Group): The HDF5 group to convert.

    Returns:
        Signal: The generated Signal object.
    """
    metadatas = [dict_to_signal_metadata(hdf5_group_to_dict(group['metadata'][key])) for key in group['metadata'].keys()]
    metadata = None
    if len(metadatas) == 1:
        metadata = metadatas[0]
    component_signals = []
    if len(metadatas) > 1:
        component_signals = [Signal(data=None, metadata=component_metadata, component_signals=[]) for component_metadata in metadatas]
    data = group['data'][()]
    return Signal(data=data, metadata=metadata, component_signals=component_signals)

class HDF5Reader(FileReader):
    """Handles reading Signal data from HDF5 files."""

    def __init__(self, root) -> None:
        """Initializes the HDF5Reader.

        Args:
            root (str): The root directory containing the HDF5 file.
        """
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file = h5py.File(self.datapath, 'r')
        self.dataset_metadata = load_dataset_metadata(self.dataset_info_filepath)

    def read(self, idx: int) -> Signal:
        """Reads a single sample and its corresponding targets from the HDF5 file.

        Args:
            idx (int): The index of the sample to read.

        Returns:
            Signal: The sample as a Signal object.
        """
        new_signal = hdf5_group_to_signal(self._file[str(idx)])
        new_signal.dataset_metadata = self.dataset_metadata
        for metadata in new_signal.get_full_metadata():
            metadata.dataset_metadata = self.dataset_metadata
        return new_signal

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self._file)

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
        elif mode == "w":
            return HDF5FileHandler.writer_class(root, **kwargs)
        else:
            raise ValueError(f"Invalid File Handler mode: {mode}")