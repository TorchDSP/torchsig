
from __future__ import annotations

# TorchSig
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.datasets.dataset_metadata import DatasetMetadata

# Third Party
import zarr
import numpy as np

# Built-In
from typing import TYPE_CHECKING, Tuple, List, Dict, Any
import os
import pickle

if TYPE_CHECKING:
    from torchsig.datasets.datasets import NewTorchSigDataset

class ZarrFileHandler(TorchSigFileHandler):
    """Handler for reading and writing data to/from a Zarr file format.

    This class extends the `TorchSigFileHandler` and provides functionality to handle 
    reading, writing, and managing Zarr-based storage for dataset samples.

    Attributes:
        datapath_filename (str): The name of the file used to store the data in Zarr format.
    """

    datapath_filename = "data.zarr"
    chunk_size = (100, )

    def __init__(
        self,
        root: str,
        dataset_metadata: DatasetMetadata,
        batch_size: int,
        train: bool = None,
    ):
        """Initializes the ZarrFileHandler with dataset metadata and write type.

        Args:
            dataset_metadata (DatasetMetadata): Metadata about the dataset, including 
                sample sizes and other configuration.
            write_type (str, optional): Specifies the write mode for the dataset ("raw" or otherwise). 
                Defaults to None.
        """
        super().__init__(
            root = root,
            dataset_metadata = dataset_metadata,
            batch_size = batch_size,
            train = train,
        )

        self.datapath = f"{self.root}/{ZarrFileHandler.datapath_filename}"

        self.data_shape = (self.dataset_metadata.num_samples, self.dataset_metadata.num_iq_samples_dataset)
        self.data_type = float
        # check data type and shape upon first batch
        self.zarr_updated = False 

        self.zarr_array = None

    def exists(self) -> bool:
        """Checks if the Zarr file exists at the specified path.

        Returns:
            bool: True if the Zarr file exists, otherwise False.
        """
        if os.path.exists(self.datapath):
            return True
        else:
            return False
    
    def _setup(self) -> None:
        """Sets up the Zarr file for writing by creating a new Zarr array.

        This method initializes the Zarr array with the specified data shape, type, 
        compression settings, and chunking.
        """
        self.zarr_array = zarr.open(
            self.datapath,
            mode = 'w', # create or overwrite if exists
            # array will be shape (num samples, num iq samples)
            shape = self.data_shape,
            # chunk array every 1000 elements
            chunks = ZarrFileHandler.chunk_size,
            # IQ data type
            dtype = self.data_type,
            # compression
            compressor = zarr.Blosc(
                cname = 'zstd', # type
                clevel = 4, # compression level
                shuffle = 2 # use bit shuffle
            )
        )

    def _update_zarr(self, data: Any) -> None:
        # first batch to be writtern
        # check and update zarr array data shape and type
        if self.data_shape[1:] != data[0].shape:
            # data is >1D, update data shape
            # print(f"current shape: {self.data_shape}, new shape: {data[0].shape}")
            self.data_shape = (self.data_shape[0],) + data[0].shape
            # print(self.data_shape)

        if self.data_type != data[0].dtype:
            self.data_type = data[0].dtype

        self._setup()

        self.zarr_updated = True

    def write(self, batch_idx: int, batch: Any) -> None:
        """Writes a sample (data and targets) to the Zarr file at the specified index.

        Args:
            idx (int): The index at which to store the data in the Zarr file.
            data (np.ndarray): The data to write to the Zarr file.
            targets (Any): The corresponding targets to write as metadata for the sample.
        
        Notes:
            If the index is greater than the current size of the array, the array is 
            expanded to accommodate the new sample.
        """

        start_idx = batch_idx * self.batch_size
        stop_idx = start_idx + len(batch[0])

        data, targets = batch

        # print(f"write: {targets}")

        if not self.zarr_updated:
            self._update_zarr(data)
        
        try:
            # set batched data into zarr array
            self.zarr_array[start_idx: stop_idx, :] = data
        except ValueError as v:
            print(v)
            raise MemoryError(f"Data too large to write to zarr array. Try a smaller batch size or smaller chunk size (ZarrFileHandler.chunk_size).")

        # add targets to zarr array attributes
        for tidx, target in enumerate(targets):
            # target index is start sample index + target index
            kidx = start_idx + tidx
            self.zarr_array.attrs[str(kidx)] = target

    @staticmethod
    def size(dataset_path: str) -> int:
        zarr_arr = zarr.open(f"{dataset_path}/{ZarrFileHandler.datapath_filename}", mode = 'r')

        return zarr_arr.shape[0]

    @staticmethod
    def static_load(filename:str, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Loads a sample from the Zarr file at the specified index.

        Args:
            filename (str): Path to the directory containing the Zarr file.
            idx (int): The index of the sample to load.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: The data and the associated metadata for the sample.
        
        Raises:
            IndexError: If the index is out of bounds.
        """
        # root/data.zarr
        zarr_arr = zarr.open(f"{filename}/{ZarrFileHandler.datapath_filename}", mode = 'r')

        data = zarr_arr[idx]
        
        targets = zarr_arr.attrs[str(idx)]

        # print(f"load: {targets}")
        # print(data)
        # breakpoint()

        if isinstance(targets, tuple) or isinstance(targets, list):
            # target has multiple outputs
            if isinstance(targets[0], list):
                # convert `wideband targets (2D list) to a list of tuples
                # also convert any nested lists into tuples
                targets = list(
                    tuple(item if not isinstance(item, list) else tuple(item) for item in target)
                    for target in targets
                )
            else:
                # convert narrowband targets (1D list) to a tuple
                targets = tuple(targets)
        # else:
            # narrowband target (single item), return itself

        # print(f"post load: {targets}")

        return data, targets
    

    def load(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]] | Tuple[Any,...]:
        """Loads a sample from the Zarr file at the specified index into memory.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]] | Tuple[Any, ...]]]: The data and the corresponding 
            targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        # # loads sample `idx` from disk into memory

        # return data, targets
        return ZarrFileHandler.static_load(self.root, idx)
