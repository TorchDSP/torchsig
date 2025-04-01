
"""
zarr==2.18.3
"""
from __future__ import annotations

# TorchSig
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.dataset_utils import writer_yaml_name

# Third Party
import zarr
import numpy as np

# Built-In
from typing import TYPE_CHECKING, Tuple, List, Dict, Any
import os
import pickle
import yaml

class ZarrFileHandler(TorchSigFileHandler):
    """Handler for reading and writing data to/from a Zarr file format.

    This class extends the `TorchSigFileHandler` and provides functionality to handle 
    reading, writing, and managing Zarr-based storage for dataset samples.

    Attributes:
        datapath_filename (str): The name of the folder used to store the data in Zarr format.
    """

    datapath_filename_base = "data"

    def __init__(
        self,
        root: str,
        batch_size: int = 1,
    ):
        """Initializes the ZarrFileHandler

        Args:
            root (str): Where to write dataset on disk.
            batch_size (int, optional): Size fo each batch write. Defaults to 1.
        """        
        super().__init__(
            root = root,
            batch_size = batch_size
        )

        self.datapath = f"{self.root}/{ZarrFileHandler.datapath_filename_base}"

        # compressor
        self.compressor = zarr.Blosc(
            cname = 'zstd', # type
            clevel = 4, # compression level
            shuffle = 2 # use bit shuffle
        )

    def exists(self) -> bool:
        """Checks if the Zarr file exists at the specified path.

        Returns:
            bool: True if the Zarr file exists, otherwise False.
        """
        num_files = len(os.listdir(self.datapath))
        if os.path.exists(self.datapath) and num_files > 0:
            return True
        else:
            return False

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

        
        # write batch of data into file
        zarr_array = zarr.open(
            # filenames will have 10 digits
            # might need to change if you have more than 1 billion batches 
            f"{self.datapath}/{batch_idx:010}.zarr",
            mode = 'w', # create or overwrite if exists
            # array will be shape (num samples, num iq samples)
            shape = (len(data),) + data[0].shape,
            # Data type
            dtype = data[0].dtype,
            # compression
            compressor = self.compressor
        )
        zarr_array[:] = np.array(data)

        # add targets to zarr array attributes
        attrs_dict = {str(start_idx + tidx): target for tidx, target in enumerate(targets)}

        zarr_array.attrs.update(attrs_dict) 



    @staticmethod
    def size(dataset_path: str) -> int:
        """Return size of dataset

        Args:
            dataset_path (str): path to dataset on disk

        Returns:
            int: size of dataset
        """        
        # find batch size
        batch_size = TorchSigFileHandler._calculate_batch_size(dataset_path)
        
        # count number of files
        all_zarr_arrays = sorted(os.listdir(f"{dataset_path}/{ZarrFileHandler.datapath_filename_base}"))
        num_zarr_files = len(all_zarr_arrays)

        # num files * batch size
        size = batch_size * (num_zarr_files - 1)

        # check last file, since it might have less than batch_size data points
        last_array = zarr.open(f"{dataset_path}/{ZarrFileHandler.datapath_filename_base}/{all_zarr_arrays[-1]}", mode = 'r')
        last_batch_size = last_array.shape[0]

        # add size of last batch file
        size += last_batch_size

        return size

    @staticmethod
    def static_load(filename:str, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Loads a sample from the Zarr file at the specified index (without instantiating a ZarrFileHandler)

        Args:
            filename (str): Path to the directory containing the Zarr file.
            idx (int): The index of the sample to load.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: The data and the associated metadata for the sample.
        
        Raises:
            IndexError: If the index is out of bounds.
        """

        # calculate batch size
        batch_size = TorchSigFileHandler._calculate_batch_size(filename)
        batch_idx = idx // batch_size
        batch_file_idx = idx % batch_size 

        # find correct file
        batch_filename = f"{batch_idx:010}.zarr"

        # load in
        # root/data/batch filename.zarr
        zarr_arr = zarr.open(f"{filename}/{ZarrFileHandler.datapath_filename_base}/{batch_filename}", mode = 'r')

        data = zarr_arr[batch_file_idx]
        
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
