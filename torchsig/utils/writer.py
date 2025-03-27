"""Dataset Writer Utils
"""

from __future__ import annotations

# TorchSig
from torch.utils.data import DataLoader
from torchsig.datasets.datasets import NewTorchSigDataset, dataset_yaml_name, writer_yaml_name
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler

from torchsig.datasets.dataset_utils import save_type
from torchsig.datasets.dataset_utils import collate_fn as default_collate_fn
from torchsig.utils.yaml import write_dict_to_yaml

# Third Party
from tqdm import tqdm
import yaml
import numpy as np

# Built-In
from typing import Callable, Dict, Any, List, Tuple
from pathlib import Path
import os
from shutil import disk_usage


class DatasetCreator:
    """Class for creating a dataset and saving it to disk in batches.

    This class generates a dataset if it doesn't already exist on disk. 
    It processes the data in batches and saves it using a specified file handler. 
    The class allows setting options like whether to overwrite existing datasets, 
    batch size, and number of worker threads.

    Attributes:
        root (Path): The root directory where the dataset will be saved.
        overwrite (bool): Flag indicating whether to overwrite an existing dataset.
        batch_size (int): The number of samples in each batch.
        num_workers (int): The number of worker threads to use for data loading.
        save_type (str): The type of dataset being saved ("raw" or "processed").
        tqdm_desc (str): A description for the progress bar.
        writer (TorchSigFileHandler): The file handler used for saving the dataset.
        dataloader (DataLoader): The DataLoader used to load data in batches.
    """
    def __init__(
        self,
        dataset: NewTorchSigDataset,
        root: str,
        overwrite: bool = False, # will overwrite any existing dataset on disk
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = default_collate_fn,
        tqdm_desc: str = None,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        train: bool = None,
    ):
        """Initializes the DatasetCreator.

        Args:
            dataset (NewTorchSigDataset): The dataset to be written to disk.
            root (str): The root directory where the dataset will be saved.
            overwrite (bool): Whether to overwrite an existing dataset (default: False).
            batch_size (int): The number of samples per batch (default: 1).
            num_workers (int): The number of workers for loading data (default: 1).
            collate_fn (Callable): Function to merge a list of samples into a batch (default: default_collate_fn).
            tqdm_desc (str): Description for the tqdm progress bar (optional).
            file_handler (TorchSigFileHandler): File handler for saving the dataset (default: ZarrFileHandler).
            train (bool): Whether the dataset is for training (optional).
        
        Raises:
            ValueError: If the dataset does not specify `num_samples`.
        """
        self.root = Path(root)
        self.overwrite = overwrite
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset.dataset_metadata.num_samples is None:
            raise ValueError("Must specify num_samples as an integer number. Cannot write infinite dataset to disk.")

        self.dataloader = DataLoader(
            dataset = dataset,
            num_workers = num_workers,
            batch_size = batch_size,
            collate_fn = collate_fn
        )
        self.writer = file_handler(
            root = self.root,
            dataset_metadata = dataset.dataset_metadata,
            batch_size = batch_size,
            train = train,
        )
        # save_type (str): What kind of data was written to disk.
        # * "raw" means data and metadata after impairments are applied, but no other transforms and target transforms.
        #     * When loaded back in, users can choose what transforms or target transforms to apply.
        #     * Choose this option if you want to create a dataset that you (or multiple people) can later choose their own transforms and target transforms.
        # * "processed" means data and targets after all transforms and target transforms are applied.
        #     * When loaded back in, users cannot change the transforms or target transform already applied to data.
        #     * Choose this option if you want to lock in the transforms and target transform applied, or if you want maximum speed and/or minimal disk space used.
        self.save_type = "raw" if save_type(
            dataset.dataset_metadata.transforms,
            dataset.dataset_metadata.target_transforms
        ) else "processed"

        self.tqdm_desc = f"Generating {self.dataloader.dataset.dataset_metadata.dataset_type.title()}" if tqdm_desc is None else tqdm_desc

        # limit in gigabytes for remaining space on disk for which writer stops writing
        self.minimum_remaining_disk_gigabytes = 1

    
    def get_writing_info_dict(self) -> Dict[str, Any]:
        """Returns a dictionary with information about the dataset being written.

        This method gathers information regarding the root, overwrite status, 
        batch size, number of workers, file handler class, and the save type 
        of the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing the dataset writing configuration.
        """
        return {
            'root':str(self.root),
            'full_root': self.writer.root,
            'overwrite': self.overwrite,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'file_handler': self.writer.__class__.__name__,
            'save_type': self.save_type
        }

    def check_yamls(self) -> List[Tuple[str, Any, Any]]:
        """Checks for differences between the dataset metadata on disk and the dataset metadata in memory.

        Compares the dataset metadata that would be written to disk against the 
        existing metadata on disk. Returns a list of differences.

        Returns:
            List[Tuple[str, Any, Any]]: List of differences between metadata on disk and in memory.
        """
        to_write_dataset_metadata = self.dataloader.dataset.dataset_metadata.to_dict()
        
        dataset_yaml = f"{self.writer.root}/{dataset_yaml_name}"
        different_params = []

        if os.path.exists(dataset_yaml):
            with open(dataset_yaml, 'r') as f:
                dataset_metadata_yaml = yaml.load(f, Loader=yaml.FullLoader)

                found_params = dataset_metadata_yaml["required"]
                found_params.update(dataset_metadata_yaml["overrides"])

                current_params = to_write_dataset_metadata["required"]
                current_params.update(to_write_dataset_metadata["overrides"])

                for k,v in found_params.items():
                    if current_params[k] != v:
                        different_params.append((k, v, current_params[k]))

        return different_params

    
    def create(self) -> None:
        """Creates the dataset on disk by writing batches to the file handler.

        This method generates the dataset in batches and saves it to disk. If the 
        dataset already exists and `overwrite` is set to False, it will skip regeneration.

        The method also writes the dataset metadata and writing information to YAML files.

        Raises:
            ValueError: If the dataset is already generated and `overwrite` is set to False.
        """
        if self.writer.exists() and not self.overwrite:
            different_params = self.check_yamls()
            if len(different_params) == 0:
                print(f"Dataset already exists in {self.writer.root}. Not regenerating.")
                return
            # else:
            # dataset exists on disk with different params
            # use dataset on disk instead
            # warn users that params are different
            print(f"Dataset exists at {self.writer.root} but is different than current dataset.")
            print("Differences:")
            for row in different_params:
                key, disk_value, current_value = row
                print(f"\t{key} = {current_value} ({disk_value} found)")
            print("If you want to overwrite dataset on disk, set overwrite = True for the DatasetCreator.")
            print("Not regenerating. Using dataset on disk.")
            return

        # set up writer
        self.writer.setup()

        # generate info yamls
        write_dict_to_yaml(f"{self.writer.root}/{dataset_yaml_name}", self.dataloader.dataset.dataset_metadata.to_dict())
        write_dict_to_yaml(f"{self.writer.root}/{writer_yaml_name}", self.get_writing_info_dict())

        # get reference to tqdm progress bar object
        pbar = tqdm()

        # update progress bar message
        self._update_tqdm_message(pbar)

        for batch_idx, batch in tqdm(enumerate(self.dataloader), total = len(self.dataloader)):

            # write to disk
            self.writer.write(batch_idx, batch)

            # update progress bar message
            self._update_tqdm_message(pbar,batch_idx)


    def _update_tqdm_message( self, pbar=tqdm(), batch_idx:int = 0 ):
        """Updates the tqdm progress bar with remaining disk space

        Informs the user how much remaining space left (in gigabytes) is
        on their disk. Includes a check to stop writing to disk in case
        the disk is at risk of being completely filled.
   
        Raises:
            ValueError: If the disk space remaining is below a threshold
        """

        # run periodically
        if (np.mod(batch_idx,10) == 0):

            # get the amount of disk space remaining
            disk_size_available_bytes = disk_usage(self.writer.root)[2]
            # convert to GB and round to two decimal places
            disk_size_available_gigabytes = np.round(disk_size_available_bytes/(1024**3),2)

            # get size of dataset written so far
            dataset_size_current_gigabytes = self._get_directory_size_gigabytes(self.writer.root)
            # estimate size per sample
            dataset_size_per_sample_gigabytes = dataset_size_current_gigabytes/(batch_idx+1)
            # number of samples left
            num_samples_remaining = len(self.dataloader)-(batch_idx+1)
            # project estimated size
            dataset_size_remaining_gigabytes = np.round(dataset_size_per_sample_gigabytes*num_samples_remaining,2)

            # concatenate disk size for progress bar message
            updated_tqdm_desc = f'{self.tqdm_desc}, dataset remaining to create = {dataset_size_remaining_gigabytes} GB, remaining disk = {disk_size_available_gigabytes} GB'

            # avoid crashing by stopping write process
            if (disk_size_available_gigabytes < self.minimum_remaining_disk_gigabytes):
                # remaining disk size is below a hard cutoff value to avoid crashing operating system
                raise ValueError(f'Disk nearly full! Remaining space is {disk_size_available_gigabytes} GB. Please make space before continuing.')
            elif (dataset_size_remaining_gigabytes > disk_size_available_gigabytes):
                # projected size of dataset too large for available disk space
                raise ValueError(f'Not enough disk space. Projected dataset size is {dataset_size_remaining_gigabytes} GB. Remaining space is {disk_size_available_gigabytes} GB. Please reduce dataset size or make space before continuing.')

            # set the progress bar message
            pbar.set_description(updated_tqdm_desc)


    def _get_directory_size_gigabytes ( self, start_path ):
        """
        Returns total size of a directory (including subdirs) in gigabytes
        """
        total_size = 0
        for path, dirs, files in os.walk(start_path):
           for f in files:
              fp = os.path.join(path, f)
              total_size += os.path.getsize(fp)
        
        total_size_GB = total_size/(1024**3)
        return total_size_GB


