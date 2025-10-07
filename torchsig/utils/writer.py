"""Dataset Writer Utils
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_utils import dataset_yaml_name, writer_yaml_name
from torchsig.utils.file_handlers.base_handler import FileWriter as TorchSigFileHandler
from torchsig.utils.file_handlers.hdf5 import HDF5Writer as DEFAULT_FILE_HANDLER
from torchsig.utils.yaml import write_dict_to_yaml
from torchsig.signals.signal_types import Signal, SignalMetadata, targets_as_metadata

# Third Party
from tqdm.auto import tqdm
import yaml
import numpy as np
from time import time
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate as torch_default_collate

# Built-In
from typing import Dict, Any, List, Tuple
from pathlib import Path
import os
from shutil import disk_usage
import concurrent.futures
import threading

def default_collate_fn(batch):
    """Collates a batch by zipping its elements together.

    Args:
        batch (tuple): A batch from the dataloader.

    Returns:
        tuple: A tuple of zipped elements, where each element corresponds to a single batch item.
    """
    return tuple(zip(*batch))

def handle_non_numpy_datatypes(data):
    if isinstance(data, Tensor):
        data = data.numpy()
    return data

def batch_as_signal_list(batch, target_labels = None, dataset_metadata = None):
    signal_list = []
    if isinstance(batch, tuple) and len(batch) == 2:
        datas, batch_targets = batch[0], batch[1]
        for i in range(len(datas)):
            data = handle_non_numpy_datatypes(datas[i])
            metadata = None
            component_signals=[]
            targets = batch_targets[i]
            if dataset_metadata.num_signals_max == 1:
                #do make on a single signal with no comonents
                metadata = targets_as_metadata(targets, target_labels, dataset_metadata)
            else:
                #many a single signal with multiple components containing the given targets
                for component_targets in targets:
                    component_metadata = targets_as_metadata(component_targets, target_labels, dataset_metadata)
                    component_signals += [Signal(data=None, metadata=component_metadata, component_signals=[])]

            signal_list += [Signal(data=data, metadata=metadata, component_signals=component_signals)]
        return signal_list
    elif isinstance(batch, np.ndarray):
        if len(batch.shape) < 2:
            batch = batch.reshape(1,-1)
        for row in batch:
            signal_list += [Signal(data=handle_non_numpy_datatypes(row), metadata=None, component_signals=[])]
        return signal_list
    elif isinstance(batch, list):
        for s in batch:
            if isinstance(s,Signal):
                signal_list += [s]
            else:
                raise ValueError("could not parse batch input as signals")
        return signal_list
    raise ValueError("could not parse batch input as signals")


class DatasetCreator():
    """Class for creating a dataset and saving it to disk in batches.

    This class generates a dataset if it doesn't already exist on disk. 
    It processes the data in batches and saves it using a specified file handler. 
    The class allows setting options like whether to overwrite existing datasets, 
    batch size, and number of worker threads.

    Attributes:
        dataloader (DataLoader): The DataLoader used to load data in batches.
        root (Path): The root directory where the dataset will be saved.
        overwrite (bool): Flag indicating whether to overwrite an existing dataset.
        tqdm_desc (str): A description for the progress bar.
        file_handler (TorchSigFileHandler): The file handler used for saving the dataset.
    """
    def __init__(
        self,
        dataloader: DataLoader = None,
        dataset_length : int = None,
        root: str = '.',
        overwrite: bool = True, # will overwrite any existing dataset on disk
        tqdm_desc: str = None,
        file_handler: TorchSigFileHandler = DEFAULT_FILE_HANDLER,
        multithreading: bool = True,
        **kwargs # any additional file handler args
    ):
        """Initializes the DatasetCreator.

        Args:
            dataloader (DataLoader): The DataLoader used to load data in batches.
            dataset_length (int): The number of samples to draw from a dataset.
            root (Path): The root directory where the dataset will be saved.
            overwrite (bool): Flag indicating whether to overwrite an existing dataset.
            tqdm_desc (str): A description for the progress bar.
            file_handler (TorchSigFileHandler): The file handler used for saving the dataset.
        """
        self.root = Path(root)
        self.dataset_info_filepath = self.root.joinpath("dataset_info.yaml")
        self.writer_info_filepath = self.root.joinpath("writer_info.yaml")
        self.dataset_length = dataset_length
        self.overwrite = overwrite
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers
        self.multithreading = multithreading
        self.num_batches = self.dataset_length//self.batch_size
        if self.dataset_length % self.batch_size != 0:
            self.num_batches += 1 # include the partial batch at the end if it can't be evenly batched

        self.dataloader = dataloader
        if self.dataloader.dataset.target_labels is None and self.dataloader.collate_fn == torch_default_collate:
            # DataLoader should just return Signal objects
            # do not use torch's default collate function
            self.dataloader.collate_fn = lambda x: x

        self.file_handler = file_handler

        # get reference to tqdm progress bar object
        self.pbar = None

        self.tqdm_desc = "Generating Dataset:" if tqdm_desc is None else tqdm_desc

        # limit in gigabytes for remaining space on disk for which writer stops writing
        self.minimum_remaining_disk_gigabytes = 1

        # Thread lock for updating tqdm message to avoid race conditions
        self._tqdm_lock = threading.Lock()

        self._msg_timer = None

    
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
            'overwrite': self.overwrite,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'complete': False,
        }

    def check_yamls(self) -> List[Tuple[str, Any, Any]]:
        """Checks for differences between the dataset metadata on disk and the dataset metadata in memory.

        Compares the dataset metadata that would be written to disk against the 
        existing metadata on disk. Returns a list of differences.

        Returns:
            List[Tuple[str, Any, Any]]: List of differences between metadata on disk and in memory.
        """
        to_write_dataset_metadata = self.dataloader.dataset.dataset_metadata.to_dict()

        complete = False
        with open(self.writer_info_filepath, 'r') as f:
            writer_dict = yaml.load(f, Loader=yaml.FullLoader)
            # check if dataset finished writing
            complete = writer_dict['complete']
        
        different_params = []

        if os.path.exists(self.dataset_info_filepath):
            with open(dataset_yaml, 'r') as f:
                dataset_metadata_yaml = yaml.load(f, Loader=yaml.FullLoader)

                found_params = dataset_metadata_yaml["required"]
                found_params.update(dataset_metadata_yaml["overrides"])

                current_params = to_write_dataset_metadata["required"]
                current_params.update(to_write_dataset_metadata["overrides"])

                for k,v in found_params.items():
                    if current_params[k] != v:
                        different_params.append((k, v, current_params[k]))

        return complete, different_params

    def _write_batch(self, writer, batch_idx: int, batch: Any):
        """Multi-threaded writer batch
        Args:
            batch_idx (int): batch index
            batch (Any): batch
        """        
        try:
            # write to disk
            writer.write(batch_idx, batch)
        finally:
            # Clear batch reference to help garbage collection
            del batch

    def create(self) -> None:
        """Creates the dataset on disk by writing batches to the file handler.

        This method generates the dataset in batches and saves it to disk. If the 
        dataset already exists and `overwrite` is set to False, it will skip regeneration.

        The method also writes the dataset metadata and writing information to YAML files.

        Raises:
            ValueError: If the dataset is already generated and `overwrite` is set to False.
        """
        with self.file_handler(root = self.root) as writer:
            if writer.exists() and not self.overwrite:
                complete, different_params = self.check_yamls()
                if np.equal(len(different_params),0) and complete:
                    print(f"Dataset already exists in {self.root}. Not regenerating.")
                    return
    
                if not complete:
                    # dataset on disk is corrupted
                    # dataset was not fully written to disk
                    raise RuntimeError(f"Dataset only partially exists in {self.root} (writing dataset to disk was cancelled early). Regenerate the dataset by setting overwrite = True for DatasetCreator")
                # else:
                # dataset exists on disk with different params
                # use dataset on disk instead
                # warn users that params are different
                print(f"Dataset exists at {self.root} but is different than current dataset.")
                print("Differences:")
                for row in different_params:
                    key, disk_value, current_value = row
                    print(f"\t{key} = {current_value} ({disk_value} found)")
                print("If you want to overwrite dataset on disk, set overwrite = True for the DatasetCreator.")
                print("Not regenerating. Using dataset on disk.")
                return
    
            # generate info yamls
            write_dict_to_yaml(self.dataset_info_filepath, self.dataloader.dataset.dataset_metadata.to_dict())
            write_dict_to_yaml(self.writer_info_filepath, self.get_writing_info_dict())
    
            # store start time
            self._msg_timer = time()

            with tqdm(total=self.num_batches, desc=self.tqdm_desc) as pbar:
                self.pbar = pbar  # Make the instance accessible to helper methods
    
                # write dataset
                if self.multithreading:
                    # write each batch as its own thread
                    # num_threads defaults to: min(32, os.cpu_count() + 4)
                    with concurrent.futures.ThreadPoolExecutor() as executor:
        
                        # Process batches in chunks to avoid memory buildup
                        batch_chunk_size = max(1, min(100, self.num_batches) // 10) # Process in smaller chunks
                        batch_iter = enumerate(self.dataloader)
                        processed_batches = 0
        
                        # Process in chunks to manage memory
                        while processed_batches < self.num_batches:
                            # Get next chunk of batches
                            chunk_futures = []
                            chunk_size = 0
        
                            for _ in range(min(batch_chunk_size, self.num_batches - processed_batches)):
                                try:
                                    batch_idx, batch = next(batch_iter)
                                    batch = batch_as_signal_list(batch, self.dataloader.dataset.target_labels, self.dataloader.dataset.dataset_metadata)
        
                                    if batch_idx == self.num_batches - 1 and not np.equal(self.dataset_length % self.batch_size,0):
                                        batch = batch[:self.dataset_length%self.batch_size]
        
                                    future = executor.submit(self._write_batch, writer, batch_idx, batch)
                                    chunk_futures.append(future)
                                    chunk_size += 1
                                except StopIteration:
                                    break
        
                            # Only process if we have futures to process
                            if chunk_futures:
                                # Wait for chunk to complete before processing next chunk
                                concurrent.futures.wait(chunk_futures)
        
                                # Clear references to help garbage collection
                                for future in chunk_futures:
                                    future.result()  # Ensure completion
                                del chunk_futures
        
                                processed_batches += chunk_size

                                # update progress bar message
                                self.pbar.update(chunk_size)
                                self._update_tqdm_message(processed_batches)
        
                                # Force garbage collection between chunks
                                import gc
                                gc.collect()
        
                            else:
                                # No more batches to process
                                break
        
                else:
                    # single threaded writing
                    itr = iter(self.dataloader)
        
                    # for batch_idx in tqdm(range(self.num_batches), total = self.num_batches):
                    for batch_idx in range(self.num_batches):
                        batch = next(itr)
                        batch = batch_as_signal_list(batch, self.dataloader.dataset.target_labels, self.dataloader.dataset.dataset_metadata)
                        
                        if batch_idx == self.num_batches - 1 and not np.equal(self.dataset_length % self.batch_size,0):
                            batch = batch[:self.dataset_length%self.batch_size]

                        try:
                            # write to disk
                            self._write_batch(writer,batch_idx,batch)
        
                            # update tqdm
                            self.pbar.update(1)
                            self._update_tqdm_message(batch_idx + 1)
        
                        finally:
                            # Clear batch reference to help garbage collection
                            del batch
        
                            # Force garbage collection every 10 batches
                            if np.equal(batch_idx % 10,0):
                                import gc
                                gc.collect()
            # update writer yaml
            # indicate writing dataset to disk was successful
            updated_writer_yaml = self.get_writing_info_dict()
            updated_writer_yaml['complete'] = True
            write_dict_to_yaml(self.writer_info_filepath, updated_writer_yaml)

    # def _update_tqdm_message(self, batch_idx:int ):
    def _update_tqdm_message(self, num_batches_processed: int) -> None:

        """Updates the tqdm progress bar with remaining disk space (thread safe)

        Informs the user how much remaining space left (in gigabytes) is
        on their disk. Includes a check to stop writing to disk in case
        the disk is at risk of being completely filled.
   
        Raises:
            ValueError: If the disk space remaining is below a threshold
        """

        # Don't run if no batches are done or pbar isn't ready
        if num_batches_processed == 0 or not hasattr(self, 'pbar') or self.pbar is None:
            return

        with self._tqdm_lock:

            # compute elapsed time since last run
            elapsed_time = time() - self._msg_timer

            # run every second, but wait until 20 iterations have
            # passed in order to create a more realiable estimate
            if self._msg_timer == 0 or elapsed_time > 1:

                num_samples_written = num_batches_processed * self.batch_size
                if num_samples_written == 0:
                    return # Avoid division by zero

                # get the amount of disk space remaining
                disk_size_available_bytes = disk_usage(self.root)[2]
                # convert to GB and round to two decimal places
                disk_size_available_gigabytes = np.round(disk_size_available_bytes/(1000**3),2)
                # get size of dataset written so far
                dataset_size_current_gigabytes = self._get_directory_size_gigabytes(self.root)
                # num samples processed and remaining
                # num_samples_written = (batch_idx+1)*self.batch_size
                num_samples_remaining = self.dataset_length - num_samples_written
                # estimate size per sample
                dataset_size_per_sample_gigabytes = dataset_size_current_gigabytes/num_samples_written
                # predict estimated size
                dataset_size_remaining_gigabytes = np.round(dataset_size_per_sample_gigabytes*num_samples_remaining,2)
                # estimate total dataset size
                dataset_size_total_gigabytes = np.round(dataset_size_per_sample_gigabytes*self.dataset_length,2)

                # concatenate disk size for progress bar message
                # updated_tqdm_desc = f'{self.tqdm_desc} estimated dataset size = {dataset_size_total_gigabytes} GB, dataset remaining = {dataset_size_remaining_gigabytes} GB, remaining disk = {disk_size_available_gigabytes} GB'

                desc = (f'{self.tqdm_desc} | Est. Size: {dataset_size_total_gigabytes} GB | Disk Free: {disk_size_available_gigabytes} GB')
                self.pbar.set_description(desc)

                # avoid crashing by stopping write process
                if disk_size_available_gigabytes < self.minimum_remaining_disk_gigabytes:
                    # remaining disk size is below a hard cutoff value to avoid crashing operating system
                    raise ValueError(f'Disk nearly full! Remaining space is {disk_size_available_gigabytes} GB. Please make space before continuing.')
                if dataset_size_remaining_gigabytes > disk_size_available_gigabytes:
                    # projected size of dataset too large for available disk space
                    raise ValueError(f'Not enough disk space. Projected dataset size is {dataset_size_remaining_gigabytes} GB. Remaining space is {disk_size_available_gigabytes} GB. Please reduce dataset size or make space before continuing.')

                # set the progress bar message
                # self.pbar.set_description(updated_tqdm_desc)
                self._msg_timer = time()


    def _get_directory_size_gigabytes ( self, start_path ):
        """
        Returns total size of a directory (including subdirs) in gigabytes
        """
        total_size = 0
        for path, _, files in os.walk(start_path):
            for f in files:
                fp = os.path.join(path, f)
                #total_size += os.path.getsize(fp)
                try:
                    total_size += os.path.getsize(fp)
                except (OSError, FileNotFoundError):
                    # file might have been deleted/moved by another thread
                    # skip it and continue
                    continue
        
        total_size_gb = total_size/(1000**3)
        return total_size_gb
