"""Dataset Writer Utils"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from pathlib import Path
from shutil import disk_usage
from time import time
import warnings

# Built-In
from typing import Any, TYPE_CHECKING

import numpy as np
from torch.utils.data._utils.collate import default_collate as torch_default_collate

# Third Party
from tqdm.auto import tqdm

# TorchSig
from torchsig.utils.file_handlers.hdf5 import HDF5Writer
from torchsig.utils.yaml import write_dict_to_yaml

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torchsig.utils.file_handlers.base_handler import FileWriter


def default_collate_fn(batch):
    """Collates a batch by zipping its elements together.

    Args:
        batch (tuple): A batch from the dataloader.

    Returns:
        tuple: A tuple of zipped elements, where each element corresponds to a single batch item.
    """
    return tuple(zip(*batch))


class DatasetCreator:
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
        file_handler (FileWriter): The file handler used for saving the dataset.
    """

    def __init__(
        self,
        dataloader: DataLoader = None,
        dataset_length: int | None = None,
        root: str = ".",
        overwrite: bool = True,  # will overwrite any existing dataset on disk
        tqdm_desc: str | None = None,
        file_handler: FileWriter = HDF5Writer,
        multithreading: bool = True,
        **kwargs,  # any additional file handler args
    ):
        """Initializes the DatasetCreator.

        Args:
            dataloader (DataLoader): The DataLoader used to load data in batches.
            dataset_length (int): The number of samples to draw from a dataset.
            root (Path): The root directory where the dataset will be saved.
            overwrite (bool): Flag indicating whether to overwrite an existing dataset.
            tqdm_desc (str): A description for the progress bar.
            file_handler (FileWriter): The file handler used for saving the dataset.
        """
        self.root = Path(root)
        self.dataset_info_filepath = self.root.joinpath("dataset_info.yaml")
        self.writer_info_filepath = self.root.joinpath("writer_info.yaml")
        self.dataset_length = dataset_length
        self.overwrite = overwrite
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers
        self.multithreading = multithreading
        self.num_batches = self.dataset_length // self.batch_size
        if not np.equal(self.dataset_length % self.batch_size, 0):
            self.num_batches += (
                1  # include the partial batch at the end if it can't be evenly batched
            )

        self.dataloader = dataloader
        if (
            self.dataloader.dataset.target_labels is None
            and self.dataloader.collate_fn == torch_default_collate
        ):
            # DataLoader should just return Signal objects
            # do not use torch's default collate function
            self.dataloader.collate_fn = lambda x: x
        self.file_handler = file_handler

        # get reference to tqdm progress bar object
        self.pbar = tqdm()

        self.tqdm_desc = "Generating Dataset:" if tqdm_desc is None else tqdm_desc

        # limit in gigabytes for remaining space on disk for which writer stops writing
        self.minimum_remaining_disk_gigabytes = 1

        # Thread lock for updating tqdm message to avoid race conditions
        self._tqdm_lock = threading.Lock()

        self._msg_timer = None

    def get_writing_info_dict(self) -> dict[str, Any]:
        """Returns a dictionary with information about the dataset being written.

        This method gathers information regarding the root, overwrite status,
        batch size, number of workers, file handler class, and the save type
        of the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing the dataset writing configuration.
        """
        return {
            "root": str(self.root),
            "overwrite": self.overwrite,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "complete": False,
        }

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
        temp_labels = self.dataloader.dataset.target_labels
        self.dataloader.dataset.target_labels = None
        with self.file_handler(root=self.root) as writer:
            if writer.exists() and not self.overwrite:
                complete, different_params = self.check_yamls()
                if np.equal(len(different_params), 0) and complete:
                    print(f"Dataset already exists in {self.root}. Not regenerating.")
                    return

                if not complete:
                    # dataset on disk is corrupted
                    # dataset was not fully written to disk
                    raise RuntimeError(
                        f"Dataset only partially exists in {self.root} (writing dataset to disk was cancelled early). Regenerate the dataset by setting overwrite = True for DatasetCreator"
                    )
                # dataset exists on disk with different params
                # use dataset on disk instead
                # warn users that params are different
                print(
                    f"Dataset exists at {self.root} but is different than current dataset."
                )
                print("Differences:")
                for row in different_params:
                    key, disk_value, current_value = row
                    print(f"\t{key} = {current_value} ({disk_value} found)")
                print(
                    "If you want to overwrite dataset on disk, set overwrite = True for the DatasetCreator."
                )
                print("Not regenerating. Using dataset on disk.")
                return

            # generate info yamls
            write_dict_to_yaml(self.writer_info_filepath, self.get_writing_info_dict())

            # store start time
            self._msg_timer = time()

            # write dataset
            if self.multithreading:
                # write each batch as its own thread
                # num_threads defaults to: min(32, os.cpu_count() + 4)
                with concurrent.futures.ThreadPoolExecutor() as executor:

                    # Process batches in chunks to avoid memory buildup
                    batch_chunk_size = max(
                        1, min(100, self.num_batches) // 10
                    )  # Process in smaller chunks

                    batch_iter = enumerate(self.dataloader)
                    processed_batches = 0
                    total_batches = self.num_batches

                    # Process in chunks to manage memory
                    while processed_batches < total_batches:
                        # Get next chunk of batches
                        chunk_futures = []
                        chunk_size = 0

                        for _ in range(
                            min(batch_chunk_size, total_batches - processed_batches)
                        ):
                            try:
                                batch_idx, batch = next(batch_iter)

                                if batch_idx == self.num_batches - 1 and not np.equal(
                                    self.dataset_length % self.batch_size, 0
                                ):
                                    batch = batch[
                                        : self.dataset_length % self.batch_size
                                    ]

                                future = executor.submit(
                                    self._write_batch, writer, batch_idx, batch
                                )
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

                            # Force garbage collection between chunks
                            import gc

                            gc.collect()

                        else:
                            # No more batches to process
                            break

            else:
                # single threaded writing
                itr = iter(self.dataloader)

                for batch_idx in tqdm(range(self.num_batches), total=self.num_batches):
                    batch = next(itr)

                    if batch_idx == self.num_batches - 1 and not np.equal(
                        self.dataset_length % self.batch_size, 0
                    ):
                        batch = batch[: self.dataset_length % self.batch_size]

                    try:
                        # write to disk
                        self._write_batch(writer, batch_idx, batch)

                        # update progress bar message
                        self._update_tqdm_message(batch_idx)

                    finally:
                        # Clear batch reference to help garbage collection
                        del batch

                        # Force garbage collection every 10 batches
                        if np.equal(batch_idx % 10, 0):
                            import gc

                            gc.collect()
            # update writer yaml
            # indicate writing dataset to disk was successful
            updated_writer_yaml = self.get_writing_info_dict()
            updated_writer_yaml["complete"] = True
            write_dict_to_yaml(self.writer_info_filepath, updated_writer_yaml)
            self.dataloader.dataset.target_labels = temp_labels

    def _update_tqdm_message(self, batch_idx: int):
        """Updates the tqdm progress bar with remaining disk space

        Informs the user how much remaining space left (in gigabytes) is
        on their disk. Includes a check to stop writing to disk in case
        the disk is at risk of being completely filled.

        Raises:
            ValueError: If the disk space remaining is below a threshold
        """
        with self._tqdm_lock:

            # compute elapsed time since last run
            elapsed_time = time() - self._msg_timer

            # run every second, but wait until 20 iterations have
            # passed in order to create a more realiable estimate
            if not batch_idx or elapsed_time > 1:

                # get the amount of disk space remaining
                disk_size_available_bytes = disk_usage(self.root)[2]
                # convert to GB and round to two decimal places
                disk_size_available_gigabytes = np.round(
                    disk_size_available_bytes / (1000**3), 2
                )
                # get size of dataset written so far
                dataset_size_current_gigabytes = self._get_directory_size_gigabytes(self.root)
                # num samples processed and remaining
                num_samples_written = (batch_idx + 1) * self.batch_size
                num_samples_remaining = self.dataset_length - num_samples_written
                # estimate size per sample
                dataset_size_per_sample_gigabytes = (
                    dataset_size_current_gigabytes / num_samples_written
                )
                # predict estimated size
                dataset_size_remaining_gigabytes = np.round(
                    dataset_size_per_sample_gigabytes * num_samples_remaining, 2
                )
                # estimate total dataset size
                dataset_size_total_gigabytes = np.round(
                    dataset_size_per_sample_gigabytes * self.dataset_length, 2
                )

                # concatenate disk size for progress bar message
                updated_tqdm_desc = f"{self.tqdm_desc} estimated dataset size = {dataset_size_total_gigabytes} GB, dataset remaining = {dataset_size_remaining_gigabytes} GB, remaining disk = {disk_size_available_gigabytes} GB"

                # avoid crashing by stopping write process
                if (
                    disk_size_available_gigabytes
                    < self.minimum_remaining_disk_gigabytes
                ):
                    # remaining disk size is below a hard cutoff value to avoid crashing operating system
                    raise ValueError(
                        f"Disk nearly full! Remaining space is {disk_size_available_gigabytes} GB. Please make space before continuing."
                    )
                if dataset_size_remaining_gigabytes > disk_size_available_gigabytes:
                    # projected size of dataset too large for available disk space
                    raise ValueError(
                        f"Not enough disk space. Projected dataset size is {dataset_size_remaining_gigabytes} GB. Remaining space is {disk_size_available_gigabytes} GB. Please reduce dataset size or make space before continuing."
                    )

                # set the progress bar message
                self.pbar.set_description(updated_tqdm_desc)

    def _get_directory_size_gigabytes(self, start_path: str | Path) -> float:
        """Calculate the total size of a directory (including subdirectories) in gigabytes.

        This function recursively walks through all files in the specified directory
        and its subdirectories, summing their sizes. Files that cannot be accessed
        (due to permissions, deletion, etc.) are skipped with a warning.

        Args:
            start_path: Path to the directory to calculate size for. Can be either
                a string or Path object.

        Returns:
            Total size of the directory in gigabytes as a float.

        Raises:
            NotADirectoryError: If the provided path is not a directory.
            FileNotFoundError: If the provided path doesn't exist.
        """
        total_size = 0
        start_path = Path(start_path)

        # Validate the path
        if not start_path.exists():
            raise FileNotFoundError(f"Path does not exist: {start_path}")
        if not start_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {start_path}")

        for path, _, files in os.walk(start_path):
            for f in files:
                fp = Path(path) / f
                try:
                    total_size += fp.stat().st_size
                except (OSError, FileNotFoundError) as e:
                    # Skip files that can't be accessed
                    warnings.warn(
                        f"Skipping file {fp} due to error: {e}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    continue

        return total_size / (1000**3)
