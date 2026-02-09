"""PyTorch Lightning DataModules
Learn More: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
If dataset does not exist at root, creates new dataset and writes to disk
If dataset does exist, simply loaded it back in
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# Built-In
# Third Party
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# TorchSig
from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.writer import DatasetCreator

if TYPE_CHECKING:
    from torchsig.utils.file_handlers import BaseFileHandler

class TorchSigDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for creating and loading TorchSig datasets.

    This DataModule handles:
      - Dataset creation or loading from disk via a file handler.
      - Splitting into train/val/test subsets.
      - Batching, collation, and worker seeding for training.

    Attributes:
        root: Directory where datasets are stored or created.
        dataset_size: Total number of samples in the dataset.
        dataset_splits: Fractions or counts for train/val/test splits.
        dataset_metadata: Metadata describing the dataset.
        impairment_level: Optional interference level for synthetic impairments.
        transforms: Transforms applied to the input data.
        target_labels: Names of target metadata fields to include.
        batch_size: Batch size for the training/validation/testing DataLoaders.
        num_workers: Number of worker processes for data loading.
        collate_fn: Custom collate function for batching.
        create_batch_size: Batch size used during on-disk dataset creation.
        create_num_workers: Number of workers used during dataset creation.
        file_writer: FileHandler class for disk I/O.
        file_reader: FileReader class for disk I/O.
        overwrite: If True, existing on-disk data will be overwritten.
        seed: Optional random seed for reproducibility.
        train: Initialized training dataset (set in `setup()`).
        val: Initialized validation dataset (set in `setup()`).
        test: Initialized test dataset (set in `setup()`).
    """

    def __init__(
        self,
        root: str,
        metadata,
        dataset_size: int,
        dataset_splits: list[float] | list[int] = [0.70, 0.20, 0.10],
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: callable | None = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: BaseFileHandler = HDF5Writer,
        file_reader: BaseFileHandler = HDF5Reader,
        overwrite: bool = False,
        # transforms
        impairment_level: int | None = None,
        transforms=[],
        target_labels: list[str] | None = None,
        seed: int | None = None,
    ):
        """Initialize the TorchSigDataModule.

        Args:
            root: Path to store or load the dataset.
            metadata: Metadata object, YAML file path, or dict describing classes and settings.
            dataset_size: Total number of samples to generate or load.
            dataset_splits: Fractions or counts for train/val/test splits. Defaults to [0.70, 0.20, 0.10].
            batch_size: Batch size for data loaders. Defaults to 1.
            num_workers: Number of worker processes for data loading. Defaults to 1.
            collate_fn: Custom function to collate batch samples. Defaults to None.
            create_batch_size: Batch size when writing data to disk. Defaults to 8.
            create_num_workers: Workers used when creating the on-disk dataset. Defaults to 4.
            file_writer: FileWriter class for disk I/O.
            file_reader: FileReader class for disk I/O.
            overwrite: If True, existing data at `root` will be overwritten. Defaults to False.
            impairment_level: Level of synthetic impairment to apply. Defaults to None (no impairment).
            transforms: List of transforms applied to each sample's input. Defaults to [].
            target_labels: Names of metadata fields to include. Defaults to None.
            seed: Seed for randomness and reproducibility. Defaults to None.

        Raises:
            ValueError: If dataset_splits don't sum to 1.0 (when using fractions).
            FileNotFoundError: If metadata file path is invalid.
        """
        # read from yaml or dataset metadata or code inputs
        super().__init__()
        # filepaths
        self.root = Path(root)
        self.dataset_size = dataset_size
        self.dataset_splits = dataset_splits
        # metadatas
        self.metadata = metadata
        # transforms
        self.impairment_level = impairment_level
        self.transforms = transforms
        if self.impairment_level is not None:
            # add impairment transforms
            impairment_transforms = Impairments(level=self.impairment_level)
            self.transforms = (
                impairment_transforms.dataset_transforms.transforms + self.transforms
            )
        self.target_labels = target_labels
        # initialize dataloader params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        # dataset creator params
        self.create_batch_size = create_batch_size
        self.create_num_workers = create_num_workers
        self.file_writer = file_writer
        self.file_reader = file_reader
        self.overwrite = overwrite

        # to be initialized in setup()
        self.train: StaticTorchSigDataset = None
        self.val: StaticTorchSigDataset = None
        self.test: StaticTorchSigDataset = None
        self.seed = seed

    def prepare_data(self) -> None:
        """Prepares the dataset by creating new datasets if they do not exist on disk.

        The datasets are created using the `DatasetCreator` class.
        If the dataset already exists on disk, it is loaded back into memory.

        Raises:
            FileNotFoundError: If the root directory cannot be created.
            RuntimeError: If dataset creation fails.
        """
        dataset = TorchSigIterableDataset(
            metadata=self.metadata, transforms=self.transforms, seed=self.seed
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.create_batch_size,
            collate_fn=self.collate_fn,
        )
        creator = DatasetCreator(
            dataloader=loader,
            dataset_length=self.dataset_size,
            root=self.root,
            overwrite=self.overwrite,
            file_writer=self.file_writer,
        )
        print(f"Full Dataset: Impairment Level {self.impairment_level}, {self.dataset_size} dataset size")
        creator.create()

    def setup(self, stage: str = "train") -> None:
        """Sets up the train and validation datasets for the given stage.

        Args:
            stage: The stage of the DataModule, typically 'train' or 'test'. Defaults to 'train'.

        Raises:
            FileNotFoundError: If the dataset files are not found at the specified root.
            ValueError: If dataset splits are invalid.
        """
        full_dataset = StaticTorchSigDataset(
            root=self.root,
            file_handler_class=self.file_reader,
            target_labels=self.target_labels,
        )
        self.train, self.val, self.test = random_split(
            full_dataset, self.dataset_splits
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            A PyTorch DataLoader for the training dataset.

        Raises:
            RuntimeError: If the training dataset is not initialized.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            A PyTorch DataLoader for the validation dataset.

        Raises:
            RuntimeError: If the validation dataset is not initialized.
        """
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset.

        Returns:
            A PyTorch DataLoader for the test dataset.

        Raises:
            RuntimeError: If the test dataset is not initialized.
        """
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
