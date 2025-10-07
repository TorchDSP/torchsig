"""PyTorch Lightning DataModules
Learn More: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
If dataset does not exist at root, creates new dataset and writes to disk
If dataset does exist, simply loaded it back in
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.datasets.dataset_utils import to_dataset_metadata
from torchsig.utils.writer import DatasetCreator
from torchsig.utils.file_handlers import BaseFileHandler
from torchsig.utils.file_handlers.hdf5 import HDF5Writer as DEFAULT_WRITER, HDF5Reader as DEFAULT_READER
from torchsig.transforms.base_transforms import Transform
from torchsig.transforms.metadata_transforms import MetadataTransform
from torchsig.transforms.impairments import Impairments
from torchsig.datasets.default_configs.loader import get_default_yaml_config

# Third Party
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# Built-In
from typing import Callable, List
from pathlib import Path

class TorchSigDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for creating and loading TorchSig datasets.

    This DataModule handles:
      - Dataset creation or loading from disk via a file handler.
      - Splitting into train/val/test subsets.
      - Batching, collation, and worker seeding for training.

    Attributes:
        root (Path): Directory where datasets are stored or created.
        dataset_size (int): Total number of samples in the dataset.
        dataset_splits (List[float] | List[int]): Fractions or counts for train/val/test splits.
        dataset_metadata (DatasetMetadata): Metadata describing the dataset.
        impairment_level (int | None): Optional interference level for synthetic impairments.
        transforms (List[Transform]): Transforms applied to the input data.
        component_transforms (List[Transform]): Transforms applied to individual signal components.
        target_labels (List[str] | None): Names of target metadata fields to include.
        batch_size (int): Batch size for the training/validation/testing DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        collate_fn (Callable | None): Custom collate function for batching.
        create_batch_size (int): Batch size used during on-disk dataset creation.
        create_num_workers (int): Number of workers used during dataset creation.
        file_writer (Type[BaseFileHandler]): FileHandler class for disk I/O.
        file_reader (Type[BaseFileHandler]): FileReader class for disk I/O.
        overwrite (bool): If True, existing on-disk data will be overwritten.
        seed (int | None): Optional random seed for reproducibility.
        train (StaticTorchSigDataset): Initialized training dataset (set in `setup()`).
        val (StaticTorchSigDataset): Initialized validation dataset (set in `setup()`).
        test (StaticTorchSigDataset): Initialized test dataset (set in `setup()`).
    """
    def __init__(
        self,
        root: str,
        dataset_metadata: DatasetMetadata | str | dict,
        dataset_size: int,
        dataset_splits: List[float] | List[int] = [0.70, 0.20, 0.10],
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,

        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: BaseFileHandler = DEFAULT_WRITER,
        file_reader: BaseFileHandler = DEFAULT_READER,
        overwrite: bool = False,
        # tranforms
        impairment_level: int = None,
        transforms: List[Transform] = [],
        component_transforms: List[Transform] = [],
        target_labels: List[str] = None,
        seed: int = None,
    ):
        """Initialize the TorchSigDataModule.

        Args:
            root (str): Path to store or load the dataset.
            dataset_metadata (DatasetMetadata | str | dict): Metadata object, YAML file path, or dict describing classes and settings.
            dataset_size (int): Total number of samples to generate or load.
            dataset_splits (List[float] | List[int], optional): Fractions or counts for train/val/test splits. Defaults to [0.70, 0.20, 0.10].
            batch_size (int, optional): Batch size for data loaders. Defaults to 1.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 1.
            collate_fn (Callable, optional): Custom function to collate batch samples. Defaults to None.
            create_batch_size (int, optional): Batch size when writing data to disk. Defaults to 8.
            create_num_workers (int, optional): Workers used when creating the on-disk dataset. Defaults to 4.
            file_writer (Type[BaseFileHandler]): FileWriter class for disk I/O.
            file_reader (Type[BaseFileHandler]): FileReader class for disk I/O.
            overwrite (bool, optional): If True, existing data at `root` will be overwritten. Defaults to False.
            impairment_level (int, optional): Level of synthetic impairment to apply. Defaults to None (no impairment).
            transforms (List[Transform], optional): List of transforms applied to each sample’s input. Defaults to [].
            component_transforms (List[Transform], optional): Transforms applied to individual signal components. Defaults to [].
            target_labels (List[str], optional): Names of metadata fields to include. Defaults to None.
            seed (int, optional): Seed for randomness and reproducibility. Defaults to None.
        """
        # read from yaml or dataset metadata or code inputs
        super().__init__()
        # filepaths
        self.root = Path(root)
        self.dataset_size = dataset_size
        self.dataset_splits = dataset_splits
        # metadatas
        self.dataset_metadata = to_dataset_metadata(dataset_metadata)
        # transforms
        self.impairment_level = impairment_level
        self.transforms = transforms
        self.component_transforms = component_transforms
        if self.impairment_level is not None:
            # add impairment transforms
            impairment_transforms = Impairments(level=self.impairment_level)
            self.transforms = impairment_transforms.dataset_transforms.transforms + self.transforms
            self.component_transforms = impairment_transforms.signal_transforms.transforms + self.component_transforms
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
        """
        Prepares the dataset by creating new datasets if they do not exist on disk.
        The datasets are created using the `DatasetCreator` class.
        If the dataset already exists on disk, it is loaded back into memory.
        """
        dataset = TorchSigIterableDataset(
            dataset_metadata = self.dataset_metadata,
            transforms = self.transforms,
            seed=self.seed
        )
        loader = DataLoader(
            dataset = dataset,
            batch_size = self.create_batch_size,
            num_workers = self.create_num_workers,
            collate_fn = self.collate_fn
        )
        creator = DatasetCreator(
            dataloader = loader,
            dataset_length = self.dataset_size,
            root = self.root,
            overwrite = self.overwrite,
            file_writer = self.file_writer,
        )
        # breakpoint()
        print(f"Full Dataset: Impairment Level {self.impairment_level}, {self.dataset_size} dataset size")
        creator.create()

    def setup(self, stage: str = 'train') -> None:
        """
        Sets up the train and validation datasets for the given stage.
        Args:
            stage (str, optional): The stage of the DataModule, typically 'train' or 'test'. Defaults to 'train'.
        """
        full_dataset = StaticTorchSigDataset(
            root = self.root,
            file_handler_class=self.file_reader,
            target_labels=self.target_labels
        )
        self.train, self.val, self.test = random_split(full_dataset, self.dataset_splits)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.
        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        return DataLoader(
            dataset = self.train,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn
        )
    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.
        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset = self.val,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.
        Returns:
            DataLoader: A PyTorch DataLoader for the test dataset.
        """
        return DataLoader(
            dataset = self.test,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn
        )