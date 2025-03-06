"""PyTorch Lightning DataModules for Narrowband and Wideband

Learn More: https://lightning.ai/docs/pytorch/stable/data/datamodule.html

If dataset does not exist at root, creates new dataset and writes to disk
If dataset does exsit, simply loaded it back in
"""

from __future__ import annotations

# TorchSig
from torch.utils.data import DataLoader
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from torchsig.datasets.narrowband import NewNarrowband, StaticNarrowband
from torchsig.datasets.wideband import NewWideband, StaticWideband
from torchsig.datasets.dataset_utils import to_dataset_metadata
from torchsig.utils.writer import DatasetCreator
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.transforms.base_transforms import Transform
from torchsig.transforms.target_transforms import TargetTransform
from torchsig.datasets.default_configs.loader import get_default_yaml_config

# Third Party
import pytorch_lightning as pl

# Built-In
from typing import Callable, List
from pathlib import Path


class TorchSigDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for managing TorchSig datasets.

    Args:
        root (str): The root directory where datasets are stored or created.
        dataset (str): The name of the dataset (either 'narrowband' or 'wideband').
        train_metadata (DatasetMetadata | str | dict): Metadata for the training dataset.
        val_metadata (DatasetMetadata | str | dict): Metadata for the validation dataset.
        batch_size (int, optional): The batch size for data loading. Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.
        collate_fn (Callable, optional): A function to collate data into batches.
        create_batch_size (int, optional): The batch size used during dataset creation. Defaults to 8.
        create_num_workers (int, optional): The number of workers used during dataset creation. Defaults to 4.
        file_handler (TorchSigFileHandler, optional): The file handler for managing data storage. Defaults to ZarrFileHandler.
        transforms (list, optional): A list of transformations to apply to the input data. Defaults to an empty list.
        target_transforms (list, optional): A list of transformations to apply to the target labels. Defaults to an empty list.
    """
    def __init__(
        self,
        root: str,
        dataset: str,
        train_metadata: DatasetMetadata | str | dict,
        val_metadata: DatasetMetadata | str | dict,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: list = [],
        target_transforms: list = [],
    ):
        """
        Initializes the TorchSigDataModule with the provided parameters.

        Args:
            root (str): The root directory where datasets are stored or created.
            dataset (str): The name of the dataset (either 'narrowband' or 'wideband').
            train_metadata (DatasetMetadata | str | dict): Metadata for the training dataset.
            val_metadata (DatasetMetadata | str | dict): Metadata for the validation dataset.
            batch_size (int, optional): The batch size for data loading. Defaults to 1.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.
            collate_fn (Callable, optional): A function to collate data into batches.
            create_batch_size (int, optional): The batch size used during dataset creation. Defaults to 8.
            create_num_workers (int, optional): The number of workers used during dataset creation. Defaults to 4.
            file_handler (TorchSigFileHandler, optional): The file handler for managing data storage. Defaults to ZarrFileHandler.
            transforms (list, optional): A list of transformations to apply to the input data. Defaults to an empty list.
            target_transforms (list, optional): A list of transformations to apply to the target labels. Defaults to an empty list.
        """
        # read from yaml or dataset metadata or code inputs
        super().__init__()

        self.root = Path(root)
        self.dataset = dataset
        self.train_metadata = to_dataset_metadata(train_metadata)
        self.val_metadata = to_dataset_metadata(val_metadata)
        self.impaired = self.train_metadata.impairment_level > 0

        self.new_dataset_class = NewNarrowband if self.dataset == "narrowband" else NewWideband
        self.static_dataset_class = StaticNarrowband if self.dataset == "narrowband" else StaticWideband

        self.transforms = transforms
        self.target_transforms = target_transforms


        # initialize dataloader params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        # dataset creator params
        self.create_batch_size = create_batch_size
        self.create_num_workers = create_num_workers
        self.file_handler = file_handler

        # to be initialized in setup()
        self.train: DataLoader = None
        self.val: DataLoader = None
        self.test: DataLoader = None

    def prepare_data(self) -> None:
        """
        Prepares the dataset by creating new datasets if they do not exist on disk.
        The datasets are created using the `DatasetCreator` class.

        If the dataset already exists on disk, it is loaded back into memory.
        """
        # Creates dataset if does not exist on disk or overwrite
        train_dataset = self.new_dataset_class(
            dataset_metadata = self.train_metadata,
        )
        train_creator = DatasetCreator(
            dataset = train_dataset,
            root = self.root,
            overwrite = False,
            file_handler = self.file_handler,
            batch_size = self.create_batch_size,
            num_workers = self.create_num_workers,
            train = True,
        )
        print(f"Train Dataset: {self.dataset.title()}, Impairment Level {self.train_metadata.impairment_level}, {self.train_metadata.num_samples} samples")
        train_creator.create()

        val_dataset = self.new_dataset_class(
            dataset_metadata = self.val_metadata
        )
        val_creator = DatasetCreator(
            dataset = val_dataset,
            root = self.root,
            overwrite = False,
            file_handler = self.file_handler,
            batch_size = self.create_batch_size,
            num_workers = self.create_num_workers,
            train = False,
        )
        print(f"Val Dataset: {self.dataset.title()}, Impairment Level {self.val_metadata.impairment_level}, {self.val_metadata.num_samples} samples")
        val_creator.create()

    def setup(self, stage: str = 'train') -> None:
        """
        Sets up the train and validation datasets for the given stage.

        Args:
            stage (str, optional): The stage of the DataModule, typically 'train' or 'test'. Defaults to 'train'.
        """
        self.train = self.static_dataset_class(
            root = self.root,
            impaired = self.impaired,
            transforms = self.transforms,
            target_transforms = self.target_transforms,
            train = True,
        )
        self.val = self.static_dataset_class(
            root = self.root,
            impaired = self.impaired,
            transforms = self.transforms,
            target_transforms = self.target_transforms,
            train = False
        )

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


### DataModules for creating custom datasets

class NarrowbandDataModule(TorchSigDataModule):
    """
    DataModule for creating and managing narrowband datasets.

    Args:
        root (str): The root directory where datasets are stored or created.
        dataset_metadata (NarrowbandMetadata | str | dict): Metadata for the narrowband dataset.
        num_samples_train (int): The number of training samples.
        num_samples_val (int, optional): The number of validation samples. Defaults to 10% of training samples if not provided.
        batch_size (int, optional): The batch size for data loading. Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.
        collate_fn (Callable, optional): A function to collate data into batches.
        create_batch_size (int, optional): The batch size used during dataset creation. Defaults to 8.
        create_num_workers (int, optional): The number of workers used during dataset creation. Defaults to 4.
        file_handler (TorchSigFileHandler, optional): The file handler for managing data storage. Defaults to ZarrFileHandler.
        transforms (Transform | List[Callable | Transform], optional): A list of transformations to apply to the input data.
        target_transforms (TargetTransform | List[Callable | TargetTransform], optional): A list of transformations to apply to the target labels.
    """
    def __init__(
        self,
        root: str,
        # dataset params
        dataset_metadata: NarrowbandMetadata | str | dict,
        num_samples_train: int,
        num_samples_val: int = None,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: Transform | List[Callable | Transform] = [],
        target_transforms: TargetTransform | List[Callable | TargetTransform] = [],
    ):

        num_samples_val = int(0.10 * num_samples_train) if num_samples_val is None else num_samples_val

        base = to_dataset_metadata(dataset_metadata)

        train_metadata = NarrowbandMetadata(
            num_iq_samples_dataset = base.num_iq_samples_dataset,
            fft_size = base.fft_size,
            impairment_level = base.impairment_level,
            sample_rate = base.sample_rate,
            num_signals_min = base.num_signals_min,
            num_signals_distribution = base.num_signals_distribution,
            snr_db_min = base.snr_db_min,
            snr_db_max = base.snr_db_max,
            signal_duration_percent_min = base.signal_duration_percent_min,
            signal_duration_percent_max = base.signal_duration_percent_max,
            signal_bandwidth_min = base.signal_bandwidth_min,
            signal_bandwidth_max = base.signal_bandwidth_max,
            signal_center_freq_min = base.signal_center_freq_min,
            signal_center_freq_max = base.signal_center_freq_max,
            transforms = base.transforms,
            target_transforms = base.target_transforms,
            class_list = base.class_list,
            class_distribution = base.class_distribution,
            num_samples = num_samples_train,
        )

        val_metadata = NarrowbandMetadata(
            num_iq_samples_dataset = base.num_iq_samples_dataset,
            fft_size = base.fft_size,
            impairment_level = base.impairment_level,
            sample_rate = base.sample_rate,
            num_signals_min = base.num_signals_min,
            num_signals_distribution = base.num_signals_distribution,
            snr_db_min = base.snr_db_min,
            snr_db_max = base.snr_db_max,
            signal_duration_percent_min = base.signal_duration_percent_min,
            signal_duration_percent_max = base.signal_duration_percent_max,
            signal_bandwidth_min = base.signal_bandwidth_min,
            signal_bandwidth_max = base.signal_bandwidth_max,
            signal_center_freq_min = base.signal_center_freq_min,
            signal_center_freq_max = base.signal_center_freq_max,
            transforms = base.transforms,
            target_transforms = base.target_transforms,
            class_list = base.class_list,
            class_distribution = base.class_distribution,
            num_samples = num_samples_val,
        )

        super().__init__(
            root = root,
            dataset = 'narrowband',
            train_metadata = train_metadata,
            val_metadata = val_metadata,
            batch_size = batch_size,
            num_workers = num_workers,
            collate_fn = collate_fn,
            create_batch_size = create_batch_size,
            create_num_workers = create_num_workers,
            file_handler = file_handler,
            transforms = transforms,
            target_transforms = target_transforms,
        )

class WidebandDataModule(TorchSigDataModule):
    """
    DataModule for creating and managing wideband datasets.

    Args:
        root (str): The root directory where datasets are stored or created.
        dataset_metadata (WidebandMetadata | str | dict): Metadata for the wideband dataset.
        num_samples_train (int): The number of training samples.
        num_samples_val (int, optional): The number of validation samples. Defaults to 10% of training samples if not provided.
        batch_size (int, optional): The batch size for data loading. Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.
        collate_fn (Callable, optional): A function to collate data into batches.
        create_batch_size (int, optional): The batch size used during dataset creation. Defaults to 8.
        create_num_workers (int, optional): The number of workers used during dataset creation. Defaults to 4.
        file_handler (TorchSigFileHandler, optional): The file handler for managing data storage. Defaults to ZarrFileHandler.
        transforms (Transform | List[Callable | Transform], optional): A list of transformations to apply to the input data.
        target_transforms (TargetTransform | List[Callable | TargetTransform], optional): A list of transformations to apply to the target labels.
    """
    def __init__(
        self,
        root: str,
        # dataset params
        dataset_metadata: WidebandMetadata | str | dict,
        num_samples_train: int,
        num_samples_val: int = None,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: Transform | List[Callable | Transform]  = [],
        target_transforms: TargetTransform | List[Callable | TargetTransform] = [],
    ):

        num_samples_val = int(0.10 * num_samples_train) if num_samples_val is None else num_samples_val

        base = to_dataset_metadata(dataset_metadata)

        train_metadata = WidebandMetadata(
            num_iq_samples_dataset = base.num_iq_samples_dataset,
            fft_size = base.num_iq_samples_dataset,
            impairment_level = base.impairment_level,
            num_signals_max = base.num_signals_max,
            sample_rate = base.sample_rate,
            num_signals_min = base.num_signals_min,
            num_signals_distribution = base.num_signals_distribution,
            snr_db_min = base.snr_db_min,
            snr_db_max = base.snr_db_max,
            signal_duration_percent_min = base.signal_duration_percent_min,
            signal_duration_percent_max = base.signal_duration_percent_max,
            signal_bandwidth_min = base.signal_bandwidth_min,
            signal_bandwidth_max = base.signal_bandwidth_max,
            signal_center_freq_min = base.signal_center_freq_min,
            signal_center_freq_max = base.signal_center_freq_max,
            transforms = base.transforms,
            target_transforms = base.target_transforms,
            class_list = base.class_list,
            class_distribution = base.class_distribution,
            num_samples = num_samples_train,
        )

        val_metadata = WidebandMetadata(
            num_iq_samples_dataset = base.num_iq_samples_dataset,
            fft_size = base.num_iq_samples_dataset,
            impairment_level = base.impairment_level,
            num_signals_max = base.num_signals_max,
            sample_rate = base.sample_rate,
            num_signals_min = base.num_signals_min,
            num_signals_distribution = base.num_signals_distribution,
            snr_db_min = base.snr_db_min,
            snr_db_max = base.snr_db_max,
            signal_duration_percent_min = base.signal_duration_percent_min,
            signal_duration_percent_max = base.signal_duration_percent_max,
            signal_bandwidth_min = base.signal_bandwidth_min,
            signal_bandwidth_max = base.signal_bandwidth_max,
            signal_center_freq_min = base.signal_center_freq_min,
            signal_center_freq_max = base.signal_center_freq_max,
            transforms = base.transforms,
            target_transforms = base.target_transforms,
            class_list = base.class_list,
            class_distribution = base.class_distribution,
            num_samples = num_samples_val,
        )

        super().__init__(
            root = root,
            dataset = 'wideband',
            train_metadata = train_metadata,
            val_metadata = val_metadata,
            batch_size = batch_size,
            num_workers = num_workers,
            collate_fn = collate_fn,
            create_batch_size = create_batch_size,
            create_num_workers = create_num_workers,
            file_handler = file_handler,
            transforms = transforms,
            target_transforms = target_transforms,
        )



### DataModules for Official Narrowband and Wideband Datasets
### uses default YAML configs in torchsig/datasets/default_configs

class OfficialTorchSigDataModdule(TorchSigDataModule):
    """
    A PyTorch Lightning DataModule for official TorchSignal datasets.
    
    This class manages the dataset metadata, configuration, and data loading process 
    for datasets with official configurations instead of using custom metadata. 
    It initializes the train and validation metadata based on the dataset type and 
    impairment level.

    Args:
        root (str): Root directory where the dataset is stored.
        dataset (str): Name of the dataset.
        impaired (bool | int): Defines the impairment level of the dataset.
        batch_size (int, optional): Batch size for the dataloaders. Default is 1.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        collate_fn (Callable, optional): Function to merge a list of samples into a batch. Default is None.
        create_batch_size (int, optional): Batch size used during dataset creation. Default is 8.
        create_num_workers (int, optional): Number of workers used during dataset creation. Default is 4.
        file_handler (TorchSigFileHandler, optional): File handler used to read/write dataset. Default is ZarrFileHandler.
        transforms (Transform | List[Callable | Transform], optional): List of transforms applied to dataset. Default is empty list.
        target_transforms (TargetTransform | List[Callable | TargetTransform], optional): List of transforms applied to targets. Default is empty list.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        impaired: bool | int,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: Transform | List[Callable | Transform]  = [],
        target_transforms: TargetTransform | List[Callable | TargetTransform] = [],
    ):
        # sets train and val metadata

        train_metadata = get_default_yaml_config(
            dataset_type = dataset,
            impairment_level = impaired,
            train = True
        )

        val_metadata = get_default_yaml_config(
            dataset_type = dataset,
            impairment_level = impaired,
            train = False
        )

        super().__init__(
            root = root,
            dataset = dataset,
            train_metadata = train_metadata,
            val_metadata = val_metadata,
            batch_size = batch_size,
            num_workers = num_workers,
            collate_fn = collate_fn,
            create_batch_size = create_batch_size,
            create_num_workers = create_num_workers,
            file_handler = file_handler,
            transforms = transforms,
            target_transforms = target_transforms,
        )


class OfficialNarrowbandDataModule(OfficialTorchSigDataModdule):
    """
    A DataModule for the official Narrowband dataset.
    
    This class extends `OfficialTorchSigDataModdule` and sets the dataset type to 'narrowband'.
    It initializes the necessary parameters for the dataset and loads the train and validation metadata 
    accordingly.

    Args:
        root (str): Root directory where the dataset is stored.
        impaired (bool | int): Defines the impairment level of the dataset.
        batch_size (int, optional): Batch size for the dataloaders. Default is 1.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        collate_fn (Callable, optional): Function to merge a list of samples into a batch. Default is None.
        create_batch_size (int, optional): Batch size used during dataset creation. Default is 8.
        create_num_workers (int, optional): Number of workers used during dataset creation. Default is 4.
        file_handler (TorchSigFileHandler, optional): File handler used to read/write dataset. Default is ZarrFileHandler.
        transforms (list, optional): List of transforms applied to dataset. Default is empty list.
        target_transforms (list, optional): List of transforms applied to targets. Default is empty list.
    """
    def __init__(
        self,
        root: str,
        impaired: bool | int,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: list = [],
        target_transforms: list = [],

    ):
        # read from yaml or dataset metadata or code inputs
        super().__init__(
            root = root,
            dataset = 'narrowband',
            impaired = impaired,
            batch_size = batch_size,
            num_workers = num_workers,
            collate_fn = collate_fn,
            create_batch_size = create_batch_size,
            create_num_workers = create_num_workers,
            file_handler = file_handler,
            transforms = transforms,
            target_transforms = target_transforms,
        )


class OfficialWidebandDataModule(OfficialTorchSigDataModdule):
    """
    A DataModule for the official Wideband dataset.
    
    This class extends `OfficialTorchSigDataModdule` and sets the dataset type to 'wideband'.
    It initializes the necessary parameters for the dataset and loads the train and validation metadata 
    accordingly.

    Args:
        root (str): Root directory where the dataset is stored.
        impaired (bool | int): Defines the impairment level of the dataset.
        batch_size (int, optional): Batch size for the dataloaders. Default is 1.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        collate_fn (Callable, optional): Function to merge a list of samples into a batch. Default is None.
        create_batch_size (int, optional): Batch size used during dataset creation. Default is 8.
        create_num_workers (int, optional): Number of workers used during dataset creation. Default is 4.
        file_handler (TorchSigFileHandler, optional): File handler used to read/write dataset. Default is ZarrFileHandler.
        transforms (list, optional): List of transforms applied to dataset. Default is empty list.
        target_transforms (list, optional): List of transforms applied to targets. Default is empty list.
    """
    def __init__(
        self,
        root: str,
        impaired: bool | int,
        # dataloader params
        batch_size: int = 1,
        num_workers: int = 1,
        collate_fn: Callable = None,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        # applied after dataset written to disk
        transforms: list = [],
        target_transforms: list = [],
    ):
        # read from yaml or dataset metadata or code inputs
        super().__init__(
            root = root,
            dataset = 'wideband',
            impaired = impaired,
            batch_size = batch_size,
            num_workers = num_workers,
            collate_fn = collate_fn,
            create_batch_size = create_batch_size,
            create_num_workers = create_num_workers,
            file_handler = file_handler,
            transforms = transforms,
            target_transforms = target_transforms,
        )
