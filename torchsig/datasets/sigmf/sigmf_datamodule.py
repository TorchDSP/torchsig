"""SigMF Data Module for PyTorch Lightning

This module provides a PyTorch Lightning DataModule for loading and processing
SigMF (Signal Metadata Format) datasets, supporting both wideband and narrowband
signal processing workflows.
"""

from pathlib import Path
from typing import Callable, Dict, Literal, Optional
import yaml
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.datasets.sigmf.sigmf_dataset_converter import SigMFDatasetConverter
from torchsig.datasets.sigmf.custom_dataset import CustomSigmfStaticTorchSigDataset
from torchsig.datasets.dataset_utils import dataset_yaml_name


class SigmfDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for SigMF datasets.

    This DataModule handles the conversion of SigMF files to TorchSig format and provides
    PyTorch DataLoaders for training. It supports both wideband (multiple overlapping signals)
    and narrowband (single signal per sample) processing modes.

    Args:
        root (str): Root directory containing SigMF files (.sigmf-meta and .sigmf-data pairs).
        dataset (Literal["narrowband", "wideband"], optional): Dataset processing mode.
            Defaults to "wideband".
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        collate_fn (Callable, optional): Custom collation function for DataLoader. Defaults to None.
        file_handler (TorchSigFileHandler, optional): File handler for dataset storage. 
            Defaults to ZarrFileHandler.
        overwrite (bool, optional): Whether to overwrite existing converted datasets. Defaults to False.
        transforms (list, optional): List of transforms to apply to signal data. Defaults to [].
        target_transforms (list, optional): List of transforms to apply to target annotations. 
            Defaults to [].
        fft_size (int, optional): FFT size for spectrogram computation. Defaults to 512.
        num_iq_samples (int, optional): Number of IQ samples per dataset sample. 
            Defaults to 512^2 (262,144).
        target_snr_db (float, optional): Target signal-to-noise ratio in dB for narrowband 
            extraction. Only used in narrowband mode. Defaults to 10.0.
    """

    def __init__(
        self,
        root: str,
        dataset: Literal["narrowband", "wideband"] = "wideband",
        batch_size: int = 8,
        num_workers: int = 4,
        collate_fn: Callable = None,
        file_handler: TorchSigFileHandler = ZarrFileHandler,
        overwrite: bool = False,
        transforms: list = [],
        target_transforms: list = [],
        fft_size: int = 512,
        num_iq_samples: int = 512 ** 2,
        target_snr_db: float = 10.0,
    ):
        """Initialize the SigMF DataModule.

        Args:
            root: Root directory containing SigMF files
            dataset: Processing mode ("narrowband" or "wideband")
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for data loading
            collate_fn: Custom collation function for DataLoader
            file_handler: File handler for dataset storage
            overwrite: Whether to overwrite existing converted datasets
            transforms: List of transforms to apply to signal data
            target_transforms: List of transforms to apply to target annotations
            fft_size: FFT size for spectrogram computation
            num_iq_samples: Number of IQ samples per dataset sample
            target_snr_db: Target SNR in dB for narrowband extraction
        """
        super().__init__()
        self.root = Path(root)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fft_size = fft_size
        self.num_iq_samples = num_iq_samples
        self.target_snr_db = target_snr_db

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.file_handler = file_handler
        self.overwrite = overwrite
        self.collate_fn = collate_fn

        self.train: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.val: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.test: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.label_mapping: Optional[Dict[int, str]] = None

    def prepare_data(self) -> None:
        """Prepare the dataset by converting SigMF files to TorchSig format.

        This method handles the conversion of SigMF files to the internal TorchSig
        Zarr format. It only runs on the main process in distributed training.
        """
        SigMFDatasetConverter(
            root=self.root,
            dataset=self.dataset,
            overwrite=self.overwrite,
            fft_size=self.fft_size,
            num_iq_samples=self.num_iq_samples,
            target_snr_db=self.target_snr_db,
        ).convert()

    def setup(self, stage: str = None) -> None:
        """Setup datasets for training, validation, and testing.

        This method creates the dataset instances and loads label mappings.
        It runs on every process in distributed training.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None for all).
        """
        self.train = CustomSigmfStaticTorchSigDataset(
            root=str(self.root / "torchsig"),
            dataset_type=self.dataset,
            transforms=self.transforms,
            target_transforms=self.target_transforms,
            file_handler_class=self.file_handler,
        )

        self.label_mapping = self._load_label_mapping()

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the training dataset
                with the specified batch size, shuffling, and number of workers.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def _load_label_mapping(self) -> Dict[int, str]:
        """Load label mapping from the dataset configuration file.

        This method reads the generated YAML configuration file and extracts
        the class list to create a mapping from class indices to class names.

        Returns:
            Dict[int, str]: Mapping from class index to class name.

        Raises:
            FileNotFoundError: If the dataset YAML configuration file is not found.
        """
        yaml_path = self.root / "torchsig" / dataset_yaml_name
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
            # Get class_list from read_only.signals section
            class_list = dataset_info.get('read_only', {}).get(
                'signals', {}).get('class_list', [])
            # Create mapping from index to class name
            label_mapping = {i: class_name for i,
                             class_name in enumerate(class_list)}
            return label_mapping
