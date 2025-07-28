
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
    """
    Data module for loading and processing SigMF data.
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
        target_snr_db: float = 10.0,  # only for narrowband extraction
    ):
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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.train: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.val: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.test: Optional[CustomSigmfStaticTorchSigDataset] = None
        self.label_mapping: Optional[Dict[int, str]] = None

    def prepare_data(self) -> None:

        SigMFDatasetConverter(
            root=self.root,
            dataset=self.dataset,
            overwrite=self.overwrite,
            fft_size=self.fft_size,
            num_iq_samples=self.num_iq_samples,
            target_snr_db=self.target_snr_db,
        ).convert()

    def setup(self, stage: str = None) -> None:
        self.train = CustomSigmfStaticTorchSigDataset(
            root=str(self.root / "torchsig"),
            dataset_type=self.dataset,
            transforms=self.transforms,
            target_transforms=self.target_transforms,
            file_handler_class=self.file_handler,
        )

        self.label_mapping = self._load_label_mapping()

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def _load_label_mapping(self) -> Dict[int, str]:
        """Load label mapping from the dataset YAML file."""
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
