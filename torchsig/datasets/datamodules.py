"""PyTorch Lightning DataModules for TorchSigNarrowband and TorchSigWideband
"""
from torch.utils.data import DataLoader
from torch.nn import Identity
import pytorch_lightning as pl
import numpy as np
from typing import Callable, Union, Optional
import os

from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.datasets.torchsig_wideband import TorchSigWideband
from torchsig.datasets import conf
from torchsig.utils.dataset import collate_fn as collate_fn_default
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.writer import DatasetCreator
from torchsig.transforms.target_transforms import DescToClassIndex#, DescToListTuple
from torchsig.transforms.transforms import (
    Compose,
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Spectrogram
)

class TorchSigDataModule(pl.LightningDataModule):
    """General TorchSig DataModule.

        Subclasses must implement `prepare_data`, `setup` functions.
    """

    def __init__(self, 
                 root: str, 
                 dataset: str, 
                 impaired: bool, 
                 qa: bool = True, 
                 eb_no: bool = False, 
                 seed: int = 12345,
                 overlap_prob: Optional[float] = .1,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 collate_fn: Optional[Callable] = None,
                ):             
        """TorchSig DataModule Init

        Args:
            root (str): Dataset root path.
            dataset (str): Dataset name, either "Narrowband" or "Wideband".
            impaired (bool): Dataset impairment setting.
            qa (bool, optional): Generate small dataset sample. Defaults to True.
            eb_no (bool, optional): Use EbNo config. Defaults to False.
            seed (int, optional): Dataset generation seed. Defaults to 12345.
            transform (Optional[Callable], optional): Data transforms. Defaults to None.
            target_transform (Optional[Callable], optional): Label transforms. Defaults to None.
            batch_size (int, optional): Dataloader batch size. Defaults to 4.
            num_workers (int, optional): Dataloader number of workers to use. Defaults to 1.
            collate_fn (Optional[Callable], optional): Dataloader custom collate function. Defaults to TorchSig collate_fn.
        """
        super().__init__()

        self.root = root
        self.dataset = dataset
        self.impaired = impaired
        self.clean = "impaired" if impaired else "clean" #: boolean: `self.impaired` as a string variable.
        self.seed = seed
        self.overlap_prob = overlap_prob

        self.train_config = self._set_config(self.dataset, self.impaired, True, qa, eb_no, self.seed) #: dict: TorshSig train config 
        self.val_config = self._set_config(self.dataset, self.impaired, False, qa, eb_no, self.seed) #: dict: TorshSig val config 

        self.train = None #: Dataset: train dataset
        self.val = None #: Dataset: validation dataset
        

        self.transform = Identity() if transform is None else transform
        
        self.target_transform = Identity() if target_transform is None else target_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn is not None else collate_fn_default

        self.data_path = None #: str: Path to downloaded dataset in root
        self.train_path = None #: str: Path to train dataset
        self.val_path = None #: str: Path to validation dataset
        

    def _set_config(self, dataset: str, impaired: bool, is_train: bool, qa: bool, eb_no: bool, seed: int) -> dict:
        """_summary_

        Args:
            dataset (str): Dataset name, either "Narrowband" or "Wideband".
            impaired (bool): Whether dataset is impaired or not (clean).
            is_train (bool): Whether dataset is train or not (val).
            qa (bool): Whether to create smaller dataset version.
            eb_no (bool): Whether to use EbNo for dataset.
            seed (int): Seed for dataset generation.

        Raises:
            ValueError: Dataset name is not Narrowband or Wideband.

        Returns:
            dict: TorchSig config file for dataset.
        """
        if not dataset in ["Narrowband", "Wideband"]:
            raise ValueError(f"Invalid dataset type: {dataset}")
        
        i = "Impaired" if impaired else "Clean"
        t = "Train" if is_train else "Val"
        q = "QA" if qa else ""
        e = "EbNo" if eb_no else ""

        config_name = f"{dataset}{i}{e}{t}{q}Config"

        return getattr(conf, config_name)

    def prepare_data(self) -> None:
        """Download datasets into self.data_path

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("prepare_data not implemented")

    # create dataset instances
    # sets self.train, self.val
    def setup(self, stage: str) -> None:
        """Set up datasets, self.train and self.val

        Args:
            stage (str): PyTorch Lightning trainer stage - fit, test, predict.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("setup not implemented")

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Returns validation dataloader.

        Returns:
            DataLoader: val dataloader
        """
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

class NarrowbandDataModule(TorchSigDataModule):
    """TorchSig Narrowband PyTorch Lightning DataModule

        Attributes:
            class_list (list): TorchSigNarrowband class list names.
    """

    class_list = list(TorchSigNarrowband._idx_to_name_dict.values())
    

    def __init__(self, 
                 root: str, 
                 impaired: bool, 
                 qa: bool = True, 
                 eb_no: bool = False, 
                 seed: int = 12345,
                 overlap_prob: Optional[float] = .1,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 collate_fn: Optional[Callable] = None
                ):
        """TorchSigNarrowband DataModule Init

        Args:
            root (str): Dataset root path.
            impaired (bool): Dataset impairment setting.
            qa (bool, optional): Generate small dataset sample. Defaults to True.
            eb_no (bool, optional): Use EbNo config. Defaults to False.
            seed (int, optional): Dataset generation seed. Defaults to 12345.
            transform (Optional[Callable], optional): Data transforms. Defaults to None.
            target_transform (Optional[Callable], optional): Label transforms. Defaults to None.
            batch_size (int, optional): Dataloader batch size. Defaults to 4.
            num_workers (int, optional): Dataloader number of workers to use. Defaults to 1.
            collate_fn (Optional[Callable], optional): Dataloader custom collate function. Defaults to TorchSig collate_fn.
        """
        super().__init__(root, "Narrowband", impaired, qa, eb_no, seed, overlap_prob, transform, target_transform, batch_size, num_workers, collate_fn)

        self.data_path = f"{self.root}/{self.dataset.lower()}_{self.clean}_"
        self.train_path = self.data_path + "train"
        self.val_path = self.data_path + "val"

    def prepare_data(self) -> None:
        """Download TorchSigNarrowband Dataset
        """
        ds_train = ModulationsDataset(
            level=self.train_config.level,
            num_samples=self.train_config.num_samples,
            num_iq_samples=self.train_config.num_iq_samples,
            use_class_idx=self.train_config.use_class_idx,
            include_snr=self.train_config.include_snr,
            eb_no=self.train_config.eb_no,
        )
        
        ds_val = ModulationsDataset(
            level=self.val_config.level,
            num_samples=self.val_config.num_samples,
            num_iq_samples=self.val_config.num_iq_samples,
            use_class_idx=self.val_config.use_class_idx, 
            include_snr=self.val_config.include_snr,
            eb_no=self.val_config.eb_no,
        )
    
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)
    
        creator_train = DatasetCreator(ds_train, seed=self.seed, path=self.train_path, num_workers = self.num_workers)
        print(f"Using {self.train_config.__name__} for train.")
        creator_train.create()
    
        creator_val = DatasetCreator(ds_val, seed=self.seed, path=self.val_path, num_workers = self.num_workers)
        print(f"Using {self.val_config.__name__} for val.")
        creator_val.create()

    def setup(self, stage: str) -> None:
        """Set up TorchSigNarrowband train and validation datasets.

        Args:
            stage (str): PyTorch Lightning trainer stage - fit, test, predict.
        """
        self.train = TorchSigNarrowband(
            self.root,
            train=True,
            impaired=self.impaired,
            transform=self.transform,
            target_transform=self.target_transform,
            use_signal_data=True,
        )
        
        self.val = TorchSigNarrowband(
            self.root,
            train=False,
            impaired=self.impaired,
            transform=self.transform,
            target_transform=self.target_transform,
            use_signal_data=True,
        )

class WidebandDataModule(TorchSigDataModule):
    """TorchSigWideband PyTorch Lightning DataModule

    """

    def __init__(self, 
                 root: str, 
                 impaired: bool, 
                 qa: bool = True,
                 seed: int = 12345,
                 overlap_prob: Optional[float] = None,
                 fft_size: int = 512,
                 num_classes: int = 53,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 collate_fn: Optional[Callable] = None
                ):
        """TorchSigWideband DataModule Init

        Args:
            root (str): Dataset root path.
            impaired (bool): Dataset impairment setting.
            qa (bool, optional): Generate small dataset sample. Defaults to True.
            eb_no (bool, optional): Use EbNo config. Defaults to False.
            seed (int, optional): Dataset generation seed. Defaults to 12345.
            overlap_prob (Optional[float], optional): Signal overlap probability. Defaults to None.
            fft_size (int, optional): Fast Fourier Transform window size. Defaults to 512.
            num_classes (int, optional): Number of classes of signals. Defaults to 53.
            transform (Optional[Callable], optional): Data transforms. Defaults to None.
            target_transform (Optional[Callable], optional): Label transforms. Defaults to None.
            batch_size (int, optional): Dataloader batch size. Defaults to 4.
            num_workers (int, optional): Dataloader number of workers to use. Defaults to 1.
            collate_fn (Optional[Callable], optional): Dataloader custom collate function. Defaults to TorchSig collate_fn.
        """
        super().__init__(root, "Wideband", impaired, qa, False, seed, overlap_prob, transform, target_transform, batch_size, num_workers, collate_fn)

        self.overlap_prob = self.train_config.overlap_prob if overlap_prob is None else overlap_prob
        self.fft_size = fft_size
        self.num_classes = num_classes

        self.data_path = f"{self.root}/wideband_{self.clean}_"
        self.train_path = self.data_path + "train"
        self.val_path = self.data_path + "val"

    def prepare_data(self) -> None:
        """Download TorchSigWideband
        """
        ds_train = WidebandModulationsDataset(
            level=self.train_config.level,
            num_samples=self.train_config.num_samples,
            num_iq_samples=self.train_config.num_iq_samples,
            seed=self.train_config.seed,
            overlap_prob=self.overlap_prob
        )
        
        ds_val = WidebandModulationsDataset(
            level=self.val_config.level,
            num_samples=self.val_config.num_samples,
            num_iq_samples=self.val_config.num_iq_samples,
            seed=self.val_config.seed,
            overlap_prob=self.overlap_prob
        )
    
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)
    
        creator_train = DatasetCreator(ds_train, seed=self.seed, path=self.train_path, num_workers = self.num_workers)
        print(f"Using {self.train_config.__name__} for train.")
        creator_train.create()
    
        creator_val = DatasetCreator(ds_val, seed=self.seed, path=self.val_path, num_workers = self.num_workers)
        print(f"Using {self.val_config.__name__} for val.")
        creator_val.create()

    def setup(self, stage: str) -> None:
        """Set up TorchSigWideband train and validation datasets.

        Args:
            stage (str): PyTorch Lightning trainer stage - fit, test, predict.
        """
        self.train = TorchSigWideband(
            self.root,
            train=True,
            impaired=self.impaired,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        self.val = TorchSigWideband(
            self.root,
            train=False,
            impaired=self.impaired,
            transform=self.transform,
            target_transform=self.target_transform,
        )