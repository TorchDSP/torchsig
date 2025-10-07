"""Helpful methods for quickly creating TorchSig datasts and dataloaders
"""

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.writer import default_collate_fn
from torchsig.utils.data_loading import WorkerSeedingDataLoader

from torch.utils.data import DataLoader

from typing import List, Callable

def default_dataset(
    num_signals_min: int = 1, 
    num_signals_max: int = 1, 
    num_iq_samples_dataset: int = 4096, 
    fft_size:int = 64,
    impairment_level: int = None,
    target_labels: list = None,
    transforms: list = [],
    component_transforms: list = [],
    **kwargs
) -> TorchSigIterableDataset:
    """creates a TorchSigIterableDataset with given params, uses impairment_level

    Args:
        num_signals_min (int, optional): min number of signals per sample. Defaults to 1.
        num_signals_max (int, optional): max number of signals per sample. Defaults to 1.
        num_iq_samples_dataset (int, optional): sample length. Defaults to 4096.
        fft_size (int, optional): fft size. Defaults to 64.
        impairment_level (int, optional): Level of impairments to use (0, 1, 2). Defaults to None.
        target_labels (list, optional): Labels for data. Defaults to None.
        transforms (list, optional): dataset Transforms to apply. Defaults to [].
        component_transforms (list, optional): Signal/busrt level of transforms to aply. Defaults to [].

    Returns:
        TorchSigIterableDataset: dataset accoriding to params.
    """
    dataset_metadata = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples_dataset,
        fft_size = fft_size,
        num_signals_max = num_signals_max,
        num_signals_min = num_signals_min,
        num_samples = None
    )
    if impairment_level is not None:
        impairments = Impairments(impairment_level)
        burst_impairments = impairments.signal_transforms
        signal_impairments = impairments.dataset_transforms
        new_transforms=[signal_impairments] + transforms
        new_component_transforms=[burst_impairments] + component_transforms
    else:
        new_transforms = transforms
        new_component_transforms = component_transforms
    return TorchSigIterableDataset(
        dataset_metadata = dataset_metadata,
        target_labels=target_labels,
        transforms=new_transforms,
        component_transforms=new_component_transforms,
        **kwargs
    )

def default_dataloader(
    seed: int = False,
    collate_fn: Callable = default_collate_fn,
    batch_size: int = 1,
    num_workers: int = 1,
    **kwargs
) -> WorkerSeedingDataLoader:
    """Shortcut for creating a WorkerSeedingDataLoader

    Args:
        seed (int, optional): Dataloder seed. Defaults to False.
        collate_fn (Callable, optional): Collate function to use. Defaults to default_collate_fn.
        batch_size (int, optional): Batch size. Defaults to 1.
        num_workers (int, optional): Number of workers. Defaults to 1.

    Returns:
        WorkerSeedingDataLoader: Dataloader according to params
    """
    dataset = default_dataset(**kwargs)
    dataloader = WorkerSeedingDataLoader(
        dataset, 
        collate_fn=collate_fn, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    if seed:
        dataloader.seed(seed)
    return dataloader