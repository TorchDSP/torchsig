from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.writer import default_collate_fn
from torchsig.utils.data_loading import WorkerSeedingDataLoader

def default_dataset(num_signals_min=1, num_signals_max=1, num_iq_samples_dataset=4096, fft_size=64, impairment_level=None, target_labels=None, transforms=[], component_transforms=[], **kwargs):
    dataset_metadata = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples_dataset,
        fft_size = fft_size,
        num_signals_max = num_signals_max,
        num_signals_min = num_signals_min,
        num_samples = None
    )
    if impairment_level != None:
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

def default_dataloader(seed=False, collate_fn=default_collate_fn, batch_size=1, num_workers=1, **kwargs):
    dataset = default_dataset(**kwargs)
    dataloader = WorkerSeedingDataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)
    if seed:
        dataloader.seed(seed)
    return dataloader