from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import default_collate_fn


class TorchSigDefaults:
    """A class for managing default values used in TorchSig.

    This class provides a centralized location for default configuration values
    used throughout the TorchSig library, particularly for dataset generation.
    """

    def __init__(self):
        """Initialize default dataset metadata values."""
        self._default_dataset_metadata = {
            "num_iq_samples_dataset": 262144,  # 512**2
            "num_signals_min": 1,
            "num_signals_max": 1,
            "fft_size": 512,
            "fft_stride": 512,
            "sample_rate": 10000000,
            "noise_power_db": 0.0,
            "cochannel_overlap_probability": 0.2,
            "signal_duration_in_samples_min": 262144 * 0.8,
            "signal_duration_in_samples_max": 262144 * 1.0,
            "bandwidth_min": 2500000,
            "bandwidth_max": 3333333,
            "signal_center_freq_min": -2500000,
            "signal_center_freq_max": 2499999,
            "frequency_min": -2500000,
            "frequency_max": 2499999,
        }

    @property
    def default_dataset_metadata(self):
        """Return a copy of the default dataset metadata.

        Returns:
            dict: A copy of the default dataset metadata dictionary.
        """
        return self._default_dataset_metadata.copy()


def default_dataset(
    impairment_level=None, transforms=[], component_transforms=[], **kwargs
):
    """Create a default TorchSigIterableDataset with optional impairments.

    This function creates a dataset with default metadata and applies any specified
    impairments and transforms. If impairment_level is provided, it adds the
    corresponding signal and dataset transforms.

    Args:
        impairment_level: Optional impairment level to apply to the dataset.
            If None, no impairments are applied.
        transforms: List of dataset-level transforms to apply.
        component_transforms: List of signal-level transforms to apply.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        TorchSigIterableDataset: A configured dataset instance.
    """
    defaults_to_use = TorchSigDefaults()
    dataset_metadata = defaults_to_use.default_dataset_metadata
    if impairment_level is not None:
        impairments = Impairments(impairment_level)
        burst_impairments = impairments.signal_transforms
        signal_impairments = impairments.dataset_transforms
        new_transforms = [signal_impairments, *transforms]
        new_component_transforms = [burst_impairments, *component_transforms]
    else:
        new_transforms = transforms
        new_component_transforms = component_transforms
    new_dataset = TorchSigIterableDataset(
        metadata=dataset_metadata, transforms=new_transforms, **kwargs
    )
    for signal_gen in new_dataset.signal_generators:
        try:
            signal_gen["transforms"] = new_component_transforms
        except:
            pass  # this object has no transforms to set
    return new_dataset


def default_dataloader(
    seed=False, collate_fn=default_collate_fn, batch_size=1, num_workers=1, **kwargs
) -> WorkerSeedingDataLoader:
    """Create a default WorkerSeedingDataLoader with optional seeding.

    This function creates a data loader with default settings and applies any
    specified configuration. If seed is provided, it initializes the loader
    with that seed value.

    Args:
        seed: Optional seed value for reproducible data loading. If False,
            no seeding is applied.
        collate_fn: Function to use for collating samples into batches.
            Defaults to default_collate_fn.
        batch_size: Number of samples per batch. Defaults to 1.
        num_workers: Number of subprocesses to use for data loading.
            Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        WorkerSeedingDataLoader: Configured data loader instance.
    """
    dataset = default_dataset(**kwargs)
    dataloader = WorkerSeedingDataLoader(
        dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers
    )
    if seed:
        dataloader.seed(seed)
    return dataloader
