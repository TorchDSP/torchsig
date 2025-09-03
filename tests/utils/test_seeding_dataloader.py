""" Testing the random seeding functionality of the Seedable class
"""

import pytest

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader

def test_dataset_seeds_correctly():
    fft_size = 64
    num_iq_samples_dataset = fft_size ** 2 # 64^2
    impairment_level = 0 # clean
    metadata = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples_dataset, # 64^2
        fft_size = fft_size,
        num_signals_max = 1,
    )

    ts_ds = TorchSigIterableDataset(metadata, target_labels=["class_name"])
    ts_ds.seed(42)
    test_value11 = next(ts_ds)[0][0]
    test_value12 = next(ts_ds)[0][0]
    ts_ds = TorchSigIterableDataset(metadata, target_labels=["class_name"])
    ts_ds.seed(42)
    test_value21 = next(ts_ds)[0][0]
    test_value22 = next(ts_ds)[0][0]
    ts_ds = TorchSigIterableDataset(metadata, target_labels=["class_name"])
    ts_ds.seed(7)
    test_value31 = next(ts_ds)[0][0]
    test_value32 = next(ts_ds)[0][0]

    assert test_value11 == test_value21
    assert test_value12 == test_value22
    assert test_value31 != test_value21
    assert test_value31 != test_value21


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
def test_dataloader_seeds_correctly():
    num_iq_samples_dataset = 4096 # 64^2
    fft_size = 64
    impairment_level = 0 # clean
    metadata = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples_dataset, # 64^2
        fft_size = fft_size,
        impairment_level = impairment_level, # clean
        num_signals_max = 1,
    )

    ts_ds = TorchSigIterableDataset(metadata, target_labels=["class_name"])
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, num_workers=2)
    dataloader.seed(42)
    test_value1 = next(iter(dataloader))[0][-1][0]
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, num_workers=2)
    dataloader.seed(42)
    test_value2 = next(iter(dataloader))[0][-1][0]
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, num_workers=2)
    dataloader.seed(7)
    test_value3 = next(iter(dataloader))[0][-1][0]

    assert test_value1 == test_value2
    assert test_value1 != test_value3

