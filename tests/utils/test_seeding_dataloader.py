"""Testing the random seeding functionality of the Seedable class"""

import pytest
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults


def test_dataset_seeds_correctly():
    dataset_metadata = TorchSigDefaults().default_dataset_metadata

    ts_ds = TorchSigIterableDataset(metadata=dataset_metadata)
    ts_ds.seed(42)
    test_value11 = next(ts_ds).data
    test_value12 = next(ts_ds).data
    ts_ds = TorchSigIterableDataset(metadata=dataset_metadata)
    ts_ds.seed(42)
    test_value21 = next(ts_ds).data
    test_value22 = next(ts_ds).data
    ts_ds = TorchSigIterableDataset(metadata=dataset_metadata)
    ts_ds.seed(7)
    test_value31 = next(ts_ds).data
    test_value32 = next(ts_ds).data

    assert (test_value11 == test_value21).all()
    assert (test_value12 == test_value22).all()
    assert (test_value31 != test_value21).all()
    assert (test_value31 != test_value21).all()


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
def test_dataloader_seeds_correctly():
    dataset_metadata = TorchSigDefaults().default_dataset_metadata

    ts_ds = TorchSigIterableDataset(metadata=dataset_metadata)
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, collate_fn=lambda x: x)
    dataloader.seed(42)
    test_value1 = next(iter(dataloader))[0].data
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, collate_fn=lambda x: x)
    dataloader.seed(42)
    test_value2 = next(iter(dataloader))[0].data
    dataloader = WorkerSeedingDataLoader(ts_ds, batch_size=4, collate_fn=lambda x: x)
    dataloader.seed(7)
    test_value3 = next(iter(dataloader))[0].data

    assert (test_value1 == test_value2).all()
    assert (test_value1 != test_value3).all()
