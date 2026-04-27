"""Testing the random seeding functionality of the Seedable class"""

import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


def test_dataset_seeds_correctly():
    # tests that TorchSigIterableDataset correctly seeds its RNG and that the seed is 
    # consistent across multiple dataset instances with the same seed
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

    assert (test_value11 == test_value21).all()
    assert (test_value12 == test_value22).all()
    assert (test_value31 != test_value21).all()


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
def test_dataloader_seeds_correctly_single_worker():
    # tests that WorkerSeedingDataLoader correctly seeds the underlying dataset and that the seed is 
    # consistent across multiple dataloader instances with the same seed
    dataset_metadata = TorchSigDefaults().default_dataset_metadata

    ds0 = TorchSigIterableDataset(metadata=dataset_metadata)
    dl0 = WorkerSeedingDataLoader(ds0, batch_size=4, collate_fn=identity_collate_fn)
    dl0.seed(42)
    test_value0 = next(iter(dl0))[0].data

    ds1 = TorchSigIterableDataset(metadata=dataset_metadata)
    dl1 = WorkerSeedingDataLoader(ds1, batch_size=4, collate_fn=identity_collate_fn)
    dl1.seed(42)
    test_value1 = next(iter(dl1))[0].data

    ds2 = TorchSigIterableDataset(metadata=dataset_metadata)
    dl2 = WorkerSeedingDataLoader(ds2, batch_size=4, collate_fn=identity_collate_fn)
    dl2.seed(7)
    test_value2 = next(iter(dl2))[0].data

    assert (test_value0 == test_value1).all()
    assert (test_value0 != test_value2).all()


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
def test_dataloader_smoke_test_mp_workers():
    # tests dataloader with multiple workers. This is a SMOKE TEST ONLY to check that no 
    # errors are raised and that the seed is set without error when using multiple workers, since 
    # true reproducibility with multiple workers is not guaranteed and will depend on the platform 
    # and PyTorch version
    dataset_metadata = TorchSigDefaults().default_dataset_metadata
    ds = TorchSigIterableDataset(metadata=dataset_metadata, target_labels=[])
    dl = WorkerSeedingDataLoader(ds, batch_size=4, num_workers=4)
    dl.seed(42)
    batch = next(iter(dl))
    assert batch is not None


def test_WorkerSeedingDataLoader_rejects_worker_init_fn():
    # tests that WorkerSeedingDataLoader raises an error if a user-provided worker_init_fn is given,
    # since this is not compatible with the internal seeding mechanism
    dataset_metadata = TorchSigDefaults().default_dataset_metadata

    ds = TorchSigIterableDataset(metadata=dataset_metadata)

    def user_worker_init_fn(_wid: int):
        return None

    with pytest.raises(ValueError, match="No worker_init_fn should be given"):
        WorkerSeedingDataLoader(ds, batch_size=2, collate_fn=identity_collate_fn, worker_init_fn=user_worker_init_fn)