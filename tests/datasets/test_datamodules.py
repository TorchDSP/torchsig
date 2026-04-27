"""Unit Tests for datamodules"""
import pytest

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
@pytest.mark.parametrize(
    "num_workers, overwrite",
    [
        (None, True),  # single worker with overwrite (should create dataset files on disk)
        (None, False), # single worker with no overwrite (should not error, just skip creation)
        (2, True),     # multiworker with overwrite (should create dataset files on disk)
        (3, False),    # multiworker with no overwrite (should not error, just skip creation)
    ],
)
def test_TorchSigDataModule_smoke_and_disk_artifacts(tmp_path, num_workers, overwrite):
    # tests that TorchSigDataModule can prepare data and set up dataloaders without error, and that 
    # it creates dataset files on disk after prepare_data/setup (multiworker)
    metadata = TorchSigDefaults().default_dataset_metadata

    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=16,
        overwrite=overwrite,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup()

    assert dm.impairment_level == 0

    dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    for dl in dls: 
        for batch_idx, data in enumerate(dl): # check all batches (multiworker issues)
            assert hasattr(data, "__len__")
            print(len(data))
            
        

    # At least ensure dataset directory is populated after prepare_data/setup.
    # (Exact filenames depend on file handler; HDF5 usually uses data.h5.)
    assert any(tmp_path.iterdir())

