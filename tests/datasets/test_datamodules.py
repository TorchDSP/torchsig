"""Unit Tests for datamodules"""

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.writer import default_collate_fn

import pytest
import copy
from pathlib import Path

filename = "data_module_test"
data_dir = Path(__file__).parent

fft_size = 64
num_iq_samples_dataset = fft_size**2


from torchsig.utils.defaults import TorchSigDefaults

metadata = TorchSigDefaults().default_dataset_metadata


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
def test_TorchSigDataModule():
    datamodule = TorchSigDataModule(root=Path.joinpath(data_dir, filename), metadata=metadata, dataset_size=50, overwrite=True, impairment_level=0, collate_fn=lambda x: x)
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.impairment_level == 0

    dataloaders = [datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()]

    for d in dataloaders:
        for batch in d:
            print(batch)
            break

    datamodule = TorchSigDataModule(root=Path.joinpath(data_dir, filename), metadata=metadata, dataset_size=50, overwrite=True, impairment_level=None, collate_fn=lambda x: x)
    datamodule.prepare_data()
    datamodule.setup()

    dataloaders = [datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()]

    for d in dataloaders:
        for batch in d:
            print(batch)
            break
