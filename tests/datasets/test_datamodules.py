"""Unit Tests for datasets/datamodules.py

Classes:
- TorchSigDataModule
- OfficialTorchSigDataModule
"""

from torchsig.datasets.datamodules import (
    TorchSigDataModule,  
    OfficialTorchSigDataModule,
)
from torchsig.datasets.dataset_metadata import DatasetMetadata

import pytest

def test_datamodule(tmpdir):

    root = tmpdir

    train_metadata = DatasetMetadata(
        num_iq_samples_dataset = 64 ** 2,
        fft_size = 64,
        impairment_level = 2,
        num_signals_max=3,
        num_samples = 10
    )

    val_metadata = DatasetMetadata(
        num_iq_samples_dataset = 64 ** 2,
        fft_size = 64,
        impairment_level = 2,
        num_signals_max=3,
        num_samples = 10
    )

    dm = TorchSigDataModule(
        root = root,
        train_metadata = train_metadata,
        val_metadata = val_metadata
    )
    dm.prepare_data()
    dm.setup()


@pytest.mark.slow
@pytest.mark.skip(reason = "Tests too slow")
def test_official_datamodule(tmpdir):
    root = tmpdir

    dm = OfficialTorchSigDataModule(
        root=root,
        impairment_level=2,
        create_batch_size=32,
        create_num_workers=16
    )
    dm.prepare_data()
    dm.setup()


