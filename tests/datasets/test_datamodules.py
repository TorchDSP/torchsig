"""Unit Tests for datasets/datamodules.py

Classes:
- NarrowbandDataModule
- WidebandDataModule
- OfficialNarrowbandDataModule
- OfficialWidebandDataModule
"""


from torchsig.datasets.datamodules import (
    NarrowbandDataModule, 
    WidebandDataModule,  
    OfficialNarrowbandDataModule, 
    OfficialWidebandDataModule,
)
from torchsig.datasets.dataset_metadata import NarrowbandMetadata, WidebandMetadata

import pytest

num_samples = 5

# fft_size = np.random.randint(128,1024)
fft_size = 512
# num_iq_samples_dataset = fft_size*np.random.randint(128,1024)
num_iq_samples_dataset = fft_size ** 2

impairment_level = 2

num_signals_min = 1

# wideband
num_signals_max = 3

def test_wideband_datamodule(tmpdir):

    root = tmpdir

    WM = WidebandMetadata(
        num_iq_samples_dataset = 64 ** 2,
        fft_size = 64,
        impairment_level = 2,
        num_signals_max=3,
    )

    dm = WidebandDataModule(
        root = root,
        dataset_metadata = WM,
        num_samples_train = 10
    )
    dm.prepare_data()
    dm.setup()

def test_narrowband_datamodule(tmpdir):

    root = tmpdir

    NM = NarrowbandMetadata(
        num_iq_samples_dataset=64 ** 2,
        fft_size = 64,
        impairment_level=2
    )

    dm = NarrowbandDataModule(
        root = root,
        dataset_metadata = NM,
        num_samples_train = 10
    )
    
    dm.prepare_data()
    dm.setup()

@pytest.mark.slow
@pytest.mark.skip(reason = "Tests too slow")
def test_official_narrowband_datamodule(tmpdir):

    root = tmpdir

    dm = OfficialNarrowbandDataModule(
        root=root,
        impaired=True,
        create_batch_size=32,
        create_num_workers=16
    )
    dm.prepare_data()
    dm.setup()

@pytest.mark.slow
@pytest.mark.skip(reason = "Tests too slow")
def test_official_wideband_datamodule(tmpdir):
    root = tmpdir

    dm = OfficialWidebandDataModule(
        root=root,
        impaired=True,
        create_batch_size=32,
        create_num_workers=16
    )
    dm.prepare_data()
    dm.setup()


