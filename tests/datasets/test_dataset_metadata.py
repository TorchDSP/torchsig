"""Unit Tests for datasets/dataset_metadata.py

Classes:
- DatasetMetadata
- NarrowbandMetadata
- WidebandMetadata
"""

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import pytest
import numpy as np

# Built-In


fft_size = 512
num_iq_samples_dataset = fft_size ** 2
narrowband_sample_rate = 10e6
wideband_sample_rate = 100e6
num_signals_max = 1
num_signals_min = 0
transforms = []
target_transforms = []
impairment_level = [0, 1, 2]
class_list = TorchSigSignalLists.all_signals


def test_DatasetMetadata():
    with pytest.raises(NotImplementedError):
        md = DatasetMetadata(
            num_iq_samples_dataset=num_iq_samples_dataset,
            sample_rate=10e6,
            fft_size = 64,
            num_signals_min=0,
            transforms=transforms,
            target_transforms=target_transforms,
            impairment_level=1,
            class_list=class_list,
            num_signals_max=5,
        )
        md.to_dict()

def test_NarrowbandMetadata():
    md = NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        sample_rate=narrowband_sample_rate,
        fft_size=fft_size,
        impairment_level=0,
    )
    md.to_dict()

def test_WidebandMetadata():
    md = WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        sample_rate=wideband_sample_rate,
        fft_size=fft_size,
        num_signals_max=5,
        impairment_level=0
    )
    md.to_dict()
