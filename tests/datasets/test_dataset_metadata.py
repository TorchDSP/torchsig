"""Unit Tests for datasets/dataset_metadata.py

Classes:
- DatasetMetadata
"""

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import pytest
import numpy as np

# Built-In


fft_size = 512
num_iq_samples_dataset = fft_size ** 2
sample_rate = 100e6
num_signals_max = 1
num_signals_min = 0
transforms = []
target_transforms = []
impairment_level = [0, 1, 2]
class_list = TorchSigSignalLists.all_signals


def test_DatasetMetadata():
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

    assert isinstance(str(md), str)
    assert isinstance(repr(md), str)
    
    assert isinstance(md.dataset_center_freq_max, float)
    assert isinstance(md.dataset_center_freq_min, float)
    assert md.dataset_center_freq_min < md.dataset_center_freq_max

    assert isinstance(md.dataset_duration_max, float)
    assert isinstance(md.dataset_duration_min, float)
    assert md.dataset_duration_min < md.dataset_duration_max

    assert isinstance(md.dataset_duration_in_samples_max, int)
    assert isinstance(md.dataset_duration_in_samples_min, int)
    assert md.dataset_duration_in_samples_min < md.dataset_duration_in_samples_max

    assert isinstance(md.dataset_bandwidth_max, float)
    assert isinstance(md.dataset_bandwidth_min, float)
    assert md.dataset_bandwidth_min < md.dataset_bandwidth_max

    assert isinstance(md.signal_duration_in_samples_max, int)
    assert isinstance(md.signal_duration_in_samples_min, int)
    assert md.signal_duration_in_samples_min < md.signal_duration_in_samples_max

    assert isinstance(md.fft_frequency_resolution, float)
    
    assert isinstance(md.fft_frequency_max, float)
    assert isinstance(md.fft_frequency_min, float)
    assert md.fft_frequency_min < md.fft_frequency_max

    assert isinstance(md.frequency_max, float)
    assert isinstance(md.frequency_min, float)
    assert md.frequency_min < md.frequency_max

