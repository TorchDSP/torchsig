"""Utility functions for transforms testing.
"""
from torchsig.signals.signal_types import Signal
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.signals.builders.constellation import ConstellationSignalBuilder
from torchsig.signals.builders.tone import ToneSignalBuilder
import torchsig.transforms.functional as F
from torchsig.utils.dsp import TorchSigComplexDataType

# Third Party
import numpy as np


def generate_test_signal(num_iq_samples: int = 10, scale: float = 1.0) -> Signal:
    """Generate a scaled, high SNR baseband QPSK Signal.

        Args:
        num_iq_samples (int, optional): Length of sample. Defaults to 10.
        scale (int, optional): scale normalized signal data. Defaults to 1.0.

        Returns:
            signal: generated Signal.

    """
    sample_rate = 10e6
    md = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples,
        fft_size = 4,
        impairment_level = 0,
        sample_rate = sample_rate,
        num_signals_min = 1,
        num_signals_max = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,
        signal_duration_min = 1.00*num_iq_samples/sample_rate,
        signal_duration_max = 1.00*num_iq_samples/sample_rate,
        signal_bandwidth_min = sample_rate/4,
        signal_bandwidth_max = sample_rate/4,
        signal_center_freq_min = 0.0,
        signal_center_freq_max = 0.0,         
        class_list = ['qpsk'],
        class_distribution = [1.0],
        seed = 42
    )

    builder = ConstellationSignalBuilder(
        dataset_metadata = md, 
        class_name = 'qpsk',
        seed = 42
    )
    signal = builder.build()

    # normalize, then scale data   
    signal.data = F.normalize(
        data = signal.data,
        norm_order = 2,
        flatten = False
    )
    signal.data = np.multiply(signal.data, scale).astype(TorchSigComplexDataType)

    return signal

def generate_tone_signal(num_iq_samples: int = 10, scale: float = 1.0) -> Signal:
    """Generate a scaled, high SNR baseband tone Signal.

        Args:
        num_iq_samples (int, optional): Length of sample. Defaults to 10.
        scale (int, optional): scale normalized signal data. Defaults to 1.0.

        Returns:
            signal: generated Signal.

    """
    sample_rate = 10e6
    md = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples,
        fft_size = 4,
        impairment_level = 0,
        sample_rate = sample_rate,
        num_signals_min = 1,
        num_signals_max = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,
        signal_duration_min = 1.00*num_iq_samples/sample_rate,
        signal_duration_max = 1.00*num_iq_samples/sample_rate,
        signal_bandwidth_min = sample_rate/4,
        signal_bandwidth_max = sample_rate/4,
        signal_center_freq_min = 0.0,
        signal_center_freq_max = 0.0,         
        class_list = ['tone'],
        class_distribution = [1.0],
        seed = 42
    )

    builder = ToneSignalBuilder(
        dataset_metadata = md, 
        class_name = 'tone',
        seed = 42
    )
    signal = builder.build()

    # normalize, then scale data   
    signal.data = F.normalize(
        data = signal.data,
        norm_order = 2,
        flatten = False
    )
    signal.data = np.multiply(signal.data, scale).astype(TorchSigComplexDataType)

    return signal
