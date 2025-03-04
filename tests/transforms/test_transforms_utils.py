"""Utility functions for transforms testing.
"""
from torchsig.signals.signal_types import Signal, DatasetSignal, DatasetDict
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from torchsig.signals.builders.constellation import ConstellationSignalBuilder
import torchsig.transforms.functional as F
from torchsig.utils.dsp import torchsig_complex_data_type

# Third Party
import numpy as np


def generate_test_dataset_dict(num_iq_samples: int = 64, scale: float = 1.0) -> DatasetDict:
    dataset_signal = generate_test_dataset_signal(
        num_iq_samples=num_iq_samples,
        scale=scale
    )
    return DatasetDict(signal = dataset_signal)


def generate_test_dataset_signal(num_iq_samples: int = 64, scale: float = 1.0) -> DatasetSignal:
    """Generate a DatasetSignal with two signals: high SNR baseband BPSK and QPSK Signals.

        Args:
            num_iq_samples (int, optional): Length of sample. Defaults to 64.
            scale (int, optional): scale normalized signal data. Defaults to 1.0.            

        Returns:
            nb_signal: generated DatatsetSignal.

    """
    rng = np.random.default_rng(42)

    # build a test scaled QPSK Signal component
    md_qpsk = NarrowbandMetadata(
        num_iq_samples_dataset = num_iq_samples,
        fft_size = 4,
        impairment_level = 0,
        sample_rate = 10e6,
        num_signals_min = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,       
        signal_duration_percent_min = 100.0,
        signal_duration_percent_max = 100.0,
        transforms = [],
        target_transforms = [],
        class_list = ['qpsk'],
        class_distribution = [1.0],
        num_samples = 1,
        seed = 1234
    )
    qpsk_builder = ConstellationSignalBuilder(
        dataset_metadata = md_qpsk, 
        class_name = 'qpsk',
        seed = 1234
    )
    qpsk_signal = qpsk_builder.build()

    # build a test scaled BPSK Signal component
    md_bpsk = NarrowbandMetadata(
        num_iq_samples_dataset = num_iq_samples,
        fft_size = 4,
        impairment_level = 0,
        sample_rate = 10e6,
        num_signals_min = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,
        signal_duration_percent_min = 100.0,
        signal_duration_percent_max = 100.0,
        class_list = ['bpsk'],
        class_distribution = [1.0],
        seed = 5678
    )

    bpsk_builder = ConstellationSignalBuilder(
        dataset_metadata = md_bpsk,
        class_name = 'bpsk',
        seed = 5678
    )
    bpsk_signal = bpsk_builder.build()

    # create test DatasetSignal
    
    # noise floor
    noise_power_lin = 10**(md_bpsk.noise_power_db / 10)
    noise_real_samples = rng.normal(0,np.sqrt(noise_power_lin/2),4*num_iq_samples)
    noise_imag_samples = rng.normal(0,np.sqrt(noise_power_lin/2),4*num_iq_samples)
    iq_samples = noise_real_samples + 1j*noise_imag_samples

    # place baseband QPSK signal at start and BPSK signal at midpoint
    qpsk_signal.metadata.start_in_samples = 0
    bpsk_signal.metadata.start_in_samples = int(2*num_iq_samples)
        
    iq_samples[qpsk_signal.metadata.start_in_samples: qpsk_signal.metadata.start_in_samples 
               + qpsk_signal.metadata.duration_in_samples] += qpsk_signal.data
    
    iq_samples[bpsk_signal.metadata.start_in_samples: bpsk_signal.metadata.start_in_samples 
               + bpsk_signal.metadata.duration_in_samples] += bpsk_signal.data   
    
    signals = [qpsk_signal, bpsk_signal] 
    ds_signal = DatasetSignal(data=iq_samples, signals=signals)
    
    # normalize, then scale data   
    ds_signal.data = F.normalize(
        data = ds_signal.data,
        norm_order = 2,
        flatten = False
    )
    ds_signal.data = np.multiply(ds_signal.data, scale).astype(torchsig_complex_data_type)    
    return ds_signal


def generate_test_signal(num_iq_samples: int = 10, scale: float = 1.0) -> Signal:
    """Generate a scaled, high SNR baseband QPSK Signal.

        Args:
        num_iq_samples (int, optional): Length of sample. Defaults to 10.
        scale (int, optional): scale normalized signal data. Defaults to 1.0.

        Returns:
            signal: generated Signal.

    """
    md = NarrowbandMetadata(
        num_iq_samples_dataset = num_iq_samples,
        fft_size = 4,
        impairment_level = 0,
        sample_rate = 10e6,
        num_signals_min = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,
        signal_duration_percent_min = 100.0,
        signal_duration_percent_max = 100.0,
        class_list = ['qpsk'],
        class_distribution = [1.0]
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
    signal.data = np.multiply(signal.data, scale).astype(torchsig_complex_data_type)

    return signal
