"""Unit Tests for signals/builder.py

Classes:
- Builder
- SignalBuilder
"""

from torchsig.signals.signal_types import Signal
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.signals.builders.constellation import ConstellationSignalBuilder
import torchsig.transforms.functional as F
from torchsig.utils.dsp import TorchSigComplexDataType

# Third Party
import numpy as np


rng = np.random.default_rng(42)

for i in range(3):
    # build a test scaled QPSK Signal component
    md_qpsk = DatasetMetadata(
        num_iq_samples_dataset = 1024,
        fft_size = 128,
        sample_rate = 10e6,
        num_signals_min = 1,
        num_signals_max = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,
        class_list = ['qpsk'],
        class_distribution = [1.0],
        seed = 1234
    )
    
    qpsk_builder = ConstellationSignalBuilder(
        dataset_metadata = md_qpsk, 
        class_name = 'qpsk',
        seed = 1234
    )
    qpsk_signal = qpsk_builder.build()
    
    print("\nqpsk_signal:",qpsk_signal.data[0:3])
