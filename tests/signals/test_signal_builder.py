"""Unit Tests for signals/builder.py

Classes:
- Builder
- SignalBuilder
"""

from torchsig.signals.signal_types import Signal, DatasetSignal, DatasetDict
from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.signals.builders.constellation import ConstellationSignalBuilder
import torchsig.transforms.functional as F
from torchsig.utils.dsp import torchsig_complex_data_type

# Third Party
import numpy as np


rng = np.random.default_rng(42)

for i in range(3):
    # build a test scaled QPSK Signal component
    md_qpsk = WidebandMetadata(
        num_iq_samples_dataset = 1024,
        fft_size = 128,
        impairment_level = 0,
        sample_rate = 10e6,
        num_signals_min = 1,
        num_signals_max = 1,
        num_signals_distribution = [1.0],
        snr_db_min = 100.0,
        snr_db_max = 100.0,       
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
    
    print("\nqpsk_signal:",qpsk_signal.data[0:3])
