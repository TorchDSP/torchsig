"""Unit Tests for signals/signal_types.py

Classes:
- SignalMetadata
- Signal
- DatasetSignal
- DatasetDict
"""

from __future__ import annotations

# TorchSig
from torchsig.signals.signal_types import (
    SignalMetadata,
    Signal,
    DatasetSignal,
    DatasetDict
)
from torchsig.utils.dsp import torchsig_complex_data_type
# from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import pytest
import numpy as np

# Built-In
import itertools
from typing import TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from torchsig.datasets.dataset_metadata import DatasetMetadata

# dataset metadata variables
fft_size = 64
num_iq_samples_dataset = fft_size ** 2
sample_rate = 10e6
snr_db_min = 0.0
snr_db_max = 50.0

def narrowband_metadata():
    from torchsig.datasets.dataset_metadata import NarrowbandMetadata

    return NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=0,
        sample_rate=sample_rate,
        snr_db_max=snr_db_max,
        snr_db_min=snr_db_min
    )

def wideband_metadata():
    from torchsig.datasets.dataset_metadata import WidebandMetadata

    return WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=0,
        num_signals_max=3,
        sample_rate=sample_rate,
        snr_db_max=snr_db_max,
        snr_db_min=snr_db_min
    )


# SignalMetadata Tests

def create_verify_signal_metadata(
    dataset_metadata: DatasetMetadata = None,
    center_freq: float = None,
    bandwidth: float = None,
    start_in_samples: int = None,
    duration_in_samples: int = None,
    snr_db: float = None,
    class_name: str = None,
    class_index: int = None,
) -> None:
    m = SignalMetadata(
        dataset_metadata,
        center_freq,
        bandwidth,
        start_in_samples,
        duration_in_samples,
        snr_db,
        class_name,
        class_index
    )
    m.verify()

good_signal_metadata = dict(
    center_freq = 0.0,
    bandwidth = 0.5,
    start_in_samples = 0,
    duration_in_samples = 10,
    snr_db = 25.0,
    class_name = "test-signal-name",
    class_index = 0
)




def run_SignalMetadata_test(
    dataset_metadata: DatasetMetadata, 
    center_freq: float, 
    bandwidth: float, 
    start_in_samples: int, 
    duration_in_samples: int, 
    snr_db: float, 
    class_name: str, 
    class_index: int, 
    is_error: bool
) -> None:
    if is_error:
        with pytest.raises(Exception):
            create_verify_signal_metadata(
                dataset_metadata,
                center_freq,
                bandwidth,
                start_in_samples,
                duration_in_samples,
                snr_db,
                class_name,
                class_index
            )
    else:
        create_verify_signal_metadata(
            dataset_metadata,
            center_freq,
            bandwidth,
            start_in_samples,
            duration_in_samples,
            snr_db,
            class_name,
            class_index
        )


test_signalmetadata_params_valid = list(itertools.product(
    # dataset_type
    ['narrowband', 'wideband'],
    # center_freq
    [0.0, 0.5, -0.5],
    # bandwidth
    [0.1, 0.2],
    # start_in_samples
    [1, 10],
    # duration_in_samples
    [2, 10, 100],
    # snr_db
    [10.0, 50.0],
    # class_name
    ['ook', 'tone'],
    # class_index
    [1, 2]
))
@pytest.mark.parametrize(
    "dataset_type, center_freq, bandwidth, start_in_samples, duration_in_samples, snr_db, class_name, class_index", 
    test_signalmetadata_params_valid
)
def test_valid_SignalMetadata(
    dataset_type: str,
    center_freq: float, 
    bandwidth: float, 
    start_in_samples: int, 
    duration_in_samples: int, 
    snr_db: float, 
    class_name: str, 
    class_index: int,
):
    signal_metadata = dict(
        center_freq=center_freq,
        bandwidth=bandwidth,
        start_in_samples=start_in_samples,
        duration_in_samples=duration_in_samples,
        snr_db=snr_db,
        class_name=class_name,
        class_index=class_index

    )

    if dataset_type == 'narrowband':
        dataset_metadata = narrowband_metadata()
    else:
        dataset_metadata = wideband_metadata()

    edge_cases = dict(
        # center_freq
        center_freq = [
            dataset_metadata.signal_center_freq_min + 1, 
            dataset_metadata.signal_center_freq_max
        ],
        # bandwidth
        bandwidth = [
            0.000001, 
            dataset_metadata.sample_rate
        ],
        # start_in_samples
        start_in_samples = [
            0, 
            dataset_metadata.num_iq_samples_dataset - 1
        ],
        # duration_in_samples
        duration_in_samples = [
            1,
            dataset_metadata.num_iq_samples_dataset,
        ],
        # snr_db
        snr_db = [
            0
        ],
    )

    signal_metadata['dataset_metadata'] = dataset_metadata
    signal_metadata['is_error'] = False

    for field in edge_cases.keys():
        for edge_case in edge_cases[field]:
            signal_metadata[field] = edge_case
            run_SignalMetadata_test(**signal_metadata)


@pytest.mark.parametrize("dataset_type", ('narrowband', 'wideband'))
def test_invalid_SignalMetadata(
    dataset_type: str
):
    if dataset_type == 'narrowband':
        dataset_metadata = narrowband_metadata()
    else:
        dataset_metadata = wideband_metadata()

    bad_params = dict(
        # dataset_metadata
        dataset_metadata = [None],
        # center_freq
        center_freq = [
            dataset_metadata.signal_center_freq_min, 
            -99999, 
            dataset_metadata.signal_center_freq_max + 1
        ],
        # bandwidth
        bandwidth = [
            0.0, 
            dataset_metadata.sample_rate + 1
        ],
        # start_in_samples
        start_in_samples = [
            -1,
            dataset_metadata.num_iq_samples_dataset,
            dataset_metadata.num_iq_samples_dataset + 1
        ],
        # duration_in_samples
        duration_in_samples = [
            0,
            dataset_metadata.num_iq_samples_dataset + 1,
        ],
        # snr_db
        snr_db = [
            -1,
            -0.1,
        ],
        # class_name
        class_name = [
            None,
            True,
            7,
        ],
        # class_index
        class_index = [
            -1,
            None,
            3.2,
            "hi"
        ]
    )

    signal_metadata = deepcopy(good_signal_metadata)
    signal_metadata['dataset_metadata'] = dataset_metadata

    for field in bad_params.keys():
        for bad_param in bad_params[field]:
            signal_metadata[field] = bad_param

            signal_metadata['is_error'] = True
    
            run_SignalMetadata_test(**signal_metadata)


# Signal Tests

@pytest.mark.parametrize("data, is_error", [
    (np.ones((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type), False),
    (np.zeros((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type), False),
    (np.random.random((good_signal_metadata['duration_in_samples'])).astype(torchsig_complex_data_type), False),

    (np.ones((good_signal_metadata['duration_in_samples']), dtype=int), True),
    (np.zeros((good_signal_metadata['duration_in_samples']), dtype=float), True),
    (np.random.random((good_signal_metadata['duration_in_samples'])).astype(float), True),

    (np.ones((good_signal_metadata['duration_in_samples'] + 1), dtype=torchsig_complex_data_type), True),
    (np.zeros((good_signal_metadata['duration_in_samples'] - 1), dtype=torchsig_complex_data_type), True),

    (np.random.random((good_signal_metadata['duration_in_samples'] + 2)).astype(float), True),
])
def test_Signal(data: np.ndarray, is_error: bool):

    signal_metadata = deepcopy(good_signal_metadata)
    signal_metadata['dataset_metadata'] = narrowband_metadata()
    if is_error:
        with pytest.raises(Exception):
            
            s = Signal(
                data = data,
                metadata = SignalMetadata(**signal_metadata)
            )
            s.verify()
    else:
        s = Signal(
            data = data,
            metadata = SignalMetadata(**signal_metadata)
        )
        s.verify()

# DatasetSignal Tests
good_signal_metadata = dict(
    center_freq = 0.0,
    bandwidth = 0.5,
    start_in_samples = 0,
    duration_in_samples = 10,
    snr_db = 25.0,
    class_name = "test-signal-name",
    class_index = 0
)

@pytest.mark.parametrize("data, is_error", [
    (np.ones((num_iq_samples_dataset), dtype=torchsig_complex_data_type), False),
    (np.zeros((num_iq_samples_dataset), dtype=torchsig_complex_data_type), False),
    (np.random.random((num_iq_samples_dataset)).astype(torchsig_complex_data_type), False),

    (np.ones((num_iq_samples_dataset + 1), dtype=torchsig_complex_data_type), True),
    (np.zeros((num_iq_samples_dataset - 1), dtype=torchsig_complex_data_type), True),

    (np.random.random((num_iq_samples_dataset + 2)).astype(float), True),
])
def test_DatasetSignal(data: np.ndarray, is_error: bool):
    signal_metadata = deepcopy(good_signal_metadata)
    signal_metadata['dataset_metadata'] = narrowband_metadata()
    s1 = Signal(
        data = np.ones((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type),
        metadata = SignalMetadata(**signal_metadata)
    )
    s2 = Signal(
        data = np.ones((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type),
        metadata = SignalMetadata(**signal_metadata)
    )
    signals = [s1, s2]

    all_signals = [
        [s1, s2],
        s1,
        [SignalMetadata(**signal_metadata), SignalMetadata(**signal_metadata)],
        SignalMetadata(**signal_metadata),
        [signal_metadata],
        [signal_metadata, deepcopy(signal_metadata)]
    ]

    for signals in all_signals:
        if is_error:
            with pytest.raises(Exception):
                
                ds = DatasetSignal(
                    data = data,
                    signals = signals,
                    dataset_metadata = signal_metadata['dataset_metadata']
                )
                ds.verify()
        else:
            ds = DatasetSignal(
                    data = data,
                    signals = signals,
                    dataset_metadata = signal_metadata['dataset_metadata']
                )
            ds.verify()

# DatasetDict

def test_DatasetDict():
    signal_metadata = deepcopy(good_signal_metadata)
    signal_metadata['dataset_metadata'] = narrowband_metadata()
    s1 = Signal(
        data = np.ones((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type),
        metadata = SignalMetadata(**signal_metadata)
    )
    s2 = Signal(
        data = np.ones((good_signal_metadata['duration_in_samples']), dtype=torchsig_complex_data_type),
        metadata = SignalMetadata(**signal_metadata)
    )
    signals = [s1, s2]
    data = np.ones((num_iq_samples_dataset), dtype=torchsig_complex_data_type)
    ds = DatasetSignal(
        data = data,
        signals = signals
    )

    dd = DatasetDict(signal = ds)
    dd.verify()
