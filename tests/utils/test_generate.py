"""Tests for Narrowband and Wideband Generation Scripts
"""

from torchsig.utils.generate import generate
from torchsig.datasets.dataset_metadata import NarrowbandMetadata, WidebandMetadata

import pytest

import itertools
import math

def run_generate(tmpdir, dataset_metadata, batch_size, num_workers):

    generate(
        root=tmpdir,
        dataset_metadata=dataset_metadata,
        batch_size=batch_size,
        num_workers=num_workers
    )

test_narrowband_params = list(itertools.product(
    # num_samples
    [1, 10],
    # num_iq_samples
    [50, 100],
    # impaired
    [True, False]
))
@pytest.mark.parametrize("num_samples, num_iq_samples, impaired", test_narrowband_params)
def test_generate_narrowband(
    tmpdir, 
    num_samples: int, 
    num_iq_samples: int, 
    impaired: bool,
    fft_size: int = -1,
):
    print(num_samples, num_iq_samples, impaired)
    dataset_metadata = NarrowbandMetadata(
        num_samples = num_samples,
        num_iq_samples_dataset=num_iq_samples,
        fft_size= int(math.sqrt(num_iq_samples)) if fft_size == -1 else fft_size,
        impairment_level= 2 if impaired else 0,
    )

    run_generate(
        tmpdir=tmpdir,
        dataset_metadata=dataset_metadata,
        batch_size=1,
        num_workers=1
    )


test_wideband_params = list(itertools.product(
    # num_signals
    [1, 3],
    # num_samples
    [1, 10],
    # num_iq_samples
    [50],
    # impaired
    [True, False]
))
@pytest.mark.parametrize("num_signals, num_samples, num_iq_samples, impaired", test_wideband_params)
def test_generate_wideband(
    tmpdir,
    num_signals: int,
    num_samples: int,
    num_iq_samples: int,
    impaired: bool,
    fft_size: int = -1,
):
    print(num_signals, num_samples, num_iq_samples, impaired)
    dataset_metadata = WidebandMetadata(
        num_signals_max=num_signals,
        num_samples = num_samples,
        num_iq_samples_dataset=num_iq_samples,
        fft_size= int(math.sqrt(num_iq_samples)) if fft_size == -1 else fft_size,
        impairment_level= 2 if impaired else 0,
    )

    run_generate(
        tmpdir=tmpdir,
        dataset_metadata=dataset_metadata,
        batch_size=1,
        num_workers=1
    )