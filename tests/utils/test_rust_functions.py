"""Test DSP Rust Code"""

from torchsig.utils.rust_functions import upfirdn as upfirdn_RUST

from scipy.signal import firwin
import pytest
import numpy as np


def test_rust_upfirdn():
    up_rate = 10000
    down_rate = 9700
    input_len = 5310

    print("up = " + str(up_rate) + ", down = " + str(down_rate))
    print("num input samples = " + str(input_len))

    # build input signal
    cfo = 1 / input_len
    in_array = np.array(np.exp(1j * 2 * np.pi * cfo * np.arange(input_len)), dtype=np.complex64)

    # number of iterations to test speed
    num_iter = 100

    # filter design params
    num_taps = 10 * up_rate
    fs = 1
    cutoff = 0.5 / up_rate
    weights = firwin(num_taps, cutoff, fs=fs)
    weights = weights.astype(np.float32)

    # rust resampler
    for this_iter in range(num_iter):
        out_array_rust = upfirdn_RUST(weights, in_array, up_rate, down_rate)

    # check for reproducibility
    for this_iter in range(num_iter):
        out_array_rust_2 = upfirdn_RUST(weights, in_array, up_rate, down_rate)

    assert out_array_rust.dtype == np.complex64
    assert all(out_array_rust == out_array_rust_2)


def test_sampling_clock_impairments():
    pass
