"""Unit tests for Bluetooth Low Energy (btle) signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.btle import (
    BTLE_ACCESS_ADDRESS,
    BTLESignalGenerator,
    btle_modulator,
    build_btle_bit_stream,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

BTLE_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_000_000,
    "bandwidth_max": 2_000_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_btle_access_address():
    """BTLE advertising access address matches the spec value."""
    assert BTLE_ACCESS_ADDRESS == 0x8E89BED6


def test_btle_bit_stream_contains_access_address():
    """The bit stream begins with the 8-bit preamble then the 32-bit access address."""
    from torchsig.signals.builders.btle import _PREAMBLE_BITS

    rng = np.random.default_rng(0)
    stream = build_btle_bit_stream(100, rng)
    assert len(stream) == 100
    # Check preamble (bipolar): 0xAA = 10101010 → bipolar [-1,+1,-1,+1,...]
    preamble_bipolar = 2.0 * _PREAMBLE_BITS - 1.0
    np.testing.assert_array_equal(stream[:8], preamble_bipolar)


def test_btle_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = btle_modulator(1_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_btle_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        btle_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        btle_modulator(6_000_000, 10_000_000, 4096)


def test_btle_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = BTLESignalGenerator(metadata=BTLE_METADATA, seed=1)()
    assert signal.class_name == "btle"
    assert len(signal.data) == BTLE_METADATA["signal_duration_in_samples_min"]


def test_btle_generator_reproducible():
    """Same seed yields identical IQ."""
    a = BTLESignalGenerator(metadata=BTLE_METADATA, seed=2).generate()
    b = BTLESignalGenerator(metadata=BTLE_METADATA, seed=2).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_btle_registered_and_in_signal_lists():
    """'btle' resolves through lookup and belongs to the 'bluetooth' family."""
    assert isinstance(lookup_signal_generator_by_string("btle"), BTLESignalGenerator)
    assert CLASS_FAMILY_DICT["btle"] == "bluetooth"
    lists = TorchSigSignalLists()
    assert "btle" in lists.bluetooth_signals
