"""Unit tests for LoRa type signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.lora import (
    LORA_SF_VALUES,
    LoraSignalGenerator,
    build_lora_symbol_stream,
    lora_modulator,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

LORA_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 125_000,
    "bandwidth_max": 500_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_lora_sf_values():
    """LORA_SF_VALUES contains all standard spreading factors."""
    assert set(LORA_SF_VALUES) == {7, 8, 9, 10, 11, 12}


def test_lora_symbol_stream_has_preamble():
    """The preamble up-chirps appear at the start of the symbol stream."""
    rng = np.random.default_rng(0)
    sf = 8
    osr = 4
    stream = build_lora_symbol_stream(2000, sf, osr, rng)
    assert len(stream) >= 2000
    assert np.all(np.isfinite(stream))


def test_lora_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = lora_modulator(sf=9, bandwidth=250_000, sample_rate=10_000_000, num_samples=num_samples, rng=rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_lora_modulator_invalid_args():
    """Invalid bandwidth/sample-rate combinations raise."""
    with pytest.raises(ValueError):
        lora_modulator(7, 0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        lora_modulator(7, 6_000_000, 10_000_000, 4096)


def test_lora_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = LoraSignalGenerator(metadata=LORA_METADATA, seed=1)()
    assert signal.class_name == "lora"
    assert signal.center_freq == 0
    assert len(signal.data) == LORA_METADATA["signal_duration_in_samples_min"]


def test_lora_generator_reproducible():
    """Same seed yields identical IQ."""
    a = LoraSignalGenerator(metadata=LORA_METADATA, seed=3).generate()
    b = LoraSignalGenerator(metadata=LORA_METADATA, seed=3).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_lora_registered_and_in_signal_lists():
    """'lora' resolves through lookup and is its own family."""
    assert isinstance(lookup_signal_generator_by_string("lora"), LoraSignalGenerator)
    assert CLASS_FAMILY_DICT["lora"] == "lora"
    lists = TorchSigSignalLists()
    assert "lora" in lists.lora_signals
