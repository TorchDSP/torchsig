"""Unit tests for Cellular type signal builders."""

import numpy as np
import pytest

from torchsig.signals.builders.cellular import (
    GSM_TSC,
    GSMSignalGenerator,
    build_gsm_burst,
    gsm_modulator,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

# --------------------------------------------------------------------------- #
# GSM
# --------------------------------------------------------------------------- #
GSM_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_500_000,
    "bandwidth_max": 2_500_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_gsm_tsc_count_and_length():
    """There are 8 TSCs, each 26 bits."""
    assert len(GSM_TSC) == 8
    assert all(len(t) == 26 for t in GSM_TSC)


def test_gsm_burst_length_and_values():
    """A GSM burst is 157 bipolar symbols containing the TSC in the middle."""
    rng = np.random.default_rng(0)
    burst = build_gsm_burst(0, rng)
    assert len(burst) == 157
    assert set(np.unique(burst)).issubset({-1.0, 1.0})
    # TSC starts at index 61: 3 tail + 57 data + 1 steal flag = 61
    tsc_bipolar = np.array([2 * b - 1 for b in GSM_TSC[0]], dtype=float)
    np.testing.assert_array_equal(burst[61:87], tsc_bipolar)


def test_gsm_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = gsm_modulator(2_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_gsm_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        gsm_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        gsm_modulator(6_000_000, 10_000_000, 4096)


def test_gsm_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = GSMSignalGenerator(metadata=GSM_METADATA, seed=1)()
    assert signal.class_name == "gsm"
    assert len(signal.data) == GSM_METADATA["signal_duration_in_samples_min"]


def test_gsm_generator_reproducible():
    """Same seed yields identical IQ."""
    a = GSMSignalGenerator(metadata=GSM_METADATA, seed=4).generate()
    b = GSMSignalGenerator(metadata=GSM_METADATA, seed=4).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_gsm_registered_and_in_signal_lists():
    """'gsm' resolves through lookup and is in the 'cellular' family."""
    assert isinstance(lookup_signal_generator_by_string("gsm"), GSMSignalGenerator)
    assert CLASS_FAMILY_DICT["gsm"] == "cellular"
    lists = TorchSigSignalLists()
    assert "gsm" in lists.cellular_signals
