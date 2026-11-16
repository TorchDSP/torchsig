"""Tests for the experimental frequency-hopping signal prototype."""

import numpy as np
import pytest

from torchsig.signals.builders import __all__ as builder_exports
from torchsig.signals.builders.fhss import (
    FrequencyHoppingSignalGenerator,
    frequency_hopping_modulator,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, SIGNALS_SHARED_LIST, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import signal_generator_lookup_table


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "bandwidth", "message"),
    [
        (0, 100.0, 20.0, "num_samples must be positive"),
        (-1, 100.0, 20.0, "num_samples must be positive"),
        (16, 0.0, 20.0, "sample_rate must be positive"),
        (16, -1.0, 20.0, "sample_rate must be positive"),
        (16, 100.0, 0.0, "bandwidth must be positive"),
        (16, 100.0, -1.0, "bandwidth must be positive"),
        (16, 100.0, 101.0, "bandwidth must not exceed sample_rate"),
    ],
)
def test_frequency_hopping_modulator_rejects_invalid_inputs(
    num_samples,
    sample_rate,
    bandwidth,
    message,
):
    with pytest.raises(ValueError, match=message):
        frequency_hopping_modulator(num_samples, sample_rate, bandwidth)


def test_frequency_hopping_modulator_shape_dtype_and_magnitude():
    data = frequency_hopping_modulator(
        257,
        1_000.0,
        400.0,
        np.random.default_rng(4),
    )

    assert data.shape == (257,)
    assert data.dtype == np.dtype(TorchSigComplexDataType)
    np.testing.assert_allclose(np.abs(data), 1.0, rtol=1e-6, atol=1e-6)


def test_frequency_hopping_modulator_is_seeded_phase_continuous_and_hops():
    first = frequency_hopping_modulator(512, 1_000.0, 400.0, np.random.default_rng(9))
    second = frequency_hopping_modulator(512, 1_000.0, 400.0, np.random.default_rng(9))

    np.testing.assert_array_equal(first, second)

    expected_rng = np.random.default_rng(9)
    num_hops = int(expected_rng.integers(2, 17))
    hop_frequencies = expected_rng.uniform(-200.0, 200.0, size=num_hops)
    hop_index = np.minimum(np.arange(512) * num_hops // 512, num_hops - 1)
    phase_steps = np.diff(np.unwrap(np.angle(first)))
    expected_phase_steps = 2 * np.pi * hop_frequencies[hop_index][:-1] / 1_000.0

    np.testing.assert_allclose(phase_steps, expected_phase_steps, rtol=1e-5, atol=1e-5)
    assert np.unique(np.round(phase_steps, decimals=5)).size > 1


def test_frequency_hopping_signal_generator_uses_standard_metadata():
    generator = FrequencyHoppingSignalGenerator(
        metadata={
            "sample_rate": 1_000.0,
            "bandwidth_min": 200.0,
            "bandwidth_max": 400.0,
            "signal_duration_in_samples_min": 128,
            "signal_duration_in_samples_max": 256,
        },
        seed=3,
    )

    generator.validate_metadata_fields()
    signal = generator()

    assert signal.class_name == "fhss"
    assert 128 <= len(signal.data) <= 256
    assert signal.data.dtype == np.dtype(TorchSigComplexDataType)
    assert 0.0 <= signal.bandwidth <= 400.0
    assert abs(signal.center_freq) <= 200.0


def test_experimental_fhss_is_not_registered_or_listed():
    signal_lists = TorchSigSignalLists()

    assert "fhss" not in builder_exports
    assert "fhss" not in CLASS_FAMILY_DICT
    assert "fhss" not in SIGNALS_SHARED_LIST
    assert "fhss" not in signal_lists.all_signals
    assert "fhss" not in signal_generator_lookup_table
    assert all(spec[0] is not FrequencyHoppingSignalGenerator for spec in signal_generator_lookup_table["all"][1])
