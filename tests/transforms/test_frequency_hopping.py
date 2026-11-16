"""Tests for time-varying frequency shifting and experimental hopping."""

import numpy as np
import pytest

import torchsig.transforms.transforms as transforms_module
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import FrequencyHopping
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    frequency_hopping_offsets,
    frequency_shift,
    time_varying_frequency_shift,
)


def make_signal(data: np.ndarray, *, center_freq: float = 0.0, bandwidth: float = 100.0) -> Signal:
    """Create a signal with the frequency metadata required by the transform."""
    return Signal(
        data=data,
        sample_rate=1_000.0,
        center_freq=center_freq,
        bandwidth=bandwidth,
        class_name="bpsk",
        custom_metadata="preserved",
    )


def test_constant_time_varying_shift_matches_frequency_shift():
    rng = np.random.default_rng(2)
    data = (rng.normal(size=128) + 1j * rng.normal(size=128)).astype(np.complex128)
    offsets = np.full(len(data), 123.0)

    actual = time_varying_frequency_shift(data, offsets, 1_000.0)
    expected = frequency_shift(data, 123.0, 1_000.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.dtype == data.dtype


def test_tone_hopping_has_expected_instantaneous_frequency():
    offsets = np.repeat(np.array([-125.0, 75.0]), 16)
    tone = np.ones(len(offsets), dtype=TorchSigComplexDataType)

    hopped = time_varying_frequency_shift(tone, offsets, 1_000.0)
    measured = np.angle(hopped[1:] * np.conj(hopped[:-1])) * 1_000.0 / (2 * np.pi)

    np.testing.assert_allclose(measured, offsets[:-1], rtol=1e-5, atol=1e-5)


def test_frequency_hopping_preserves_bpsk_modulation_and_metadata():
    symbols = np.tile(np.array([1, -1, -1, 1], dtype=TorchSigComplexDataType), 64)
    original = make_signal(symbols.copy(), center_freq=25.0, bandwidth=100.0)
    expected_offsets = frequency_hopping_offsets(len(symbols), 400.0, np.random.default_rng(17))

    hopped = FrequencyHopping(400.0, seed=17)(original)
    recovered = time_varying_frequency_shift(hopped.data, -expected_offsets, hopped.sample_rate)

    np.testing.assert_allclose(recovered, symbols, rtol=2e-6, atol=2e-6)
    assert hopped.data.shape == symbols.shape
    assert hopped.data.dtype == symbols.dtype
    assert hopped.class_name == "bpsk"
    assert hopped.custom_metadata == "preserved"
    expected_lower = 25.0 - 50.0 + np.min(expected_offsets)
    expected_upper = 25.0 + 50.0 + np.max(expected_offsets)
    assert hopped.center_freq == pytest.approx((expected_lower + expected_upper) / 2)
    assert hopped.bandwidth == pytest.approx(expected_upper - expected_lower)


def test_frequency_hopping_is_reproducible_and_phase_continuous():
    first = FrequencyHopping(300.0, seed=8)(make_signal(np.ones(256, dtype=np.complex128)))
    second = FrequencyHopping(300.0, seed=8)(make_signal(np.ones(256, dtype=np.complex128)))
    offsets = frequency_hopping_offsets(256, 300.0, np.random.default_rng(8))

    np.testing.assert_array_equal(first.data, second.data)
    measured_steps = np.angle(first.data[1:] * np.conj(first.data[:-1]))
    np.testing.assert_allclose(measured_steps, 2 * np.pi * offsets[:-1] / 1_000.0, atol=1e-12)


@pytest.mark.parametrize(
    "symbols",
    [
        np.array([1, -1, 1, 1, -1], dtype=np.complex64),
        np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2),
    ],
    ids=["bpsk", "qpsk"],
)
def test_inverse_hopping_recovers_deterministic_modulations(symbols):
    data = np.repeat(symbols, 7).astype(np.complex64)
    offsets = np.concatenate(
        (
            np.full(8, -175.0),
            np.full(13, 60.0),
            np.full(len(data) - 21, 140.0),
        )
    )

    hopped = time_varying_frequency_shift(data, offsets, 1_000.0)
    recovered = time_varying_frequency_shift(hopped, -offsets, 1_000.0)
    max_error = np.max(np.abs(data - recovered))

    assert max_error < 2e-6
    assert recovered.shape == data.shape
    assert recovered.dtype == data.dtype


def test_hop_boundary_phase_uses_previous_offset_without_reset():
    offsets = np.array([100.0, 100.0, -200.0, -200.0])
    phasor = time_varying_frequency_shift(np.ones(4, dtype=np.complex128), offsets, 1_000.0)

    expected_boundary_sample = np.exp(2j * np.pi * (100.0 + 100.0) / 1_000.0)
    assert phasor[2] == pytest.approx(expected_boundary_sample)
    assert np.angle(phasor[2] * np.conj(phasor[1])) == pytest.approx(2 * np.pi * offsets[1] / 1_000.0)
    assert np.angle(phasor[3] * np.conj(phasor[2])) == pytest.approx(2 * np.pi * offsets[2] / 1_000.0)


def test_frequency_hopping_offsets_use_approximately_equal_unequal_intervals():
    offsets = frequency_hopping_offsets(10, 400.0, np.random.default_rng(5))
    _, counts = np.unique(offsets, return_counts=True)

    assert np.max(counts) - np.min(counts) == 1


def test_frequency_hopping_supports_one_sample():
    signal = make_signal(np.array([1j], dtype=TorchSigComplexDataType), bandwidth=0.0)

    hopped = FrequencyHopping(100.0, seed=4)(signal)

    np.testing.assert_array_equal(hopped.data, np.array([1j], dtype=TorchSigComplexDataType))
    assert hopped.bandwidth == 0.0


@pytest.mark.parametrize(
    ("data", "offsets", "sample_rate", "error", "message"),
    [
        (np.ones((2, 2), dtype=np.complex64), np.ones(4), 1.0, ValueError, "signal must be one dimensional"),
        (np.ones(2), np.ones(2), 1.0, TypeError, "complex IQ"),
        (np.ones(2, dtype=np.complex64), np.ones((1, 2)), 1.0, ValueError, "frequencies must be one dimensional"),
        (np.ones(2, dtype=np.complex64), np.ones(1), 1.0, ValueError, "one value per signal sample"),
        (np.ones(2, dtype=np.complex64), np.ones(2), 0.0, ValueError, "positive and finite"),
        (np.ones(2, dtype=np.complex64), np.array([0.0, np.nan]), 1.0, ValueError, "only finite"),
    ],
)
def test_time_varying_frequency_shift_rejects_invalid_inputs(data, offsets, sample_rate, error, message):
    with pytest.raises(error, match=message):
        time_varying_frequency_shift(data, offsets, sample_rate)


@pytest.mark.parametrize(
    ("offsets", "expected_center", "expected_bandwidth"),
    [
        (np.array([100.0, 200.0]), 175.0, 200.0),
        (np.array([-200.0, -100.0]), -125.0, 200.0),
        (np.array([-150.0, 225.0]), 62.5, 475.0),
    ],
    ids=["positive", "negative", "mixed"],
)
def test_frequency_hopping_metadata_uses_actual_offsets(
    monkeypatch,
    offsets,
    expected_center,
    expected_bandwidth,
):
    expanded_offsets = np.repeat(offsets, 16)
    monkeypatch.setattr(transforms_module, "frequency_hopping_offsets", lambda *_: expanded_offsets)
    signal = make_signal(np.ones(32, dtype=np.complex64), center_freq=25.0, bandwidth=100.0)

    hopped = FrequencyHopping(600.0, seed=3)(signal)

    assert hopped.center_freq == pytest.approx(expected_center)
    assert hopped.bandwidth == pytest.approx(expected_bandwidth)


def test_frequency_hopping_allows_nyquist_boundary_and_rejects_overflow(monkeypatch):
    boundary_offsets = np.repeat(np.array([-200.0, 200.0]), 16)
    monkeypatch.setattr(transforms_module, "frequency_hopping_offsets", lambda *_: boundary_offsets)
    boundary = make_signal(np.ones(32, dtype=np.complex64), bandwidth=600.0)
    FrequencyHopping(400.0, seed=3)(boundary)

    overflow_offsets = np.repeat(np.array([-200.0, 200.01]), 16)
    monkeypatch.setattr(transforms_module, "frequency_hopping_offsets", lambda *_: overflow_offsets)
    overflow = make_signal(np.ones(32, dtype=np.complex64), bandwidth=600.0)
    with pytest.raises(ValueError, match="beyond Nyquist"):
        FrequencyHopping(400.0, seed=3)(overflow)
