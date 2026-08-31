"""Tests for the DSP utilities."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from torchsig.utils import dsp
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    compute_spectrogram,
)

FFT_SIZE = 64
ZEROS = np.zeros(512, dtype=TorchSigComplexDataType)


def _tone(bin_index, num_samples=512, fft_size=FFT_SIZE, amplitude=1.0):
    """Complex exponential centered exactly on FFT bin ``bin_index``."""
    n = np.arange(num_samples)
    tone = amplitude * np.exp(2j * np.pi * bin_index * n / fft_size)
    return tone.astype(TorchSigComplexDataType)


# compute_spectrogram tests
@pytest.mark.parametrize(
    "samples,fft_size,fft_stride",
    [
        (ZEROS, 0, FFT_SIZE),
        (ZEROS, -FFT_SIZE, FFT_SIZE),
        (ZEROS, FFT_SIZE, 0),
        (ZEROS, FFT_SIZE, -1),
        (ZEROS.reshape(2, -1), FFT_SIZE, FFT_SIZE),
    ],
)
def test_compute_spectrogram_invalid_arguments_raise(samples, fft_size, fft_stride):
    """Non-positive sizes and non-1D input raise ValueError."""
    with pytest.raises(ValueError):
        compute_spectrogram(samples, fft_size, fft_stride)


@pytest.mark.parametrize(
    "num_samples,fft_stride,expected_frames",
    [
        (1024, FFT_SIZE, 16),  # no overlap
        (1024, FFT_SIZE // 2, 31),  # 50% overlap
        (1024, 2 * FFT_SIZE, 8),  # stride > fft_size, subset sampling
        (100, FFT_SIZE, 1),  # trailing partial frame discarded
        (FFT_SIZE // 4, FFT_SIZE, 1),  # short input, zero padded
    ],
)
def test_compute_spectrogram_output_geometry(num_samples, fft_stride, expected_frames):
    """Shape is (fft_size, 1 + (num_samples - fft_size) // fft_stride), float32."""
    spec = compute_spectrogram(_tone(3, num_samples), FFT_SIZE, fft_stride)
    assert spec.shape == (FFT_SIZE, expected_frames)
    assert spec.dtype == np.float32


@pytest.mark.parametrize("bin_index", [0, 1, -1, 7, FFT_SIZE // 2 - 1, -FFT_SIZE // 2])
def test_compute_spectrogram_frequency_maps_to_descending_rows(bin_index):
    """A tone on bin k peaks at row fft_size // 2 - 1 - k, in every frame."""
    spec = compute_spectrogram(_tone(bin_index), FFT_SIZE, FFT_SIZE)
    expected_row = FFT_SIZE // 2 - 1 - bin_index
    np.testing.assert_array_equal(np.argmax(spec, axis=0), np.full(spec.shape[1], expected_row))


def test_compute_spectrogram_output_is_contiguous_and_torch_safe():
    """No negative strides, so torch.from_numpy accepts the result."""
    spec = compute_spectrogram(_tone(5), FFT_SIZE, FFT_SIZE)
    assert spec.flags["C_CONTIGUOUS"]
    assert all(stride > 0 for stride in spec.strides)
    torch.from_numpy(spec)  # raises on negative strides


def test_compute_spectrogram_zero_power_bins_are_floored():
    """Exactly-zero bins clamp to peak - 100 dB rather than -inf."""
    silent = np.zeros(256, dtype=TorchSigComplexDataType)
    spec = compute_spectrogram(np.concatenate([_tone(4, 256), silent]), FFT_SIZE, FFT_SIZE)
    assert np.all(np.isfinite(spec))
    np.testing.assert_allclose(spec.min(), spec.max() - 100.0, atol=1e-3)


def test_compute_spectrogram_all_zero_input_is_finite():
    """Degenerate all-zero input yields a finite constant, no divide-by-zero."""
    with np.errstate(divide="raise", invalid="raise"):
        spec = compute_spectrogram(ZEROS, FFT_SIZE, FFT_SIZE)
    assert np.all(np.isfinite(spec))
    assert np.all(spec == spec.flat[0])


def test_compute_spectrogram_values_are_power_db():
    """Doubling amplitude raises the peak by 10*log10(4), not 20*log10(4)."""
    quiet = compute_spectrogram(_tone(5), FFT_SIZE, FFT_SIZE)
    loud = compute_spectrogram(_tone(5, amplitude=2.0), FFT_SIZE, FFT_SIZE)
    np.testing.assert_allclose(loud.max() - quiet.max(), 10.0 * np.log10(4.0), atol=1e-3)


def test_compute_spectrogram_input_not_modified():
    """The caller's array is never written to, despite the copy-free asarray."""
    samples = _tone(3)
    original = samples.copy()
    compute_spectrogram(samples, FFT_SIZE, FFT_SIZE)
    np.testing.assert_array_equal(samples, original)


PROPAGATION_SPEED = 2.9979e8


def _actual_fractional_rate(monkeypatch, requested_rate: float) -> float:
    """Capture the integer rate selected by the fractional resampler."""
    selected_rate = None

    def fake_upfirdn(weights, data, up, down):
        nonlocal selected_rate
        selected_rate = up / down
        return np.zeros(data.size + len(weights), dtype=np.complex64)

    monkeypatch.setattr(
        dsp,
        "prototype_polyphase_filter_interpolation",
        lambda _num_branches, dtype=np.float64: np.ones(20_001, dtype=dtype),
    )
    monkeypatch.setattr(dsp.sp, "upfirdn", fake_upfirdn)

    dsp.polyphase_fractional_resampler(
        np.ones(16, dtype=np.complex64),
        requested_rate,
    )

    return selected_rate


def test_fractional_resampler_preserves_near_unity_rate(monkeypatch) -> None:
    """A small Doppler shift should not be quantized into a larger shift."""
    velocity = 10.0
    alpha = PROPAGATION_SPEED / (PROPAGATION_SPEED - velocity)
    requested_rate = 1 / alpha

    actual_rate = _actual_fractional_rate(monkeypatch, requested_rate)

    assert actual_rate == pytest.approx(requested_rate, rel=1e-6)


def test_fractional_resampler_is_symmetric_around_unity(monkeypatch) -> None:
    """Equal approaching and receding speeds should have symmetric errors."""
    velocity = 10.0
    approaching_alpha = PROPAGATION_SPEED / (PROPAGATION_SPEED - velocity)
    receding_alpha = PROPAGATION_SPEED / (PROPAGATION_SPEED + velocity)

    approaching_rate = _actual_fractional_rate(
        monkeypatch,
        1 / approaching_alpha,
    )
    receding_rate = _actual_fractional_rate(
        monkeypatch,
        1 / receding_alpha,
    )

    assert abs(approaching_rate - 1.0) == pytest.approx(
        abs(receding_rate - 1.0),
        rel=1e-2,
        abs=1e-8,
    )


def test_prototype_polyphase_filter_uses_in_memory_cache(monkeypatch) -> None:
    """Repeated filter requests should not reload identical weights from disk."""
    load = Mock(return_value=np.ones(8, dtype=np.float32))
    monkeypatch.setattr(dsp.np, "load", load)
    monkeypatch.setattr(dsp.Path, "is_file", lambda _path: True)

    first = dsp.prototype_polyphase_filter(num_branches=12_345)
    second = dsp.prototype_polyphase_filter(num_branches=12_345)

    np.testing.assert_array_equal(first, second)
    assert load.call_count == 1
    assert first is not second
    assert first.flags.writeable


@pytest.mark.parametrize(
    ("filter_function", "scale"),
    [
        (dsp.prototype_polyphase_filter_interpolation, 7.0),
        (dsp.prototype_polyphase_filter_decimation, 1 / 7),
    ],
)
def test_finalized_polyphase_filters_are_cached_and_immutable(monkeypatch, filter_function, scale: float) -> None:
    """Scaled filters should be constructed once and safely reused."""
    base_weights = np.arange(1, 9, dtype=np.float32)
    get_base_weights = Mock(return_value=base_weights)
    monkeypatch.setattr(dsp, "_prototype_polyphase_filter_cached", get_base_weights)
    filter_function.cache_clear()

    first = filter_function(num_branches=7, attenuation_db=91)
    second = filter_function(num_branches=7, attenuation_db=91)

    assert first is second
    assert not first.flags.writeable
    np.testing.assert_allclose(first, base_weights * scale)
    np.testing.assert_array_equal(base_weights, np.arange(1, 9, dtype=np.float32))
    get_base_weights.assert_called_once_with(7, 91)
    filter_function.cache_clear()


@pytest.mark.parametrize(
    "filter_function",
    [
        dsp.prototype_polyphase_filter_interpolation,
        dsp.prototype_polyphase_filter_decimation,
    ],
)
def test_finalized_polyphase_filter_cache_distinguishes_dtype(monkeypatch, filter_function) -> None:
    """Finalized float32 and float64 coefficients should be cached separately."""
    monkeypatch.setattr(
        dsp,
        "_prototype_polyphase_filter_cached",
        Mock(return_value=np.arange(1, 9, dtype=np.float64)),
    )
    filter_function.cache_clear()

    weights32 = filter_function(7, dtype=np.float32)
    weights64 = filter_function(7, dtype=np.float64)

    assert weights32.dtype == np.dtype(np.float32)
    assert weights64.dtype == np.dtype(np.float64)
    assert weights32 is filter_function(7, dtype=np.float32)
    assert weights64 is filter_function(7, dtype=np.float64)
    filter_function.cache_clear()


@pytest.mark.parametrize(
    ("resampler", "rate"),
    [
        (dsp.polyphase_fractional_resampler, 1.01),
        (dsp.polyphase_integer_interpolator, 2),
        (dsp.polyphase_decimator, 2),
    ],
)
def test_polyphase_resamplers_preserve_complex64_dtype(resampler, rate) -> None:
    """Float32 filter coefficients should prevent promotion to complex128."""
    result = resampler(np.ones(32, dtype=np.complex64), rate)
    assert result.dtype == np.dtype(np.complex64)


@pytest.mark.parametrize("rate", [0.91, 0.9967, 1.0033, 1.1])
def test_multistage_polyphase_resampler_preserves_precision(rate: float) -> None:
    """Float32 processing should remain close to the float64 reference."""
    rng = np.random.default_rng(0)
    data64 = rng.standard_normal(1_024) + 1j * rng.standard_normal(1_024)
    result32 = dsp.multistage_polyphase_resampler(data64.astype(np.complex64), rate)
    result64 = dsp.multistage_polyphase_resampler(data64, rate)

    assert result32.dtype == np.dtype(np.complex64)
    assert result64.dtype == np.dtype(np.complex128)
    assert result32.shape == result64.shape
    np.testing.assert_allclose(result32, result64, rtol=1e-5, atol=1e-6)


def test_float32_polyphase_filter_preserves_spectral_performance() -> None:
    """Coefficient quantization should preserve transition-edge performance."""
    num_branches = 32
    frequencies = None
    responses_db = {}
    for dtype in (np.float32, np.float64):
        weights = dsp.prototype_polyphase_filter_interpolation(num_branches, dtype=dtype).astype(np.float64)
        frequencies, response = dsp.sp.freqz(weights / num_branches, worN=131_072, fs=1.0)
        responses_db[dtype] = 20 * np.log10(np.maximum(np.abs(response), np.finfo(np.float64).tiny))

    passband_edge = 1 / (4 * num_branches)
    stopband_edge = 3 / (4 * num_branches)
    response32_db = responses_db[np.float32]
    assert np.ptp(response32_db[frequencies <= passband_edge]) < 1e-3
    assert np.max(response32_db[frequencies >= stopband_edge]) < -119.5
    for edge in (passband_edge, stopband_edge):
        edge_index = int(np.argmin(np.abs(frequencies - edge)))
        assert response32_db[edge_index] == pytest.approx(responses_db[np.float64][edge_index], abs=0.1)
