"""Unit Tests: transforms/functional"""

from typing import Any

import numpy as np
import pytest
import scipy as sp
from numba.core.errors import TypingError
from numpy.testing import assert_allclose, assert_array_equal
from test_transforms_utils import generate_test_signal, generate_tone_signal

from torchsig.transforms.functional import (
    _build_full_profile,
    _fft_filter,
    add_slope,
    additive_noise,
    adjacent_channel_interference,
    awgn,
    carrier_frequency_drift,
    carrier_phase_noise,
    channel_swap,
    clock_jitter,
    coarse_gain_change,
    cochannel_interference,
    cut_out,
    digital_agc,
    doppler,
    drop_samples,
    fading,
    interleave_complex,
    intermodulation_products,
    iq_imbalance,
    nonlinear_amplifier,
    nonlinear_amplifier_table,
    normalize,
    passband_ripple,
    patch_shuffle,
    phase_offset,
    quantize,
    shadowing,
    spectral_inversion,
    spectrogram,
    spectrogram_drop_samples,
    spurs,
    time_reversal,
    time_varying_noise,
)
from torchsig.utils import dsp
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    TorchSigRealDataType,
    compute_spectrogram,
)

RTOL = 1e-6
TEST_DATA = generate_test_signal(num_iq_samples=8192, scale=1.0).data


@pytest.mark.parametrize("data, expected, is_error", [(0, ValueError, True), (TEST_DATA.copy(), True, False)])
def test_add_slope(data: Any, expected: bool | ValueError, is_error: bool) -> None:
    """Test the add_slope functional with pytest.

    Args:
        data (Any): Data input.
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = add_slope(data)
    else:
        data_test = data.copy()
        diff = np.diff(data_test)
        diff = np.insert(diff, 0, 0)
        data_test += diff

        data = add_slope(data)

        assert np.allclose(data, data_test, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize("velocity", [10.0, 1e6])
def test_doppler_preserves_length_for_long_inputs(velocity: float) -> None:
    """Doppler resampling should not shorten long input signals."""
    data = np.ones(65_536, dtype=TorchSigComplexDataType)
    assert doppler(data, velocity, 2.9979e8).shape == data.shape


def test_doppler_does_not_append_zeros_to_preserve_length() -> None:
    """Length preservation should retain resampled signal content at the tail."""
    result = doppler(np.ones(65_536, dtype=TorchSigComplexDataType), 10.0, 2.9979e8)
    assert np.count_nonzero(result[-5:]) == 5


@pytest.mark.parametrize(
    ("velocity", "propagation_speed"),
    [(2.9979e8, 2.9979e8), (2.9979e8 + 1, 2.9979e8), (np.nan, 2.9979e8), (1.0, 0.0)],
)
def test_doppler_rejects_invalid_physical_parameters(velocity: float, propagation_speed: float) -> None:
    """Invalid propagation speeds and velocities should raise a clear error."""
    with pytest.raises(ValueError, match="velocity|propagation_speed"):
        doppler(np.ones(16, dtype=TorchSigComplexDataType), velocity, propagation_speed)


def test_doppler_keeps_complex64_input_during_resampling(monkeypatch) -> None:
    """Approaching-velocity padding should not promote complex64 input."""
    resampler_input_dtype = None

    def capture_resampler(data: np.ndarray, _rate: float) -> np.ndarray:
        nonlocal resampler_input_dtype
        resampler_input_dtype = data.dtype
        return data

    monkeypatch.setattr(
        "torchsig.transforms.functional.multistage_polyphase_resampler",
        capture_resampler,
    )
    doppler(np.ones(32, dtype=TorchSigComplexDataType), 1e6, 2.9979e8)
    assert resampler_input_dtype == np.dtype(TorchSigComplexDataType)


def test_doppler_avoids_copy_when_resampler_returns_complex64(monkeypatch) -> None:
    """The final dtype normalization should reuse complex64 resampler output."""
    resampled = np.arange(32, dtype=TorchSigComplexDataType)
    monkeypatch.setattr(
        "torchsig.transforms.functional.multistage_polyphase_resampler",
        lambda _data, _rate: resampled,
    )
    result = doppler(resampled.copy(), -1e6, 2.9979e8)
    assert result.dtype == np.dtype(TorchSigComplexDataType)
    assert np.shares_memory(result, resampled)


def test_doppler_rejects_multidimensional_input() -> None:
    """Doppler should reject arrays without a single sample axis."""
    with pytest.raises(ValueError, match="one-dimensional"):
        doppler(np.ones((2, 16), dtype=TorchSigComplexDataType))


def test_doppler_skips_resampling_for_effective_unity_rate(monkeypatch) -> None:
    """An effective 1:1 rate should bypass polyphase filtering."""
    data = np.ones(32, dtype=TorchSigComplexDataType)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("resampler should not be called for a 1:1 rate")

    monkeypatch.setattr(
        "torchsig.transforms.functional.multistage_polyphase_resampler",
        fail_if_called,
    )
    result = doppler(data, 0.0, 2.9979e8)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA.copy(), {"noise_power": -42.0, "noise_color": "white", "continuous": False}, ValueError, True),
        (TEST_DATA.copy(), {"noise_power": 1.0, "noise_color": "purple", "continuous": False}, ValueError, True),
        (TEST_DATA.copy(), {"noise_power": 42.4, "noise_color": "white", "continuous": False}, True, False),
        (TEST_DATA.copy(), {"noise_power": 4.2, "noise_color": "white", "continuous": True}, True, False),
        (TEST_DATA.copy(), {"noise_power": 0.1, "noise_color": "pink", "continuous": True}, True, False),
        (TEST_DATA.copy(), {"noise_power": 3.14, "noise_color": "red", "continuous": True}, True, False),
    ],
)
def test_additive_noise(data: Any, params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the additive_noise functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    power = params["noise_power"]
    color = params["noise_color"]
    continuous = params["continuous"]

    if is_error:
        with pytest.raises(expected):
            data = additive_noise(data=data, power=power, color=color, continuous=continuous, rng=rng)
    else:
        data_test = data.copy()
        data = additive_noise(data=data, power=power, color=color, continuous=continuous, rng=rng)

        input_power = np.sum(np.abs(data_test) ** 2) / len(data_test)
        output_power = np.sum(np.abs(data) ** 2) / len(data)
        power_delta = output_power - input_power

        assert (len(data) == len(data_test)) == expected
        assert (np.abs(power_delta - power) < 10 ** (0.1 / 10)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "params, expected, is_error",
    [
        ({"N": 8192, "sample_rate": 4.0, "power": 0.5, "adj_fc": 0.2, "tone_freq": 0.042, "phase_sigma": 0.0, "time_sigma": 0.0, "filter_weights": dsp.low_pass(0.25, 0.25, 4.0)}, True, False),
        ({"N": 8192, "sample_rate": 2.5, "power": 0.25, "adj_fc": -0.12, "tone_freq": 0.1, "phase_sigma": 1.0, "time_sigma": 4.0, "filter_weights": dsp.low_pass(0.11, 0.18, 2.5)}, True, False),
    ],
)
def test_adjacent_channel_interference(params: dict, expected: bool, is_error: bool) -> None:
    """Test the adjacent_channel_interference functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    N = params["N"]
    sample_rate = params["sample_rate"]
    adj_power = params["power"]
    tone_freq = params["tone_freq"]
    center_frequency = params["adj_fc"]
    filter_weights = params["filter_weights"]
    phase_sigma = params["phase_sigma"]

    # tone signal: freq = tone_freq, power = 1.0 W
    tone_baseband = generate_tone_signal(num_iq_samples=N, scale=1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N) / sample_rate)  # *np.sqrt(N)

    if is_error:
        with pytest.raises(expected):
            data = adjacent_channel_interference(
                data=data, sample_rate=sample_rate, power=adj_power, center_frequency=center_frequency, filter_weights=filter_weights, phase_sigma=phase_sigma, rng=rng
            )
    else:
        data_test = data.copy()
        data = adjacent_channel_interference(data=data, sample_rate=sample_rate, power=adj_power, center_frequency=center_frequency, filter_weights=filter_weights, phase_sigma=phase_sigma, rng=rng)

        est_power = np.sum(np.abs(data) ** 2) / len(data)

        D = np.abs(np.fft.fft(data, norm="ortho"))
        freqs = np.fft.fftfreq(N) * sample_rate
        peaks, _ = sp.signal.find_peaks(D, height=1.0, distance=N / 20)
        top_two_indices = np.argsort(D[peaks])[-2:][::-1]
        freqs0 = freqs[peaks[top_two_indices[0]]]
        freqs1 = freqs[peaks[top_two_indices[1]]]

        assert (np.abs(est_power - (adj_power + 1.0)) < 10 ** (0.1 / 10)) == expected
        assert (np.abs(freqs0 - tone_freq) < (3 / N)) == expected
        assert (np.abs(freqs1 - (tone_freq + center_frequency)) < 0.01) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error", [(0, {"noise_power_db": 0.0}, AttributeError, True), (np.zeros(1024, dtype=TorchSigComplexDataType), {"noise_power_db": 3.0}, True, False)]
)
def test_awgn(data: Any, params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the awgn functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)
    noise_power_db = params["noise_power_db"]

    if is_error:
        with pytest.raises(expected):
            data = awgn(data, noise_power_db=noise_power_db, rng=rng)
    else:
        noise_power_linear = 10 ** (noise_power_db / 10.0)
        data_test = data.copy()

        data = awgn(data, noise_power_db=noise_power_db, rng=rng)
        power_est = np.mean(np.abs(data) ** 2)

        assert (abs(power_est - noise_power_linear) < 1e-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (generate_tone_signal(num_iq_samples=8192, scale=1.0).data, {"drift_ppm": 0.1}, True, False),
        (generate_tone_signal(num_iq_samples=8192, scale=1.0).data, {"drift_ppm": 1}, True, False),
    ],
)
def test_carrier_frequency_drift(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the carrier_frequency_drift functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    drift_ppm = params["drift_ppm"]

    if is_error:
        with pytest.raises(expected):
            data = carrier_frequency_drift(data=data, drift_ppm=drift_ppm, rng=rng)
    else:
        data_test = data.copy()
        data = carrier_frequency_drift(data=data, drift_ppm=drift_ppm, rng=rng)

        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA.copy(), {"phase_noise_degrees": 1}, True, False),
    ],
)
def test_carrier_phase_noise(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the carrier_phase_noise functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    phase_noise_degrees = params["phase_noise_degrees"]

    if is_error:
        with pytest.raises(expected):
            data = carrier_phase_noise(data=data, phase_noise_degrees=phase_noise_degrees, rng=rng)
    else:
        data_test = data.copy()
        data = carrier_phase_noise(data=data, phase_noise_degrees=phase_noise_degrees, rng=rng)

        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize("data, expected, is_error", [(0, AttributeError, True), (TEST_DATA.copy(), True, False)])
def test_channel_swap(data: Any, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the channel_swap functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = channel_swap(data)
    else:
        data_test = data.copy()
        test_real = data_test.real
        test_imag = data_test.imag

        data = channel_swap(data)

        assert np.allclose(data.real, test_imag, RTOL) == expected
        assert np.allclose(data.imag, test_real, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [(TEST_DATA.copy(), {"jitter_ppm": 50}, True, False)])
def test_clock_jitter(data: Any, params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the clock_jitter functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    jitter_ppm = params["jitter_ppm"]

    if is_error:
        with pytest.raises(expected):
            data = clock_jitter(data=data, jitter_ppm=jitter_ppm, rng=rng)
    else:
        data_test = data.copy()

        data = clock_jitter(data=data, jitter_ppm=jitter_ppm, rng=rng)

        assert (type(data) == type(data_test)) == expected
        assert (data != data_test).any() == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error", [(TEST_DATA.copy(), {"start_idx": 5, "gain_change_db": 0.25}, True, False), (TEST_DATA.copy(), {"start_idx": -17, "gain_change_db": -15.7}, True, False)]
)
def test_coarse_gain_change(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the coarse_gain_change functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    start_idx = params["start_idx"]
    gain_change_db = params["gain_change_db"]

    if is_error:
        with pytest.raises(expected):
            data = coarse_gain_change(data=data, gain_change_db=gain_change_db, start_idx=start_idx)
    else:
        data_test = data.copy()

        data = coarse_gain_change(data=data, gain_change_db=gain_change_db, start_idx=start_idx)

        gain_change_linear = 10 ** (gain_change_db / 20)
        assert (np.allclose(data[start_idx:], gain_change_linear * data_test[start_idx:], RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "params, expected, is_error",
    [
        ({"N": 8192, "sample_rate": 4.0, "power": 0.1, "tone_freq": 0.2, "filter_weights": dsp.low_pass(0.25, 0.25, 4.0), "color": "white", "continuous": True}, True, False),
        ({"N": 16384, "sample_rate": 2.42, "power": 0.01, "tone_freq": -0.04, "filter_weights": dsp.low_pass(0.1, 0.15, 2.42), "color": "white", "continuous": True}, True, False),
    ],
)
def test_cochannel_interference(params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the cochannel_interference functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    N = params["N"]
    sample_rate = params["sample_rate"]
    intf_power = params["power"]
    tone_freq = params["tone_freq"]
    filter_weights = params["filter_weights"]
    color = params["color"]
    continuous = params["continuous"]

    # tone signal: freq = tone_freq, power = 1.0 W
    tone_baseband = generate_tone_signal(num_iq_samples=N, scale=1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N) / sample_rate)  # *np.sqrt(N)

    if is_error:
        with pytest.raises(expected):
            data = cochannel_interference(data=data, power=intf_power, filter_weights=filter_weights, color=color, continuous=continuous)
    else:
        data_test = data.copy()
        data = cochannel_interference(data=data, power=intf_power, filter_weights=filter_weights, color=color, continuous=continuous)

        est_power = np.sum(np.abs(data) ** 2) / len(data)

        D = np.abs(np.fft.fft(data, norm="ortho"))
        freqs = np.fft.fftfreq(N) * sample_rate
        peaks, _ = sp.signal.find_peaks(D, height=10.0, distance=N / 2)
        est_freq = freqs[peaks[0]]

        assert (np.abs(est_power - (intf_power + 1.0)) < 10 ** (0.1 / 10)) == expected
        assert (np.abs(est_freq - tone_freq) < (3 / N)) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (0, {"start": 0.25, "duration": 0.5, "cut_type": "zeros"}, AttributeError, True),
        (TEST_DATA.copy(), {"start": 0.25, "duration": 0.5, "cut_type": "invalid_cut_type"}, ValueError, True),
        (TEST_DATA.copy(), {"start": 0.25, "duration": 0.5, "cut_type": "zeros"}, True, False),
        (TEST_DATA.copy(), {"start": 0.0, "duration": 0.99, "cut_type": "ones"}, True, False),
        (TEST_DATA.copy(), {"start": 0.75, "duration": 0.1, "cut_type": "low_noise"}, True, False),
        (TEST_DATA.copy(), {"start": 0.90, "duration": 0.42, "cut_type": "avg_noise"}, True, False),
        (TEST_DATA.copy(), {"start": 0.5, "duration": 1.0, "cut_type": "high_noise"}, True, False),
    ],
)
def test_cut_out(data: Any, params: dict, expected: bool | AttributeError | ValueError, is_error: bool) -> None:
    """Test the cut_out functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)
    start = params["start"]
    duration = params["duration"]
    cut_type = params["cut_type"]

    if is_error:
        with pytest.raises(expected):
            data = cut_out(data, start, duration, cut_type, rng)
    else:
        data_test = data.copy()
        data = cut_out(data, start, duration, cut_type, rng)

        cut_inds = np.where(data != data)[0]
        duration_samples = int(duration * data.size)

        if np.any(cut_inds):
            assert duration_samples == cut_inds[-1] - cut_inds[0] + 1

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (
            0,
            {
                "initial_gain_db": 0.0,
                "alpha_smooth": 0.0,
                "alpha_track": np.log(0.1),
                "alpha_overflow": np.log(0.1),
                "alpha_acquire": np.log(0.1),
                "ref_level": 0.0,
                "ref_level_db": np.log(0.1),
                "track_range_db": np.log(0.1),
                "low_level_db": 0.0,
                "high_level_db": 0.0,
            },
            TypingError,
            True,
        ),
        (
            0.2 + generate_test_signal(num_iq_samples=8192, scale=0.01).data,
            {
                "initial_gain_db": 0.0,
                "alpha_smooth": 0.1,
                "alpha_track": np.log(1.1),
                "alpha_overflow": np.log(1.1),
                "alpha_acquire": np.log(1.1),
                "ref_level": 10.0,
                "ref_level_db": np.log(10.0),
                "track_range_db": np.log(4.0),
                "low_level_db": -200.0,
                "high_level_db": 200.0,
            },
            True,
            False,
        ),
    ],
)
def test_digital_agc(data: Any, params: dict, expected: bool | TypeError, is_error: bool) -> None:
    """Test the agc functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = digital_agc(
                data,
                initial_gain_db=params["initial_gain_db"],
                alpha_smooth=params["alpha_smooth"],
                alpha_track=params["alpha_track"],
                alpha_overflow=params["alpha_overflow"],
                alpha_acquire=params["alpha_acquire"],
                ref_level_db=params["ref_level_db"],
                track_range_db=params["track_range_db"],
                low_level_db=params["low_level_db"],
                high_level_db=params["high_level_db"],
            )
    else:
        reference_level = params["ref_level"]
        data_type = type(data)

        data = digital_agc(
            data,
            initial_gain_db=params["initial_gain_db"],
            alpha_smooth=params["alpha_smooth"],
            alpha_track=params["alpha_track"],
            alpha_overflow=params["alpha_overflow"],
            alpha_acquire=params["alpha_acquire"],
            ref_level_db=params["ref_level_db"],
            track_range_db=params["track_range_db"],
            low_level_db=params["low_level_db"],
            high_level_db=params["high_level_db"],
        )
        mean_level_est = np.round(np.mean(np.abs(data[-128:])))

        assert (abs(mean_level_est - reference_level) < 1e-1) == expected
        assert (type(data) == data_type) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "params, expected, is_error",
    [
        ({"N": 8192, "tone_freq": 0.2, "velocity": 1e7}, True, False),
        ({"N": 8192, "tone_freq": 0.2, "velocity": 1e6}, True, False),
    ],
)
def test_doppler(params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the doppler functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    N = params["N"]
    tone_freq = params["tone_freq"]
    velocity = params["velocity"]

    tone_baseband = generate_tone_signal(num_iq_samples=N, scale=1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N))

    if is_error:
        with pytest.raises(expected):
            data = doppler(data=data, velocity=velocity, propagation_speed=2.9979e8)
    else:
        data_test = data.copy()
        data = doppler(data=data, velocity=velocity, propagation_speed=2.9979e8)

        alpha = 2.9979e8 / (2.9979e8 - velocity)  # scaling factor
        D = np.abs(np.fft.fft(data, norm="ortho"))
        freqs = np.fft.fftfreq(N)
        peaks, _ = sp.signal.find_peaks(D, height=0.5, distance=N / 10)
        est_freq = freqs[peaks[0]]

        assert (np.abs(est_freq - tone_freq * alpha) < 0.01) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (0, {"drop_starts": [8], "drop_sizes": [2], "fill": "zero"}, TypeError, True),
        (TEST_DATA.copy(), {"drop_starts": [8], "drop_sizes": [2], "fill": "invalid_fill"}, ValueError, True),
        (TEST_DATA.copy(), {"drop_starts": [8], "drop_sizes": [2], "fill": "zero"}, True, False),
        (TEST_DATA.copy(), {"drop_starts": [2, 7], "drop_sizes": [2, 3], "fill": "mean"}, True, False),
        (TEST_DATA.copy(), {"drop_starts": [3, 11], "drop_sizes": [4, 3], "fill": "ffill"}, True, False),
        (TEST_DATA.copy(), {"drop_starts": [4], "drop_sizes": [10], "fill": "bfill"}, True, False),
    ],
)
def test_drop_samples(data: Any, params: dict, expected: bool | TypeError | ValueError, is_error: bool) -> None:
    """Test the drop_samples functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    drop_starts = params["drop_starts"]
    drop_sizes = params["drop_sizes"]
    fill = params["fill"]

    if is_error:
        with pytest.raises(expected):
            data = drop_samples(data, drop_starts, drop_sizes, fill)
    else:
        data_test = data.copy()
        data = drop_samples(data, drop_starts, drop_sizes, fill)

        drop_inds = np.empty(0, dtype=int)
        drop_stops = np.empty(0, dtype=int)
        for idx, drop_start in enumerate(drop_starts):
            drop_stops = np.append(drop_stops, drop_start + drop_sizes[idx])
            drop_inds = np.append(drop_inds, np.arange(drop_start, drop_stops[-1], dtype=int))

        if np.any(drop_inds):
            fill_inds = np.where(data != data_test)[0]
            assert np.allclose(drop_inds, fill_inds, RTOL) == expected

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (0, {"coherence_bandwidth": 0.1, "power_delay_profile": np.array([0.5, 0.25])}, IndexError, True),
        (TEST_DATA.copy(), {"coherence_bandwidth": 0.1, "power_delay_profile": np.array([0.5, 0.25])}, True, False),
    ],
)
def test_fading(data: Any, params: dict, expected: bool | IndexError, is_error: bool) -> None:
    """Test the fading functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    coherence_bandwidth = params["coherence_bandwidth"]
    power_delay_profile = params["power_delay_profile"]

    if is_error:
        with pytest.raises(expected):
            data = fading(data, coherence_bandwidth=coherence_bandwidth, power_delay_profile=power_delay_profile, rng=rng)
    else:
        data_test = data.copy()

        data = fading(data, coherence_bandwidth=coherence_bandwidth, power_delay_profile=power_delay_profile, rng=rng)

        data_mean_power = np.mean(np.abs(data) ** 2)
        data_test_mean_power = np.mean(np.abs(data_test) ** 2)
        assert (abs(data_mean_power - data_test_mean_power) < 1e-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA.copy(), {"coeffs": np.array([])}, IndexError, True),
        (TEST_DATA.copy(), {"coeffs": np.array([0.5, 1.0])}, ValueError, True),
        (TEST_DATA.copy(), {"coeffs": np.array([0.2, 0, 0.1])}, True, False),
    ],
)
def test_intermodulation_products(data: Any, params: dict, expected: bool | IndexError, is_error: bool) -> None:
    """Test the intermodulation_products functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    coeffs = params["coeffs"]

    if is_error:
        with pytest.raises(expected):
            data = intermodulation_products(data=data, coeffs=coeffs)
    else:
        data_test = data.copy()
        data = intermodulation_products(data=data, coeffs=coeffs)

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"amplitude_imbalance": 0.1, "phase_imbalance": -np.pi / 4, "dc_offset_db": 10, "dc_offset_phase_rads": -np.pi / 2}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"amplitude_imbalance": 3.4, "phase_imbalance": np.pi / 8, "dc_offset_db": 3, "dc_offset_phase_rads": np.pi / 8}, True, False),
    ],
)
def test_iq_imbalance(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the iq_imbalance functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    amplitude_imbalance = params["amplitude_imbalance"]
    amplitude_imbalance_linear = 10 ** (amplitude_imbalance / 10.0)
    phase_imbalance = params["phase_imbalance"]
    dc_offset_db = params["dc_offset_db"]
    dc_offset_phase_rads = params["dc_offset_phase_rads"]

    if is_error:
        with pytest.raises(expected):
            data = iq_imbalance(data, amplitude_imbalance=amplitude_imbalance, phase_imbalance=phase_imbalance, dc_offset_db=dc_offset_db, dc_offset_phase_rads=dc_offset_phase_rads)
    else:
        data_test = data.copy()

        data = iq_imbalance(data, amplitude_imbalance=amplitude_imbalance, phase_imbalance=phase_imbalance, dc_offset_db=dc_offset_db, dc_offset_phase_rads=dc_offset_phase_rads)

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA, {}, True, False),
    ],
)
def test_interleave_complex(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the interleave_complex functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = interleave_complex(data)
    else:
        data_test = data.copy()

        data = interleave_complex(data_test)

        assert (data.dtype == TorchSigRealDataType) == expected
        assert (len(data) == len(data_test) * 2) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA.copy(), {"gain": 1.0, "psat_backoff": 10.0, "phi_max": 0.02, "phi_slope": 0.1, "auto_scale": True}, True, False),
        (TEST_DATA.copy(), {"gain": 7.4, "psat_backoff": 3.0, "phi_max": -0.1, "phi_slope": 0.01, "auto_scale": False}, True, False),
    ],
)
def test_nonlinear_amplifier(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the nonlinear_amplifier functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    gain = params["gain"]
    psat_backoff = params["psat_backoff"]
    phi_max = params["phi_max"]
    phi_slope = params["phi_slope"]
    auto_scale = params["auto_scale"]

    if is_error:
        with pytest.raises(expected):
            data = nonlinear_amplifier(data=data, gain=gain, psat_backoff=psat_backoff, phi_max=phi_max, phi_slope=phi_slope, auto_scale=auto_scale)
    else:
        data_test = data.copy()

        data = nonlinear_amplifier(data=data, gain=gain, psat_backoff=psat_backoff, phi_max=phi_max, phi_slope=phi_slope, auto_scale=auto_scale)

        input_power = np.mean(np.abs(data_test) ** 2)
        output_power = np.mean(np.abs(data) ** 2)
        psat = input_power * psat_backoff
        input_phase_rad = np.angle(data_test)
        output_phase_rad = np.angle(data)
        phase_diff = abs(np.mean(np.unwrap(output_phase_rad - input_phase_rad)))

        if auto_scale:
            assert (abs(output_power - input_power) < 10 ** (0.1 / 10)) == expected
        else:
            assert (np.all(output_power <= psat)) == expected
        assert (phase_diff <= (abs(phi_max) + RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (np.zeros((2,)), {"p_in": np.zeros((3,)), "p_out": np.zeros((4,)), "phi": np.zeros((5,)), "p_ratio": 0.0, "phase_shift": 0.0}, ValueError, True),
        (
            TEST_DATA.copy(),
            {
                "p_in": 10 ** ((np.array([-100.0, -50.0, 0.0, 50.0])) / 10),
                "p_out": 10 ** ((np.array([-97.0, -47.0, 3.0, 53.0])) / 10),
                "phi": np.deg2rad(np.array([0.1, 0.1, 0.1, 0.1])),
                "p_ratio": 10 ** (3.0 / 10),
                "phase_shift": np.deg2rad(0.1),
            },
            True,
            False,
        ),
    ],
)
def test_nonlinear_amplifier_table(data: Any, params: dict, expected: bool | ValueError, is_error: bool) -> None:
    """Test the nonlinear_amplifier_table functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    p_in = params["p_in"]
    p_out = params["p_out"]
    phi = params["phi"]
    p_ratio = params["p_ratio"]
    phase_shift = params["phase_shift"]

    if is_error:
        with pytest.raises(expected):
            data = nonlinear_amplifier_table(data=data, p_in=p_in, p_out=p_out, phi=phi, auto_scale=False)
    else:
        data_test = data.copy()

        data = nonlinear_amplifier_table(data=data, p_in=p_in, p_out=p_out, phi=phi, auto_scale=False)

        input_power = np.mean(np.abs(data_test) ** 2)
        input_phase_rad = np.angle(data_test)
        output_power = np.mean(np.abs(data) ** 2)
        output_phase_rad = np.angle(data)

        assert (abs(output_power / input_power - p_ratio) < RTOL) == expected
        assert (abs(np.mean(np.unwrap(output_phase_rad - input_phase_rad)) - phase_shift) < RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (0, {"norm": 2, "flatten": False}, ValueError, True),
        (generate_test_signal(num_iq_samples=8192, scale=42.0).data, {"norm": 2, "flatten": False}, TEST_DATA.copy(), False),
        (np.reshape(generate_test_signal(num_iq_samples=8192, scale=0.4).data, (1, -1)), {"norm": 2, "flatten": True}, np.reshape(TEST_DATA.copy(), (1, -1)), False),
    ],
)
def test_normalize(data: Any, params: dict, expected: np.ndarray | ValueError, is_error: bool) -> None:
    """Test the normalize functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (np.ndarray | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    norm = params["norm"]
    flatten = params["flatten"]

    if is_error:
        with pytest.raises(expected):
            data = normalize(data, norm, flatten)
    else:
        data_test = data.copy()

        data = normalize(data, norm, flatten)
        expected = normalize(expected, norm, flatten)

        assert np.allclose(data, expected, RTOL)
        assert type(data) == type(data_test)
        assert data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "params, expected_success, is_error, is_warning",
    [
        # Test Case 1: Basic functionality, smooth phases
        ({"num_taps": 101, "max_ripple_db": 1.0, "ripple_freq": 4.5, "passband_fuzz": "smooth", "stopband_fuzz": "smooth", "fallback": "original"}, True, False, False),
        # Test Case 2: Random phases, larger filter
        ({"num_taps": 255, "max_ripple_db": 3.0, "ripple_freq": 4.5, "passband_fuzz": "random", "stopband_fuzz": "random", "fallback": "original"}, True, False, False),
        # Test Case 3: Even num_taps (should be handled internally by incrementing to odd)
        ({"num_taps": 100, "max_ripple_db": 2.0, "ripple_freq": 4.5, "passband_fuzz": "random", "stopband_fuzz": "smooth", "fallback": "original"}, True, False, False),
        # Test Case 4: Error handling - force a failure by passing an invalid phase (will trigger the try-except)
        ({"num_taps": 101, "max_ripple_db": 1.0, "ripple_freq": 4.5, "passband_fuzz": "invalid_option", "stopband_fuzz": "smooth", "fallback": "raise"}, ValueError, True, False),
        # Test Case 5: Error handling - invalid num_taps
        ({"num_taps": -55, "max_ripple_db": 1.0, "ripple_freq": 4.5, "passband_fuzz": "smooth", "stopband_fuzz": "smooth", "fallback": "raise"}, ValueError, True, False),
        # Test Case 6: input argument handling, automatically handling below minimum num_taps while producing warning
        ({"num_taps": 14, "max_ripple_db": 1.0, "ripple_freq": 4.5, "passband_fuzz": "smooth", "stopband_fuzz": "smooth", "fallback": "raise"}, True, False, True),
    ],
)
def test_passband_ripple(params: dict, expected_success: Any, is_error: bool, is_warning: bool) -> None:
    """Test the passband_ripple function with pytest.

    Args:
        params (dict): Function call parameters.
        expected_success (bool | type): Expected result (True if success, or Exception type if is_error is True).
        is_error (bool): Whether a test error is expected.
        is_warning (bool): Whether a test warning is expected.
    """
    rng = np.random.default_rng(42)

    # create impulse response
    data = dsp.noise_generator(num_samples=128, power=1.0, color="white", continuous=False, rng=rng)

    data_original = data.copy()

    if is_error:
        # We expect a specific error (e.g., RuntimeError) when fallback="raise"
        with pytest.raises(expected_success):
            passband_ripple(data=data, **params)
    elif is_warning:
        num_taps = params["num_taps"]
        with pytest.warns(UserWarning, match=f"The num_taps '{num_taps}' is below the enforced minimum '65', and will be increased."):
            result = passband_ripple(data=data, **params)
    else:
        # Execute the function
        result = passband_ripple(data=data, **params)

        # 1. Check that the output type and dtype are preserved
        assert type(result) == type(data_original), "Output type should match input type"
        assert result.dtype == data_original.dtype, "Output dtype should match input dtype"

        # 2. Check that the shape is preserved (mode="same")
        assert result.shape == data_original.shape, "Output shape should be the same as input shape"

        # 3. Basic sanity check: The data should actually be modified
        # (Unless ripple is 0, but here it is > 0)
        assert not np.array_equal(result, data_original), "Result should be different from input data"

        # 4. Ensure no NaNs or Infs were introduced
        assert np.all(np.isfinite(result)), "Result contains NaNs or Infs"


def test_passband_ripple_output_shape():
    """Ensure output shape matches input shape for both real and complex data."""
    rng = np.random.default_rng(42)

    fft_len = 1024
    num_taps = 128

    # Test Real
    data_real = np.random.randn(fft_len)
    out_real = passband_ripple(data_real, num_taps=num_taps, rng=rng)
    assert out_real.shape == data_real.shape
    assert np.isrealobj(out_real)

    # Test Complex
    data_complex = np.random.randn(fft_len) + 1j * np.random.randn(fft_len)
    out_complex = passband_ripple(data_complex, num_taps=num_taps, rng=rng)
    assert out_complex.shape == data_complex.shape
    assert np.iscomplexobj(out_complex)


def test_passband_ripple_gain():
    """Verify that the filter preserves the designed DC gain."""
    data = np.ones(1024)
    rng = np.random.default_rng(42)

    out = passband_ripple(data, num_taps=128, rng=rng, passband_fuzz="smooth", stopband_fuzz="smooth")

    # DC gain should be roughly 1.0 +/- ripple_amp
    # Since max_ripple_db=2.0 -> ripple_amp approx 0.11
    # So we check if it's between 0.8 and 1.2
    avg_gain = np.mean(out[100:-100])

    assert 0.8 <= avg_gain <= 1.2, f"DC gain {avg_gain} is outside expected ripple range"
    # Check that the filter is actually applying a constant gain (no drifting)
    assert_allclose(out[100:-100], avg_gain, atol=1e-3)


def test_fallback_on_calculation_error(monkeypatch):
    """Test that if a calculation error occurs INSIDE the try block,
    the function catches it and returns the original data.
    """
    data = np.random.randn(1024)
    rng = np.random.default_rng(42)

    # 1. Define a "poison" function
    def mock_fail(*args, **kwargs):
        raise RuntimeError("Simulated internal math failure")

    # 2. Monkeypatch _build_full_profile since that is where the logic starts now
    from torchsig.transforms import functional

    monkeypatch.setattr(functional, "_build_full_profile", mock_fail)

    # 3. Execute. fallback="original" should return input data
    result = passband_ripple(data, fallback="original", rng=rng)
    assert_array_equal(result, data)


def test_fallback_raise_mode(monkeypatch):
    """Verify that when fallback='raise', the internal error is propagated."""
    data = np.random.randn(1024)
    rng = np.random.default_rng(42)

    from torchsig.transforms import functional

    def mock_fail(*args, **kwargs):
        raise RuntimeError("Simulated internal math failure")

    monkeypatch.setattr(functional, "_build_full_profile", mock_fail)

    with pytest.raises(RuntimeError, match="Filter failed: Simulated internal math failure"):
        passband_ripple(data, fallback="raise", rng=rng)


@pytest.mark.parametrize("num_taps", [513, 1025])
@pytest.mark.parametrize("trim_tol", [0, 1e-6, 1e-4])
@pytest.mark.parametrize("ripple_amp", [0.0, 0.1, 0.5, 5])
def test_fft_filter_reconstruction_error(num_taps, trim_tol, ripple_amp):
    """Verify that the frequency response reconstructed from the generated taps
    is sufficiently close to the intended profile.
    """
    # 1. Setup
    ripple_freq = 5.0
    rng = np.random.default_rng(42)

    # Generate the target frequency profile
    profile_standard = _build_full_profile(num_taps=num_taps, ripple_amp=ripple_amp, ripple_freq=ripple_freq, passband_fuzz="smooth", stopband_fuzz="smooth", rng=rng)

    # 2. Generate Taps using the filter function
    # fit_metric='rms' is used to calculate the error during generation
    taps, fit_err = _fft_filter(profile_standard, trim_tol=trim_tol, fit_metric="rms")

    # 3. Perform Round-Trip Analysis
    # The actual response is the FFT of the taps we just generated
    # We ensure n=num_taps to match the original profile length
    actual_resp = np.fft.fft(taps, n=num_taps)

    # 4. Calculate Magnitude Error
    # We compare the magnitude of the intended profile vs the actual response
    intended_mag = np.abs(profile_standard)
    actual_mag = np.abs(actual_resp)

    # Calculate RMS Error manually to verify the 'fit_err' returned by the function
    rmse = np.sqrt(np.mean((intended_mag - actual_mag) ** 2))

    # 5. Assertions
    # Tolerance logic:
    # If trim_tol=0, the reconstruction should be nearly perfect (float precision).
    # If trim_tol > 0, we allow a small margin of error.
    if trim_tol == 0:
        assert rmse < 1e-10, f"Perfect reconstruction failed for tol=0. RMSE: {rmse:.2e}"
    else:
        # Threshold can be adjusted based on your specific filter requirements
        max_allowed_rmse = 1e-3
        assert rmse < max_allowed_rmse, f"Trimmed reconstruction error too high. tol={trim_tol}, RMSE: {rmse:.2e}"

    # Additionally, verify that the fit_err returned by the function matches our calculation
    assert np.isclose(fit_err, rmse, rtol=1e-5), f"Function reported fit_err {fit_err:.2e} differs from actual RMSE {rmse:.2e}"


def test_build_full_profile_hermitian_symmetry():
    from torchsig.transforms.functional import _build_full_profile

    rng = np.random.default_rng(42)
    N = 1025

    full_profile = _build_full_profile(num_taps=N, rng=rng)

    # Create a shifted version for the DC check
    profile_shifted = np.fft.fftshift(full_profile)

    # 2. The DC component must be real
    # In a shifted profile of length 1025, DC is exactly at index 512
    dc_val = profile_shifted[N // 2]
    assert np.isclose(dc_val.imag, 0), f"DC value should be real, got {dc_val}"

    # 3. Verify Hermitian Symmetry on the standard profile
    # Standard order: [DC, pos_1, pos_2, ..., neg_2, neg_1]
    # The positive frequencies are full_profile[1 : n_pos+1]
    # The negative frequencies are full_profile[n_pos+1 :]

    # A cleaner way to check Hermitian symmetry:
    # The profile should be equal to its own conjugate reversed,
    # but we must account for the fact that the DC bin is not mirrored.

    # Extract positive frequencies (1 to 512)
    pos_freqs = full_profile[1 : (N // 2) + 1]
    # Extract negative frequencies (513 to 1024)
    neg_freqs = full_profile[(N // 2) + 1 :]

    # H(f) == conj(H(-f))
    # neg_freqs are stored in order [H(-f_max), ..., H(-f_1)]
    # So we reverse them to align with [H(f_1), ..., H(f_max)]
    assert np.allclose(pos_freqs, np.conj(neg_freqs[::-1])), "Hermitian Symmetry failed"


@pytest.mark.parametrize(
    "data, params, expected, is_error", [(0, {"patch_size": 3, "patches_to_shuffle": [2, 7]}, TypeError, True), (TEST_DATA.copy(), {"patch_size": 3, "patches_to_shuffle": [2, 7]}, True, False)]
)
def test_patch_shuffle(data: Any, params: dict, expected: bool | TypeError, is_error: bool) -> None:
    """Test the patch_shuffle functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)
    patch_size = params["patch_size"]
    patches_to_shuffle = params["patches_to_shuffle"]

    if is_error:
        with pytest.raises(expected):
            data = patch_shuffle(data, patch_size, patches_to_shuffle, rng)
    else:
        data_test = data.copy()

        data = patch_shuffle(data, patch_size, patches_to_shuffle, rng)

        patch_inds = np.where(data != data_test)[0]
        assert ((patch_inds[0] + patch_size - 1) in patch_inds) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [("invalid_input", {"phase": -np.pi / 4}, TypeError, True), (TEST_DATA.copy(), {"phase": -np.pi / 4}, True, False)])
def test_phase_offset(data: Any, params: dict, expected: bool | TypeError, is_error: bool) -> None:
    """Test the phase_offset functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    phase = params["phase"]

    if is_error:
        with pytest.raises(expected):
            data = phase_offset(data, phase=phase)
    else:
        data_test = data.copy()

        data = phase_offset(data, phase=phase)

        data_restored = data * np.exp(-1j * phase)
        assert (np.allclose(data_restored, data_test, rtol=RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (2.0 * np.sqrt(2) * (np.ones((16,)) + 1j * np.ones((16,))), {"num_bits": 8}, 2.0 * (np.ones((16,)) + 1j * np.ones((16,))), False),
        (np.sqrt(2) * (np.ones((16,)) + 1j * np.ones((16,))), {"num_bits": 8}, 2.0 * (np.ones((16,)) + 1j * np.ones((16,))), False),
        ((np.zeros((16,)) + 1j * np.zeros((16,))), {"num_bits": 16}, (np.zeros((16,)) + 1j * np.zeros((16,))), False),  # zero values should be quantized to zero
        (np.array([np.nan + 1j]), {"num_bits": 10}, ValueError, True),  # NaN or Inf values should raise an error
    ],
)
def test_quantize(data: Any, params: dict, expected: np.ndarray | ValueError, is_error: bool) -> None:
    """Test the quantize functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (np.ndarray | TypeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    num_bits = params["num_bits"]

    if is_error:
        with pytest.raises(expected):
            data = quantize(
                data,
                num_bits=num_bits,
            )
    else:
        data = quantize(
            data,
            num_bits=num_bits,
        )

        assert type(data) == type(expected)
        assert data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize("data, params, expected, is_error", [(TEST_DATA.copy(), {"mean_db": 4.0, "sigma_db": 2.0}, True, False), (TEST_DATA.copy(), {"mean_db": 0.0, "sigma_db": 0.42}, True, False)])
def test_shadowing(data: Any, params: dict, expected: bool, is_error: bool) -> None:
    """Test the shadowing functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    mean_db = params["mean_db"]
    sigma_db = params["sigma_db"]

    if is_error:
        with pytest.raises(expected):
            data = shadowing(data=data, mean_db=mean_db, sigma_db=sigma_db, rng=rng)
    else:
        data_test = data.copy()
        n_iterations = 30
        results = [10 * np.log10(np.mean(np.abs(shadowing(data=data, mean_db=mean_db, sigma_db=sigma_db, rng=rng) ** 2))) for _ in range(n_iterations)]
        results_array = np.array(results)

        # Shapiro-Wilk test for normality
        stat, p_value = sp.stats.shapiro(results_array)

        assert (p_value > 0.05) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize("data, expected, is_error", [(TEST_DATA.copy(), True, False)])
def test_spectral_inversion(data: Any, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the spectral_inversion functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        expected (bool | AttributeError]): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = spectral_inversion(data)
    else:
        data_test = data.copy()
        test_real = data_test.real
        test_imag = data_test.imag

        data = spectral_inversion(data)

        assert np.allclose(data.real, test_real, RTOL) == expected
        assert np.allclose(data.imag, -test_imag, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [(0, {"fft_size": 16, "fft_stride": 4}, ValueError, True), (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"fft_size": 16, "fft_stride": 4}, True, False)],
)
def test_spectrogram(data: Any, params: dict, expected: bool | ValueError, is_error: bool) -> None:
    """Test the spectrogram functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    fft_size = params["fft_size"]
    fft_stride = params["fft_stride"]

    if is_error:
        with pytest.raises(expected):
            spec_data = spectrogram(data, fft_size, fft_stride)
    else:
        spec_test = compute_spectrogram(iq_samples=data, fft_size=fft_size, fft_stride=fft_stride)

        spec_data = spectrogram(
            data,
            fft_size,
            fft_stride,
        )

        assert np.allclose(spec_data, spec_test, RTOL) == expected
        assert (type(spec_data) == type(spec_test)) == expected
        assert (spec_data.dtype == spec_test.dtype) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [8], "drop_sizes": [2], "fill": "invalid_fill_type"}, ValueError, True),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [8], "drop_sizes": [2], "fill": "zero"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [2, 7], "drop_sizes": [2, 1], "fill": "mean"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [3, 11], "drop_sizes": [4, 3], "fill": "ffill"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [4], "drop_sizes": [10], "fill": "bfill"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [1, 2, 5], "drop_sizes": [1, 1, 1], "fill": "min"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [13], "drop_sizes": [3], "fill": "max"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [2, 4], "drop_sizes": [1, 7], "fill": "low"}, True, False),
        (generate_test_signal(num_iq_samples=8192, scale=1.0).data, {"drop_starts": [1, 2, 3, 4], "drop_sizes": [1, 1, 1, 1], "fill": "ones"}, True, False),
    ],
)
def test_spectrogram_drop_samples(data: Any, params: dict, expected: bool | ValueError, is_error: bool) -> None:
    """Test the spectrogram_drop_samples functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    drop_starts = params["drop_starts"]
    drop_sizes = params["drop_sizes"]
    fill = params["fill"]

    spec_data = compute_spectrogram(
        iq_samples=data,
        fft_size=16,
        fft_stride=4,
    )
    spec_data = np.tile(spec_data, (16, 1, 1))
    spec_test = spec_data.copy()

    if is_error:
        with pytest.raises(expected):
            spec_data = spectrogram_drop_samples(spec_data, drop_starts, drop_sizes, fill)
    else:
        spec_data = spectrogram_drop_samples(spec_data, drop_starts, drop_sizes, fill)

        drop_inds = []
        for idx, drop_start in enumerate(drop_starts):
            drop_inds = np.append(drop_inds, np.arange(drop_start, drop_start + drop_sizes[idx]))

        changed_inds = np.unique(np.where(spec_data != spec_test)[2])

        if np.any(drop_inds):
            assert (sorted(drop_inds) == sorted(changed_inds)) == expected

        assert (type(spec_data) == type(spec_test)) == expected
        assert (spec_data.dtype == spec_test.dtype) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (TEST_DATA, {"sample_rate": 1, "center_freqs": [-0.3, 0.1], "relative_power_db": [5, 10]}, TEST_DATA, False),
        (TEST_DATA, {"sample_rate": 1, "center_freqs": [-3, 2], "relative_power_db": [5, 10]}, ValueError, True),
        (TEST_DATA, {"sample_rate": 1, "center_freqs": [-3], "relative_power_db": [5, 10]}, ValueError, True),
    ],
)
def test_spurs(data: Any, params: dict, expected: np.ndarray | TypeError | ValueError, is_error: bool) -> None:
    """Test the spurs functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (np.ndarray | TypeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    sample_rate = params["sample_rate"]
    center_freqs = params["center_freqs"]
    relative_power_db = params["relative_power_db"]

    if is_error:
        with pytest.raises(expected):
            data = spurs(data, sample_rate=sample_rate, center_freqs=center_freqs, relative_power_db=relative_power_db)
    else:
        data = spurs(data, sample_rate=sample_rate, center_freqs=center_freqs, relative_power_db=relative_power_db)

        assert data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "sample_rate, center_freq",
    [(1, 0.25), (1, 0.123), (1, -0.3), (1, 0.5 / 4096), (10e6, 1.234e6), (1, -0.4999)],
)
def test_spurs_analytic_peak_matches_fft(sample_rate: float, center_freq: float) -> None:
    """The analytic spur peak (replacing the per-spur FFT) matches the FFT peak.

    spurs() computes the spur's spectral peak in closed form instead of taking
    an FFT of the pure tone; this verifies the two agree.
    """
    n = 4096
    spur = np.exp(2j * np.pi * (center_freq / sample_rate) * np.arange(0, n))
    fft_peak_db = np.max(20 * np.log10(np.abs(np.fft.fft(spur))))

    bin_offset = (center_freq / sample_rate) * n
    frac = bin_offset - np.round(bin_offset)
    if np.isclose(frac, 0.0):
        spur_max = float(n)
    else:
        spur_max = np.abs(np.sin(np.pi * frac) / np.sin(np.pi * frac / n))
    analytic_peak_db = 20 * np.log10(spur_max)

    np.testing.assert_allclose(analytic_peak_db, fft_peak_db, atol=1e-9)


@pytest.mark.parametrize("data, expected, is_error", [(0, ValueError, True), (TEST_DATA.copy(), True, False)])
def test_time_reversal(data: Any, expected: bool | ValueError, is_error: bool) -> None:
    """Test the time_reversal functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(expected):
            data = time_reversal(data)
    else:
        data_test = data.copy()

        data = time_reversal(data)

        assert np.allclose(data, np.flip(data_test, axis=0), RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "data, params, expected, is_error",
    [
        (0, {"noise_power_low": 3.0, "noise_power_high": 3.0, "inflections": 4, "random_regions": False}, AttributeError, True),
        (np.zeros(1024, dtype=TorchSigComplexDataType), {"noise_power_low": 3.0, "noise_power_high": 3.0, "inflections": 4, "random_regions": False}, True, False),
    ],
)
def test_time_varying_noise(data: Any, params: dict, expected: bool | AttributeError, is_error: bool) -> None:
    """Test the time_varying_noise functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)
    noise_power_low = params["noise_power_low"]
    noise_power_high = params["noise_power_high"]
    noise_power_high_linear = 10 ** (noise_power_high / 10.0)
    inflections = params["inflections"]
    random_regions = params["random_regions"]

    if is_error:
        with pytest.raises(expected):
            data = time_varying_noise(data, noise_power_low=noise_power_low, noise_power_high=noise_power_high, inflections=inflections, random_regions=random_regions, rng=rng)
    else:
        data_test = data.copy()

        data = time_varying_noise(data, noise_power_low=noise_power_low, noise_power_high=noise_power_high, inflections=inflections, random_regions=random_regions, rng=rng)

        power_est = np.mean(np.abs(data) ** 2)
        assert (abs(power_est - noise_power_high_linear) < 1e-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == TorchSigComplexDataType) == expected
