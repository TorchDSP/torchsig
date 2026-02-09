"""Unit Tests: transforms/transforms.py"""

from torchsig.transforms.transforms import (
    SignalTransform,
    AWGN,
    AddSlope,
    AdditiveNoise,
    AdjacentChannelInterference,
    CarrierFrequencyDrift,
    CarrierPhaseNoise,
    CarrierPhaseOffset,
    ChannelSwap,
    ClockDrift,
    ClockJitter,
    CoarseGainChange,
    CochannelInterference,
    ComplexTo2D,
    CutOut,
    DigitalAGC,
    Doppler,
    Fading,
    IntermodulationProducts,
    IQImbalance,
    NonlinearAmplifier,
    PassbandRipple,
    PatchShuffle,
    Quantize,
    RandomDropSamples,
    Shadowing,
    SpectralInversion,
    Spectrogram,
    SpectrogramDropSamples,
    SpectrogramImage,
    Spurs,
    TimeReversal,
    TimeVaryingNoise,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import compute_spectrogram, low_pass, TorchSigComplexDataType, TorchSigRealDataType
from test_transforms_utils import generate_test_signal

# Third Party
import numpy as np
import pytest

# Built-In
from typing import Union


RTOL = 1e-6
TEST_SIGNAL = generate_test_signal(num_iq_samples=64, scale=1.0)


# test fixtures
def new_test_signal() -> Signal:
    return TEST_SIGNAL.copy()


# pytests
@pytest.mark.parametrize("is_error", [False])
def test_SignalTransform(is_error: bool) -> None:
    """Test the parent transform with pytest.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = SignalTransform()
    else:
        T = SignalTransform()

        assert isinstance(T, SignalTransform)
        assert isinstance(T.random_generator, np.random.Generator)


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (generate_test_signal(num_iq_samples=1024, scale=1.0), {"noise_power_db": 3.0, "precise": False}, False),
        (generate_test_signal(num_iq_samples=1024, scale=1.0), {"noise_power_db": 0.1, "precise": False}, False),
        # (generate_test_signal(num_iq_samples = 1024, scale = 1.0), {'noise_power_db': 0.5, 'precise': True}, False),
    ],
)
def test_AWGN(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the AWGN transform with pytest.

    Args:
        signal: input signal.
        params (dict): AWGN parameters (see functional AWGN description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    noise_power_db = params["noise_power_db"]
    add_noise_power_linear = 10 ** (noise_power_db / 10)
    precise = params["precise"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AWGN(noise_power_db=noise_power_db, precise=precise, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = AWGN(noise_power_db=noise_power_db, precise=precise, seed=42)
        signal = T(signal)

        if precise:
            orig_power = np.sum(np.abs(signal_test.data) ** 2) / len(signal_test.data)
            orig_snr_linear = 10 ** (signal_test.metadata.snr_db / 10)
            out_power = np.sum(np.abs(signal.data) ** 2) / len(signal.data)
            add_noise_power = out_power - orig_power
            sig_power = orig_power / (1 + 1 / orig_snr_linear)
            noise_power = sig_power / orig_snr_linear
            new_snr_db = 10 * np.log10(sig_power / (noise_power + add_noise_power))
            assert np.abs(signal.metadata.snr_db - new_snr_db) < 10 ** (1.0 / 10)
        else:
            pass  # assert signal.metadata.snr_db == signal_test.metadata.snr_db

        assert isinstance(T, AWGN)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.noise_power_db, float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, is_error",
    [
        (generate_test_signal(num_iq_samples=6400, scale=1.0), False),
        (generate_test_signal(num_iq_samples=256, scale=1.0), False),
    ],
)
def test_AddSlope(signal: Signal, is_error: bool) -> None:
    """Test AddSlope transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AddSlope(seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = AddSlope(seed=42)
        signal = T(signal)

        assert isinstance(T, AddSlope)
        assert isinstance(T.random_generator, np.random.Generator)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (generate_test_signal(num_iq_samples=1024, scale=1.0), {"power_range": (0.01, 10.0), "color": "white", "continuous": True, "precise": False}, False),
        (generate_test_signal(num_iq_samples=1024, scale=1.0), {"power_range": (0.5, 2.0), "color": "pink", "continuous": False, "precise": False}, False),
        # (generate_test_signal(num_iq_samples = 1024, scale = 1.0),{'power_range': (2.0, 2.0),'color': 'white','continuous': True, 'precise': True}, False),
    ],
)
def test_AdditiveNoise(signal: Signal, params: dict, is_error: bool) -> None:
    """Test AdditiveNoise transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    power_range = params["power_range"]
    color = params["color"]
    continuous = params["continuous"]
    precise = params["precise"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AdditiveNoise(power_range=power_range, color=color, continuous=continuous, precise=precise, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = AdditiveNoise(power_range=power_range, color=color, continuous=continuous, precise=precise, seed=42)
        signal = T(signal)

        if precise:
            orig_power = np.sum(np.abs(signal_test.data) ** 2) / len(signal_test.data)
            orig_snr_linear = 10 ** (signal_test.snr_db / 10)
            out_power = np.sum(np.abs(signal.data) ** 2) / len(signal.data)
            add_noise_power = out_power - orig_power
            sig_power = orig_power / (1 + 1 / orig_snr_linear)
            noise_power = sig_power / orig_snr_linear
            new_snr_db = 10 * np.log10(sig_power / (noise_power + add_noise_power))

            assert np.abs(signal.snr_db - new_snr_db) < 10 ** (1.0 / 10)
        else:
            pass

        assert isinstance(T, AdditiveNoise)
        assert isinstance(T.power_distribution(), float)
        assert isinstance(T.color, str)
        assert isinstance(T.continuous, bool)
        # assert isinstance(T.precise, bool)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, expected, is_error",
    [
        (
            new_test_signal(),
            {
                "sample_rate": 1.0,
                "power_range": (0.5, 2.0),
                "center_frequency_range": (0.25, 0.25),
                "phase_sigma_range": (0.0, 0.0),
                "time_sigma_range": (0.0, 0.0),
                "filter_weights": low_pass(0.125, 0.125, 1.0),
            },
            True,
            False,
        ),
        (
            new_test_signal(),
            {
                "sample_rate": 2.4,
                "power_range": (0.01, 100.0),
                "center_frequency_range": (-0.5, -0.07),
                "phase_sigma_range": (0.0, 1.0),
                "time_sigma_range": (0.0, 10.0),
                "filter_weights": low_pass(0.04, 0.16, 2.4),
            },
            True,
            False,
        ),
    ],
)
def test_AdjacentChannelInterference(signal: Signal, params: dict, expected: Union[bool, ValueError], is_error: bool) -> None:
    """Test AdjacentChannelInterference transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    sample_rate = params["sample_rate"]
    power_range = params["power_range"]
    center_frequency_range = params["center_frequency_range"]
    phase_sigma_range = params["phase_sigma_range"]
    time_sigma_range = params["time_sigma_range"]
    filter_weights = params["filter_weights"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AdjacentChannelInterference(
                sample_rate=sample_rate,
                power_range=power_range,
                center_frequency_range=center_frequency_range,
                phase_sigma_range=phase_sigma_range,
                time_sigma_range=time_sigma_range,
                filter_weights=filter_weights,
                seed=42,
            )
            signal = T(signal)
    else:
        T = AdjacentChannelInterference(
            sample_rate=sample_rate,
            power_range=power_range,
            center_frequency_range=center_frequency_range,
            phase_sigma_range=phase_sigma_range,
            time_sigma_range=time_sigma_range,
            filter_weights=filter_weights,
            seed=42,
        )
        signal = T(signal)

        assert isinstance(T, AdjacentChannelInterference) == expected
        assert isinstance(T.sample_rate, float) == expected
        assert isinstance(T.power_distribution(), float) == expected
        assert isinstance(T.center_frequency_distribution(), float) == expected
        assert isinstance(T.phase_sigma_distribution(), float) == expected
        assert isinstance(T.time_sigma_distribution(), float) == expected
        assert isinstance(T.filter_weights, np.ndarray) == expected

        assert (signal.data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"drift_ppm": (0.1, 1)}, False),
        (new_test_signal(), {"drift_ppm": (0.1, 1)}, False),
    ],
)
def test_CarrierFrequencyDrift(signal: Signal, params: dict, is_error: bool) -> None:
    """Test CarrierFrequencyDrift transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    drift_ppm = params["drift_ppm"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CarrierFrequencyDrift(drift_ppm=drift_ppm, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = CarrierFrequencyDrift(drift_ppm=drift_ppm, seed=42)
        signal = T(signal)

        assert isinstance(T, CarrierFrequencyDrift)
        assert isinstance(T.drift_ppm_distribution(), float)

        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (
            new_test_signal(),
            {
                "phase_noise_degrees": (0.25, 1),
            },
            False,
        ),
    ],
)
def test_CarrierPhaseNoise(signal: Signal, params: dict, is_error: bool) -> None:
    """Test CarrierPhaseNoise transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    phase_noise_degrees = params["phase_noise_degrees"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CarrierPhaseNoise(phase_noise_degrees=phase_noise_degrees, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = CarrierPhaseNoise(phase_noise_degrees=phase_noise_degrees, seed=42)
        signal = T(signal)

        assert isinstance(T, CarrierPhaseNoise)
        assert isinstance(T.phase_noise_degrees, tuple)
        assert isinstance(T.phase_noise_degrees_distribution(), float)

        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize("signal, is_error", [(generate_test_signal(num_iq_samples=6400, scale=1.0), False), (generate_test_signal(num_iq_samples=256, scale=1.0), False)])
def test_CarrierPhaseOffset(signal: Signal, is_error: bool) -> None:
    """Test CarrierPhaseOffset transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CarrierPhaseOffset(seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = CarrierPhaseOffset(seed=42)
        signal = T(signal)

        assert isinstance(T, CarrierPhaseOffset)
        assert isinstance(T.phase_offset_distribution(), float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, is_error",
    [
        (generate_test_signal(num_iq_samples=6400, scale=1.0), False),
        (generate_test_signal(num_iq_samples=256, scale=1.0), False),
    ],
)
def test_ChannelSwap(signal: Signal, is_error: bool) -> None:
    """Test ChannelSwap transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = ChannelSwap()
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = ChannelSwap()
        signal = T(signal)

        assert isinstance(T, ChannelSwap)

        assert len(signal.data) == len(signal_test.data)
        assert np.not_equal(signal.data, signal_test.data).any()
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"drift_ppm": (0.1, 1.0)}, False),
        (new_test_signal(), {"drift_ppm": (1.0, 10.0)}, False),
    ],
)
def test_ClockDrift(signal: Signal, params: dict, is_error: bool) -> None:
    """Test ClockDrift transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    drift_ppm = params["drift_ppm"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = ClockDrift(drift_ppm=drift_ppm, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = ClockDrift(drift_ppm=drift_ppm, seed=42)
        signal = T(signal)

        assert isinstance(T, ClockDrift)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.drift_ppm_distribution(), float)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"jitter_ppm": (0.1, 1.0)}, False),
        (new_test_signal(), {"jitter_ppm": (1.0, 10.0)}, False),
    ],
)
def test_ClockJitter(signal: Signal, params: dict, is_error: bool) -> None:
    """Test ClockJitter transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    jitter_ppm = params["jitter_ppm"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = ClockJitter(jitter_ppm=jitter_ppm, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = ClockJitter(jitter_ppm=jitter_ppm, seed=42)
        signal = T(signal)

        assert isinstance(T, ClockJitter)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.jitter_ppm_distribution(), float)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"gain_change_db": (-3.0, 3.0)}, False),
        (new_test_signal(), {"gain_change_db": (-10.0, 10.0)}, False),
    ],
)
def test_CoarseGainChange(signal: Signal, params: dict, is_error: bool) -> None:
    """Test CoarseGainChange transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    gain_change_db = params["gain_change_db"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CoarseGainChange(gain_change_db=gain_change_db, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = CoarseGainChange(gain_change_db=gain_change_db, seed=42)
        signal = T(signal)

        assert isinstance(T, CoarseGainChange)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.gain_change_db_distribution(), float)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, expected, is_error",
    [
        (
            new_test_signal(),
            {
                "power_range": (0.5, 2.0),
                "filter_weights": low_pass(0.125, 0.125, 1.0),
                "color": "white",
                "continuous": True,
                "precise": False,
            },
            True,
            False,
        ),
        (
            new_test_signal(),
            {
                "power_range": (0.01, 100.0),
                "filter_weights": low_pass(0.04, 0.16, 2.4),
                "color": "pink",
                "continuous": False,
                "precise": False,
            },
            True,
            False,
        ),
    ],
)
def test_CochannelInterference(signal: Signal, params: dict, expected: bool, is_error: bool) -> None:
    """Test CochannelInterference with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    power_range = params["power_range"]
    filter_weights = params["filter_weights"]
    color = params["color"]
    continuous = params["continuous"]
    precise = params["precise"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CochannelInterference(power_range=power_range, filter_weights=filter_weights, color=color, continuous=continuous, precise=precise, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = CochannelInterference(power_range=power_range, filter_weights=filter_weights, color=color, continuous=continuous, precise=precise, seed=42)
        signal = T(signal)

        assert isinstance(T, CochannelInterference) == expected
        assert isinstance(T.power_range, tuple) == expected
        assert isinstance(T.power_distribution(), float) == expected
        assert isinstance(T.filter_weights, np.ndarray) == expected
        assert isinstance(T.color, str) == expected
        assert isinstance(T.continuous, bool) == expected
        assert isinstance(T.precise, bool) == expected

        assert (signal.data.dtype == TorchSigComplexDataType) == expected


@pytest.mark.parametrize(
    "signal, is_error",
    [
        (new_test_signal(), False),
    ],
)
def test_ComplexTo2D(signal: Signal, is_error: bool) -> None:
    """Test ComplexTo2D transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = ComplexTo2D()
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = ComplexTo2D()
        signal = T(signal)

        assert isinstance(T, ComplexTo2D)

        assert signal.data.dtype == TorchSigRealDataType


@pytest.mark.parametrize(
    "signal, is_error",
    [
        (new_test_signal(), False),
    ],
)
def test_CutOut(signal: Signal, is_error: bool) -> None:
    """Test CutOut transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CutOut()
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = CutOut()
        signal = T(signal)

        assert isinstance(T, CutOut)
        assert isinstance(T.duration_distribution(), float)
        assert isinstance(T.cut_type_distribution(), str)

        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (
            new_test_signal(),
            {
                "initial_gain_db": (-3, 3),
                "alpha_smooth": (1e-5, 1e-3),
                "alpha_track": (1e-4, 1e-2),
                "alpha_overflow": (1e-1, 3e-1),
                "alpha_acquire": (1e-4, 1e-3),
                "track_range_db": (0.5, 2),
            },
            False,
        ),
    ],
)
def test_DigitalAGC(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the DigitalAGC transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): AGC parameters (see DigitalAGC description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = DigitalAGC(
                initial_gain_db=params["initial_gain_db"],
                alpha_smooth=params["alpha_smooth"],
                alpha_track=params["alpha_track"],
                alpha_overflow=params["alpha_overflow"],
                alpha_acquire=params["alpha_acquire"],
                track_range_db=params["track_range_db"],
                seed=42,
            )
            signal = T(signal)
    else:
        T = DigitalAGC(
            initial_gain_db=params["initial_gain_db"],
            alpha_smooth=params["alpha_smooth"],
            alpha_track=params["alpha_track"],
            alpha_overflow=params["alpha_overflow"],
            alpha_acquire=params["alpha_acquire"],
            track_range_db=params["track_range_db"],
            seed=42,
        )
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, DigitalAGC)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"velocity_range": (0.0, 10.0), "propagation_speed": 2.9979e8}, False),
    ],
)
def test_Doppler(signal: Signal, params: dict, is_error: bool) -> None:
    """Test Doppler with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    velocity_range = params["velocity_range"]
    propagation_speed = params["propagation_speed"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Doppler(velocity_range=velocity_range, propagation_speed=propagation_speed, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = Doppler(velocity_range=velocity_range, propagation_speed=propagation_speed, seed=42)
        signal = T(signal)

        assert isinstance(T, Doppler)
        assert isinstance(T.velocity_distribution(), float)
        assert isinstance(T.propagation_speed, float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"coherence_bandwidth": (0.01, 0.1), "power_delay_profile": [0.5, 0.25, 0.125]}, False),
    ],
)
def test_Fading(signal: Signal, params: dict, is_error: bool) -> None:
    """Test Fading transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    coherence_bandwidth = params["coherence_bandwidth"]
    power_delay_profile = params["power_delay_profile"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Fading(coherence_bandwidth=coherence_bandwidth, power_delay_profile=power_delay_profile, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = Fading(coherence_bandwidth=coherence_bandwidth, power_delay_profile=power_delay_profile, seed=42)
        signal = T(signal)

        assert isinstance(T, Fading)
        assert isinstance(T.coherence_bandwidth_distribution(), float)
        assert np.allclose(T.power_delay_profile, power_delay_profile)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (
            new_test_signal(),
            {
                "model_order": [3, 5],
                "coeffs_range": (1e-3, 1e-1),
            },
            False,
        ),
    ],
)
def test_IntermodulationProducts(signal: Signal, params: dict, is_error: bool) -> None:
    """Test IntermodulationProducts transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    model_order = params["model_order"]
    coeffs_range = params["coeffs_range"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = IntermodulationProducts(model_order=model_order, coeffs_range=coeffs_range, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = IntermodulationProducts(model_order=model_order, coeffs_range=coeffs_range, seed=42)
        signal = T(signal)

        assert isinstance(T, IntermodulationProducts)
        assert isinstance(T.model_order_distribution(), np.int64)
        assert isinstance(T.coeffs_distribution(), float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (
            new_test_signal(),
            {"amplitude_imbalance": (0.0, 6.0), "phase_imbalance": (-np.pi, np.pi), "dc_offset": ((-0.2, 0.2))},
            False,
            # new_test_signal(),
            # {
            #     'amplitude_imbalance': (0.0, 6.0),
            #     'phase_imbalance': (-np.pi, np.pi),
            #     'dc_offset': ((-0.2, 0.2),(-0.2, 0.2))
            # },
            # False
        ),
    ],
)
def test_IQImbalance(signal: Signal, params: dict, is_error: bool) -> None:
    """Test IQImbalance with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    amplitude_imbalance = params["amplitude_imbalance"]
    phase_imbalance = params["phase_imbalance"]
    dc_offset = params["dc_offset"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = IQImbalance(amplitude_imbalance=amplitude_imbalance, phase_imbalance=phase_imbalance, dc_offset_db=dc_offset, seed=42)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = IQImbalance(amplitude_imbalance=amplitude_imbalance, phase_imbalance=phase_imbalance, dc_offset_db=dc_offset, seed=42)
        signal = T(signal)

        assert isinstance(T, IQImbalance)
        assert isinstance(T.amplitude_imbalance_distribution(), float)
        assert isinstance(T.phase_imbalance_distribution(), float)
        assert isinstance(T.dc_offset_db_distribution(), float)
        assert isinstance(T.dc_offset_phase_rads_distribution(), float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"gain_range": (1.0, 4.0), "psat_backoff_range": (5.0, 20.0), "phi_max_range": (0.001, 0.001), "phi_slope_range": (0.0001, 0.0001), "auto_scale": True}, False),
    ],
)
def test_NonlinearAmplifier(signal: Signal, params: dict, is_error: bool) -> None:
    """Test NonlinearAmplifier with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    gain_range = params["gain_range"]
    psat_backoff_range = params["psat_backoff_range"]
    phi_max_range = params["phi_max_range"]
    phi_slope_range = params["phi_slope_range"]
    auto_scale = params["auto_scale"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = NonlinearAmplifier(gain_range=gain_range, psat_backoff_range=psat_backoff_range, phi_max_range=phi_max_range, phi_slope_range=phi_slope_range, auto_scale=auto_scale, seed=42)
            signal = T(signal)
    else:
        T = NonlinearAmplifier(gain_range=gain_range, psat_backoff_range=psat_backoff_range, phi_max_range=phi_max_range, phi_slope_range=phi_slope_range, auto_scale=auto_scale, seed=42)
        signal = T(signal)

        assert isinstance(T, NonlinearAmplifier)
        assert isinstance(T.gain_distribution(), float)
        assert isinstance(T.psat_backoff_distribution(), float)
        assert isinstance(T.phi_max_distribution(), float)
        assert isinstance(T.phi_slope_distribution(), float)

        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (
            new_test_signal(),
            {
                "max_ripple_db": (1, 2),
                "num_taps": [2, 3],
                "coefficient_decay_rate": (1, 5),
            },
            False,
        ),
    ],
)
def test_PassbandRipple(signal: Signal, params: dict, is_error: bool) -> None:
    """Test PassbandRipple transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    max_ripple_db = params["max_ripple_db"]
    num_taps = params["num_taps"]
    coefficient_decay_rate = params["coefficient_decay_rate"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = PassbandRipple(max_ripple_db=max_ripple_db, num_taps=num_taps, coefficient_decay_rate=coefficient_decay_rate)
            signal = T(signal)
    else:
        signal_test = signal.copy()
        T = PassbandRipple(max_ripple_db=max_ripple_db, num_taps=num_taps, coefficient_decay_rate=coefficient_decay_rate)
        signal = T(signal)

        assert isinstance(T, PassbandRipple)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"patch_size": [2, 2], "shuffle_ratio": 0.5}, False),
    ],
)
def test_PatchShuffle(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the PatchShuffle transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): PatchShuffle parameters (see functional PatchShuffle description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    patch_size = params["patch_size"]
    shuffle_ratio = params["shuffle_ratio"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = PatchShuffle(patch_size=patch_size, shuffle_ratio=shuffle_ratio, seed=42)
            signal = T(signal)
    else:
        T = PatchShuffle(patch_size=patch_size, shuffle_ratio=shuffle_ratio, seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, PatchShuffle)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.patch_size_distribution(), np.int_)
        assert isinstance(T.shuffle_ratio_distribution(), float)

        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"num_bits": [4]}, False),
    ],
)
def test_Quantize(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the Quantize transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): Quantize parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    num_bits = params["num_bits"]
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Quantize(num_bits=num_bits, seed=42)
            signal = T(signal)
    else:
        T = Quantize(num_bits=num_bits, seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, Quantize)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.num_bits_distribution(), np.int_)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"drop_rate": (0.01, 0.02), "size": (5, 7), "fill": ["zero"]}, False),
    ],
)
def test_RandomDropSamples(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the RandomDropSamples transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): RandomDropSamples parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    drop_rate = params["drop_rate"]
    size = params["size"]
    fill = params["fill"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = RandomDropSamples(drop_rate=drop_rate, size=size, fill=fill, seed=42)
            signal = T(signal)
    else:
        T = RandomDropSamples(drop_rate=drop_rate, size=size, fill=fill, seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, RandomDropSamples)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.drop_rate_distribution(), float)
        assert isinstance(T.size_distribution(), float)
        assert isinstance(T.fill_distribution(), str)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(size, list):  # assume constant value list
            size_samples = size[0]
        else:
            size_samples = size

        fill_inds = np.where(signal.data != signal_test.data)[0]
        assert (fill_inds[0] + size_samples[0] - 1) in fill_inds


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"mean_db_range": (0.0, 4.0), "sigma_db_range": (2.0, 6.0)}, False),
    ],
)
def test_Shadowing(signal: Signal, params: dict, is_error: bool) -> None:
    """Test Shadowing transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    mean_db_range = params["mean_db_range"]
    sigma_db_range = params["sigma_db_range"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Shadowing(mean_db_range=mean_db_range, sigma_db_range=sigma_db_range)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = Shadowing(mean_db_range=mean_db_range, sigma_db_range=sigma_db_range)
        signal = T(signal)

        assert isinstance(T, Shadowing)
        assert isinstance(T.mean_db_distribution(), float)
        assert isinstance(T.sigma_db_distribution(), float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize("signal, is_error", [(generate_test_signal(num_iq_samples=6400, scale=1.0), False), (generate_test_signal(num_iq_samples=256, scale=1.0), False)])
def test_SpectralInversion(signal: Signal, is_error: bool) -> None:
    """Test SpectralInversion transform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = SpectralInversion()
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = SpectralInversion()
        signal = T(signal)

        assert signal.center_freq == -1 * signal_test.center_freq

        assert isinstance(T, SpectralInversion)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == TorchSigComplexDataType


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"fft_size": 16, "fft_stride": 4}, False),
    ],
)
def test_Spectrogram(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the Spectrogram transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): Spectrogram parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    fft_size = params["fft_size"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            signal.data = np.tile(signal.data, (16, 1, 1))
            T = Spectrogram(fft_size=fft_size)
            signal = T(signal)
    else:
        signal_test = signal.copy()

        T = Spectrogram(fft_size=fft_size)
        signal = T(signal)

        assert isinstance(T, Spectrogram)
        assert isinstance(T.fft_size, int)


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"drop_rate": [0.1], "size": [5], "fill": ["zero"]}, False),
    ],
)
def test_SpectrogramDropSamples(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the SpectrogramDropSamples transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): SpectrogramDropSamples parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    drop_rate = params["drop_rate"]
    size = params["size"]
    fill = params["fill"]
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            spec_data = compute_spectrogram(iq_samples=signal.data, fft_size=16, fft_stride=4)
            signal.data = np.tile(spec_data, (16, 1, 1))

            T = SpectrogramDropSamples(drop_rate=drop_rate, size=size, fill=fill, seed=42)
            signal = T(signal)
    else:
        spec_data = compute_spectrogram(iq_samples=signal.data, fft_size=16, fft_stride=4)
        signal.data = np.tile(spec_data, (16, 1, 1))
        signal_test = signal.copy()

        T = SpectrogramDropSamples(drop_rate=drop_rate, size=size, fill=fill, seed=42)
        signal = T(signal)

        assert isinstance(T, SpectrogramDropSamples)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.drop_rate_distribution(), float)
        assert isinstance(T.size_distribution(), np.int_)
        assert isinstance(T.fill_distribution(), str)

        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(size, list):  # assume constant value list
            size_samples = size[0]
        else:
            size_samples = size

        fill_inds = np.where(signal.data != signal_test.data)
        assert (fill_inds[2][0] + size_samples - 1) in fill_inds[2]


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"fft_size": 16}, False),
    ],
)
def test_SpectrogramImage(signal: Signal, params: dict, is_error: bool) -> None:
    """Test SpectrogramImage transform with pytest.

    Args:
        signal (Signal): Input signal to transform.

        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.
    """
    fft_size = params["fft_size"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = SpectrogramImage(fft_size=fft_size)
            signal = T(signal)
    else:
        T = SpectrogramImage(fft_size=fft_size)
        signal = T(signal)

        assert isinstance(T, SpectrogramImage)

        # assert (signal.data.dtype == 'uint8')


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"num_spurs": [1, 4], "relative_power_db": [5, 15]}, False),
    ],
)
def test_Spurs(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the Spurs transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): Spurs parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    num_spurs = params["num_spurs"]
    relative_power_db = params["relative_power_db"]
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Spurs(num_spurs=num_spurs, relative_power_db=relative_power_db, seed=42)
            signal = T(signal)
    else:
        T = Spurs(num_spurs=num_spurs, relative_power_db=relative_power_db, seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, Spurs)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.num_spurs_distribution(), np.int_)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"allow_spectral_inversion": False}, False),
    ],
)
def test_TimeReversal(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the TimeReversal transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): TimeReversal parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = TimeReversal(allow_spectral_inversion=params["allow_spectral_inversion"], seed=42)
            signal = T(signal)
    else:
        T = TimeReversal(allow_spectral_inversion=params["allow_spectral_inversion"], seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, TimeReversal)

        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize(
    "signal, params, is_error",
    [
        (new_test_signal(), {"noise_power_low": (2.0, 3.0), "noise_power_high": (3.0, 4.0), "inflections": [int(0), int(10)], "random_regions": False}, False),
    ],
)
def test_TimeVaryingNoise(signal: Signal, params: dict, is_error: bool) -> None:
    """Test the TimeVaryingNoise transform with pytest.

    Args:
        signal (Signal): input dataset.
        params (dict): TimeVaryingNoise parameters (see functional description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    rng = np.random.default_rng(42)

    noise_power_low = params["noise_power_low"]
    noise_power_high = params["noise_power_high"]
    inflections = params["inflections"]
    random_regions = params["random_regions"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = TimeVaryingNoise(noise_power_low=noise_power_low, noise_power_high=noise_power_high, inflections=inflections, random_regions=random_regions, seed=42)
            signal = T(signal)
    else:
        T = TimeVaryingNoise(noise_power_low=noise_power_low, noise_power_high=noise_power_high, inflections=inflections, random_regions=random_regions, seed=42)
        signal_test = signal.copy()
        signal = T(signal)

        assert isinstance(T, TimeVaryingNoise)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.noise_power_low_distribution(), float)
        assert isinstance(T.noise_power_high_distribution(), float)
        assert isinstance(T.inflections_distribution(), np.int_)
        assert isinstance(T.random_regions_distribution(), float)

        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()
