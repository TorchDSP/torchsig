"""Functional transforms for reuse and custom fine-grained control."""

from copy import copy
from typing import Literal

import cv2

# Third Party
import numpy as np
from scipy import signal as sp
from scipy.constants import c as speed_of_light
from scipy.interpolate import interp1d as sp_interp1d

# TorchSig
from torchsig.utils import dsp
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    TorchSigRealDataType,
    is_even,
    multistage_polyphase_resampler,
    prototype_polyphase_filter,
)
from torchsig.utils.rust_functions import sampling_clock_impairments

__all__ = [
    "add_slope",
    "additive_noise",
    "adjacent_channel_interference",
    "awgn",
    "carrier_frequency_drift",
    "carrier_phase_noise",
    "channel_swap",
    "clock_drift",
    "clock_jitter",
    "coarse_gain_change",
    "cochannel_interference",
    "complex_to_2d",
    "cut_out",
    "digital_agc",
    "doppler",
    "drop_samples",
    "fading",
    "interleave_complex",
    "intermodulation_products",
    "iq_imbalance",
    "nonlinear_amplifier",
    "nonlinear_amplifier_table",
    "normalize",
    "passband_ripple",
    "patch_shuffle",
    "phase_offset",
    "quantize",
    "shadowing",
    "spectral_inversion",
    "spectrogram",
    "spectrogram_drop_samples",
    "spectrogram_image",
    "spurs",
    "time_reversal",
    "time_varying_noise",
]


def add_slope(data: np.ndarray) -> np.ndarray:
    """Add slope between each sample and its preceding sample is added to every sample.

    Augmentation has the effect of amplifying high frequency component more than lower
    frequency components.

    Args:
        data: IQ data.

    Returns:
        IQ data with added slope.
    """
    slope = np.diff(data)
    slope = np.insert(slope, 0, 0)
    return (data + slope).astype(TorchSigComplexDataType)


def additive_noise(
    data: np.ndarray,
    power: float = 1.0,
    color: str = "white",
    continuous: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Additive complex noise with specified parameters.

    Args:
        data: Complex valued IQ data samples.
        power: Desired noise power (linear, positive). Defaults to 1.0 W (0 dBW).
        color: Noise color, supports 'white', 'pink', or 'red' noise frequency spectrum types. Defaults to 'white'.
        continuous: Sets noise to continuous (True) or impulsive (False). Defaults to True.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with complex noise samples with specified power added.
    """
    rng = rng if rng else np.random.default_rng()
    n = len(data)
    noise_samples = dsp.noise_generator(n, power, color, continuous, rng)
    return (data + noise_samples).astype(TorchSigComplexDataType)


def adjacent_channel_interference(
    data: np.ndarray,
    sample_rate: float = 4.0,
    power: float = 1.0,
    center_frequency: float = 0.2,
    filter_weights: np.ndarray | None = None,
    phase_sigma: float = 1.0,
    time_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adds adjacent channel interference to the baseband data at a specified center frequency and power level.

    The adjacent channel signal is a filtered, frequency-offset, randomly block time-shifted, randomly phase-perturbed
    baseband copy that has similar bandwidth and modulation properties, but degrades phase and time coherence with the
    original baseband signal.

    Args:
        data: Complex valued IQ data samples.
        sample_rate: Sampling rate (Fs). Default 4.0
        power: Adjacent interference signal power (linear, positive). Default 1.0 W (0 dBW).
        center_frequency: Adjacent interference signal center frequency (normalized relative to Fs). Default 0.2.
        filter_weights: Lowpass filter weights applied to baseband signal data to band limit prior to creating
            adjacent signal. Default low_pass(0.25,0.25,4.0).
        phase_sigma: Standard deviation of Gaussian phase noise. Default 1.0.
        time_sigma: Standard deviation of Gaussian block time shift in samples. Default 0.0.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with added adjacent interference.
    """
    rng = rng if rng else np.random.default_rng()
    filter_weights = dsp.low_pass(0.25, 0.25, 4.0) if filter_weights is None else filter_weights

    n = len(data)
    t = np.arange(n) / sample_rate

    data_filtered = np.convolve(data, filter_weights)[
        -n:
    ]  # band limit original data (maintain data size)
    phase_noise = rng.normal(0, phase_sigma, n)  # Gaussian phase noise
    interference = data_filtered * np.exp(
        1j * (2 * np.pi * center_frequency * t + phase_noise)
    )  # note: does not check aliasing

    time_shift = int(
        np.round(rng.normal(0, time_sigma, 1))[0]
    )  # Gaussian block time shift for data (nearest sample)
    if time_shift > 0:  # time shift with zero fill; # note: may produce discontinuities
        interference = np.roll(interference, time_shift)
        interference[0:time_shift] = 0 + 1j * 0
    elif time_shift < 0:
        interference = np.roll(interference, time_shift)
        interference[time_shift:0] = 0 + 1j * 0

    # set interference power
    est_power = np.sum(np.abs(interference) ** 2) / len(interference)
    interference = np.sqrt(power / est_power) * interference

    return (data + interference).astype(TorchSigComplexDataType)


def awgn(
    data: np.ndarray, noise_power_db: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Adds zero-mean complex additive white Gaussian noise with power of noise_power_db.

    Args:
        data: (batch_size, vector_length, ...)-sized data.
        noise_power_db: Defined as 10*log10(E[|n|^2]).
        random_generator: Random Generator to use. Defaults to None (new generator created internally).

    Returns:
        Data with added noise.
    """
    rng = rng if rng else np.random.default_rng()

    real_noise = rng.standard_normal(*data.shape)
    imag_noise = rng.standard_normal(*data.shape)
    return (
        data
        + (10.0 ** (noise_power_db / 20.0))
        * (real_noise + 1j * imag_noise)
        / np.sqrt(2)
    ).astype(TorchSigComplexDataType)


def channel_swap(data: np.ndarray) -> np.ndarray:
    """Swap I and Q channels of IQ data.

    Args:
        data: IQ data.

    Returns:
        IQ data with channels swapped.
    """
    real_component = data.real
    imag_component = data.imag
    new_data = np.empty(data.shape, dtype=data.dtype)
    new_data.real = imag_component
    new_data.imag = real_component
    return new_data.astype(TorchSigComplexDataType)


def clock_drift(
    data: np.ndarray,
    drift_ppm: float = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Clock drift from a Local Oscillator (LO), modeled as accumulated gaussian random noise impacting the
    sampling rate.

    The drift applies a randomness to the sampling rate, and by accumulating the gaussian RV
    over time it will slightly increase or decrease the sampling rate of the data, and thereby changing the
    number of samples by a very small number.

    Args:
        data: Complex valued IQ data samples.
        drift_ppm: Clock drift in parts per million (ppm). Default 10.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with LO drift applied.
    """
    rng = rng if rng else np.random.default_rng()

    # enforce data to be the correct complex type
    data = data.astype(TorchSigComplexDataType)

    # create a random seed for rust
    rust_seed = rng.integers(low=0, high=2**32)

    # define up/down rates
    uprate = 5000
    downrate = copy(uprate)

    # build the prototype filter
    pfb_prototype_filter = prototype_polyphase_filter(num_branches=uprate)

    # convert to real data type
    pfb_prototype_filter = pfb_prototype_filter.astype(TorchSigRealDataType)

    # call the impairment
    data_with_drift = sampling_clock_impairments(
        h=pfb_prototype_filter,
        x=data,
        uprate=uprate,
        drate=downrate,
        jitter_ppm=0,
        drift_ppm=drift_ppm,
        seed=rust_seed,
    )

    # discard extra samples from resampling process, or zero-pad if too short
    num_samples_to_discard = len(data_with_drift) - len(data)

    if num_samples_to_discard > 0:
        if is_even(num_samples_to_discard):
            slice_front = num_samples_to_discard // 2
            slice_back = num_samples_to_discard // 2
        else:
            slice_front = (num_samples_to_discard + 1) // 2
            slice_back = num_samples_to_discard // 2
        data_with_drift = data_with_drift[slice_front:-slice_back]
    else:
        # calculate number of zeros to pad
        num_samples_to_pad = len(data) - len(data_with_drift)
        if is_even(num_samples_to_pad):
            pad_front = num_samples_to_pad // 2
            pad_back = num_samples_to_pad // 2
        else:
            pad_front = (num_samples_to_pad + 1) // 2
            pad_back = num_samples_to_pad // 2
        data_with_drift = np.concatenate(
            (np.zeros(pad_front), data_with_drift, np.zeros(pad_back))
        )

    # ensure data type
    return data_with_drift.astype(TorchSigComplexDataType)


def clock_jitter(
    data: np.ndarray,
    jitter_ppm: float = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Clock jitter from a Local Oscillator (LO), modeled as gaussian random noise impacting the
    sampling phase.

    The jitter applies a randomness to the sampling phase, applying a slight
    increment or decrement to the sampling phase and therefore potentially changing the number of
    samples by a very small number.

    Args:
        data: Complex valued IQ data samples.
        jitter_ppm: Jitter in parts per million (ppm). Default 10.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with LO drift applied.
    """
    rng = rng if rng else np.random.default_rng()

    # enforce data to be the correct complex type
    data = data.astype(TorchSigComplexDataType)

    # create a random seed for rust
    rust_seed = rng.integers(low=0, high=2**32)

    # define up/down rates
    uprate = 5000
    downrate = copy(uprate)

    # build the prototype filter
    pfb_prototype_filter = prototype_polyphase_filter(num_branches=uprate)

    # convert to real data type
    pfb_prototype_filter = pfb_prototype_filter.astype(TorchSigRealDataType)

    # call the impairment
    data_with_jitter = sampling_clock_impairments(
        h=pfb_prototype_filter,
        x=data,
        uprate=uprate,
        drate=downrate,
        jitter_ppm=jitter_ppm,
        drift_ppm=0,
        seed=rust_seed,
    )

    # discard extra samples from resampling process, or zero-pad if too short
    num_samples_to_discard = len(data_with_jitter) - len(data)

    if num_samples_to_discard > 0:
        if is_even(num_samples_to_discard):
            slice_front = num_samples_to_discard // 2
            slice_back = num_samples_to_discard // 2
        else:
            slice_front = (num_samples_to_discard + 1) // 2
            slice_back = num_samples_to_discard // 2
        data_with_jitter = data_with_jitter[slice_front:-slice_back]
    else:
        # calculate number of zeros to pad
        num_samples_to_pad = len(data) - len(data_with_jitter)
        if is_even(num_samples_to_pad):
            pad_front = num_samples_to_pad // 2
            pad_back = num_samples_to_pad // 2
        else:
            pad_front = (num_samples_to_pad + 1) // 2
            pad_back = num_samples_to_pad // 2
        data_with_jitter = np.concatenate(
            (np.zeros(pad_front), data_with_jitter, np.zeros(pad_back))
        )

    # ensure data type
    return data_with_jitter.astype(TorchSigComplexDataType)


def cochannel_interference(
    data: np.ndarray,
    power: float = 1.0,
    filter_weights: np.ndarray | None = None,
    color: str = "white",
    continuous: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Applies uncorrelated co-channel interference to the baseband data, modeled as shaped noise with specified parameters.

    Args:
        data: Complex valued IQ data samples.
        power: Interference power (linear, positive). Default 1.0 W (0 dBW).
        filter_weights: Lowpass interference shaping filter weights. Default low_pass(0.25, 0.25, 4.0).
        color: Base noise color, supports 'white', 'pink', or 'red' noise frequency spectrum types. Default 'white'.
        continuous: Sets noise to continuous (True) or impulsive (False). Default True.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with added uncorrelated co-channel interference.
    """
    rng = rng if rng else np.random.default_rng()
    filter_weights = dsp.low_pass(0.25, 0.25, 4.0) if filter_weights is None else filter_weights

    n = len(data)
    noise_samples = dsp.noise_generator(n, power, color, continuous, rng)
    shaped_noise = np.convolve(noise_samples, filter_weights)[-n:]

    # correct shaped noise power (do not assume filter is prescaled)
    est_power = np.sum(np.abs(shaped_noise) ** 2) / len(shaped_noise)
    interference = np.sqrt(power / est_power) * shaped_noise
    return (data + interference).astype(TorchSigComplexDataType)


def coarse_gain_change(
    data: np.ndarray, gain_change_db: float, start_idx: int
) -> np.ndarray:
    """Implements a large instantaneous jump in receiver gain.

    Args:
        data: IQ data.
        gain_change_db: Gain value to change in dB.
        start_idx: Start index for IQ data.

    Returns:
        IQ data with instantaneous gain change applied.
    """
    # convert db to linear units
    gain_change_linear = 10 ** (gain_change_db / 10)

    # create copy of signal
    output_data = copy(data)
    output_data[start_idx:] *= gain_change_linear

    return output_data.astype(TorchSigComplexDataType)


def complex_to_2d(data: np.ndarray) -> np.ndarray:
    """Converts IQ data to two channels (real and imaginary parts)."""
    return np.stack([data.real, data.imag], axis=0)


def cut_out(
    data: np.ndarray,
    cut_start: float,
    cut_duration: float,
    cut_type: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Performs CutOut: replacing values with fill.

    Args:
        data: IQ data
        cut_start: Normalized start of cut region [0.0, 1.0)
        cut_duration: Normalized duration of cut region (0.0, 1.0)
        cut_type: Type of data to fill cut region.
            * zeros
            * ones
            * low_noise
            * avg_noise
            * high_noise
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Raises:
        ValueError: Invalid `cut_type`.

    Returns:
        CutOut IQ data.
    """
    rng = rng if rng else np.random.default_rng()

    num_iq_samples = data.shape[0]
    cut_start = int(cut_start * num_iq_samples)

    # Create cut mask
    cut_mask_length = int(num_iq_samples * cut_duration)
    if cut_mask_length + cut_start > num_iq_samples:
        cut_mask_length = num_iq_samples - cut_start

    if cut_type == "zeros":
        cut_mask = np.zeros(cut_mask_length, dtype=np.complex64)
    elif cut_type == "ones":
        cut_mask = np.ones(cut_mask_length) + 1j * np.ones(cut_mask_length)
    elif cut_type == "low_noise":
        real_noise = rng.standard_normal(cut_mask_length)
        imag_noise = rng.standard_normal(cut_mask_length)
        noise_power_db = -100
        cut_mask = (
            (10.0 ** (noise_power_db / 20.0))
            * (real_noise + 1j * imag_noise)
            / np.sqrt(2)
        )
    elif cut_type == "avg_noise":
        real_noise = rng.standard_normal(cut_mask_length)
        imag_noise = rng.standard_normal(cut_mask_length)
        avg_power = np.mean(np.abs(data) ** 2)
        cut_mask = avg_power * (real_noise + 1j * imag_noise) / np.sqrt(2)
    elif cut_type == "high_noise":
        real_noise = rng.standard_normal(cut_mask_length)
        imag_noise = rng.standard_normal(cut_mask_length)
        noise_power_db = 40
        cut_mask = (
            (10.0 ** (noise_power_db / 20.0))
            * (real_noise + 1j * imag_noise)
            / np.sqrt(2)
        )
    else:
        raise ValueError(
            f"cut_type must be: zeros, ones, low_noise, avg_noise, or high_noise. Found: {cut_type}"
        )

    # Insert cut mask into data
    data[cut_start : cut_start + cut_mask_length] = cut_mask

    return data.astype(TorchSigComplexDataType)


def digital_agc(
    data: np.ndarray,
    initial_gain_db: float = 0.0,
    alpha_smooth: float = 1e-4,
    alpha_track: float = 1e-3,
    alpha_overflow: float = 0.1,
    alpha_acquire: float = 1e-3,
    ref_level_db: float = 0.0,
    track_range_db: float = 1.0,
    low_level_db: float = -80,
    high_level_db: float = 10,
) -> np.ndarray:
    """Automatic Gain Control algorithm (deterministic).

    Args:
        data: IQ data samples.
        initial_gain_db: Inital gain value in dB.
        alpha_smooth: Alpha for avergaing the measure signal level `level_n = level_n * alpha + level_n-1(1-alpha)`
        alpha_track: Amount to adjust gain when in tracking state.
        alpha_overflow: Amount to adjust gain when in overflow state `[level_db + gain_db] >= max_level`.
        alpha_acquire: Amount to adjust gain when in acquire state.
        ref_level_db: Reference level goal for algorithm to achieve, in dB units.
        track_range_db: dB range for operating in tracking state.
        low_level_db: minimum magnitude value (dB) to perform any gain control adjustment.
        high_level_db: magnitude value (dB) to enter overflow state.

    Returns:
        IQ data adjusted sample-by-sample by the AGC algorithm.
    """
    output = np.zeros_like(data)
    gain_db = initial_gain_db
    level_db = 0.0
    for sample_idx, sample in enumerate(data):
        if not np.abs(sample):  # sample == 0
            level_db = -200
        elif not sample_idx:  # first sample == 0, no smoothing
            level_db = np.log(np.abs(sample))
        else:
            level_db = level_db * alpha_smooth + np.log(np.abs(sample)) * (
                1 - alpha_smooth
            )
        output_db = level_db + gain_db
        diff_db = ref_level_db - output_db

        if level_db <= low_level_db:
            alpha_adjust = 0.0
        elif output_db >= high_level_db:
            alpha_adjust = alpha_overflow
        elif abs(diff_db) > track_range_db:
            alpha_adjust = alpha_acquire
        else:
            alpha_adjust = alpha_track

        gain_db += diff_db * alpha_adjust
        output[sample_idx] = sample * np.exp(gain_db)

    return output.astype(TorchSigComplexDataType)


def doppler(
    data: np.ndarray, velocity: float = 1e1, propagation_speed: float = speed_of_light
) -> np.ndarray:
    """Applies wideband Doppler effect through time scaling.

    Args:
        data: Complex valued IQ data samples.
        velocity: Relative velocity in m/s (positive = approaching). Default 10 m/s.
        propagation_speed: Wave speed in medium. Default 2.9979e8 m/s (speed_of_light).

    Returns:
        Data with wideband Doppler.
    """
    n = data.size

    # time scaling factor
    alpha = propagation_speed / (propagation_speed - velocity)

    # if necessary, pad with zeros to maintain size
    if alpha > 1.0:
        num_zeros = int(np.ceil(n * (alpha - 1)) + 1)
        data = np.concatenate((data, np.zeros(num_zeros)))

    data = multistage_polyphase_resampler(data, 1 / alpha)[:n]
    return data.astype(TorchSigComplexDataType)


def drop_samples(
    data: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str,
) -> np.ndarray:
    """Drop samples at given locations/durations with fill technique.

    Supported Fill Techniques:
        ffill: Forward Fill. Use value at sample one before start.
        bfill: Backwards Fill. Use value at sample one after end.
        mean: Mean Fill. Use data mean.
        zero: Zero Fill. Use 0.

    Args:
        data: IQ data.
        drop_starts: Start indicies of drops.
        drop_sizes: Durations for each start index.
        fill: Drop sample replacement method.

    Raises:
        ValueError: Invalid fill type.

    Returns:
        data array with fill values during drops.
    """
    for idx, start in enumerate(drop_starts):
        stop = start + drop_sizes[idx]

        if fill == "ffill":
            drop_region = np.full(drop_sizes[idx], data[start - 1], dtype=np.complex128)
        elif fill == "bfill":
            drop_region = np.full(
                drop_sizes[idx],
                data[stop],
                dtype=np.complex128,
            )
        elif fill == "mean":
            drop_region = np.full(drop_sizes[idx], np.mean(data), dtype=np.complex128)
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx], dtype=np.complex128)
        else:
            raise ValueError(
                f"{fill} fill type unsupported. Must be ffill, bfill, mean, or zero."
            )

        data[start:stop] = drop_region

    return data.astype(TorchSigComplexDataType)


def fading(
    data: np.ndarray,
    coherence_bandwidth: float,
    power_delay_profile: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply fading channel to signal. Currently only does Rayleigh fading.

    Taps are generated by interpolating and filtering Gaussian taps.

    Args:
        data: IQ data.
        coherence_bandwidth: coherence bandwidth relative to sample rate [0, 1.0].
        power_delay_profile: power delay profile assign to channel.
        rng: Random Generator to use. Defaults to None (new generator created internally).

    Returns:
        IQ data with fading applied.
    """
    rng = rng if rng else np.random.default_rng()

    num_taps = int(
        np.ceil(1.0 / coherence_bandwidth)
    )  # filter length to get desired coherence bandwidth

    power_taps = np.sqrt(
        np.interp(
            np.linspace(0, 1.0, 100 * num_taps),
            np.linspace(0, 1.0, len(power_delay_profile)),
            power_delay_profile,
        )
    )

    # Generate initial taps
    rayleigh_taps = rng.standard_normal(num_taps) + 1j * rng.standard_normal(
        num_taps
    )  # multi-path channel

    # Linear interpolate taps by a factor of 100 -- so we can get accurate coherence bandwidths
    old_time = np.linspace(0, 1.0, num_taps, endpoint=True)
    real_tap_function = sp_interp1d(old_time, rayleigh_taps.real)
    imag_tap_function = sp_interp1d(old_time, rayleigh_taps.imag)

    new_time = np.linspace(0, 1.0, 100 * num_taps, endpoint=True)
    rayleigh_taps = real_tap_function(new_time) + 1j * imag_tap_function(new_time)
    rayleigh_taps *= power_taps

    # Ensure that we maintain the same amount of power before and after the transform
    input_power = np.linalg.norm(data)
    data = sp.upfirdn(rayleigh_taps, data, up=100, down=100)[-data.shape[0] :]
    output_power = np.linalg.norm(data)
    data = np.multiply(input_power / output_power, data).astype(np.complex64)

    return data.astype(TorchSigComplexDataType)


def intermodulation_products(
    data: np.ndarray, coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Pass IQ data through an optimized memoryless nonlinear response model
    that creates local intermodulation distortion (IMD) products.

    Note that since only odd-order IMD products effectively fall in spectrum near the
    first-order (original) signal, only these are calculated.

    Args:
        data: Complex valued IQ data samples.
        coeffs: coefficients of memoryless IMD response such that
            y(t) = coeffs[0]*x(t) + coeffs[1]*(x(t)**2) + coeffs[2]*(x(t)**3) + ...
            Defaults to a third-order model: np.array([1.0, 1.0, 1.0]).

    Returns:
        IQ data with local IMD products.
    """
    coeffs = np.array([1.0, 0.0, 0.1]) if coeffs is None else coeffs
    if np.equal(coeffs.size, 0):
        raise IndexError("Coeffs has length zero.")

    model_order = coeffs.size
    distorted_data = np.zeros(len(data), dtype=TorchSigComplexDataType)

    # only odd-order distortion products are relevant local contributors
    for i in range(0, model_order, 1):
        if i > 0 and np.equal(np.mod(i, 2), 1) and not np.equal(coeffs[i], 0.0):
            raise ValueError("Even-order coefficients must be zero.")

        i_order_distortion = (np.abs(data) ** (i)) * data
        distorted_data += coeffs[i] * i_order_distortion

    # compute the change in spectral magnitudes in order to maintain the same SNR
    # on the other side of the transform
    win = sp.windows.blackmanharris(len(data))
    input_power = np.max(np.abs(np.fft.fft(data * win)))
    output_power = np.max(np.abs(np.fft.fft(distorted_data * win)))
    distorted_data *= input_power / output_power

    return distorted_data.astype(TorchSigComplexDataType)


def iq_imbalance(
    data: np.ndarray,
    amplitude_imbalance: float,
    phase_imbalance: float,
    dc_offset_db: float,
    dc_offset_phase_rads: float,
    noise_power_db: float | None = None,
) -> np.ndarray:
    """Applies IQ imbalance to IQ data.

    Args:
        data: IQ data.
        amplitude_imbalance: IQ amplitude imbalance in dB.
        phase_imbalance: IQ phase imbalance in radians [-pi, pi].
        dc_offset_db: Relative power of additive DC offset in dB.
        dc_offset_phase_rads: Phase of additive DC offset in radians.
        noise_power_db: Noise floor power in dB. Estimated internally if not provided. Defaults to None.

    Returns:
        IQ data with IQ Imbalance applied.
    """
    # amplitude imbalance
    data = 10 ** (amplitude_imbalance / 10.0) * np.real(data) + 1j * 10 ** (
        amplitude_imbalance / 10.0
    ) * np.imag(data)

    # phase imbalance
    data = np.exp(-1j * phase_imbalance / 2.0) * np.real(data) + np.exp(
        1j * (np.pi / 2.0 + phase_imbalance / 2.0)
    ) * np.imag(data)

    # DC offset

    if noise_power_db is not None:
        noise_floor_db = noise_power_db + 10 * np.log10(len(data))
    else:
        # compute FFT of receive signal
        data_fft_linear = np.abs(np.fft.fft(data))
        # apply smoothing
        avg_len = int(len(data_fft_linear) / 8)
        if np.equal(np.mod(avg_len, 2), 0):
            avg_len += 1
        avg = np.ones(avg_len) / avg_len
        data_fft_linear = sp.convolve(data_fft_linear, avg)[avg_len:-avg_len]
        # estimate noise floor
        noise_floor_db = 20 * np.log10(np.min(data_fft_linear))

    # create the DC offset
    dc_offset_tone = np.ones(len(data)) * np.exp(1j * dc_offset_phase_rads)
    # compute peak of FFT of DC offset
    dc_offset_tone_max_db = 20 * np.log10(np.max(np.abs(np.fft.fft(dc_offset_tone))))
    # calculate change to set DC offset power properly
    gain_change_db = (noise_floor_db - dc_offset_tone_max_db) + dc_offset_db
    # scale spur power accordingly
    gain_change = 10 ** (gain_change_db / 20)
    dc_offset_tone *= gain_change

    # add DC offset to signal
    data += dc_offset_tone

    return data.astype(TorchSigComplexDataType)


def interleave_complex(
    data: np.ndarray,
) -> np.ndarray:
    """Converts complex vectors into real interleaved IQ vector.

    Args:
        data: Input array of complex samples

    Returns:
        Real-valued array of interleaved IQ samples
    """
    output = np.empty(len(data) * 2)
    output[::2] = np.real(data)
    output[1::2] = np.imag(data)
    return output.astype(TorchSigRealDataType)


def carrier_frequency_drift(
    data: np.ndarray,
    drift_ppm: float = 1,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Carrier frequency drift from a Local Oscillator (LO), with drift modeled as accumulated gaussian random phase.

    Args:
        data: Complex valued IQ data samples.
        drift_ppm: Drift in parts per million (ppm). Default 1.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with LO drift applied.
    """
    rng = rng if rng else np.random.default_rng()
    n = data.size

    # convert drift PPM units
    drift = drift_ppm * 1e-6

    # randomize the instantaneous change in frequency
    frequency_drift = rng.normal(0, drift, n)

    # accumulate the changes into the frequency
    carrier_phase = np.cumsum(frequency_drift)

    # frequency drift effect now contained within the complex sinusoid
    drift_effect = np.exp(2j * np.pi * carrier_phase)

    # apply frequency drift effect
    data = data * drift_effect
    return data.astype(TorchSigComplexDataType)


def carrier_phase_noise(
    data: np.ndarray,
    phase_noise_degrees: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Carrier phase noise from a Local Oscillator (LO) with the noise modeled as a Gaussian RV.

    Args:
        data: Complex valued IQ data samples.
        phase_noise_degrees: Phase noise in degrees. Used as standard deviation for Gaussian distribution. Defaults to 1.0.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data mixed with noisy LO.
    """
    rng = rng if rng else np.random.default_rng()
    n = data.size

    # generate phase noise with given standard deviation
    phase_noise_degrees_array = rng.normal(0, phase_noise_degrees, n)

    # convert to radians
    phase_noise_radians_array = phase_noise_degrees_array * np.pi / 180

    # phase noise effect contained with a complex sinusoid
    phase_noise_effect = np.exp(1j * phase_noise_radians_array)

    # apply phase noise effect
    data = data * phase_noise_effect
    return data.astype(TorchSigComplexDataType)


def nonlinear_amplifier(
    data: np.ndarray,
    gain: float = 1.0,
    psat_backoff: float = 10.0,
    phi_max: float = 0.1,
    phi_slope: float = 0.01,
    auto_scale: bool = True,
) -> np.ndarray:
    """A memoryless AM/AM, AM/PM nonlinear amplifier function-based model using a
    hyperbolic tangent output power response defined by gain and saturation power,
    and a hyperbolic tangent phase response defined by maximum relative phase shift.

    Args:
        data: Complex valued IQ data samples.
        gain: Small-signal linear gain. Default 1.0.
        psat_backoff: Saturated output power factor relative to the input signal
            mean power. That is, Psat = psat_backoff * Pavg. For example, operating at
            a 2.0 psat_backoff factor with a 1 W mean power signal has saturation power
            level at 2.0 W. Default 10.0.
        phi_max: Signal maximum relative phase shift in saturation (radians). Default 0.1.
        phi_slope: Absolute slope of relative phase linear response region (W/radian). Default 0.01.
        auto_scale: Automatically rescale output power to match full-scale peak
            input power prior to transform, based on peak estimates. Default True.

    Returns:
        Nonlinearly distorted IQ data.
    """
    n = len(data)
    magnitude = np.abs(data)
    phase = np.angle(data)
    in_power = magnitude**2
    mean_power_est = np.mean(in_power)

    # amplitude-to-amplitude modulation (AM/AM)
    # hyperbolic tangent power response passes
    # through (0,0) and asymptotically approaches psat
    psat = mean_power_est * psat_backoff
    scale_factor = psat / gain
    out_power = psat * np.tanh(in_power / scale_factor)
    out_magnitude = out_power**0.5

    # amplitude-to-phase modulation (AM/PM)
    # hyperbolic tangent phase response approaches
    # zero relative phase shift at low power input
    # and approaches phimax phase shift in saturation
    phi_shift = 0.0
    if not np.equal(phi_max, 0.0) and not np.equal(phi_slope, 0.0):
        phi_slope = np.abs(phi_slope)

        # align AM and PM responses
        # place linear phase shift regime origin where ideal
        # amplifier linear gain would have output psat
        pin_origin = scale_factor  # ie, psat / gain

        # relative phase shift: tanh response on input power scale
        phi_shift = (phi_max / 2) * (np.tanh((in_power - pin_origin) / phi_slope) + 1)

    # reconstruct complex valued data format
    amp_data = out_magnitude * np.exp(1j * (phase + phi_shift))

    # auto_scale: rescale output power to match full-scale input power
    # by estimating peaks for input and output power
    if auto_scale:
        win = sp.windows.blackmanharris(n)
        input_power = np.max(np.abs(np.fft.fft(data * win)))
        output_power = np.max(np.abs(np.fft.fft(amp_data * win)))
        amp_data *= input_power / output_power

    return amp_data.astype(TorchSigComplexDataType)


def nonlinear_amplifier_table(
    data: np.ndarray,
    p_in: np.ndarray | None = None,
    p_out: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    auto_scale: bool = False,
) -> np.ndarray:
    """A nonlinear amplifier (AM/AM, AM/PM) memoryless model that distorts an input
    complex signal to simulate an amplifier response, based on interpolating a table of
    provided power input, power output, and phase change data points.

    Default very small model parameters depict a 10 dB gain amplifier with P1dB = 9.0 dBW.
        p_in =  10**((np.array([-100., -20., -10.,  0.,  5., 10. ]) / 10))
        p_out = 10**((np.array([ -90., -10.,   0.,  9., 9.9, 10. ]) / 10))
        phi = np.deg2rad(np.array([0., -2., -4., 7., 12., 23.]))

    Args:
        data: Complex valued IQ data samples.
        p_in: Model signal power input points. Assumes sorted ascending linear values (Watts).
        p_out: Model power out corresponding to p_in points (Watts).
        phi: Model output phase shift values (radians) corresponding to p_in points.
        auto_scale: Automatically rescale output power to match full-scale peak
            input power prior to transform, based on peak estimates. Default False.

    Raises:
        ValueError: If model array arguments are not the same size.

    Returns:
        Nonlinearly distorted IQ data.
    """
    p_in = 10 ** (np.array([-100.0, -20.0, -10.0, 0.0, 5.0, 10.0]) / 10) if p_in is None else p_in
    p_out = 10 ** (np.array([-90.0, -10.0, 0.0, 9.0, 9.9, 10.0]) / 10) if p_out is None else p_out
    phi = np.deg2rad(np.array([0.0, -2.0, -4.0, 7.0, 12.0, 23.0])) if phi is None else phi

    if len(p_in) != len(p_out) or len(p_in) != len(phi):
        raise ValueError("Model array arguments are not the same size.")

    magnitude = np.abs(data)
    phase = np.angle(data)

    # amplitude-to-amplitude modulation (AM/AM)
    in_power = magnitude**2
    out_power = np.interp(in_power, p_in, p_out)
    out_magnitude = out_power**0.5

    # amplitude-to-phase modulation (AM/PM)
    out_phase_shift_rad = np.interp(in_power, p_in, phi)

    amp_data = out_magnitude * np.exp(1j * (phase + out_phase_shift_rad))

    # auto_scale: rescale output power to match full-scale input power
    # by estimating peaks for input and output power
    if auto_scale:
        win = sp.windows.blackmanharris(len(data))
        input_power = np.max(np.abs(np.fft.fft(data * win)))
        output_power = np.max(np.abs(np.fft.fft(amp_data * win)))
        amp_data *= input_power / output_power

    return amp_data.astype(TorchSigComplexDataType)


def normalize(
    data: np.ndarray,
    norm_order: float | Literal["fro", "nuc"] | None = 2,
    flatten: bool = False,
) -> np.ndarray:
    """Scale data so that a specified norm computes to 1.

    For detailed information, see :func:`numpy.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        data: (batch_size, vector_length, ...)-sized data to be normalized.
        norm_order: norm order to be passed to np.linalg.norm. Defaults to 2.
        flatten: boolean specifying if the input array's norm should be calculated on the flattened representation of the input data. Defaults to False.

    Returns:
        Normalized complex array data.
    """
    if flatten:
        flat_data = data.reshape(data.size)
        norm = np.linalg.norm(flat_data, norm_order, keepdims=True)
        return np.multiply(data, 1.0 / norm)

    norm = np.linalg.norm(data, norm_order, keepdims=True)
    return np.multiply(data, 1.0 / norm).astype(TorchSigComplexDataType)


def passband_ripple(
    data: np.ndarray,
    num_taps: int = 2,
    max_ripple_db: float = 2.0,
    coefficient_decay_rate: float = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Functional for passband ripple transforms.

    This function applies a passband ripple effect to the input data by designing a filter
    with specified ripple characteristics and applying it to the data.

    Args:
        data: Complex valued IQ data samples.
        num_taps: Number of taps in simulated filter. Defaults to 2.
        max_ripple_db: Maximum allowed ripple in the simulated filter (in dB). Defaults to 2.0.
        coefficient_decay_rate: The decay rate of the exponential weighting in the filter. Defaults to 1.0.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Raises:
        ValueError: When filter cannot meet ripple spec within a set number of iterations.

    Returns:
        Filtered data with passband ripple applied.
    """
    rng = rng if rng else np.random.default_rng()
    # counter avoids infinite loop
    counter = 0
    max_counter = 1000
    # initialize estimate such that it always enters loop
    estimate_ripple_db = max_ripple_db + 1

    while estimate_ripple_db > max_ripple_db and counter < 1000:
        # designs the weights: complex gaussian with exponential decay
        gaussian = rng.normal(0, 1, num_taps) + 1j * rng.normal(0, 1, num_taps)
        decay = np.exp(-coefficient_decay_rate * np.arange(0, num_taps))
        weights = gaussian * decay
        # scale weights to have average 0 dB level
        weights /= np.max(np.abs(weights))
        # determine if passband ripple meets spec
        fft_db = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(weights, 1024))))
        estimate_ripple_db = np.max(fft_db) - np.min(fft_db)
        # increment counter
        counter += 1

    if counter >= max_counter:
        raise ValueError("Passband ripple was unable to meet ripple specs.")

    # apply filter
    data = dsp.convolve(data, weights)

    return data.astype(TorchSigComplexDataType)


def patch_shuffle(
    data: np.ndarray,
    patch_size: int,
    patches_to_shuffle: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`.

    This function divides the input data into patches of specified size and shuffles
    the data within each patch according to the provided indices.

    Args:
        data: (batch_size, vector_length, ...)-sized data.
        patch_size: Size of each patch to shuffle.
        patches_to_shuffle: Index of each patch of size patch_size to shuffle.
        rng: Random Generator to use. Defaults to None (new generator created internally).

    Returns:
        Data that has undergone patch shuffling.
    """
    rng = rng if rng else np.random.default_rng()

    for patch_idx in patches_to_shuffle:
        patch_start = int(patch_idx * patch_size)
        patch = data[patch_start : patch_start + patch_size]
        rng.shuffle(patch)
        data[patch_start : patch_start + patch_size] = patch

    return data.astype(TorchSigComplexDataType)


def phase_offset(data: np.ndarray, phase: float) -> np.ndarray:
    """Applies a phase rotation to data.

    This function multiplies the input data by a complex exponential to apply a phase rotation.

    Args:
        data: IQ data.
        phase: phase to rotate sample in [-pi, pi].

    Returns:
        Data that has undergone a phase rotation.
    """
    return (data * np.exp(1j * phase)).astype(TorchSigComplexDataType)


def quantize(
    data: np.ndarray,
    num_bits: int,
    ref_level_adjustment_db: float = 0.0,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """Quantize input to number of levels specified.

    This function quantizes the input data to a specified number of bits, with options
    for reference level adjustment and rounding mode.

    Default implementation is ceiling.

    Args:
        data: IQ data.
        num_bits: Number of bits to simulate.
        ref_level_adjustment_db: Changes the relative scaling of the input. For example, ref_level_adjustment_db = 3.0,
            the average power is now 3 dB *above* full scale and into saturation. For ref_level_adjustment_db = -3.0, the average
            power is now 3 dB *below* full scale and simulates a loss of dynamic range. Default is 0.
        rounding_mode: Represents either rounding to 'floor' or 'ceiling'. Default is 'floor'.

    Raises:
        ValueError: Invalid round type.
        TypeError: If num_bits is not an integer.

    Returns:
        Quantized IQ data.
    """
    if not isinstance(num_bits, int):
        raise TypeError("quantize() num_bits must be an integer.")

    # calculate number of levels
    num_levels = int(2**num_bits)

    # establish quantization levels
    quant_levels = np.arange(-num_levels // 2, num_levels // 2) / (num_levels // 2)

    # the distance between two quantization levels
    quant_level_distance = quant_levels[1] - quant_levels[0]

    # determine threshold levels
    if rounding_mode == "floor":
        threshold_levels = quant_levels + (quant_level_distance / 2)
    elif rounding_mode == "ceiling":
        threshold_levels = quant_levels - (quant_level_distance / 2)
    else:
        raise ValueError(
            f"quantize() rounding mode is: {rounding_mode}, must be ceiling or floor"
        )

    # determine maximum value of signal amplitude
    max_value_signal_real = np.max(np.abs(data.real))
    max_value_signal_imag = np.max(np.abs(data.imag))
    max_value_signal = np.max((max_value_signal_real, max_value_signal_imag))

    # convert the reference level adjustment into a linear value.
    # +3 dB -> 3 dB above max scaling (saturation)
    # -3 dB -> 3 dB below max scaling (dynamic range loss)
    ref_level_adjustment_linear = 10 ** (ref_level_adjustment_db / 20)

    # scale the input signal
    input_signal_scaled = data * ref_level_adjustment_linear / max_value_signal

    # quantize real and imag seperately
    quant_signal_real = np.zeros(len(data), dtype=TorchSigRealDataType)
    quant_signal_imag = np.zeros(len(data), dtype=TorchSigRealDataType)

    input_signal_scaled_real = input_signal_scaled.real
    input_signal_scaled_imag = input_signal_scaled.imag

    # check for saturated values minimum
    real_saturation_neg_index = np.where(
        input_signal_scaled_real <= threshold_levels[0]
    )[0]
    imag_saturation_neg_index = np.where(
        input_signal_scaled_imag <= threshold_levels[0]
    )[0]
    quant_signal_real[real_saturation_neg_index] = quant_levels[0]
    quant_signal_imag[imag_saturation_neg_index] = quant_levels[0]

    # check for saturated values maximum
    real_saturation_pos_index = np.where(
        input_signal_scaled_real >= threshold_levels[-1]
    )[0]
    imag_saturation_pos_index = np.where(
        input_signal_scaled_imag >= threshold_levels[-1]
    )[0]
    quant_signal_real[real_saturation_pos_index] = quant_levels[-1]
    quant_signal_imag[imag_saturation_pos_index] = quant_levels[-1]

    # calculate which remaining indicies have not yet been quantized
    all_index = np.arange(0, len(data))
    remaining_real_index = np.setdiff1d(all_index, real_saturation_neg_index)
    remaining_real_index = np.setdiff1d(remaining_real_index, real_saturation_pos_index)
    remaining_imag_index = np.setdiff1d(all_index, imag_saturation_neg_index)
    remaining_imag_index = np.setdiff1d(remaining_imag_index, imag_saturation_pos_index)

    # quantize all other levels. by default implements "ceiling"
    real_index_subset = np.digitize(
        input_signal_scaled_real[remaining_real_index], threshold_levels
    )
    imag_index_subset = np.digitize(
        input_signal_scaled_imag[remaining_imag_index], threshold_levels
    )

    quant_signal_real[remaining_real_index] = quant_levels[real_index_subset]
    quant_signal_imag[remaining_imag_index] = quant_levels[imag_index_subset]

    # form the quantized IQ samples
    quantized_data = quant_signal_real + 1j * quant_signal_imag

    # undo quantization-based scaling
    data_unscaled = quantized_data * max_value_signal / ref_level_adjustment_linear

    return data_unscaled.astype(TorchSigComplexDataType)


def shadowing(
    data: np.ndarray,
    mean_db: float = 4.0,
    sigma_db: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Applies RF shadowing to the data, assuming the channel obstructions' loss are lognormal.

    This function models RF shadowing effects by applying lognormal fading to the input data.

    Refer to T.S. Rappaport, Wireless Communications, Prentice Hall, 2002.

    Args:
        data: Complex valued IQ data samples.
        mean_db: Mean value of shadowing in dB. Default 4.0.
        sigma_db: Shadowing standard deviation. Default 2.0.
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        Data with shadowing applied.
    """
    rng = rng if rng else np.random.default_rng()
    power_db = rng.normal(mean_db, sigma_db)  # normal distribution in log domain
    data = data * 10 ** (power_db / 20)
    return data.astype(TorchSigComplexDataType)


def spectral_inversion(data: np.ndarray) -> np.ndarray:
    """Applies a spectral inversion to input data.

    This function performs spectral inversion by complex conjugation of the input data.

    Args:
        data: IQ data.

    Returns:
        Spectrally inverted data.
    """
    # apply conjugation to perform spectral inversion
    return np.conj(data)


def spectrogram(data: np.ndarray, fft_size: int, fft_stride: int) -> np.ndarray:
    """Computes spectrogram from IQ data.

    This function computes the spectrogram by applying the Short-Time Fourier Transform (STFT)
    to the input IQ data.

    Directly uses `compute_spectrogram` inside of utils/dsp.py.

    Args:
        data: IQ samples.
        fft_size: The FFT size (number of bins) in the spectrogram.
        fft_stride: The number of data points to move or "hop" over when computing the next FFT.

    Returns:
        Spectrogram computed from IQ data.
    """
    return dsp.compute_spectrogram(data, fft_size=fft_size, fft_stride=fft_stride)


def spectrogram_drop_samples(
    data: np.ndarray, drop_starts: np.ndarray, drop_sizes: np.ndarray, fill: str
) -> np.ndarray:
    """Drop samples at given locations/durations with fill technique.

    This function drops samples at specified locations and fills them with the specified technique.

    Supported Fill Techniques:
        ffill: Forward Fill. Use value at sample one before start.
        bfill: Backwards Fill. Use value at sample one after end.
        mean: Mean Fill. Use data mean.
        zero: Zero Fill. Use 0.
        min: Minimum observed value fill.
        max: Maximum observed value fill
        low: Fixed low value fill. Use np.ones * 1e-3.
        ones: Ones fill. Use np.ones.

    Args:
        data: IQ data.
        drop_starts: Start indices of drops.
        drop_sizes: Durations for each start index.
        fill: Drop sample replacement method.

    Raises:
        ValueError: Invalid fill type.

    Returns:
        Data array with fill values during drops.
    """
    flat_spec = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region_real = np.ones(drop_sizes[idx]) * flat_spec[0, drop_start - 1]
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start - 1]
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[1, drop_start : drop_start + drop_sizes[idx]] = (
                drop_region_complex
            )
        elif fill == "bfill":
            drop_region_real = (
                np.ones(drop_sizes[idx]) * flat_spec[0, drop_start + drop_sizes[idx]]
            )
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start + drop_sizes[idx]]
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[1, drop_start : drop_start + drop_sizes[idx]] = (
                drop_region_complex
            )
        elif fill == "mean":
            drop_region_real = np.ones(drop_sizes[idx]) * np.mean(flat_spec[0])
            drop_region_complex = np.ones(drop_sizes[idx]) * np.mean(flat_spec[1])
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[1, drop_start : drop_start + drop_sizes[idx]] = (
                drop_region_complex
            )
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx])
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        elif fill == "min":
            drop_region_real = np.ones(drop_sizes[idx]) * np.min(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.min(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[1, drop_start : drop_start + drop_sizes[idx]] = (
                drop_region_complex
            )
        elif fill == "max":
            drop_region_real = np.ones(drop_sizes[idx]) * np.max(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.max(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[1, drop_start : drop_start + drop_sizes[idx]] = (
                drop_region_complex
            )
        elif fill == "low":
            drop_region = np.ones(drop_sizes[idx]) * 1e-3
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        elif fill == "ones":
            drop_region = np.ones(drop_sizes[idx])
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        else:
            raise ValueError(
                f"fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {fill}"
            )
    return flat_spec.reshape(data.shape[0], data.shape[1], data.shape[2])


def spectrogram_image(
    data: np.ndarray, fft_size: int, fft_stride: int, black_hot: bool = True
) -> np.ndarray:
    """Creates spectrogram from IQ samples.

    This function computes the spectrogram and converts it to a grayscale image.

    Args:
        data: IQ samples.
        fft_size: The FFT size (number of bins) in the spectrogram.
        fft_stride: The number of data points to move or "hop" over when computing the next FFT.
        black_hot: Toggles black hot spectrogram. Defaults to True (black hot).

    Returns:
        Spectrogram image in BGR format.
    """
    # compute the spectrogram in dB
    spectrogram_db = spectrogram(data, fft_size=fft_size, fft_stride=fft_stride)

    # convert to grey-scale image
    img = np.zeros(
        (spectrogram_db.shape[0], spectrogram_db.shape[1], 3), dtype=np.float32
    )
    img = cv2.normalize(spectrogram_db, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if black_hot:
        img = cv2.bitwise_not(img, img)

    return img


def spurs(
    data: np.ndarray,
    sample_rate: float = 1,
    center_freqs=[0.25],
    relative_power_db=[3],
    noise_power_db: float | None = None,
) -> np.ndarray:
    """Adds spurs to the input data.

    This function adds spurious signals (tones) at specified frequencies with specified power levels.

    Args:
        data: IQ data samples.
        sample_rate: Sample rate associated with the samples. Defaults to 1.
        center_freqs: Center frequencies for the spurs. Defaults to [0.25].
        relative_power_db: Relative power of spurs in dB to noise floor. Defaults to [3].
        noise_power_db: Noise floor power in dB. Estimated internally if not provided. Defaults to None.

    Returns:
        IQ data with spurs (tones) added.

    Raises:
        ValueError: If center_freqs are outside the valid range or if lengths don't match.
    """
    # convert center_freqs and relative_power to arrays if received as scalars
    center_freqs_array = np.array([center_freqs]) if np.isscalar(center_freqs) else center_freqs
    relative_power_db_array = np.array([relative_power_db]) if np.isscalar(relative_power_db) else relative_power_db

    # error checking
    if (np.array(center_freqs_array) >= sample_rate / 2).any():
        raise ValueError(f"center_freqs must be < sample rate / 2 = {sample_rate/2}")
    if (np.array(center_freqs_array) <= -sample_rate / 2).any():
        raise ValueError(f"center_freqs must be >= -sample rate / 2 = {-sample_rate/2}")
    if len(relative_power_db_array) != len(center_freqs_array):
        raise ValueError(
            f"len(center_freqs) = {len(center_freqs_array)}, must be same length as len(relative_power_db) = {len(relative_power_db_array)}"
        )

    # create copy of data since it will be modified
    output = copy(data)

    if noise_power_db is not None:
        noise_floor_db = noise_power_db + 10 * np.log10(len(data))
    else:
        # compute FFT of receive signal
        data_fft_db = 20 * np.log10(np.abs(np.fft.fft(data)))
        # apply smoothing
        avg_len = int(len(data_fft_db) / 8)
        if np.equal(np.mod(avg_len, 2), 0):
            avg_len += 1
        avg = np.ones(avg_len) / avg_len
        data_fft_db = sp.convolve(data_fft_db, avg)[avg_len:-avg_len]
        # estimate noise floor
        noise_floor_db = np.min(data_fft_db)

    # generate spurs
    for spur_index, center_freq in enumerate(center_freqs_array):
        # create the spur
        spur = np.exp(
            2j * np.pi * (center_freq / sample_rate) * np.arange(0, len(data))
        )
        # compute FFT of spur
        spur_fft_db = 20 * np.log10(np.abs(np.fft.fft(spur)))
        # calculate peak value
        spur_max_db = np.max(spur_fft_db)
        # calculate change to set spur power properly
        gain_change_db = (noise_floor_db - spur_max_db) + relative_power_db_array[
            spur_index
        ]
        # scale spur power accordingly
        gain_change = 10 ** (gain_change_db / 20)
        spur *= gain_change
        # add spur to signal
        output += spur

    return output.astype(TorchSigComplexDataType)


def time_reversal(data: np.ndarray) -> np.ndarray:
    """Applies time reversal to data (flips horizontally).

    This function reverses the time axis of the input data.

    Args:
        data: IQ data.

    Returns:
        Time flipped IQ data.
    """
    return np.flip(data, axis=0).astype(TorchSigComplexDataType)


def time_varying_noise(
    data: np.ndarray,
    noise_power_low: float,
    noise_power_high: float,
    inflections: int,
    random_regions: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adds time-varying complex additive white Gaussian noise.

    This function adds noise with power levels that vary over time, with specified
    minimum and maximum power levels and number of inflection points.

    Args:
        data: IQ data.
        noise_power_low: Minimum noise power in dB.
        noise_power_high: Maximum noise power in dB.
        inflections: Number of inflection points over IQ data.
        random_regions: Inflection points spread randomly (True) or evenly (False).
        rng: Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        IQ data with time-varying noise added.
    """
    rng = rng if rng else np.random.default_rng()
    real_noise = rng.standard_normal(*data.shape)
    imag_noise = rng.standard_normal(*data.shape)
    noise_power = np.zeros(data.shape)

    if not inflections:  # inflections == 0:
        inflection_indices = np.array([0, data.shape[0]])
    elif random_regions:
        inflection_indices = np.sort(
            rng.choice(data.shape[0], size=inflections, replace=False)
        )
        inflection_indices = np.append(inflection_indices, data.shape[0])
        inflection_indices = np.insert(inflection_indices, 0, 0)
    else:
        inflection_indices = np.arange(inflections + 2) * int(
            data.shape[0] / (inflections + 1)
        )

    for idx in range(len(inflection_indices) - 1):
        start_idx = inflection_indices[idx]
        stop_idx = inflection_indices[idx + 1]
        duration = stop_idx - start_idx
        start_power = (
            noise_power_low if not idx % 2 else noise_power_high
        )  # idx % 2 == 0
        stop_power = noise_power_high if not idx % 2 else noise_power_low
        noise_power[start_idx:stop_idx] = np.linspace(start_power, stop_power, duration)

    return (
        data
        + (10.0 ** (noise_power / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)
    ).astype(TorchSigComplexDataType)
