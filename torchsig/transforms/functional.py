"""Functional transforms for reuse and custom fine-grained control
"""
from typing import Literal, Optional, Tuple

# TorchSig
import torchsig.utils.dsp as dsp
from torchsig.utils.dsp import torchsig_complex_data_type

# Third Party
import scipy
from scipy import signal as sp
import numpy as np

__all__ = [
    "add_slope",
    "agc",
    "block_agc",
    "channel_swap",
    "complex_to_2d",    
    "cut_out",
    "drop_samples",
    "fading",
    "iq_imbalance",
    "local_oscillator_frequency_drift",
    "local_oscillator_phase_noise",
    "mag_rescale",
    "nonlinear_amplifier",
    "normalize",    
    "passband_ripple",
    "patch_shuffle",
    "phase_offset",
    "quantize",
    "spectral_inversion",
    "spectrogram",
    "spectrogram_drop_samples",
    "time_reversal",
    "time_varying_noise"
]


def add_slope(
    data: np.ndarray
) -> np.ndarray:
    """Add slope between each sample and its preceding sample is added to every sample.

    Augmentation has the effect of amplifying high frequency component more than lower
    frequency components.

    Args:
        data (np.ndarray): IQ data.

    Returns:
        np.ndarray: IQ data with added slope.

    """  
    slope = np.diff(data)
    slope = np.insert(slope, 0, 0)
    return (data + slope).astype(torchsig_complex_data_type)


def agc(
    data: np.ndarray,
    initial_gain_db: float,
    alpha_smooth: float,
    alpha_track: float,
    alpha_overflow: float,
    alpha_acquire: float,
    ref_level_db: float,
    track_range_db: float,
    low_level_db: float,
    high_level_db: float,
) -> np.ndarray:
    """Automatic Gain Control algorithm (deterministic).

    Args:
        data (np.ndarray): IQ data samples.
        initial_gain_db (float): Inital gain value in dB.
        alpha_smooth (float): Alpha for avergaing the measure signal level `level_n = level_n * alpha + level_n-1(1-alpha)`
        alpha_track (float): Amount to adjust gain when in tracking state.
        alpha_overflow (float): Amount to adjust gain when in overflow state `[level_db + gain_db] >= max_level`.
        alpha_acquire (float): Amount to adjust gain when in acquire state.
        ref_level_db (float): Reference level goal for algorithm to achieve, in dB units. 
        track_range_db (float): dB range for operating in tracking state.
        low_level_db (float): minimum magnitude value (dB) to perform any gain control adjustment.
        high_level_db (float): magnitude value (dB) to enter overflow state.

    Returns:
        np.ndarray: IQ data adjusted sample-by-sample by the AGC algorithm.

    """
    output = np.zeros_like(data)
    gain_db = initial_gain_db
    level_db = 0.0
    for sample_idx, sample in enumerate(data):
        if not np.abs(sample): # sample == 0
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
        output[sample_idx] = data[sample_idx] * np.exp(gain_db)

    return output.astype(torchsig_complex_data_type)


def awgn(data: np.ndarray, 
         noise_power_db: float,
         rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
    """Adds zero-mean complex additive white Gaussian noise with power of
    noise_power_db.

    Args:
        data: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized data.

        noise_power_db (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

        random_generator (Optional[np.random.Generator], optional): 
            Random Generator to use. Defaults to None (new generator 
            created internally).

    Returns:
        np.ndarray: Data with added noise.

    """
    rng = rng if rng else np.random.default_rng()

    real_noise = rng.standard_normal(*data.shape)
    imag_noise = rng.standard_normal(*data.shape)
    return (data + (10.0 ** (noise_power_db / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)).astype(torchsig_complex_data_type)


# TODO: block_agc is very similar to mag_rescale, and
# does not actually perform any AGC - consider consolidating
def block_agc(
    data: np.ndarray,
    gain_change_db: float,
    start_idx: int
) -> np.ndarray:
    """Implements a large instantaneous jump in receiver gain.

    Args:
        data (np.ndarray): IQ data.
        gain_change_db (float): Gain value to change in dB.
        start_idx (np.ndarray): Start index for IQ data.

    Returns:
        np.ndarray: IQ data with Block AGC applied.

    """    
    # convert db to linear units
    gain_change_linear = 10**(gain_change_db/10)

    data[start_idx:] *= gain_change_linear

    return data.astype(torchsig_complex_data_type)


def channel_swap(
    data: np.ndarray
) -> np.ndarray:
    """Swap I and Q channels of IQ data.

    Args:
        data (np.ndarray): IQ data.

    Returns:
        np.ndarray: IQ data with channels swapped.

    """
    real_component = data.real
    imag_component = data.imag
    new_data = np.empty(data.shape, dtype=data.dtype)
    new_data.real = imag_component
    new_data.imag = real_component
    return new_data.astype(torchsig_complex_data_type)


def complex_to_2d(data: np.ndarray) -> np.ndarray:
    """Converts IQ data to two channels (real and imaginary parts).
    """
    return np.stack([data.real, data.imag], axis=0)
    

def cut_out(
    data: np.ndarray,
    cut_start: float,
    cut_duration: float,
    cut_type: str,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Performs CutOut: replacing values with fill.

    Args:
        data (np.ndarray): IQ data
        cut_start (float): Normalized start of cut region [0.0, 1.0)
        cut_duration (float): Normalized duration of cut region (0.0, 1.0)
        cut_type (str): Type of data to fill cut region.
        * zeros
        * ones
        * low_noise
        * avg_noise
        * high_noise

    Raises:
        ValueError: Invalid `cut_type`.

    Returns:
        np.ndarray: CutOut IQ data.

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
        raise ValueError(f"cut_type must be: zeros, ones, low_noise, avg_noise, or high_noise. Found: {cut_type}")

    # Insert cut mask into data
    data[cut_start : cut_start + cut_mask_length] = cut_mask

    return data.astype(torchsig_complex_data_type)


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
        data (np.ndarray): IQ data.
        drop_starts (np.ndarray): Start indicies of drops.
        drop_sizes (np.ndarray): Durations for each start index.
        fill (str): Drop sample replacement method.

    Raises:
        ValueError: Invalid fill type.

    Returns:
        np.ndarray: data array with fill values during drops.

    """
    for idx, start in enumerate(drop_starts):
        stop = start + drop_sizes[idx]

        if fill == "ffill":
            drop_region = np.full(
                drop_sizes[idx], 
                data[start - 1], 
                dtype=np.complex128
            )
        elif fill == "bfill":
            drop_region = np.full(
                drop_sizes[idx],
                data[stop],
                dtype=np.complex128,
            )
        elif fill == "mean":
            drop_region = np.full(
                drop_sizes[idx], 
                np.mean(data), 
                dtype=np.complex128
            )
        elif fill == "zero":
            drop_region = np.zeros(
                drop_sizes[idx], 
                dtype=np.complex128
            )
        else:
            raise ValueError(f"{fill} fill type unsupported. Must be ffill, bfill, mean, or zero.")
        
        data[start:stop] = drop_region

    return data.astype(torchsig_complex_data_type)


def fading(
    data: np.ndarray, 
    coherence_bandwidth: float, 
    power_delay_profile: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """Apply fading channel to signal. Currently only does Rayleigh fading.

    Taps are generated by interpolating and filtering Gaussian taps.

    TODO:
        implement other types of fading.

    Args:
        data (np.ndarray): IQ data.
        coherence_bandwidth (float): coherence bandwidth relative to sample rate [0, 1.0].
        power_delay_profile (np.ndarray): power delay profile assign to channel.
        rng (Optional[np.random.Generator], optional): 
            Random Generator to use. Defaults to None (new generator 
            created internally).

    Returns:
        np.ndarray: IQ data with fading applied.

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
    rayleigh_taps = rng.standard_normal(num_taps) + 1j * rng.standard_normal(num_taps)  # multi-path channel

    # Linear interpolate taps by a factor of 100 -- so we can get accurate coherence bandwidths
    old_time = np.linspace(0, 1.0, num_taps, endpoint=True)
    real_tap_function = scipy.interpolate.interp1d(old_time, rayleigh_taps.real)
    imag_tap_function = scipy.interpolate.interp1d(old_time, rayleigh_taps.imag)

    new_time = np.linspace(0, 1.0, 100 * num_taps, endpoint=True)
    rayleigh_taps = real_tap_function(new_time) + 1j * imag_tap_function(new_time)
    rayleigh_taps *= power_taps

    # Ensure that we maintain the same amount of power before and after the transform
    input_power = np.linalg.norm(data)
    data = sp.upfirdn(rayleigh_taps, data, up=100, down=100)[-data.shape[0] :]
    output_power = np.linalg.norm(data)
    data = np.multiply(input_power / output_power, data).astype(np.complex64)

    return data.astype(torchsig_complex_data_type)


def iq_imbalance(
    data: np.ndarray, 
    amplitude_imbalance: float, 
    phase_imbalance: float, 
    dc_offset: Tuple[float, float]
) -> np.ndarray:
    """Applies IQ imbalance to IQ data.

    Args:
        data (np.ndarray): IQ data.
        amplitude_imbalance (float): IQ amplitude imbalance in dB.
        phase_imbalance (float): IQ phase imbalance in radians [-pi, pi].
        dc_offset (Tuple[float, float]): IQ DC (linear) offsets (In-Phase, Quadrature).

    Returns:
        np.ndarray: IQ data with IQ Imbalance applied.

    """
    # amplitude imbalance
    data = 10 ** (amplitude_imbalance / 10.0) * np.real(data) + 1j * 10 ** (amplitude_imbalance / 10.0) * np.imag(data)

    # phase imbalance
    data = np.exp(-1j * phase_imbalance / 2.0) * np.real(data) + np.exp(1j * (np.pi / 2.0 + phase_imbalance / 2.0)) * np.imag(data)

    # DC offset
    data = (dc_offset[0] + np.real(data)) + 1j * (dc_offset[1] + np.imag(data))

    return data.astype(torchsig_complex_data_type)

def local_oscillator_frequency_drift():
    """Unimplemented Functional for modeling Local Oscillator drift in frequency.
    """   

def local_oscillator_phase_noise():
    """Unimplemented Functional for modeling Local Oscillator phase noise.
    """   

def mag_rescale(
    data: np.ndarray,
    start: float | int,
    scale: float
) -> np.ndarray:
    """Apply rescaling of input `rescale` starting at time `start`.

    Args:
        data (np.ndarray): IQ data.
        start (float | int): Start time of rescaling.
        * If int, treated as array index.
        * If float, treated as normalized start time.
        scale (float): Scaling factor.

    Returns:
        np.ndarray: data rescaled.

    """    
    if isinstance(start, float):
        start = int(data.shape[0] * start)
    
    data[start:] *= scale

    return data.astype(torchsig_complex_data_type)


def nonlinear_amplifier():
    """Unimplemented Functional for memoryless nonlinear amplifier response.
    """     


def normalize(
    data: np.ndarray,
    norm_order: Optional[float | int | Literal["fro", "nuc"]] = 2,
    flatten: bool = False,
) -> np.ndarray:
    """Scale data so that a specfied norm computes to 1. For detailed information, see :func:`numpy.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        data (np.ndarray):
            (batch_size, vector_length, ...)-sized data to be normalized.

        norm_order (int):
            norm order to be passed to np.linalg.norm

        flatten (bool):
            boolean specifying if the input array's norm should be calculated on the flattened representation of the input data

    Returns:
        np.ndarray: Normalized complex array data.

    """
    if flatten:
        flat_data = data.reshape(data.size)
        norm = np.linalg.norm(flat_data, norm_order, keepdims=True)
        return np.multiply(data, 1.0 / norm)

    norm = np.linalg.norm(data, norm_order, keepdims=True)
    return np.multiply(data, 1.0 / norm).astype(torchsig_complex_data_type)


def passband_ripple():
    """Unimplemented Functional to create passband ripple filter effects within the sampling bandwidth.
    """  


def patch_shuffle(
    data: np.ndarray,
    patch_size: int,
    patches_to_shuffle: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`.

    Args:
        data: (np.ndarray):
            (batch_size, vector_length, ...)-sized data.

        patch_size (int):
            Size of each patch to shuffle.

        patches_to_shuffle (np.ndarray):
            Index of each patch of size patch_size to shuffle.

        random_generator (Optional[np.random.Generator], optional): 
            Random Generator to use. Defaults to None (new generator 
            created internally).

    Returns:
        np.ndarray: Data that has undergone patch shuffling.

    """
    rng = rng if rng else np.random.default_rng()

    for patch_idx in patches_to_shuffle:
        patch_start = int(patch_idx * patch_size)
        patch = data[patch_start : patch_start + patch_size]
        rng.shuffle(patch)
        data[patch_start : patch_start + patch_size] = patch

    return data.astype(torchsig_complex_data_type)


def phase_offset(
        data: np.ndarray, 
        phase: float
) -> np.ndarray:
    """Applies a phase rotation to data.

    Args:
        data (np.ndarray): IQ data.

        phase (float): phase to rotate sample in [-pi, pi].

    Returns:
        np.ndarray: Data that has undergone a phase rotation.

    """
    return (data * np.exp(1j * phase)).astype(torchsig_complex_data_type)


def quantize(
    data: np.ndarray,
    num_levels: int,
    round_type: str = "ceiling"
) -> np.ndarray:
    """Quantize input to number of levels specified.

    Default implementation is ceiling.

    Args:
        data (np.ndarray): IQ data.
        num_levels (int): Number of quantization levels
        round_type (str, optional): Quantization rounding. Must be one of 
            'floor', 'nearest' or 'ceiling'. Defaults to 'ceiling'.

    Raises:
        ValueError: Invalid round type.
        
    Returns:
        np.ndarray: Quantized IQ data.

    """
    if round_type not in ("floor", "nearest", "ceiling"):
        raise ValueError(f"Invalid rounding type {round_type}. Must be 'floor', 'nearest' or 'ceiling'.")
    
    # Setup quantization resolution/bins
    max_value = max(np.abs(data)) + 1e-9
    bins = np.linspace(-max_value, max_value, num_levels + 1)

    # Digitize to bins
    quantized_real = np.digitize(data.real, bins)
    quantized_imag = np.digitize(data.imag, bins)

    if round_type == "floor":
        quantized_real -= 1
        quantized_imag -= 1

    # Revert to values
    quantized_real = bins[quantized_real]
    quantized_imag = bins[quantized_imag]

    if round_type == "nearest":
        bin_size = np.diff(bins)[0]
        quantized_real -= bin_size / 2
        quantized_imag -= bin_size / 2

    quantized_data = quantized_real + 1j * quantized_imag

    return quantized_data.astype(torchsig_complex_data_type)


def spectral_inversion(
        data: np.ndarray
) -> np.ndarray:
    """Applies a spectral inversion to input data.

    Args:
        data (np.ndarray): IQ data.

    Returns:
        np.ndarray: Spectrally inverted data.

    """
    data.imag *= -1
    return data


def spectrogram(
    data: np.ndarray,
    fft_size: int,
    fft_stride: int
) -> np.ndarray:
    """Computes spectrogram from IQ data. 
    Directly uses `compute_spectrogram` inside of utils/dsp.py.

    Args:
        data (np.ndarray): IQ samples.
        fft_size (int): The FFT size (number of bins) in the spectrogram.
        fft_stride (int): The number of data points to move or "hop" over when
            computing the next FFT.
        rng (np.random.Generator): Optional random generator.            

    Returns:
        np.ndarray: Spectrogram computed from IQ data.

    """
    return dsp.compute_spectrogram(
        data,
        fft_size=fft_size,
        fft_stride=fft_stride
    )


# TODO this function needs clean up
def spectrogram_drop_samples(
    data: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str
) -> np.ndarray:
    """Drop samples at given locations/durations with fill technique.

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
        data (np.ndarray): IQ data.
        drop_starts (np.ndarray): Start indicies of drops.
        drop_sizes (np.ndarray): Durations for each start index.
        fill (str): Drop sample replacement method.

    Raises:
        ValueError: Invalid fill type.

    Returns:
        np.ndarray: data array with fill values during drops.

    """    
    flat_spec = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region_real = np.ones(drop_sizes[idx]) * flat_spec[0, drop_start - 1]
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start - 1]
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start : drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "bfill":
            drop_region_real = (
                np.ones(drop_sizes[idx]) * flat_spec[0, drop_start + drop_sizes[idx]]
            )
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start + drop_sizes[idx]]
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start : drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "mean":
            drop_region_real = np.ones(drop_sizes[idx]) * np.mean(flat_spec[0])
            drop_region_complex = np.ones(drop_sizes[idx]) * np.mean(flat_spec[1])
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start : drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx])
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        elif fill == "min":
            drop_region_real = np.ones(drop_sizes[idx]) * np.min(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.min(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start : drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "max":
            drop_region_real = np.ones(drop_sizes[idx]) * np.max(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.max(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start : drop_start + drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start : drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "low":
            drop_region = np.ones(drop_sizes[idx]) * 1e-3
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        elif fill == "ones":
            drop_region = np.ones(drop_sizes[idx])
            flat_spec[:, drop_start : drop_start + drop_sizes[idx]] = drop_region
        else:
            raise ValueError(f"fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {fill}")
    new_data = flat_spec.reshape(data.shape[0], data.shape[1], data.shape[2])
    return new_data


def time_reversal(
    data: np.ndarray
) -> np.ndarray:
    """Applies time reversal to data (flips horizontally).

    Args:
        data (np.ndarray): IQ data.

    Returns:
        np.ndarray: Time flipped IQ data.

    """    
    return np.flip(data, axis=0).astype(torchsig_complex_data_type)


def time_varying_noise(
    data: np.ndarray,
    noise_power_low: float,
    noise_power_high: float,
    inflections: int,
    random_regions: bool,
    rng: np.random.Generator = np.random.default_rng(seed=None)
) -> np.ndarray :
    """Adds time-varying complex additive white Gaussian noise with power
    levels in range (`noise_power_low`, `noise_power_high`) dB and with
    `inflections` number of inflection points spread over the input iq data
    randomly if `random_regions` is True or evenly spread if False.

    Args:
        data (np.ndarray): IQ data.
        noise_power_low (float): Minimum noise power in dB.
        noise_power_high (float): Maximum noise power in dB.
        inflections (int): Number of inflection points over IQ data.
        random_regions (bool): Inflections points spread randomly (True) or not (False).
        rng (np.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(seed=None).

    Returns:
        np.ndarray: IQ data with time-varying noise.
        
    """    
    real_noise = rng.standard_normal(*data.shape)
    imag_noise = rng.standard_normal(*data.shape)
    noise_power = np.zeros(data.shape)

    if not inflections: #inflections == 0:
        inflection_indices = np.array([0, data.shape[0]])
    else:
        if random_regions:
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
        start_power = noise_power_low if not idx % 2 else noise_power_high #idx % 2 == 0
        stop_power = noise_power_high if not idx % 2 else noise_power_low
        noise_power[start_idx:stop_idx] = np.linspace(
            start_power, stop_power, duration
        )

    return ( data + (10.0 ** (noise_power / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2) ).astype(torchsig_complex_data_type)
