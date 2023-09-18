from typing import Callable, List, Literal, Optional, Tuple, Union
from numba import complex64, float64, int64, njit
from torchsig.utils.types import RandomDistribution
from torchsig.utils.dsp import low_pass
from scipy import interpolate
from scipy import signal as sp
import numpy as np
import pywt


import cv2


__all__ = [
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "normalize",
    "resample",
    "make_sinc_filter",
    "awgn",
    "time_varying_awgn",
    "impulsive_interference",
    "rayleigh_fading",
    "phase_offset",
    "interleave_complex",
    "complex_to_2d",
    "real",
    "imag",
    "complex_magnitude",
    "wrapped_phase",
    "discrete_fourier_transform",
    "spectrogram",
    "continuous_wavelet_transform",
    "time_shift",
    "time_crop",
    "freq_shift",
    "freq_shift_avoid_aliasing",
    "_fractional_shift_helper",
    "fractional_shift",
    "iq_imbalance",
    "spectral_inversion",
    "channel_swap",
    "time_reversal",
    "amplitude_reversal",
    "roll_off",
    "add_slope",
    "mag_rescale",
    "drop_samples",
    "quantize",
    "clip",
    "random_convolve",
    "agc",
    "cut_out",
    "patch_shuffle",
    "drop_spec_samples",
    "spec_patch_shuffle",
    "spec_translate",
]


FloatParameter = Union[RandomDistribution, float, Tuple[float, float], List]
IntParameter = Union[RandomDistribution, int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def normalize(
    tensor: np.ndarray,
    norm_order: Optional[Union[float, int, Literal["fro", "nuc"]]] = 2,
    flatten: bool = False,
) -> np.ndarray:
    """Scale a tensor so that a specfied norm computes to 1. For detailed information, see :func:`numpy.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        tensor (:class:`numpy.ndarray`)):
            (batch_size, vector_length, ...)-sized tensor to be normalized.

        norm_order (:class:`int`)):
            norm order to be passed to np.linalg.norm

        flatten (:class:`bool`)):
            boolean specifying if the input array's norm should be calculated on the flattened representation of the input tensor

    Returns:
        Tensor:
            Normalized complex array.
    """
    if flatten:
        flat_tensor = tensor.reshape(tensor.size)
        norm = np.linalg.norm(flat_tensor, norm_order, keepdims=True)
    else:
        norm = np.linalg.norm(tensor, norm_order, keepdims=True)
    return np.multiply(tensor, 1.0 / norm)


def resample(
    tensor: np.ndarray,
    up_rate: int,
    down_rate: int,
    num_iq_samples: int,
    keep_samples: bool,
    anti_alias_lpf: bool = False,
) -> np.ndarray:
    """Resample a tensor by rational value

    Args:
        tensor (:class:`numpy.ndarray`):
            tensor to be resampled.

        up_rate (:class:`int`):
            rate at which to up-sample the tensor

        down_rate (:class:`int`):
            rate at which to down-sample the tensor

        num_iq_samples (:class:`int`):
            number of IQ samples to have after resampling

        keep_samples (:class:`bool`):
            boolean to specify if the resampled data should be returned as is

        anti_alias_lpf (:class:`bool`)):
            boolean to specify if an additional anti aliasing filter should be
            applied

    Returns:
        Tensor:
            Resampled tensor
    """
    if anti_alias_lpf:
        new_rate = up_rate / down_rate
        taps = low_pass(
            cutoff=new_rate * 0.98 / 2,
            transition_bandwidth=(0.5 - (new_rate * 0.98) / 2) / 4,
        )
        tensor = sp.convolve(tensor, taps, mode="same")

    # Resample
    resampled = sp.resample_poly(tensor, up_rate, down_rate)

    # Handle extra or not enough IQ samples
    if keep_samples:
        new_tensor = resampled
    elif resampled.shape[0] > num_iq_samples:
        new_tensor = resampled[-num_iq_samples:]
    else:
        new_tensor = np.zeros((num_iq_samples,), dtype=np.complex128)
        new_tensor[: resampled.shape[0]] = resampled

    return new_tensor


@njit(cache=False)
def make_sinc_filter(beta, tap_cnt, sps, offset=0):
    """
    return the taps of a sinc filter
    """
    ntap_cnt = tap_cnt + ((tap_cnt + 1) % 2)
    t_index = np.arange(-(ntap_cnt - 1) // 2, (ntap_cnt - 1) // 2 + 1) / np.double(sps)

    taps = np.sinc(beta * t_index + offset)
    taps /= np.sum(taps)

    return taps[:tap_cnt]


def awgn(tensor: np.ndarray, noise_power_db: float) -> np.ndarray:
    """Adds zero-mean complex additive white Gaussian noise with power of
    noise_power_db.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        noise_power_db (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_noise = np.random.randn(*tensor.shape)
    imag_noise = np.random.randn(*tensor.shape)
    return tensor + (10.0 ** (noise_power_db / 20.0)) * (
        real_noise + 1j * imag_noise
    ) / np.sqrt(2)


def time_varying_awgn(
    tensor: np.ndarray,
    noise_power_db_low: float,
    noise_power_db_high: float,
    inflections: int,
    random_regions: bool,
) -> np.ndarray:
    """Adds time-varying complex additive white Gaussian noise with power
    levels in range (`noise_power_db_low`, `noise_power_db_high`) and with
    `inflections` number of inflection points spread over the input tensor
    randomly if `random_regions` is True or evely spread if False

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        noise_power_db_low (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

        noise_power_db_high (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

        inflections (:obj:`int`):
            Number of inflection points for time-varying nature

        random_regions (:obj:`bool`):
            Specify if inflection points are randomly spread throughout tensor
            or if evenly spread

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_noise: np.ndarray = np.random.randn(*tensor.shape)
    imag_noise: np.ndarray = np.random.randn(*tensor.shape)
    noise_power_db: np.ndarray = np.zeros(tensor.shape)

    if inflections == 0:
        inflection_indices = np.array([0, tensor.shape[0]])
    else:
        if random_regions:
            inflection_indices = np.sort(
                np.random.choice(tensor.shape[0], size=inflections, replace=False)
            )
            inflection_indices = np.append(inflection_indices, tensor.shape[0])
            inflection_indices = np.insert(inflection_indices, 0, 0)
        else:
            inflection_indices = np.arange(inflections + 2) * int(
                tensor.shape[0] / (inflections + 1)
            )

    for idx in range(len(inflection_indices) - 1):
        start_idx = inflection_indices[idx]
        stop_idx = inflection_indices[idx + 1]
        duration = stop_idx - start_idx
        start_power = noise_power_db_low if idx % 2 == 0 else noise_power_db_high
        stop_power = noise_power_db_high if idx % 2 == 0 else noise_power_db_low
        noise_power_db[start_idx:stop_idx] = np.linspace(
            start_power, stop_power, duration
        )

    return tensor + (10.0 ** (noise_power_db / 20.0)) * (
        real_noise + 1j * imag_noise
    ) / np.sqrt(2)


@njit(cache=False)
def impulsive_interference(
    tensor: np.ndarray,
    amp: float,
    per_offset: float,
) -> np.ndarray:
    """Applies an impulsive interferer to tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        amp (:obj:`float`):
            Maximum vector magnitude of complex interferer signal

        per_offset (:obj:`float`)
            Interferer offset into the tensor as expressed in a fraction of the tensor length.

    """
    beta = 0.3
    num_samps = len(tensor)
    sinc_pulse = make_sinc_filter(beta, num_samps, 0.1, 0)
    imp = amp * np.roll(sinc_pulse / np.max(sinc_pulse), int(per_offset * num_samps))
    rand_phase = np.random.uniform(0, 2 * np.pi)
    imp = np.exp(1j * rand_phase) * imp
    output: np.ndarray = tensor + imp
    return output


def rayleigh_fading(
    tensor: np.ndarray,
    coherence_bandwidth: float,
    power_delay_profile: np.ndarray,
) -> np.ndarray:
    """Applies Rayleigh fading channel to tensor. Taps are generated by
    interpolating and filtering Gaussian taps.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        coherence_bandwidth (:obj:`float`):
            coherence_bandwidth relative to the sample rate in [0, 1.0]

        power_delay_profile (:obj:`float`):
            power_delay_profile assigned to channel

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone Rayleigh Fading.

    """
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
    rayleigh_taps = np.random.randn(num_taps) + 1j * np.random.randn(
        num_taps
    )  # multi-path channel

    # Linear interpolate taps by a factor of 100 -- so we can get accurate coherence bandwidths
    old_time = np.linspace(0, 1.0, num_taps, endpoint=True)
    real_tap_function = interpolate.interp1d(old_time, rayleigh_taps.real)
    imag_tap_function = interpolate.interp1d(old_time, rayleigh_taps.imag)

    new_time = np.linspace(0, 1.0, 100 * num_taps, endpoint=True)
    rayleigh_taps = real_tap_function(new_time) + 1j * imag_tap_function(new_time)
    rayleigh_taps *= power_taps

    # Ensure that we maintain the same amount of power before and after the transform
    input_power = np.linalg.norm(tensor)
    tensor = sp.upfirdn(rayleigh_taps, tensor, up=100, down=100)[-tensor.shape[0] :]
    output_power = np.linalg.norm(tensor)
    tensor = np.multiply(input_power / output_power, tensor)
    return tensor


def phase_offset(tensor: np.ndarray, phase: float) -> np.ndarray:
    """Applies a phase rotation to tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        phase (:obj:`float`):
            phase to rotate sample in [-pi, pi]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone a phase rotation

    """
    return tensor * np.exp(1j * phase)


def interleave_complex(tensor: np.ndarray) -> np.ndarray:
    """Converts complex vectors to real interleaved IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Interleaved vectors.
    """
    new_tensor = np.empty((tensor.shape[0] * 2,))
    new_tensor[::2] = np.real(tensor)
    new_tensor[1::2] = np.imag(tensor)
    return new_tensor


def complex_to_2d(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to two channels representing real and imaginary

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Expanded vectors
    """

    new_tensor = np.zeros((2, tensor.shape[0]), dtype=np.float64)
    new_tensor[0] = np.real(tensor).astype(np.float64)
    new_tensor[1] = np.imag(tensor).astype(np.float64)
    return new_tensor


def real(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a real-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            real(tensor)
    """
    return np.real(tensor)


def imag(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a imaginary-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            imag(tensor)
    """
    return np.imag(tensor)


def complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a complex magnitude vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            abs(tensor)
    """
    return np.abs(tensor)


def wrapped_phase(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a wrapped-phase vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            angle(tensor)
    """
    return np.angle(tensor)


def discrete_fourier_transform(tensor: np.ndarray) -> np.ndarray:
    """Computes DFT of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            fft(tensor). normalization is 1/sqrt(n)
    """
    return np.fft.fft(tensor, norm="ortho")


def spectrogram(
    tensor: np.ndarray,
    nperseg: int,
    noverlap: int,
    nfft: int,
    detrend: Optional[str],
    scaling: Optional[str],
    window_fcn: Callable[[int], np.ndarray],
    mode: str,
) -> np.ndarray:
    """Computes spectrogram of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        nperseg (:obj:`int`):
            Length of each segment. If window is str or tuple, is set to 256,
            and if window is array_like, is set to the length of the window.

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 8.

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            If None, the FFT length is nperseg.

        detrend : str or function or False, optional
            Specifies how to detrend each segment. If detrend is a string, it is passed as the type
            argument to the detrend function. If it is a function, it takes a segment and returns a
            detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.

        scaling : { ‘density’, ‘spectrum’ }, optional
            Selects between computing the power spectral density (‘density’) where Sxx has units of
            V**2/Hz and computing the power spectrum (‘spectrum’) where Sxx has units of V**2, if
            x is measured in V and fs is measured in Hz. Defaults to ‘density’.

        window_fcn (:obj:`Callable`):
            Function generating the window for each FFT

        mode (:obj:`str`):
            Mode of the spectrogram to be computed.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Spectrogram of tensor along time dimension
    """
    _, _, spectrograms = sp.spectrogram(
        tensor,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
        window=window_fcn(nperseg),
        return_onesided=False,
        mode=mode,
        axis=0,
    )
    return np.fft.fftshift(spectrograms, axes=0)


def continuous_wavelet_transform(
    tensor: np.ndarray, wavelet: str, nscales: int, sample_rate: float
) -> np.ndarray:
    """Computes the continuous wavelet transform resulting in a Scalogram of the complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        wavelet (:obj:`str`):
            Name of the mother wavelet.
            If None, wavename = 'mexh'.

        nscales (:obj:`int`):
            Number of scales to use in the Scalogram.
            If None, nscales = 33.

        sample_rate (:obj:`float`):
            Sample rate of the signal.
            If None, fs = 1.0.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Scalogram of tensor along time dimension
    """
    scales = np.arange(1, nscales)
    cwtmatr, _ = pywt.cwt(
        tensor, scales=scales, wavelet=wavelet, sampling_period=1.0 / sample_rate
    )

    # if the dtype is complex then return the magnitude
    if np.iscomplexobj(cwtmatr):
        cwtmatr = abs(cwtmatr)

    return cwtmatr


def time_shift(tensor: np.ndarray, t_shift: int) -> np.ndarray:
    """Shifts tensor in the time dimension by tshift samples. Zero-padding is
    applied to maintain input size.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted.

        t_shift (:obj:`int` or :class:`numpy.ndarray`):
            Number of samples to shift right or left (if negative)

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor shifted in time of size tensor.shape
    """
    # Valid Range Error Checking
    if np.max(np.abs(t_shift)) >= tensor.shape[0]:
        return np.zeros_like(tensor, dtype=np.complex64)

    # This overwrites tensor as side effect, modifies inplace
    if t_shift > 0:
        tmp = tensor[:-t_shift]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (t_shift, 0), "constant", constant_values=0 + 0j)
    elif t_shift < 0:
        tmp = tensor[-t_shift:]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (0, -t_shift), "constant", constant_values=0 + 0j)
    return tensor


def time_crop(tensor: np.ndarray, start: int, length: int) -> np.ndarray:
    """Crops a tensor in the time dimension from index start(inclusive) for length samples.

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be cropped.

        start (:obj:`int` or :class:`numpy.ndarray`):
            index to begin cropping

        length (:obj:`int`):
            number of samples to include

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor cropped in time of size (tensor.shape[0], length)
    """
    # Type and Size checking
    if length < 0:
        raise ValueError("Length must be greater than 0")

    if np.any(start < 0):
        raise ValueError("Start must be greater than 0")

    if np.max(start) >= tensor.shape[0] or length == 0:
        return np.empty(shape=(1, 1))

    crop_len = min(length, tensor.shape[0] - np.max(start))

    return tensor[start : start + crop_len]


def freq_shift(tensor: np.ndarray, f_shift: float) -> np.ndarray:
    """Shifts each tensor in freq by freq_shift along the time dimension

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    sinusoid = np.exp(
        2j * np.pi * f_shift * np.arange(tensor.shape[0], dtype=np.float64)
    )
    return np.multiply(tensor, np.asarray(sinusoid))


def freq_shift_avoid_aliasing(
    tensor: np.ndarray,
    f_shift: float,
) -> np.ndarray:
    """Similar to `freq_shift` function but performs the frequency shifting at
    a higher sample rate with filtering to avoid aliasing

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    # Match output size to input
    num_iq_samples = tensor.shape[0]

    # Interpolate up to avoid frequency wrap around during shift
    up = 2
    down = 1
    tensor = sp.resample_poly(tensor, up, down)

    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    tensor = sp.convolve(tensor, taps, mode="same")

    # Freq shift to desired center freq
    time_vector = np.arange(tensor.shape[0], dtype=np.float64)
    tensor = tensor * np.exp(2j * np.pi * f_shift / up * time_vector)

    # Filter to remove out-of-band regions
    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    tensor = sp.convolve(tensor, taps, mode="same")
    tensor = tensor[
        : int(num_iq_samples * up)
    ]  # prune to be correct size out of filter

    # Decimate back down to correct sample rate
    tensor = sp.resample_poly(tensor, down, up)

    return tensor[:num_iq_samples]


@njit(cache=False)
def _fractional_shift_helper(
    taps: np.ndarray, raw_iq: np.ndarray, stride: int, offset: int
) -> np.ndarray:
    """Fractional shift. First, we up-sample by a large, fixed amount. Filter with 1/upsample_rate/2.0,
    Next we down-sample by the same, large fixed amount with a chosen offset. Doing this efficiently means not actually zero-padding.

    The efficient way to do this is to decimate the taps and filter the signal with some offset in the taps.
    """
    # We purposely do not calculate values within the group delay.
    group_delay = ((taps.shape[0] - 1) // 2 - (stride - 1)) // stride + 1
    if offset < 0:
        offset += stride
        group_delay -= 1

    # Decimate the taps.
    taps = taps[offset::stride]

    # Determine output size
    num_taps = taps.shape[0]
    num_raw_iq = raw_iq.shape[0]
    output = np.zeros(((num_taps + num_raw_iq - 1 - group_delay),), dtype=np.complex128)

    # This is a just convolution of taps and raw_iq
    for o_idx in range(output.shape[0]):
        idx_mn = o_idx - (num_raw_iq - 1) if o_idx >= num_raw_iq - 1 else 0
        idx_mx = o_idx if o_idx < num_taps - 1 else num_taps - 1
        for f_idx in range(idx_mn, idx_mx):
            output[o_idx - group_delay] += taps[f_idx] * raw_iq[o_idx - f_idx]
    return output


def fractional_shift(
    tensor: np.ndarray, taps: np.ndarray, stride: int, delay: float
) -> np.ndarray:
    """Applies fractional sample delay of delay using a polyphase interpolator

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted in time.

        taps (:obj:`float` or :class:`numpy.ndarray`):
            taps to use for filtering

        stride (:obj:`int`):
            interpolation rate of internal filter

        delay (:obj:`float` ):
            Delay in number of samples in [-1, 1]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been fractionally-shifted along time dimension of size tensor.shape
    """
    real_part = _fractional_shift_helper(
        taps, tensor.real, stride, int(stride * float(delay))
    )
    imag_part = _fractional_shift_helper(
        taps, tensor.imag, stride, int(stride * float(delay))
    )
    tensor = real_part[: tensor.shape[0]] + 1j * imag_part[: tensor.shape[0]]
    zero_idx = -1 if delay < 0 else 0  # do not extrapolate, zero-pad.
    tensor[zero_idx] = 0
    return tensor


def iq_imbalance(
    tensor: np.ndarray,
    iq_amplitude_imbalance_db: float,
    iq_phase_imbalance: float,
    iq_dc_offset_db: float,
) -> np.ndarray:
    """Applies IQ imbalance to tensor

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted in time.

        iq_amplitude_imbalance_db (:obj:`float` or :class:`numpy.ndarray`):
            IQ amplitude imbalance in dB

        iq_phase_imbalance (:obj:`float` or :class:`numpy.ndarray`):
            IQ phase imbalance in radians [-pi, pi]

        iq_dc_offset_db (:obj:`float` or :class:`numpy.ndarray`):
            IQ DC Offset in dB

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has an IQ imbalance applied across the time dimension of size tensor.shape
    """
    # amplitude imbalance
    tensor = 10 ** (iq_amplitude_imbalance_db / 10.0) * np.real(tensor) + 1j * 10 ** (
        iq_amplitude_imbalance_db / 10.0
    ) * np.imag(tensor)

    # phase imbalance
    tensor = np.exp(-1j * iq_phase_imbalance / 2.0) * np.real(tensor) + np.exp(
        1j * (np.pi / 2.0 + iq_phase_imbalance / 2.0)
    ) * np.imag(tensor)

    tensor = (
        tensor
        + 10 ** (iq_dc_offset_db / 10.0) * np.real(tensor)
        + 1j * 10 ** (iq_dc_offset_db / 10.0) * np.imag(tensor)
    )
    return tensor


def spectral_inversion(tensor: np.ndarray) -> np.ndarray:
    """Applies a spectral inversion

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone a spectral inversion

    """
    tensor.imag *= -1
    return tensor


def channel_swap(tensor: np.ndarray) -> np.ndarray:
    """Swap the I and Q channels of input complex data

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone channel swapping

    """
    real_component = tensor.real
    imag_component = tensor.imag
    new_tensor = np.empty(tensor.shape, dtype=tensor.dtype)
    new_tensor.real = imag_component
    new_tensor.imag = real_component
    return new_tensor


def time_reversal(tensor: np.ndarray) -> np.ndarray:
    """Applies a time reversal to the input tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone a time reversal

    """
    return np.flip(tensor, axis=0)


def amplitude_reversal(tensor: np.ndarray) -> np.ndarray:
    """Applies an amplitude reversal to the input tensor by multiplying by -1

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone an amplitude reversal

    """
    return tensor * -1


def roll_off(
    tensor: np.ndarray,
    lowercutfreq: float,
    uppercutfreq: float,
    fltorder: int,
) -> np.ndarray:
    """Applies front-end filter to tensor. Rolls off lower/upper edges of bandwidth

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        lowercutfreq (:obj:`float`):
            lower bandwidth cut-off to begin linear roll-off

        uppercutfreq (:obj:`float`):
            upper bandwidth cut-off to begin linear roll-off

        fltorder (:obj:`int`):
            order of each FIR filter to be applied

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone front-end filtering.

    """
    if (lowercutfreq == 0) & (uppercutfreq == 1):
        return tensor

    elif uppercutfreq == 1:
        if fltorder % 2 == 0:
            fltorder += 1
    bandwidth = uppercutfreq - lowercutfreq
    center_freq = lowercutfreq - 0.5 + bandwidth / 2
    taps = low_pass(
        cutoff=bandwidth / 2, transition_bandwidth=(0.5 - bandwidth / 2) / 4
    )
    sinusoid = np.exp(
        2j * np.pi * center_freq * np.linspace(0, len(taps) - 1, len(taps))
    )
    taps = taps * sinusoid
    return sp.convolve(tensor, taps, mode="same")


def add_slope(tensor: np.ndarray) -> np.ndarray:
    """The slope between each sample and its preceeding sample is added to
    every sample

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    slope = np.diff(tensor)
    slope = np.insert(slope, 0, 0)
    return tensor + slope


def mag_rescale(
    tensor: np.ndarray,
    start: float,
    scale: float,
) -> np.ndarray:
    """Apply a rescaling of input `scale` starting at time `start`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        start (:obj:`float`):
            Normalized start time of rescaling

        scale (:obj:`float`):
            Scaling factor

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone rescaling

    """
    start = int(tensor.shape[0] * start)
    tensor[start:] *= scale
    return tensor


def drop_samples(
    tensor: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str,
) -> np.ndarray:
    """Drop samples at specified input locations/durations with fill technique

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        drop_starts (:class:`numpy.ndarray`):
            Indices of where drops start

        drop_sizes (:class:`numpy.ndarray`):
            Durations of each drop instance

        fill (:obj:`str`):
            String specifying how the dropped samples should be replaced

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone the dropped samples

    """
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region = (
                np.ones(drop_sizes[idx], dtype=np.complex64) * tensor[drop_start - 1]
            )
        elif fill == "bfill":
            drop_region = (
                np.ones(drop_sizes[idx], dtype=np.complex64)
                * tensor[drop_start + drop_sizes[idx]]
            )
        elif fill == "mean":
            drop_region = np.ones(drop_sizes[idx], dtype=np.complex64) * np.mean(tensor)
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx], dtype=np.complex64)
        else:
            raise ValueError(
                "fill expects ffill, bfill, mean, or zero. Found {}".format(fill)
            )

        # Update drop region
        tensor[drop_start : drop_start + drop_sizes[idx]] = drop_region

    return tensor


def quantize(
    tensor: np.ndarray,
    num_levels: int,
    round_type: str = "floor",
) -> np.ndarray:
    """Quantize the input to the number of levels specified

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        num_levels (:obj:`int`):
            Number of quantization levels

        round_type (:obj:`str`):
            Quantization rounding. Options: 'floor', 'middle', 'ceiling'

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone quantization

    """
    # Setup quantization resolution/bins
    max_value = (
        max(np.abs(np.real(np.copy(tensor)), np.imag(np.copy(tensor))))
        + np.finfo(np.dtype(np.float64)).eps
    )
    bins = np.linspace(-max_value, max_value, num_levels + 1)

    # Digitize to bins
    quantized_real = np.digitize(tensor.real, bins)
    quantized_imag = np.digitize(tensor.imag, bins)

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

    quantized_tensor = quantized_real + 1j * quantized_imag

    return quantized_tensor


def clip(tensor: np.ndarray, clip_percentage: float) -> np.ndarray:
    """Clips input tensor's values above/below a specified percentage of the
    max/min of the input tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        clip_percentage (:obj:`float`):
            Percentage of max/min values to clip

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_tensor = tensor.real
    max_val = np.max(real_tensor) * clip_percentage
    min_val = np.min(real_tensor) * clip_percentage
    real_tensor[real_tensor > max_val] = max_val
    real_tensor[real_tensor < min_val] = min_val

    imag_tensor = tensor.imag
    max_val = np.max(imag_tensor) * clip_percentage
    min_val = np.min(imag_tensor) * clip_percentage
    imag_tensor[imag_tensor > max_val] = max_val
    imag_tensor[imag_tensor < min_val] = min_val

    new_tensor = real_tensor + 1j * imag_tensor
    return new_tensor


def random_convolve(
    tensor: np.ndarray,
    num_taps: int,
    alpha: float,
) -> np.ndarray:
    """Create a complex-valued filter with `num_taps` number of taps, convolve
    the random filter with the input data, and sum the original data with the
    randomly-filtered data using an `alpha` weighting factor.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        num_taps: (:obj:`int`):
            Number of taps in random filter

        alpha: (:obj:`float`):
            Weighting for the summation between the original data and the
            randomly-filtered data, following:

            `output = (1 - alpha) * tensor + alpha * filtered_tensor`

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with weighted random filtering

    """
    filter_taps = np.random.rand(num_taps) + 1j * np.random.rand(num_taps)
    return (1 - alpha) * tensor + alpha * sp.convolve(tensor, filter_taps, mode="same")


@njit(
    complex64[:](
        complex64[:],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=False,
)
def agc(
    tensor: np.ndarray,
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
    """AGC implementation

    Args:
         tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be agc'd

        initial_gain_db (:obj:`float`):
            Initial gain value in linear units

        alpha_smooth (:obj:`float`):
            Alpha for averaging the measured signal level level_n = level_n*alpha + level_n-1*(1 - alpha)

        alpha_track (:obj:`float`):
            Amount by which to adjust gain when in tracking state

        alpha_overflow (:obj:`float`):
            Amount by which to adjust gain when in overflow state [level_db + gain_db] >= max_level

        alpha_acquire (:obj:`float`):
            Amount by which to adjust gain when in acquire state abs([ref_level_db - level_db - gain_db]) >= track_range_db

        ref_level_db (:obj:`float`):
            Level to which we intend to adjust gain to achieve

        track_range_db (:obj:`float`):
            Range from ref_level_linear for which we can deviate before going into acquire state

        low_level_db (:obj:`float`):
            Level below which we disable AGC

        high_level_db (:obj:`float`):
            Level above which we go into overflow state

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with AGC applied

    """
    output = np.zeros_like(tensor)
    gain_db = initial_gain_db
    level_db = 0.0
    for sample_idx, sample in enumerate(tensor):
        if np.abs(sample) == 0:
            level_db = -200
        elif sample_idx == 0:  # first sample, no smoothing
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
        output[sample_idx] = tensor[sample_idx] * np.exp(gain_db)
    return output


def cut_out(
    tensor: np.ndarray,
    cut_start: float,
    cut_dur: float,
    cut_type: str,
) -> np.ndarray:
    """Performs the CutOut using the input parameters

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        cut_start: (:obj:`float`):
            Start of cut region in range [0.0,1.0)

        cut_dur: (:obj:`float`):
            Duration of cut region in range (0.0,1.0]

        cut_type: (:obj:`str`):
            String specifying type of data to fill in cut region with

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone cut out

    """
    num_iq_samples = tensor.shape[0]
    cut_start = int(cut_start * num_iq_samples)

    # Create cut mask
    cut_mask_length = int(num_iq_samples * cut_dur)
    if cut_mask_length + cut_start > num_iq_samples:
        cut_mask_length = num_iq_samples - cut_start

    if cut_type == "zeros":
        cut_mask = np.zeros(cut_mask_length, dtype=np.complex64)
    elif cut_type == "ones":
        cut_mask = np.ones(cut_mask_length) + 1j * np.ones(cut_mask_length)
    elif cut_type == "low_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        noise_power_db = -100
        cut_mask = (
            (10.0 ** (noise_power_db / 20.0))
            * (real_noise + 1j * imag_noise)
            / np.sqrt(2)
        )
    elif cut_type == "avg_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        avg_power = np.mean(np.abs(tensor) ** 2)
        cut_mask = avg_power * (real_noise + 1j * imag_noise) / np.sqrt(2)
    elif cut_type == "high_noise":
        real_noise = np.random.randn(cut_mask_length)
        imag_noise = np.random.randn(cut_mask_length)
        noise_power_db = 40
        cut_mask = (
            (10.0 ** (noise_power_db / 20.0))
            * (real_noise + 1j * imag_noise)
            / np.sqrt(2)
        )
    else:
        raise ValueError(
            "cut_type must be: zeros, ones, low_noise, avg_noise, or high_noise. Found: {}".format(
                cut_type
            )
        )

    # Insert cut mask into tensor
    tensor[cut_start : cut_start + cut_mask_length] = cut_mask

    return tensor


def patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        patch_size (:obj:`int`):
            Size of each patch to shuffle

        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    num_patches = int(tensor.shape[0] / patch_size)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        patch_start = int(patch_idx * patch_size)
        patch = tensor[patch_start : patch_start + patch_size]
        np.random.shuffle(patch)
        tensor[patch_start : patch_start + patch_size] = patch

    return tensor


def drop_spec_samples(
    tensor: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str,
) -> np.ndarray:
    """Drop samples at specified input locations/durations with fill technique

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        drop_starts (:class:`numpy.ndarray`):
            Indices of where drops start

        drop_sizes (:class:`numpy.ndarray`):
            Durations of each drop instance

        fill (:obj:`str`):
            String specifying how the dropped samples should be replaced

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone the dropped samples

    """
    flat_spec = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
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
            raise ValueError(
                "fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {}".format(
                    fill
                )
            )
    new_tensor = flat_spec.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return new_tensor


def spec_patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        patch_size (:obj:`int`):
            Size of each patch to shuffle

        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    channels, height, width = tensor.shape
    num_freq_patches = int(height / patch_size)
    num_time_patches = int(width / patch_size)
    num_patches = int(num_freq_patches * num_time_patches)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        freq_idx = np.floor(patch_idx / num_freq_patches)
        time_idx = patch_idx % num_time_patches
        patch = tensor[
            :,
            int(freq_idx * patch_size) : int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size) : int(time_idx * patch_size + patch_size),
        ]
        patch = patch.reshape(int(2 * patch_size * patch_size))
        np.random.shuffle(patch)
        patch = patch.reshape(2, int(patch_size), int(patch_size))
        tensor[
            :,
            int(freq_idx * patch_size) : int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size) : int(time_idx * patch_size + patch_size),
        ] = patch
    return tensor


def spec_translate(
    tensor: np.ndarray,
    time_shift: int,
    freq_shift: int,
) -> np.ndarray:
    """Apply time/freq translation to input spectrogram

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        time_shift (:obj:`int`):
            Time shift

        freq_shift (:obj:`int`):
            Frequency shift

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone time/freq translation

    """
    # Pre-fill the data with background noise
    new_tensor = np.random.rand(*tensor.shape) * np.percentile(np.abs(tensor), 50)

    # Apply translation
    channels, height, width = tensor.shape
    if time_shift >= 0 and freq_shift >= 0:
        valid_dur = width - time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:, time_shift:] = tensor[:, :valid_bw, :valid_dur]
    elif time_shift < 0 and freq_shift >= 0:
        valid_dur = width + time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:, :valid_dur] = tensor[:, :valid_bw, -time_shift:]
    elif time_shift >= 0 and freq_shift < 0:
        valid_dur = width - time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, time_shift:] = tensor[:, -freq_shift:, :valid_dur]
    elif time_shift < 0 and freq_shift < 0:
        valid_dur = width + time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, :valid_dur] = tensor[:, -freq_shift:, -time_shift:]

    return new_tensor


def spectrogram_image(
    tensor: np.ndarray,
    nperseg=512,
    noverlap=0,
    nfft=None,
    mode="psd",
    colormap="viridis",
) -> np.ndarray:
    """Computes spectrogram of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        nperseg (:obj:`int`):
            Length of each segment.
            Default 512

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            Default 0

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            Default same as nperseg

        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default psd

        colormap (:obj:'str'):
            Define OpenCV colormap to use for spectrogram image
            Default twilight

    Returns:
        transformed (:class:`numpy.ndarray`):
            Spectrogram of tensor along time dimension
    """
    if nfft is None:
        nfft = nperseg

    spec_data = spectrogram(
        tensor,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window_fcn=np.blackman,
        scaling="density",
        detrend="constant",
        mode=mode,
    )
    flattened = spec_data.flatten()
    spec_data = spec_data / np.linalg.norm(flattened, ord=np.inf, keepdims=True)
    spec = 20 * np.log10(spec_data)
    img = np.zeros((spec.shape[0], spec.shape[1], 3), dtype=np.float32)
    img = cv2.normalize(spec, img, 0, 255, cv2.NORM_MINMAX)
    colormap = colormap.upper()
    return cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS + colormap)
