import numpy as np
from scipy import signal as sp
from numba import njit, int64, float64, complex64


def time_shift(tensor: np.ndarray, t_shift: float) -> np.ndarray:
    """Shifts tensor in the time dimension by tshift samples. Zero-padding is applied to maintain input size.

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


def freq_shift_avoid_aliasing(tensor: np.ndarray, f_shift: float) -> np.ndarray:
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

    # Filter around center to remove original alias effects
    num_taps = int(
        2 * np.ceil(50 * 2 * np.pi / (1 / up) / 0.125 / 22)
    )  # fred harris rule of thumb * 2
    taps = sp.firwin(
        num_taps,
        (1 / up),
        width=(1 / up) * 0.02,
        window=sp.get_window("blackman", num_taps),
        scale=True,
    )
    tensor = sp.fftconvolve(tensor, taps, mode="same")

    # Freq shift to desired center freq
    time_vector = np.arange(tensor.shape[0], dtype=np.float64)
    tensor = tensor * np.exp(2j * np.pi * f_shift / up * time_vector)

    # Filter to remove out-of-band regions
    num_taps = int(
        2 * np.ceil(50 * 2 * np.pi / (1 / up) / 0.125 / 22)
    )  # fred harris rule-of-thumb * 2
    taps = sp.firwin(
        num_taps,
        1 / up,
        width=(1 / up) * 0.02,
        window=sp.get_window("blackman", num_taps),
        scale=True,
    )
    tensor = sp.fftconvolve(tensor, taps, mode="same")
    tensor = tensor[
        : int(num_iq_samples * up)
    ]  # prune to be correct size out of filter

    # Decimate back down to correct sample rate
    tensor = sp.resample_poly(tensor, down, up)

    return tensor[:num_iq_samples]


@njit(cache=False)
def _fractional_shift_helper(
    taps: np.ndarray, raw_iq: np.ndarray, stride: int, offset: int
):
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
    tensor: np.ndarray, taps: np.ndarray, stride: int, delay: int
) -> np.ndarray:
    """Applies fractional sample delay of delay using a polyphase interpolator

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be shifted in time.

        taps (:obj:`float` or :class:`numpy.ndarray`):
            taps to use for filtering

        stride (:obj:`int`):
            interpolation rate of internal filter

        delay (:obj:`int` ):
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

    tensor += 10 ** (iq_dc_offset_db / 10.0) * np.real(tensor) + 1j * 10 ** (
        iq_dc_offset_db / 10.0
    ) * np.imag(tensor)
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
    new_tensor = np.empty(*tensor.shape, dtype=tensor.dtype)
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
    num_taps = fltorder
    sinusoid = np.exp(2j * np.pi * center_freq * np.linspace(0, num_taps - 1, num_taps))
    taps = sp.firwin(
        num_taps,
        bandwidth,
        width=bandwidth * 0.02,
        window=sp.get_window("blackman", num_taps),
        scale=True,
    )
    taps = taps * sinusoid
    return sp.fftconvolve(tensor, taps, mode="same")


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
    max_value = max(np.abs(tensor)) + 1e-9
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
    return (1 - alpha) * tensor + alpha * np.convolve(tensor, filter_taps, mode="same")


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
    for sample_idx, sample in enumerate(tensor):
        if np.abs(sample) == 0:
            level_db = -200
        else:
            level_db = level_db * alpha_smooth + np.log(np.abs(sample)) * (
                1 - alpha_smooth
            )
        output_db = level_db + gain_db
        diff_db = ref_level_db - output_db

        if level_db <= low_level_db:
            alpha_adjust = 0
        elif output_db >= high_level_db:
            alpha_adjust = alpha_overflow
        elif abs(diff_db) > track_range_db:
            alpha_adjust = alpha_acquire
        else:
            alpha_adjust = alpha_track

        gain_db += diff_db * alpha_adjust
        output[sample_idx] = tensor[sample_idx] * np.exp(gain_db)
    return output
