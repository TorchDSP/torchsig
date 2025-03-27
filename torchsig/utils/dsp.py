"""Digital Signal Processing (DSP) Utils
"""


from scipy import signal as sp
import numpy as np
from copy import copy
import torchaudio
import torch

# common reference for the complex data type to allow for
# standardization across the different algorithms
torchsig_complex_data_type = np.complex64

# common reference for the float data type to allow for
# standardization across the different algorithms
torchsig_float_data_type = np.float32


def slice_tail_to_length(input_signal:np.ndarray, num_samples:int) -> np.ndarray:
    """Slices the tail of a signal

    Args:
        input_signal (np.ndarray): Input signal
        num_samples (int): Maximum number of samples for a signal

    Raises:
        ValueError: If signal is too short to be sliced

    Returns:
        np.ndarray: Signal with length sliced to `num_samples`
    """

    output_signal = copy(input_signal)

    # too many samples, requires slicing
    if len(output_signal) > num_samples:
        # slice off the tail
        output_signal = output_signal[0:num_samples]

    # too few samples, return error
    elif len(output_signal) < num_samples:
        raise ValueError('signal too short to be sliced')

    # else: valid length, do nothing

    return output_signal

def slice_head_tail_to_length(input_signal:np.ndarray, num_samples:int) -> np.ndarray:
    """Slices the head and tail of a signal

    Args:
        input_signal (np.ndarray): Input signal
        num_samples (int): Maximum number of samples for a signal

    Raises:
        ValueError: If signal is too short to be sliced

    Returns:
        np.ndarray: Signal with length sliced to `num_samples`
    """

    output_signal = copy(input_signal)

    # too many samples, requires slicing
    if len(output_signal) > num_samples:
        num_extra_samples = len(output_signal)-num_samples
        subtract_off_begin = int(np.ceil(num_extra_samples/2))
        subtract_off_end = int(np.floor(num_extra_samples/2))

        # slice off the beginning
        output_signal = output_signal[subtract_off_begin:]

        if subtract_off_end > 0:
            # a check to avoid subtract_off_end=0 must be done  such that it will slice
            # signal down to an empty array
            output_signal = output_signal[:-subtract_off_end]
        # else: array proper length, do nothing

    # too few samples, return error
    elif len(output_signal) < num_samples:
        raise ValueError('signal too short to be sliced')

    # else: valid length, do nothing

    return output_signal

def pad_head_tail_to_length(input_signal:np.ndarray, num_samples:int)-> np.ndarray:
    """Zero pads the head and tail of a signal

    Args:
        input_signal (np.ndarray): Input signal
        num_samples (int): Desired length of signal

    Raises:
        ValueError: If signal is too long to be padded

    Returns:
        np.ndarray: Signal with length padded to `num_samples`
    """

    output_signal = copy(input_signal)

    # signal is too short, so zero pad
    if len(output_signal) < num_samples:
        # zero-pad to proper length
        num_zeros = num_samples-len(output_signal)
        num_zeros_begin = int(np.ceil(num_zeros/2))
        num_zeros_end = int(np.floor(num_zeros/2))
        output_signal = np.concatenate((np.zeros(num_zeros_begin),output_signal,np.zeros(num_zeros_end)))

    # signal is too long, throw error
    elif len(output_signal) > num_samples:
        raise ValueError('signal is too long to be zero padded')

    # else: signal correct length, do nothing
    

    return output_signal


# apply an anti-aliasing filter to a signal which has aliased and wrapped around the
# -fs/2 or +fs/2 boundary due to upconversion
def upconversion_anti_aliasing_filter(input_signal:np.ndarray, center_freq:float, bandwidth:float, sample_rate:float, frequency_max:float, frequency_min:float):
    """Applies a BPF to avoid aliasing

    Upconversion of a signal away from baseband can force energy to alias around the
    -fs/2 or +fs/2 boundary, depending on the amount of sidelobes and the amount the
    signal has been frequency shifted by. The function checks the combination of the
    center frequency and bandwidth to see if any of the energy exceeds the limits
    specified by `frequency_max` and `frequency_min`, and if so, builds and applies
    the filter.

    Args:
        input_signal (np.ndarray): Input signal modulated to band-pass
        center_freq (float): Center frequency of the signal
        bandwidth (float): Bandwidth of the signal
        sample_rate (float): Sample rate of the signal
        frequency_max (float): The maximum frequency where energy content can reside
        frequency_min (float): The minimum frequency where energy content can reside

    Returns:
        np.ndarray: Anti-aliased signal
        float: Updated center frequency for the bounding box
        float: Updated bandwidth for the bounding box
    """

    # determine the boundaries for where the signal currently resides.
    # these values are used to determine if aliasing has occured
    upper_freq = upper_freq_from_center_freq_bandwidth(center_freq, bandwidth)
    lower_freq = lower_freq_from_center_freq_bandwidth(center_freq, bandwidth)

    # define the boundary for the upper and lower frequencies
    # upon which a BPF will be designed to limit aliasing
    upper_boundary = 0.99 * frequency_max
    lower_boundary = 0.99 * frequency_min

    # determine if aliasing has occured, and if so, in which direction
    if upper_freq > frequency_max:
        # params used in filtering
        sliced_upper_signal_edge = upper_boundary
        sliced_lower_signal_edge = lower_freq
        # params for bounding box
        box_upper_signal_edge = copy(frequency_max)
        box_lower_signal_edge = lower_freq
    elif lower_freq < frequency_min:
        # params used in filtering
        sliced_lower_signal_edge = lower_boundary
        sliced_upper_signal_edge = upper_freq
        # params for bounding box
        box_upper_signal_edge = upper_freq
        box_lower_signal_edge = copy(frequency_min)
    else: # no aliasing occurs
        return input_signal

    # compute the bandwidth and center frequency of the non-aliased portion of the signal
    # to support the anti-aliasing filter design
    sliced_bandwidth = bandwidth_from_lower_upper_freq(sliced_lower_signal_edge,sliced_upper_signal_edge)
    sliced_center_freq = center_freq_from_lower_upper_freq(sliced_lower_signal_edge,sliced_upper_signal_edge)
    # compute center freq and bandwith to support bounding box. this is slightly wider than the 
    # sliced numbers above because the box should run all the way to the boundary
    box_bandwidth = bandwidth_from_lower_upper_freq(box_lower_signal_edge,box_upper_signal_edge)
    box_center_freq = center_freq_from_lower_upper_freq(box_lower_signal_edge,box_upper_signal_edge)


    # map the bandwidth of the signal to the filter passband spec
    passband_edge = sliced_bandwidth/2
    # derive the transition bandwidth to be some small proportion of the remaining bandwidth
    transition_bandwidth = ((sample_rate/2)-passband_edge)/64
    # calculate the filter cutoff
    cutoff = 0.999*(passband_edge + (transition_bandwidth/2))
    # design the LPF
    lpf_weights = low_pass(cutoff=cutoff,transition_bandwidth=transition_bandwidth,sample_rate=sample_rate)
    # modulate the LPF to BPF
    num_lpf_weights = len(lpf_weights)
    n = np.arange(-int((num_lpf_weights-1)/2),int((num_lpf_weights-1)/2)+1)
    bpf_weights = lpf_weights * np.exp(2j*np.pi*(sliced_center_freq/sample_rate)*n)
    # apply BPF
    output = convolve(input_signal,bpf_weights)

    return output, box_center_freq, box_bandwidth


def is_even(number):
    """Is the number even?

    Returns true if the number is even, false if the number is odd

    Args:
        number: Any number

    Returns:
        bool: Returns true if number is even, false if number is odd
    """
    return np.mod(number,2) == 0

def is_odd(number):
    """Is the number odd?

    Returns true if the number is odd, false if the number is even

    Args:
        number: Any number

    Returns:
        bool: Returns true if number is odd, false if number is even
    """
    return not is_even(number)

def is_multiple_of_4(number):
    """Is the number a multiple of 4?

    Returns true if the number is a multiple of 4, false otherwise.
    A number os a multiple of 4 if both the number is even and the
    number divided by 2 is even.

    Args:
        number: Any number

    Returns:
        bool: Returns true if number is a multiple of 4, false otherwise
    """
    return is_even(number) and is_even(int(number/2))



def interpolate_power_of_2_resampler(input_signal:np.ndarray, interpolation_rate:int)-> np.ndarray:
    """Applies power of 2 resampling

    Args:
        input_signal (np.ndarray): Input signal to be resampled
        interpolation_rate (_type_): Interpolate rate, must be greater than 0. For interpolation,
        interpolate_rate >= 2.

    Raises:
        ValueError: Throws error if the interpolation rate is not an integer
        ValueError: Throws error if the interpolation rate is not >= 2.
        ValueError: Throws error if the interpolation rate is not a power of 2.

    Returns:
        np.ndarray: Interpolated signal
    """

    # interpolation_rate must be an integer
    if not isinstance(interpolation_rate,int):
        raise ValueError('interpolation_rate: ' + str(interpolation_rate) + ', must be an integer')

    # interpolation rate must be >= 2
    if interpolation_rate < 2:
        raise ValueError('interpolation_rate: ' + str(interpolation_rate) + ', must be >= 2')

    # interpolation_rate must be a power of 2
    if is_odd(interpolation_rate):
        raise ValueError('interpolation_rate: ' + str(interpolation_rate) + ', must be power of 2')

    # determine how many 1;2 stages are needed
    num_stages = int(np.log2(interpolation_rate))

    # universal filter params
    attenuation_db = 120
    passband_percentage = 0.8 # the percent of bandwidth that the passband edge represents

    # sample buffer to be continually processed, perform a copy
    # as to not modifying the input variable by reference
    sample_buffer = copy(input_signal)

    # iterate through each stage
    for stage_number in range(num_stages):
        # design the interpolator filter for current stage
        weights = design_half_band_filter(stage_number, passband_percentage, attenuation_db)
        # scale by 2 to account maintain consistent power level through the 1:2 interpolation
        weights *= 2
        # apply 1:2 interpolate
        sample_buffer = sp.upfirdn(weights,sample_buffer,up=2,down=1)
        # discard transition samples due to filter
        discard_number = int((len(weights)-1)/2)
        sample_buffer = sample_buffer[discard_number:-discard_number+1]

    return sample_buffer


def design_half_band_filter(stage_number:int=0, passband_percentage:float=0.8, attenuation_db:float=120)-> np.ndarray:
    """Designs half band filter weights for dyadic resampling

    Implements the filter design for dyadic (power of 2) resampling, see fred
    harris, Multirate Signal Processing for Communication Systems, 2nd Edition,
    Chapter 8.7.

    The dyadic filter uses a series of stages, a multi-stage structure, to
    efficiently implement large resampling rates. For interpolation, each additional
    stage increases the resampling rate by a factor of 2, and therefore the 
    signal bandwidth becomes a smaller relative proportion of the sample rate.
    Therefore, both the passband edge will be decreased for each successive stage,
    which also allows for the transition bandwidth to be increased for each successive
    stage, thereby also reducing the amount of computation needed.

    Args:
        stage_number (int, optional): Stage number in the cascade, must be greater than
            or equal to zero. Defaults to 0.
        passband_percentage (float, optional): The proportion of the available bandwidth
            used for the passband edge. The default of 0.8 translates into the passband
            edge being 80% of maximum passband of fs/4, or 0.8*fs/4.
        attenuation_db (float, optional): The sidelobe attenuation level, must be greater
            than zero. Defaults to 120.

    Raises:
        ValueError: Checks to ensure that the filter length has the appropriate length

    Returns:
        np.ndarray: Half band filter weights
    """

    sample_rate = 1
    cutoff = sample_rate/4
    fpass = cutoff * passband_percentage / (2**stage_number)
    transition_bandwidth = 2*(cutoff - fpass)
    fstop = fpass + transition_bandwidth
  
    # initial filter length estimation 
    filter_length_estim = estimate_filter_length(transition_bandwidth,attenuation_db,sample_rate)

    # a properly sized half band filter will be such that filter_length_estim+1 is a multiple of 4
    filter_length_plus_1 = int(np.ceil((filter_length_estim+1)/4))*4
    filter_length = filter_length_plus_1 - 1

    # design the filter
    weights = sp.firwin(
        filter_length,
        cutoff,
        width=transition_bandwidth,
        scale=True,
        fs=sample_rate,
    )

    # find the index corresponding to the middle weight
    middle_weight_index = int(len(weights-1)/2)
    # compute the indices of zero weights to the left and right of the middle weight
    left_indices = np.arange(middle_weight_index-2,0,-2)
    right_indices = np.arange(middle_weight_index+2,filter_length,2)
    # replace the approximately-zero weights with literal zeros
    weights[left_indices] = 0
    weights[right_indices] = 0

    # check if filter_length+1 is multiple of 4
    if not is_multiple_of_4(filter_length+1):
        raise ValueError('filter length: ' + str(filter_length) + ', filter_length+1 must be an multiple of 4')

    return weights

def multistage_polyphase_resampler(input_signal:np.ndarray, resample_rate:float) -> np.ndarray:
    """Multi-stage polyphase filterbank-based resampling.

    If the resampling rate is 1.0, then nothing is done and then same input
    signal is returned. If the resampling rate is greater than 1, then it
    performs interpolation using `multistage_polyphase_interpolator`. If
    the resampling rate is less than 1, then it performs interpolation using
    `multistage_polyphase_decimator`.

    Args:
        input_signal (np.ndarray): The input signal to be resampled
        resample_rate (float): The resampling rate. Must be greater than 0.

    Returns:
        np.ndarray: The resampled signal
    """
    resample_out = input_signal
    if resample_rate == 1:
        # no resampling, pass through
        resample_out = input_signal
    if resample_rate > 1: # interpolation
        # call the multi-stage polyphase interpolator
        resample_out = multistage_polyphase_interpolator(input_signal, resample_rate)
    elif resample_rate < 1: # decimation
        # apply the decimation
        resample_out = multistage_polyphase_decimator(input_signal, 1/resample_rate)

    return resample_out

def multistage_polyphase_decimator(input_signal:np.ndarray, decimation_rate:float) -> np.ndarray:
    """Multi-stage polyphase filterbank-based decimation

    The decimation is applied with two possible stages. The first stage implements the
    an integer rate portion and the second stage implements the fractional rate portion.

    For example, a resampling rate of 0.4 is a decimation by 2.5. The decimation of 2.5 
    is represented by an integer decimation of 2, and the fractional rate is therefore 
    2.5/2 = 1.25. Therefore a decimation by 2 is applied followed by a decimation of 1.25.

    Args:
        input_signal (np.ndarray): The input signal
        decimation_rate (float): The decimation rate. Must be greater or equal to 1.

    Returns:
        np.ndarray: The decimated signal
    """

    # calculate the integer interpolation rate, will be implemented by one function
    decimation_integer_rate = int(decimation_rate)
    # calculate the fractional or remainder interpolation rate, implemented by a second function
    decimation_fractional_rate = decimation_rate/decimation_integer_rate

    if decimation_integer_rate > 1:
        # decimate by integer rate
        decimation_integer_out = polyphase_decimator(input_signal, decimation_integer_rate)
    else:
        # no integer decimation, pass through only
        decimation_integer_out = input_signal

    if decimation_fractional_rate > 1:
        # apply fractional rate resampling
        decimation_fractional_out = polyphase_fractional_resampler(decimation_integer_out, 1/decimation_fractional_rate)
    else:
        # no resampling, pass through
        decimation_fractional_out = decimation_integer_out


    return decimation_fractional_out

def multistage_polyphase_interpolator (input_signal:np.ndarray, resample_rate_ideal:float) -> np.ndarray:
    """Multi-stage polyphase filterbank-based interpolation

    The interpolation is applied with two possible stages. The first stage implements the
    the fractional rate portion and the the second stage implements the integer rate portion.

    For example, a resampling rate of 2.5 is an interpolation of 2.5. The interpolation of 2.5
    is represented by an integer interpolation of 2, and the fractional rate is therefore 
    2.5/2 = 1.25. Therefore an interpolation of of 1.25 is applied followed by an interpolation
    of 2.

    Args:
        input_signal (np.ndarray): The input signal
        decimation_rate (float): The interpolation rate. Must be greater or equal to 1.

    Returns:
        np.ndarray: The interpolated signal
    """

    # change variable name
    interpolation_rate = resample_rate_ideal
    # calculate the integer interpolation rate, will be implemented by one function
    interpolation_integer_rate = int(interpolation_rate)
    # calculate the fractional or remainder interpolation rate, implemented by a second function
    interpolation_fractional_rate = interpolation_rate/interpolation_integer_rate

    if interpolation_fractional_rate > 1:
        # interpolate by a fractional rate
        interpolate_fractional_out = polyphase_fractional_resampler (input_signal, interpolation_fractional_rate)
    else:
        # no rate change, just a pass through
        interpolate_fractional_out = input_signal

    if interpolation_integer_rate > 1:
        # interpolate by an integer rate
        interpolate_integer_out = polyphase_integer_interpolator(interpolate_fractional_out, interpolation_integer_rate)
    else:
        # no rate change, just a pass through
        interpolate_integer_out = interpolate_fractional_out

    return interpolate_integer_out

def polyphase_fractional_resampler (input_signal:np.ndarray, fractional_rate:float) -> np.ndarray:
    """Fractional rate polyphase resampler

    Implements a fractional rate resampler through the SciPy upfirdn() function
    with a large number of branches. A fixed "up" rate of 10,000 is used and the
    fractional rate then deterimes the "down" rate, such that up/down reasonably
    approximates the desired fractional resampling rate.

    Args:
        input_signal (np.ndarray): Input signal to be resampled
        fractional_rate (float): The fractional interpolation rate, must be greater than 0.

    Returns:
        np.ndarray: Resampled signal
    """

    # map the fractional part to an up and down rate
    base_rate = 10000
    up_rate = base_rate
    down_rate = int(np.ceil(base_rate/fractional_rate))

    # design the prototype filter
    prototype_filter = prototype_polyphase_filter_interpolation(base_rate)
    filter_length = len(prototype_filter)
    taps_per_branch = (filter_length-1)/base_rate

    # apply the interpolator
    fractional_interp_out = sp.upfirdn(prototype_filter,input_signal,up_rate,down_rate)

    # discard transition period at beginning and end
    total_subtract_off = taps_per_branch*fractional_rate
    subtract_begin = int(np.floor(total_subtract_off/2))
    subtract_end = int(np.ceil(total_subtract_off/2))
    fractional_interp_out = fractional_interp_out[subtract_begin:-subtract_end]

    return fractional_interp_out

def prototype_polyphase_filter_interpolation (num_branches:int, attenuation_db=120) -> np.ndarray:
    """Designs polyphase filterbank weights for interpolation

    Args:
        num_branches (int): Number of branches in the polyphase filterbank.
        attenuation_db (int, optional): Sidelobe attenuation level in dB. Defaults to 120.

    Returns:
        np.ndarray: Filter weights
    """
    # design the prototype filter
    weights = prototype_polyphase_filter (num_branches, attenuation_db)
    # scale the weights for interpolation
    weights *= num_branches
    return weights

def prototype_polyphase_filter_decimation (num_branches:int, attenuation_db=120) -> np.ndarray:
    """Designs polyphase filterbank weights for decimation

    Args:
        num_branches (int): Number of branches in the polyphase filterbank.
        attenuation_db (int, optional): Sidelobe attenuation level in dB. Defaults to 120.

    Returns:
        np.ndarray: Filter weights
    """
    # design the prototype filter
    weights = prototype_polyphase_filter(num_branches, attenuation_db)
    # scale the weights for decimation
    weights /= num_branches
    return weights

def prototype_polyphase_filter (num_branches:int, attenuation_db=120) -> np.ndarray:
    """Designs the prototype filter for a polyphase filter bank

    Args:
        num_branches (int): Number of branches in the polyphase filterbank
        attenuation_db (int, optional): Sidelobe attenuation level. Defaults to 120.

    Returns:
        np.ndarray: Filter weights
    """

    # design filter
    sample_rate = 1.0
    cutoff = sample_rate/(2*num_branches)
    transition_bandwidth = sample_rate/(2*num_branches)

    # design prototype filter weights
    filter_weights = low_pass_iterative_design(cutoff,transition_bandwidth,sample_rate,attenuation_db)

    return filter_weights


def polyphase_integer_interpolator (input_signal:np.ndarray, interpolation_rate:int) -> np.ndarray:
    """Integer-rate polyphase filterbank-based interpolation

    Args:
        input_signal (np.ndarray): Input signal to be interpolated
        interpolation_rate (int): The interpolation rate

    Raises:
        ValueError: Throws an error if the right number of samples are not produced

    Returns:
        np.ndarray: Interpolated output signal
    """

    # update variable name
    num_branches = interpolation_rate

    # design the prototype polyphase filter
    interpolation_filter = prototype_polyphase_filter_interpolation(num_branches)

    # apply interpolation
    interpolate_out = sp.upfirdn(interpolation_filter,input_signal,interpolation_rate,1)
    
    # subtract off transition periods
    half_filter_length = int((len(interpolation_filter)-1)/2)
    if is_even(num_branches):
        subtract_off_begin = half_filter_length - int(num_branches/2)
        subtract_off_end = half_filter_length - int(num_branches/2) + 1
    else:
        subtract_off_begin = half_filter_length - int(np.floor(num_branches/2))
        subtract_off_end = half_filter_length - int(np.ceil(num_branches/2)) + 1

    interpolate_out = interpolate_out[subtract_off_begin:-subtract_off_end]

    # length check for even interpolation rates
    equal_lengths_boolean = len(interpolate_out) == int(num_branches*len(input_signal))
    # length check for odd interpolation rates
    lengths_off_by_one_boolean = len(interpolate_out) == int((num_branches*len(input_signal))+1)

    if not (equal_lengths_boolean or lengths_off_by_one_boolean):
        raise ValueError('polyphase_integer_interpolator() does not have proper number of samples')

    return interpolate_out


def polyphase_decimator (input_signal:np.ndarray, decimation_rate:int) -> np.ndarray:
    """Integer-rate polyphase filterbank-based decimation

    Args:
        input_signal (np.ndarray): Input signal to be decimated
        decimation_rate (int): The decimation rate

    Raises:
        ValueError: Throws an error if the right number of samples are not produced

    Returns:
        np.ndarray: Decimated output signal
    """

    # update variable name
    num_branches = decimation_rate

    # design the prototype polyphase filter
    decimation_filter = prototype_polyphase_filter_decimation(num_branches)

    # apply interpolation
    decimate_out = sp.upfirdn(decimation_filter,input_signal,1,decimation_rate)
    
    # subtract off transition periods
    half_filter_length = (len(decimation_filter)-1)/2

    if is_even(num_branches):
        subtract_off_begin = int(np.floor(half_filter_length/decimation_rate))
        subtract_off_end = int(np.ceil(half_filter_length/decimation_rate))
    else:
        subtract_off_begin = int(np.floor(half_filter_length/decimation_rate))
        subtract_off_end = int(np.ceil(half_filter_length/decimation_rate))+1

    decimate_out = decimate_out[subtract_off_begin:-subtract_off_end]

    # length checks, have to account the ceil() and floor() round-off
    fractional_length = len(input_signal)/num_branches
    length_floor_boolean = len(decimate_out) == int(np.floor(fractional_length))
    length_ceil_boolean = len(decimate_out) == int(np.ceil(fractional_length))

    length_off_by_one_floor_boolean = len(decimate_out) == int(np.floor(fractional_length)-1)
    length_off_by_one_ceil_boolean = len(decimate_out) == int(np.ceil(fractional_length)-1)

    if not (length_floor_boolean or length_ceil_boolean or length_off_by_one_floor_boolean or length_off_by_one_ceil_boolean):
        raise ValueError('polyphase_decimator() does not have proper number of samples')

    return decimate_out

def upsample (signal:np.ndarray, rate:int) -> np.ndarray:
    """Upsamples a signal

    Upsamples a signal by insertion of zeros. Ex: upsample by
    2 produces: sample, 0, sample, 0, sample 0, etc., and
    upsample by 3 produces sample, 0, 0, sample, 0, 0, etc.

    Args:
        signal (np.ndarray): The input signal
        rate (int): The upsampling rate, must be > 1

    Raises:
        ValueError: Throws an error when the rate is less or equal to 1
        ValueError: Throws an error when the rate is not an integer

    Returns:
        np.ndarray: The upsampled signal
    """

    if rate <= 1:
        raise ValueError('rate for upsample() must be > 1')
    if not isinstance(rate,int):
        raise ValueError('rate for upsample() must be an integer')

    # check if array is either real or complex
    is_real = np.isrealobj(signal)
    is_complex = np.iscomplexobj(signal)

    # set up variable to establish proper data type for return array
    dtype = float
    if is_real:
        dtype = float
    elif is_complex:
        dtype = complex

    # create the upsampled signal
    signal_upsampled = np.zeros(rate*len(signal),dtype=dtype)
    signal_upsampled[::rate] = signal

    return signal_upsampled


def center_freq_from_lower_upper_freq (lower_freq:float, upper_freq:float) -> float:
    """Calculates center frequency from lower frequency and upper frequency

    Args:
        lower_freq (float): The lower frequency corresponding to the 3 dB bandwidth of the signal
        upper_freq (float): The upper frequency corresponding to the 3 dB bandwidth of the signal

    Returns:
        float: The center frequency
    """
    center_freq = (lower_freq + upper_freq)/2
    return center_freq

def bandwidth_from_lower_upper_freq (lower_freq:float, upper_freq:float) -> float:
    """Calculates bandwidth from lower frequency and upper frequency

    Args:
        lower_freq (float): The lower frequency corresponding to the 3 dB bandwidth of the signal
        upper_freq (float): The upper frequency corresponding to the 3 dB bandwidth of the signal

    Returns:
        float: The bandwidth
    """
    bandwidth = upper_freq - lower_freq
    return bandwidth


def lower_freq_from_center_freq_bandwidth (center_freq:float, bandwidth:float) -> float:
    """Calculates the lower frequency from center frequency and bandwidth

    Args:
        center_freq (float): The center frequency of the signal
        bandwidth (float): The bandwidth of the signal

    Returns:
        float: The lower frequency
    """
    lower_freq = center_freq - (bandwidth/2)
    return lower_freq

def upper_freq_from_center_freq_bandwidth (center_freq:float, bandwidth:float) -> float:
    """Calculates upper frequency from center frequency and bandwidth

    Args:
        center_freq (float): The center frequency of the signal
        bandwidth (float): The bandwidth of the signal

    Returns:
        float: The upper frequency
    """
    upper_freq = center_freq + (bandwidth/2)
    return upper_freq

#def calculate_signal_power_from_snr(snr_db: float, noise_power_db:float, oversampling_rate:float):
#    """
#    Calculate the signal power based on the SNR in dB. Accounts for the oversampling rate
#    in the calculation.
#
#    Args:
#        snr_db (float): The SNR (in dB) of the signal
#        noise_power_db (float): The noise power (in dB), used in signal power calculation
#        oversampling_rate (float): The amount of oversampling of the signal, need to 
#            account for power changes in signal since it modifies the measured signal power
#
#    Returns the signal power in dB
#    """
#    signal_power_db = snr_db - noise_power_db - 10*np.log10(oversampling_rate)
#    return signal_power_db


def frequency_shift(signal:np.ndarray, frequency:float, sample_rate:float) -> np.ndarray:
    """Performs a frequency shift

    Args:
        signal (np.ndarray): Input signal
        frequency (float): The frequency to shift by. Must have the same units
            as `sample_rate`.
        sample_rate (float): The sample rate of the signal. Must have the same
            units as `frequency`.

    Returns:
        np.ndarray: The frequency shifted signal
    """
    # build mixer
    mixer = np.exp(2j*np.pi*(frequency/sample_rate)*np.arange(0,len(signal)))
    return signal*mixer

def compute_spectrogram(
        iq_samples:np.ndarray, 
        fft_size:int, 
        fft_stride:int
) -> np.ndarray:
    """Computes two-dimensional spectrogram values in dB.

    Args:
        iq_samples (np.ndarray): Input signal.
        fft_size (int): The size of the FFT in number of bins.
        fft_stride (int): The stride is the amount by which the input sample
            pointer increases for each FFT. When fft_stride=fft_size, then there is
            no overlap of input samples in successive FFTs. When fft_stride=fft_size/2,
            there is 50% overlap of input samples between successive FFTs.

    Raises:
        ValueError: Throws an error if fft_stride is less than 0 or greater than `fft_size`.

    Returns:
        np.ndarray: Two-dimensional array of spectrogram values in dB.
    """

    # error handling
    if fft_stride <= 0:
        raise ValueError(f'0 < {fft_stride} <= {fft_size}')

    # input signal is too short and needs to be zero-padded
    if len(iq_samples) < fft_size:
        # number of zeros to be padded
        num_zeros = fft_size-len(iq_samples)
        # form the zero array
        zero_padding = np.zeros(num_zeros,dtype=torchsig_complex_data_type)
        # put zeros at the end
        iq_samples_formatted = np.concatenate((iq_samples,zero_padding))
    else:
        # do not modify input samples
        iq_samples_formatted = copy(iq_samples)

    # get reference to spectrogram function
    spectrogram_function = torchaudio.transforms.Spectrogram(n_fft=fft_size, window_fn=torch.blackman_window, win_length=fft_size, hop_length=fft_stride, normalized=True, center=False, onesided=False, power=2)

    # compute the spectrogram in linear units
    spectrogram_linear = spectrogram_function(torch.from_numpy(iq_samples_formatted))

    # apply FFT shift
    spectrogram_linear_fftshift = torch.fft.fftshift(spectrogram_linear, dim=0)

    # convert to numpy types
    spectrogram_linear_numpy = spectrogram_linear_fftshift.numpy()

    # calculate a small epsilon value to replace all zero values
    epsilon = np.max(np.max(np.abs(spectrogram_linear_numpy)))*np.sqrt(1e-20)

    # find the zero locations, and replace them with tiny values
    zero_ind_rows, zero_ind_cols = np.where(spectrogram_linear_numpy == 0)
    spectrogram_linear_numpy[zero_ind_rows,zero_ind_cols] = epsilon

    # convert to dB
    spectrogram_db = 10*np.log10(spectrogram_linear_numpy)

    # reverse bins order of FFT bins
    spectrogram_db = spectrogram_db[::-1,:]

    return spectrogram_db

def estimate_tone_bandwidth(num_samples:int, sample_rate:float):
    """Estimate the bandwidth of a tone

    The bandwidth of a tone is completely defined by the number
    of samples in the time-series.

    Args:
        num_samples (int): The length of the tone in samples.
        sample_rate (float): The sample rate associated with the tone.

    Returns:
        np.ndarray: Bandwidth estimate of the tone

    """    
    return sample_rate/num_samples

def convolve(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Wrapper function to implement convolution()

    A wrapped version of SciPy's convolve(), which discards trasition regions
    resulting from the convolution process.

    Args:
        signal (np.ndarray): The input signal
        taps (np.ndarray): The filter weights

    Returns:
        np.ndarray: The convolution output
    """

    filtered = sp.convolve(signal, taps, mode='full')
    if is_even(len(taps)): # even-length filter
        slice_length = int(len(taps)/2)
    else: # odd-length filter
        slice_length = int((len(taps)-1)/2)
    return filtered[slice_length:-slice_length]

def low_pass(cutoff: float, transition_bandwidth: float, sample_rate:float, attenuation_db:float=120) -> np.ndarray:
    """Low-pass filter design

    Args:
        cutoff (float): The filter cutoff, 0 < `cutoff` < sample_rate/2. Must
            be in the same units as `sample_rate`.
        transition_bandwidth (float): The transition bandwidth of the filter,
            0 < `transition_bandwidth` < `sample_rate`/2. Must be in the same units
            as `sample_rate`.
        sample_rate (float): The sampling rate associated with the filter design.
        attenuation_db (float, optional): Sidelobe attenuation level. Defaults to 120.

    Returns:
        np.ndarray: Filter weights
    """
    num_taps = estimate_filter_length(transition_bandwidth,attenuation_db,sample_rate)
    return sp.firwin(
        num_taps,
        cutoff,
        width=transition_bandwidth,
        scale=True,
        fs=sample_rate,
    )

def estimate_filter_length(transition_bandwidth: float, attenuation_db:float, sample_rate:float) -> int:
    """Estimates FIR filter length

    Estimate the length of an FIR filter using fred harris' approximation,
    Multirate Signal Processing for Communication Systems, Second Edition, p.59.

    Args:
        transition_bandwidth (float): The transition bandwidth of the filter,
            0 < `transition_bandwidth` < sample_rate/2.
        attenuation_db (float): Sidelobe attenuation level in dB.
        sample_rate (float): The sampling rate associated with the filter design.

    Returns:
        int: The estimated filter length
    """

    filter_length = int(np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22)))

    # odd-length filters are desirable because they do not introduce a half-sample delay
    if np.mod(filter_length, 2) == 0:
        filter_length += 1

    return filter_length


def srrc_taps(iq_samples_per_symbol: int, filter_span_in_symbols: int, alpha: float = 0.35) -> np.ndarray:
    """Designs square-root raised cosine (SRRC) pulse shaping filter

    Args:
        iq_samples_per_symbol (int): The samples-per-symbol (SPS) of the underlying modulation,
            equivalent to the oversampling rate.
        filter_span_in_symbols (int): The filter span in number of symbols.
        alpha (float, optional): The alpha roll-off value of the pulse shaping filter, which
            is the amount of excess bandwidth. Defaults to 0.35.

    Returns:
        np.ndarray: SRRC filter weights
    """

    m = filter_span_in_symbols
    n_s = float(iq_samples_per_symbol)
    n = np.arange(-m * n_s, m * n_s + 1)
    taps = np.zeros(int(2 * m * n_s + 1))
    for i in range(int(2 * m * n_s + 1)):
        # handle the discontinuity at t=+-n_s/(4*alpha)
        if n[i] * 4 * alpha == n_s or n[i] * 4 * alpha == -n_s:
            taps[i] = (
                1
                / 2.0
                * (
                    (1 + alpha) * np.sin((1 + alpha) * np.pi / (4.0 * alpha))
                    - (1 - alpha) * np.cos((1 - alpha) * np.pi / (4.0 * alpha))
                    + (4 * alpha) / np.pi * np.sin((1 - alpha) * np.pi / (4.0 * alpha))
                )
            )
        else:
            taps[i] = 4 * alpha / (np.pi * (1 - 16 * alpha**2 * (n[i] / n_s) ** 2))
            taps[i] = taps[i] * (
                np.cos((1 + alpha) * np.pi * n[i] / n_s)
                + np.sinc((1 - alpha) * n[i] / n_s) * (1 - alpha) * np.pi / (4.0 * alpha)
            )
    return taps


def gaussian_taps(samples_per_symbol: int, bt: float = 0.35) -> np.ndarray:
    """Designs Gaussian filter weights

    Args:
        samples_per_symbol (int): Samples-per-symbol (SPS) for the underlying
            modulation, equivalent to the oversampling rate.
        bt (float, optional): Time-bandwidth product. Defaults to 0.35.

    Returns:
        np.ndarray: Gaussian filter weights
    """
    
    m = 4  # filter span in symbols
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / float(samples_per_symbol)) ** 2)
    p = p / np.sum(p)
    return p


def low_pass_iterative_design(cutoff:float, transition_bandwidth:float, sample_rate:float, desired_attenuation_db:float=120)-> np.ndarray:

    # estimate the filter length
    filter_length = estimate_filter_length( transition_bandwidth, desired_attenuation_db, sample_rate)
    #print('filter length = ' + str(filter_length))
    
    # initialize design counter
    iterations = 0

    # maximum number of designs, avoids an infinite loop.
    max_iterations = 2*filter_length

    while True:

        # design the filter
        lpf = sp.firwin(
            filter_length,
            cutoff,
            width=transition_bandwidth,
            scale=True,
            fs=sample_rate,
        )

        # hold onto the initial filter design in case the design
        if iterations == 0:
            lpf_init = copy(lpf)

        # get FFT of filter from 0 to fs/2
        fft_size = 4096
        fft_linear = np.abs(np.fft.fftshift(np.fft.fft(lpf,fft_size*2)))
        fft_linear[np.where(fft_linear == 0)[0]] = 1e-15 # replace all zeros with tiny value
        fft_db = 20*np.log10(fft_linear)
        fft_db = fft_db[fft_size:]
        f = np.linspace(0,0.5,fft_size)*sample_rate
        
        # calculate the stopband edge
        stopband_freq_init = cutoff + (transition_bandwidth/2)
        
        # find the closest bin matching the stopband edge
        stopband_bin = np.argmin(np.abs(stopband_freq_init - f)) 
        stopband_freq = f[stopband_bin]

        # get the maximum sidelobe level from stopband to fs/2
        measured_attenuation_db = np.abs(np.max(fft_db[stopband_bin:]))

        if iterations > max_iterations:
            # hit too many iterations, exit to avoid infinite loop
            raise Warning('low_pass_iterative_design has trouble converging, using initial design.')
            return lpf_init
 
        if desired_attenuation_db  > measured_attenuation_db:
            # the filter is below speed and needs an increase to filter length

            # because filter length is roughly proportional to sidelobe
            # levels, estimate the increase in filter length by the ratio
            # of the sidelobes
            sidelobe_ratio = desired_attenuation_db/measured_attenuation_db

            # estimate a new filter length
            new_filter_length = int(filter_length*sidelobe_ratio)

            if new_filter_length > filter_length:
                # the new filter length is larger, so assign it
                filter_length = copy(new_filter_length)
            else:
                # force an increase in filter length; add 2 to filter 
                # length since odd-length is desirable and no need to 
                # check evens
                filter_length += 2

            # include check to ensure the filter length is odd
            if is_even(filter_length):
                filter_length += 1
        else:
            # the design is within spec and completed before reaching max
            # number of iterations, so leave the 
            break

        # increment the count for how many designs have been completed
        iterations += 1

    return lpf

