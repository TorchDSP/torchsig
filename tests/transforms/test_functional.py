"""Unit Tests: transforms/functional
"""

from torchsig.transforms.functional import (
    add_slope,
    additive_noise,
    adjacent_channel_interference,
    awgn,
    carrier_frequency_drift,
    carrier_phase_noise,    
    channel_swap,
    coarse_gain_change,
    cochannel_interference,    
    cut_out,
    doppler,    
    drop_samples,
    fading,
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
    time_reversal,
    time_varying_noise,
    tracking_agc,
)
from test_transforms_utils import (
    generate_test_signal,
    generate_tone_signal
)
import torchsig.utils.dsp as dsp
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    compute_spectrogram,
)

# Third Party
from typing import Any
import numpy as np
import scipy as sp
from copy import deepcopy
import pytest


RTOL = 1E-6
TEST_DATA = generate_test_signal(num_iq_samples = 64, scale = 1.0).data


@pytest.mark.parametrize("data, expected, is_error", [
    (0, ValueError, True),
    (deepcopy(TEST_DATA), True, False)
])
def test_add_slope(
    data: Any, 
    expected: bool | ValueError, 
    is_error: bool
    ) -> None:
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
        data_test = deepcopy(data)
        diff = np.diff(data_test)
        diff = np.insert(diff, 0, 0)
        data_test += diff

        data = add_slope(data)

        assert np.allclose(data, data_test, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (deepcopy(TEST_DATA), {'noise_power': -42.0,'noise_color': 'white', 'continuous': False}, ValueError,True),
    (deepcopy(TEST_DATA), {'noise_power': 1.0, 'noise_color': 'purple','continuous': False}, ValueError, True),
    (deepcopy(TEST_DATA), {'noise_power': 42.4, 'noise_color': 'white', 'continuous': False}, True, False), 
    (deepcopy(TEST_DATA), {'noise_power': 4.2, 'noise_color': 'white', 'continuous': True}, True, False),
    (deepcopy(TEST_DATA), {'noise_power': 0.1, 'noise_color': 'pink','continuous': True}, True, False),
    (deepcopy(TEST_DATA), {'noise_power': 3.14, 'noise_color': 'red','continuous': True}, True, False)
])
def test_additive_noise(
    data: Any, 
    params: dict,
    expected: bool | AttributeError,
    is_error: bool
    ) -> None:
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

    power = params['noise_power']
    color = params['noise_color']
    continuous = params['continuous']

    if is_error:
        with pytest.raises(expected):
            data = additive_noise(
                data  = data,
                power = power, 
                color = color,
                continuous = continuous,
                rng   = rng 
            )
    else:
        data_test = deepcopy(data)
        data = additive_noise(
            data  = data,
            power = power, 
            color = color,
            continuous = continuous,
            rng   = rng 
        )
        
        input_power = np.sum(np.abs(data_test)**2)/len(data_test)
        output_power = np.sum(np.abs(data)**2)/len(data)
        power_delta = output_power - input_power

        assert (len(data) == len(data_test)) == expected
        assert (np.abs(power_delta - power) < 10**(0.1/10)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("params, expected, is_error", [
    ({'N': 8192, 'sample_rate': 4.0, 'power': 0.5, 'adj_fc': 0.2, 'tone_freq': 0.042, 'phase_sigma': 0.0, 'time_sigma': 0.0, 'filter_weights': dsp.low_pass(0.25, 0.25, 4.0)}, True, False),
    ({'N': 16384, 'sample_rate': 2.5, 'power': 0.25, 'adj_fc': -0.12, 'tone_freq': 0.1, 'phase_sigma': 1.0, 'time_sigma': 4.0, 'filter_weights': dsp.low_pass(0.11, 0.18, 2.5)}, True, False),
])
def test_adjacent_channel_interference(
    params: dict,
    expected: bool,
    is_error: bool
    ) -> None:
    """Test the adjacent_channel_interference functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """     
    rng = np.random.default_rng(42)

    N = params['N']
    sample_rate = params['sample_rate']
    adj_power = params['power']
    tone_freq = params['tone_freq']
    center_frequency = params['adj_fc']
    filter_weights = params['filter_weights']
    phase_sigma = params['phase_sigma']

    # tone signal: freq = tone_freq, power = 1.0 W
    tone_baseband = generate_tone_signal(num_iq_samples = N, scale = 1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N) / sample_rate) *np.sqrt(N)

    if is_error:
        with pytest.raises(expected):
            data = adjacent_channel_interference(
                data = data,
                sample_rate = sample_rate,
                power = adj_power,
                center_frequency = center_frequency,
                filter_weights = filter_weights,
                phase_sigma = phase_sigma,
                rng = rng
            )
    else:
        data_test = deepcopy(data)
        data = adjacent_channel_interference(
            data = data,
            sample_rate = sample_rate,
            power = adj_power,
            center_frequency = center_frequency,
            filter_weights = filter_weights,
            phase_sigma = phase_sigma,
            rng = rng
        )

        est_power = np.sum(np.abs(data)**2)/len(data)

        D = np.abs(np.fft.fft(data, norm='ortho'))
        freqs = np.fft.fftfreq(N) * sample_rate
        peaks, _ = sp.signal.find_peaks(D, height=1.0, distance=N/20)
        top_two_indices = np.argsort(D[peaks])[-2:][::-1]
        freqs0 = freqs[peaks[top_two_indices[0]]]
        freqs1 = freqs[peaks[top_two_indices[1]]]

        assert (np.abs(est_power - (adj_power + 1.0)) < 10**(0.1/10)) == expected
        assert (np.abs(freqs0 - tone_freq) < (3/N)) == expected
        assert (np.abs(freqs1 - (tone_freq + center_frequency)) < 0.01) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (0, {'noise_power_db' : 0.0}, AttributeError, True),
    (np.zeros(1024, dtype=torchsig_complex_data_type), {'noise_power_db' : 3.0}, True, False)
])
def test_awgn(
    data: Any, 
    params: dict, 
    expected: bool | AttributeError, 
    is_error: bool
    ) -> None:
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
    noise_power_db = params['noise_power_db']
    
    if is_error:
        with pytest.raises(expected): 
            data = awgn(
                data, 
                noise_power_db  = noise_power_db, 
                rng             = rng 
            )
    else:
        noise_power_linear = 10 ** (noise_power_db / 10.0)
        data_test = deepcopy(data)

        data = awgn(
            data, 
            noise_power_db  = noise_power_db, 
            rng             = rng 
        )
        power_est = np.mean(np.abs(data)**2)
        
        assert (abs(power_est - noise_power_linear) < 1E-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (generate_tone_signal(num_iq_samples = 1024, scale = 1.0).data, {'drift_ppm': 0.1}, True, False),
    (generate_tone_signal(num_iq_samples = 1024, scale = 1.0).data, {'drift_ppm': 1}, True, False),
])
def test_carrier_frequency_drift(
    data: Any, 
    params: dict, 
    expected: bool, 
    is_error: bool
    ) -> None:
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

    drift_ppm = params['drift_ppm']

    if is_error:
        with pytest.raises(expected): 
            data = carrier_frequency_drift(
                data = data,
                drift_ppm = drift_ppm,
                rng = rng
            )
    else:
        data_test = deepcopy(data)
        data = carrier_frequency_drift(
            data = data,
            drift_ppm = drift_ppm,
            rng = rng
        )

        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (deepcopy(TEST_DATA), {'phase_noise_degrees': 1}, True, False),
])
def test_carrier_phase_noise(
    data: Any, 
    params: dict, 
    expected: bool, 
    is_error: bool
    ) -> None:
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

    phase_noise_degrees = params['phase_noise_degrees']

    if is_error:
        with pytest.raises(expected): 
            data = carrier_phase_noise(
                data = data,
                phase_noise_degrees = phase_noise_degrees,
                rng = rng
            )
    else:
        data_test = deepcopy(data)
        data = carrier_phase_noise(
            data = data,
            phase_noise_degrees = phase_noise_degrees,
            rng = rng
        )

        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, expected, is_error", [
    (0, AttributeError, True),
    (deepcopy(TEST_DATA), True, False)
])
def test_channel_swap(
    data: Any, 
    expected: bool | AttributeError, 
    is_error: bool
    ) -> None:
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
        data_test = deepcopy(data)
        test_real = data_test.real
        test_imag = data_test.imag

        data = channel_swap(data)

        assert np.allclose(data.real, test_imag, RTOL) == expected
        assert np.allclose(data.imag, test_real, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        deepcopy(TEST_DATA),
        {'start_idx': 5, 'gain_change_db': 0.25}, 
        True,
        False
    ),
        (
        deepcopy(TEST_DATA),
        {'start_idx': -17, 'gain_change_db': -15.7}, 
        True,
        False
    )    
])
def test_coarse_gain_change(
    data: Any,
    params: dict,
    expected: bool, 
    is_error: bool
    ) -> None:
    """Test the coarse_gain_change functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    start_idx = params['start_idx']
    gain_change_db = params['gain_change_db']
    
    if is_error:
        with pytest.raises(expected):   
            data = coarse_gain_change(
                data = data,
                gain_change_db = gain_change_db,
                start_idx = start_idx
            )
    else:
        data_test = deepcopy(data)

        data = coarse_gain_change(
            data = data,
            gain_change_db = gain_change_db,
            start_idx = start_idx
        )
        
        gain_change_linear = 10**(gain_change_db/10)
        assert (np.allclose(data[start_idx:], gain_change_linear * data_test[start_idx:], RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("params, expected, is_error", [
    ({'N': 8192, 'sample_rate': 4.0, 'power': 0.1, 'tone_freq': 0.2, 'filter_weights': dsp.low_pass(0.25, 0.25, 4.0), 'color': 'white', 'continuous': True}, True, False),
    ({'N': 16384, 'sample_rate': 2.42, 'power': 0.01, 'tone_freq': -0.04, 'filter_weights': dsp.low_pass(0.1, 0.15, 2.42), 'color': 'white', 'continuous': True}, True, False),
])
def test_cochannel_interference(
    params: dict,
    expected: bool | AttributeError,
    is_error: bool
    ) -> None:
    """Test the cochannel_interference functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """     
    rng = np.random.default_rng(42)

    N = params['N']
    sample_rate = params['sample_rate']
    intf_power = params['power']
    tone_freq = params['tone_freq']
    filter_weights = params['filter_weights']
    color = params['color']
    continuous = params['continuous']

    # tone signal: freq = tone_freq, power = 1.0 W
    tone_baseband = generate_tone_signal(num_iq_samples = N, scale = 1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N) / sample_rate) *np.sqrt(N)

    if is_error:
        with pytest.raises(expected):
            data = cochannel_interference(
                data = data,
                power = intf_power,
                filter_weights = filter_weights,
                color = color,
                continuous = continuous
            )
    else:
        data_test = deepcopy(data)
        data = cochannel_interference(
            data = data,
            power = intf_power,
            filter_weights = filter_weights,
            color = color,
            continuous = continuous
        )

        est_power = np.sum(np.abs(data)**2)/len(data)
        
        D = np.abs(np.fft.fft(data, norm='ortho'))
        freqs = np.fft.fftfreq(N) * sample_rate
        peaks, _ = sp.signal.find_peaks(D, height=10.0, distance=N/2)
        est_freq = freqs[peaks[0]]

        assert (np.abs(est_power - (intf_power + 1.0)) < 10**(0.1/10)) == expected
        assert (np.abs(est_freq - tone_freq) < (3/N)) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'start': 0.25, 'duration': 0.5, 'cut_type': 'zeros'},
        AttributeError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.25, 'duration': 0.5, 'cut_type': 'invalid_cut_type'},
        ValueError,
        True
    ),    
    (
        deepcopy(TEST_DATA),
        {'start': 0.25, 'duration': 0.5, 'cut_type': 'zeros'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.0, 'duration': 0.99, 'cut_type': 'ones'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.75, 'duration': 0.1, 'cut_type': 'low_noise'},
        True,
        False        
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.90, 'duration': 0.42, 'cut_type': 'avg_noise'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.5, 'duration': 1.0, 'cut_type': 'high_noise'},
        True,
        False
    ),           
])
def test_cut_out(
    data: Any,
    params: dict,
    expected: bool | AttributeError | ValueError, 
    is_error: bool
    ) -> None:
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
    start = params['start']
    duration = params['duration']
    cut_type = params['cut_type']
    
    if is_error:
        with pytest.raises(expected): 
            data = cut_out(data, start, duration, cut_type, rng)
    else:
        data_test = deepcopy(data)    
        data = cut_out(data, start, duration, cut_type, rng)

        cut_inds = np.where(data != data)[0]
        duration_samples = int(duration * data.size)
    
        if np.any(cut_inds):
            assert duration_samples == cut_inds[-1] - cut_inds[0] + 1

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("params, expected, is_error", [
    ({'N': 10000, 'sampling_rate': 4.0,'tone_freq': 0.2, 'velocity': 1e7}, True, False),
    ({'N': 1000, 'sampling_rate': 2.0,'tone_freq': 0.42, 'velocity': 1e6}, True, False),
])
def test_doppler(
    params: dict,
    expected: bool | AttributeError,
    is_error: bool
    ) -> None:
    """Test the doppler functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """     
    rng = np.random.default_rng(42)

    N = params['N']
    sampling_rate = params['sampling_rate']
    tone_freq = params['tone_freq']
    velocity = params['velocity']

    tone_baseband = generate_tone_signal(num_iq_samples = N, scale = 1.0).data
    data = tone_baseband * np.exp(2j * np.pi * tone_freq * np.arange(N) / sampling_rate)

    if is_error:
        with pytest.raises(expected):
            data = doppler(
                data = data,
                velocity = velocity,
                propagation_speed = 2.9979e8,
                sampling_rate = sampling_rate
            )
    else:
        data_test = deepcopy(data)
        data = doppler(
            data = data,
            velocity = velocity,
            propagation_speed = 2.9979e8,
            sampling_rate = sampling_rate
        )

        alpha = 2.9979e8 / (2.9979e8 - velocity) # scaling factor
        D = np.abs(np.fft.fft(data, norm='ortho'))
        freqs = np.fft.fftfreq(N) * sampling_rate
        peaks, _ = sp.signal.find_peaks(D, height=0.5, distance=N/10)
        est_freq = freqs[peaks[0]]
        
        assert (np.abs(est_freq - alpha*tone_freq) < (3/N)) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'drop_starts': [8], 'drop_sizes': [2], 'fill': 'zero'},
        TypeError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'drop_starts': [8], 'drop_sizes': [2], 'fill': 'invalid_fill'},
        ValueError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'drop_starts': [8], 'drop_sizes': [2], 'fill': 'zero'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'drop_starts': [2, 7], 'drop_sizes': [2, 3], 'fill': 'mean'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'drop_starts': [3, 11], 'drop_sizes': [4, 3], 'fill': 'ffill'},
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'drop_starts': [4], 'drop_sizes': [10], 'fill': 'bfill'},
        True,
        False        
    ),       
])
def test_drop_samples(
    data: Any,
    params: dict,
    expected: bool | TypeError | ValueError,
    is_error: bool
    ) -> None:
    """Test the drop_samples functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """  
    drop_starts = params['drop_starts']
    drop_sizes = params['drop_sizes']
    fill = params['fill']

    if is_error:
        with pytest.raises(expected): 
            data = drop_samples(data, drop_starts, drop_sizes, fill)
    else:
        data_test = deepcopy(data)
        data = drop_samples(data, drop_starts, drop_sizes, fill)

        drop_inds = np.empty(0, dtype=int)
        drop_stops = np.empty(0, dtype=int)
        for idx, drop_start in enumerate(drop_starts):
            drop_stops = np.append(drop_stops, drop_start + drop_sizes[idx])
            drop_inds = np.append(
                drop_inds, 
                np.arange(drop_start, drop_stops[-1], dtype=int)
            )

        if np.any(drop_inds):
            fill_inds = np.where(data != data_test)[0]  
            assert np.allclose(drop_inds, fill_inds, RTOL) == expected

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'coherence_bandwidth': 0.1, 'power_delay_profile': np.array([0.5, 0.25])},
        IndexError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'coherence_bandwidth': 0.1, 'power_delay_profile': np.array([0.5, 0.25])}, 
        True,
        False
    )
])
def test_fading(
    data: Any,
    params: dict,
    expected: bool | IndexError, 
    is_error: bool
    ) -> None:
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

    coherence_bandwidth = params['coherence_bandwidth']
    power_delay_profile = params['power_delay_profile']

    if is_error:
        with pytest.raises(expected): 
            data = fading(
                data, 
                coherence_bandwidth = coherence_bandwidth,
                power_delay_profile = power_delay_profile,
                rng = rng
            )            
    else:
        data_test = deepcopy(data)

        data = fading(
            data, 
            coherence_bandwidth = coherence_bandwidth,
            power_delay_profile = power_delay_profile,
            rng = rng
        )
    
        data_mean_power = np.mean(np.abs(data)**2)
        data_test_mean_power = np.mean(np.abs(data_test)**2)
        assert (abs(data_mean_power - data_test_mean_power) < 1E-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (deepcopy(TEST_DATA), {'coeffs': np.array([])}, IndexError, True),
    (deepcopy(TEST_DATA), {'coeffs': np.array([0.5, 1.0])}, ValueError, True),
    (deepcopy(TEST_DATA), {'coeffs': np.array([0.2, 0, 0.1])}, True, False)
])
def test_intermodulation_products(
    data: Any, 
    params: dict, 
    expected: bool | IndexError, 
    is_error: bool
    ) -> None:
    """Test the intermodulation_products functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    coeffs = params['coeffs']
    
    if is_error:
        with pytest.raises(expected): 
            data = intermodulation_products(data = data, coeffs = coeffs)
    else:
        data_test = deepcopy(data)
        data = intermodulation_products(data = data, coeffs = coeffs)

        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected        

        
@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        generate_test_signal(num_iq_samples = 1024, scale = 1.0).data,
        {
            'amplitude_imbalance': 0.1, 
            'phase_imbalance': -np.pi/4,
            'dc_offset': (-0.1, -0.22)
        },
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 1024, scale = 1.0).data,
        {
            'amplitude_imbalance': 3.4, 
            'phase_imbalance': np.pi/8,
            'dc_offset': (0.27, 0.11)
        },
        True,
        False 
    )    
])
def test_iq_imbalance(
    data: Any,
    params: dict,
    expected: bool, 
    is_error: bool
    ) -> None:
    """Test the iq_imbalance functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    amplitude_imbalance = params['amplitude_imbalance']
    amplitude_imbalance_linear = 10 ** (amplitude_imbalance / 10.0)
    phase_imbalance = params['phase_imbalance']
    dc_offset = params['dc_offset']
    
    if is_error:
        with pytest.raises(expected):     
            data = iq_imbalance(
                data, 
                amplitude_imbalance = amplitude_imbalance,
                phase_imbalance = phase_imbalance,
                dc_offset = dc_offset
            )
    else:
        data_test = deepcopy(data)

        data = iq_imbalance(
            data, 
            amplitude_imbalance = amplitude_imbalance,
            phase_imbalance = phase_imbalance,
            dc_offset = dc_offset
        )

        dc_real_est = np.mean(data.real)
        dc_imag_est = np.mean(data.imag)
        ac_data = (data.real - dc_offset[0]) + 1j * (data.imag - dc_offset[1])
        
        mean_mag_ac_data = np.mean(np.abs(ac_data))
        mean_mag_data_test = np.mean(np.abs(data_test))
        amp_imbal_est = mean_mag_ac_data / mean_mag_data_test
        norm_ac_data = np.divide(ac_data, amplitude_imbalance_linear)

        restored_data = np.multiply(norm_ac_data.real, 1/ np.exp(-1j * phase_imbalance / 2.0)) + \
                        np.multiply(norm_ac_data.imag, 1 / np.exp(1j * (np.pi / 2.0 + phase_imbalance / 2.0)))

        assert (abs(dc_real_est - dc_offset[0]) < 1E-1) == expected
        assert (abs(dc_imag_est - dc_offset[1]) < 1E-1) == expected
        assert (abs(amp_imbal_est - amplitude_imbalance_linear) < 1E-1) == expected
        assert (abs(np.mean(restored_data - data_test)) < 1E-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        deepcopy(TEST_DATA),
        {
            'gain': 1.0,
            'psat_backoff' : 10.0,
            'phi_rad': 0.0,
            'auto_scale': True
        }, 
        True, 
        False
    ),
    (
        deepcopy(TEST_DATA),
        {
            'gain': 7.4,
            'psat_backoff' : 3.0,
            'phi_rad': 0.2,
            'auto_scale': False
        }, 
        True, 
        False
    ),    
])
def test_nonlinear_amplifier(
    data: Any, 
    params: dict, 
    expected: bool, 
    is_error: bool
    ) -> None:
    """Test the nonlinear_amplifier functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    gain = params['gain']
    psat_backoff = params['psat_backoff']
    phi_rad = params['phi_rad']
    auto_scale = params['auto_scale']
    
    if is_error:
        with pytest.raises(expected): 
            data = nonlinear_amplifier(
                data = data,
                gain = gain,
                psat_backoff = psat_backoff,
                phi_rad = phi_rad,
                auto_scale = auto_scale
            )
    else:
        data_test = deepcopy(data)

        data = nonlinear_amplifier(
            data = data,
            gain = gain,
            psat_backoff = psat_backoff,
            phi_rad = phi_rad,
            auto_scale = auto_scale
        )

        input_power = np.mean(np.abs(data_test)**2)
        output_power = np.mean(np.abs(data)**2)        
        psat = input_power * psat_backoff
        input_phase_rad = np.angle(data_test)
        output_phase_rad = np.angle(data)
        phase_diff = abs(np.mean(np.unwrap(output_phase_rad - input_phase_rad)))
        
        if auto_scale:
            assert (abs(output_power - input_power) < 10**(0.1/10)) == expected
        else:
            assert (np.all(output_power <= psat)) == expected
        assert (phase_diff <= (abs(phi_rad) + RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        np.zeros((2,)), 
        {
            'Pin': np.zeros((3,)), 
            'Pout': np.zeros((4,)), 
            'Phi': np.zeros((5,)),
            'p_ratio': 0.,
            'phase_shift': 0.
        }, 
        ValueError, 
        True
    ),
    (
        deepcopy(TEST_DATA),
        {
            'Pin':     10**((np.array([-100., -50.,  0., 50.])) / 10), 
            'Pout':    10**((np.array([ -97., -47.,  3., 53.])) / 10), 
            'Phi': np.deg2rad(np.array([ 0.1,  0.1, 0.1, 0.1])),
            'p_ratio': 10**(3./10),
            'phase_shift': np.deg2rad(0.1)
        }, 
        True, 
        False
    ),
])
def test_nonlinear_amplifier_table(
    data: Any, 
    params: dict, 
    expected: bool | ValueError, 
    is_error: bool
    ) -> None:
    """Test the nonlinear_amplifier_table functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    Pin = params['Pin']
    Pout = params['Pout']
    Phi = params['Phi']
    p_ratio = params['p_ratio']
    phase_shift = params['phase_shift']
    
    if is_error:
        with pytest.raises(expected): 
            data = nonlinear_amplifier_table(
                data = data,
                Pin  = Pin,
                Pout = Pout,
                Phi  = Phi,
                auto_scale = False
            )
    else:
        data_test = deepcopy(data)

        data = nonlinear_amplifier_table(
            data = data,
            Pin  = Pin,
            Pout = Pout,
            Phi  = Phi,
            auto_scale = False
        )

        input_power = np.mean(np.abs(data_test)**2)
        input_phase_rad = np.angle(data_test)
        output_power = np.mean(np.abs(data)**2)
        output_phase_rad = np.angle(data)

        assert (abs(output_power/input_power - p_ratio) < RTOL) == expected
        assert (abs(np.mean(np.unwrap(output_phase_rad - input_phase_rad)) - phase_shift) < RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'norm': 2, 'flatten': False},
        ValueError,
        True
    ),   
    (
        generate_test_signal(num_iq_samples=64, scale=42.0).data,
        {'norm': 2, 'flatten': False},
        deepcopy(TEST_DATA),
        False
    ),   
    (
        np.reshape(generate_test_signal(num_iq_samples=64, scale=0.4).data,(2, -1)),
        {'norm': 2, 'flatten': True},
        np.reshape(deepcopy(TEST_DATA),(2, -1)),
        False
    )
])
def test_normalize(
    data: Any,
    params: dict,
    expected: np.ndarray | ValueError, 
    is_error: bool
    ) -> None:
    """Test the normalize functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (np.ndarray | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    norm = params['norm']
    flatten = params['flatten']

    if is_error:
        with pytest.raises(expected):  
            data = normalize(data, norm, flatten)
    else:
        data_test = deepcopy(data)    

        data = normalize(data, norm, flatten)

        assert np.allclose(data, expected, RTOL)
        assert type(data) == type(data_test) 
        assert data.dtype == torchsig_complex_data_type


@pytest.mark.parametrize("params, expected, is_error", [
    ({'N': 1024, 'ripple_db': 6.0, 'order': 5, 'cutoff': 0.2, 'numtaps': 128}, True, False),
    ({'N': 1111,'ripple_db': 3.2, 'order': 7, 'cutoff': 0.3, 'numtaps': 67}, True, False),
    ({'N': 101,'ripple_db': 0.1, 'order': 10, 'cutoff': 0.1, 'numtaps': 255}, True, False),
    ({'N': 100,'ripple_db': 4.2, 'order': 5, 'cutoff': 0.15, 'numtaps': 63}, True, False),
])
def test_passband_ripple(
    params: dict, 
    expected: bool, 
    is_error: bool
    ) -> None:
    """Test the passband_ripple functional with pytest.

    Args:
        params (dict): Function call parameters (see description).
        expected (bool | IndexError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    rng = np.random.default_rng(42)

    N = params['N']
    ripple_db = params['ripple_db']
    order = params['order']
    cutoff = params['cutoff']
    numtaps = params['numtaps']

    # create impulse response
    data = dsp.noise_generator(
        N       = N,
        power   = 1.0, 
        color   = 'white',
        continuous = False,
        rng     = rng 
    )

    # design filter
    b, a = sp.signal.cheby1(order, ripple_db, cutoff, fs=1.0, btype='low')
    t, h = sp.signal.dimpulse((b, a, 1/1.0), n=numtaps)
    fir_coeffs = h[0].squeeze()

    if is_error:
        with pytest.raises(expected): 
            data = passband_ripple(
                data = data,
                filter_coeffs = fir_coeffs,
                normalize = True
            )
    else:
        data_test = deepcopy(data)   
        data = passband_ripple(
            data = data,
            filter_coeffs = fir_coeffs,
            normalize = True
        )
        D = np.abs(np.fft.fft(data, norm='ortho'))
        mag = np.abs(D)
        M = len(D)
        
        peak_inds, _ = sp.signal.find_peaks(mag, height=0.1, distance=M/20)
        peak_vals = mag[peak_inds]
        trough_inds, _ = sp.signal.find_peaks(-mag, height=-10.0, distance=M/20)
        trough_vals = mag[trough_inds]
        ripple_est = np.mean(peak_vals[peak_vals > 0.5]) - np.mean(trough_vals[trough_vals > 0.5])
        ripple_est_db = 20*np.log10(1 + ripple_est)

        assert (np.abs(ripple_est_db - ripple_db) < 10**(0.5/20)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'patch_size': 3, 'patches_to_shuffle': [2, 7]}, 
        TypeError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'patch_size': 3, 'patches_to_shuffle': [2, 7]}, 
        True,
        False
    )
])
def test_patch_shuffle(
    data: Any,
    params: dict,
    expected: bool | TypeError, 
    is_error: bool
    ) -> None:
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
    patch_size = params['patch_size']
    patches_to_shuffle = params['patches_to_shuffle']

    if is_error:
        with pytest.raises(expected):  
            data = patch_shuffle(data, patch_size, patches_to_shuffle, rng)
    else:
        data_test = deepcopy(data)    
    
        data = patch_shuffle(data, patch_size, patches_to_shuffle, rng)

        patch_inds = np.where(data != data_test)[0]
        assert ((patch_inds[0] + patch_size - 1) in patch_inds) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        'invalid_input', 
        {'phase' : -np.pi / 4},
        TypeError,
        True
    ),
    (
        deepcopy(TEST_DATA), 
        {'phase' : -np.pi / 4},
        True,
        False
    )
])
def test_phase_offset(
    data: Any,
    params: dict,
    expected: bool | TypeError, 
    is_error: bool
    ) -> None:
    """Test the phase_offset functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    phase = params['phase']

    if is_error:
        with pytest.raises(expected):  
            data = phase_offset(data, phase = phase)
    else:
        data_test = deepcopy(data)

        data = phase_offset(data, phase = phase)
        
        data_restored = data * np.exp(-1j * phase)
        assert (np.allclose(data_restored, data_test, rtol=RTOL)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        2.0 * np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_bits': 8},
        2.0 * (np.ones((16,)) + 1j*np.ones((16,))),
        False
    ),
    (
        np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_bits': 8},
        2.0 * (np.ones((16,)) + 1j*np.ones((16,))),
        False
    )      
])
def test_quantize(
    data: Any,
    params: dict,
    expected: np.ndarray | TypeError | ValueError, 
    is_error: bool
    ) -> None:
    """Test the quantize functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (np.ndarray | TypeError | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    num_bits = params['num_bits']

    if is_error:
        with pytest.raises(expected):      
            data = quantize(
                data,
                num_bits  = num_bits,
            )
    else:
        data = quantize(
            data,
            num_bits  = num_bits,
        )

        assert type(data) == type(expected)
        assert data.dtype == torchsig_complex_data_type


@pytest.mark.parametrize("data, params, expected, is_error", [
    (deepcopy(TEST_DATA), {'mean_db': 4.0, 'sigma_db': 2.0}, True, False),
    (deepcopy(TEST_DATA), {'mean_db': 0.0, 'sigma_db': 0.42}, True, False)
])
def test_shadowing(
    data: Any, 
    params: dict, 
    expected: bool, 
    is_error: bool
    ) -> None:
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

    mean_db = params['mean_db']
    sigma_db = params['sigma_db']

    if is_error:
        with pytest.raises(expected): 
            data = shadowing(
                data = data,
                mean_db = mean_db,
                sigma_db = sigma_db,
                rng = rng
            )
    else:
        data_test = deepcopy(data)
        n_iterations = 30
        results = [
            10*np.log10(np.mean(np.abs(
                shadowing(
                    data = data,
                    mean_db = mean_db,
                    sigma_db = sigma_db,
                    rng = rng
                )**2)
            ))
            for _ in  range(n_iterations)
        ]
        results_array = np.array(results)
        
        # Shapiro-Wilk test for normality
        stat, p_value = sp.stats.shapiro(results_array)
        
        assert (p_value > 0.05) == expected
        assert (len(data) == len(data_test)) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, expected, is_error", [
    (0, AttributeError, True),
    (deepcopy(TEST_DATA), True, False)
])
def test_spectral_inversion(
    data: Any,
    expected: bool | AttributeError,
    is_error: bool
    ) -> None:
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
        data_test = deepcopy(data)
        test_real = data_test.real
        test_imag = data_test.imag

        data = spectral_inversion(data)

        assert np.allclose(data.real, test_real, RTOL) == expected
        assert np.allclose(data.imag, -test_imag, RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {'fft_size': 16, 'fft_stride': 4},
        TypeError,
        True
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'fft_size': 16, 'fft_stride': 4},
        True,
        False
    )
])
def test_spectrogram(
    data: Any,
    params: dict,
    expected: bool | TypeError, 
    is_error: bool
    ) -> None:
    """Test the spectrogram functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """  
    fft_size = params['fft_size']
    fft_stride = params['fft_stride']
    
    if is_error:
        with pytest.raises(expected):
            spec_data = spectrogram(data, fft_size, fft_stride)
    else:
        spec_test = compute_spectrogram(
            iq_samples = data, 
            fft_size = fft_size, 
            fft_stride = fft_stride
        )
        
        spec_data = spectrogram(
            data, 
            fft_size, 
            fft_stride,
        )

        assert np.allclose(spec_data, spec_test, RTOL) == expected        
        assert (type(spec_data) == type(spec_test)) == expected
        assert (spec_data.dtype == spec_test.dtype) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [8], 'drop_sizes': [2], 'fill': 'invalid_fill_type'},
        ValueError,
        True
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [8], 'drop_sizes': [2], 'fill': 'zero'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [2, 7], 'drop_sizes': [2, 1], 'fill': 'mean'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [3, 11], 'drop_sizes': [4, 3], 'fill': 'ffill'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [4], 'drop_sizes': [10], 'fill': 'bfill'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [1, 2, 5], 'drop_sizes': [1, 1, 1], 'fill': 'min'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [13], 'drop_sizes': [3], 'fill': 'max'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [2, 4], 'drop_sizes': [1, 7], 'fill': 'low'},
        True,
        False
    ),
    (
        generate_test_signal(num_iq_samples = 128, scale = 1.0).data,
        {'drop_starts': [1, 2, 3, 4], 'drop_sizes': [1, 1, 1, 1], 'fill': 'ones'},
        True,
        False
    )         
])
def test_spectrogram_drop_samples(
    data: Any,
    params: dict,
    expected: bool | ValueError, 
    is_error: bool
    ) -> None:
    """Test the spectrogram_drop_samples functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """  
    drop_starts = params['drop_starts']
    drop_sizes = params['drop_sizes']
    fill = params['fill']
    
    spec_data = compute_spectrogram(
        iq_samples = data, 
        fft_size = 16, 
        fft_stride = 4, 
    )
    spec_data = np.tile(spec_data, (16, 1, 1))
    spec_test = deepcopy(spec_data)

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


@pytest.mark.parametrize("data, expected, is_error", [
    (0, ValueError, True),
    (deepcopy(TEST_DATA), True, False)
])
def test_time_reversal(
    data: Any,
    expected: bool | ValueError,
    is_error: bool
    ) -> None:
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
        data_test = deepcopy(data)
        
        data = time_reversal(data)

        assert np.allclose(data, np.flip(data_test, axis=0), RTOL) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0,
        {
            'noise_power_low' : 3.0, 
            'noise_power_high': 3.0,
            'inflections' : 4,
            'random_regions' : False
        },
        AttributeError,
        True
    ),
    (
        np.zeros(1024, dtype=torchsig_complex_data_type),
        {
            'noise_power_low' : 3.0, 
            'noise_power_high': 3.0,
            'inflections' : 4,
            'random_regions' : False
        },
        True,
        False
    )      
])
def test_time_varying_noise(
    data: Any,
    params: dict,
    expected: bool | AttributeError,
    is_error: bool
    ) -> None:
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
    noise_power_low = params['noise_power_low']
    noise_power_high = params['noise_power_high']
    noise_power_high_linear = 10 ** (noise_power_high / 10.0)
    inflections = params['inflections']
    random_regions = params['random_regions']

    if is_error:
        with pytest.raises(expected):
            data = time_varying_noise(
                data, 
                noise_power_low  = noise_power_low, 
                noise_power_high = noise_power_high, 
                inflections      = inflections, 
                random_regions   = random_regions,
                rng              = rng 
            )
    else:
        data_test = deepcopy(data)

        data = time_varying_noise(
            data, 
            noise_power_low  = noise_power_low, 
            noise_power_high = noise_power_high, 
            inflections      = inflections, 
            random_regions   = random_regions,
            rng              = rng 
        )

        power_est = np.mean(np.abs(data)**2)
        assert (abs(power_est - noise_power_high_linear) < 1E-1) == expected
        assert (type(data) == type(data_test)) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("data, params, expected, is_error", [
    (
        0, 
        {  
            'initial_gain_db' : 0.,
            'alpha_smooth'    : 0.,
            'alpha_track'     : np.log(0.1),
            'alpha_overflow'  : np.log(0.1),
            'alpha_acquire'   : np.log(0.1),
            'ref_level'       : 0.,
            'ref_level_db'    : np.log(0.1),
            'track_range_db'  : np.log(0.1),
            'low_level_db'    : 0.,
            'high_level_db'   : 0.
        },
        TypeError,
        True
    ),
    (
        0.2 + generate_test_signal(num_iq_samples = 1024, scale = 0.01).data,
        {  
            'initial_gain_db' : 0.0,
            'alpha_smooth'    : 0.1,
            'alpha_track'     : np.log(1.1),
            'alpha_overflow'  : np.log(1.1),
            'alpha_acquire'   : np.log(1.1),
            'ref_level'       : 10.0,
            'ref_level_db'    : np.log(10.0),
            'track_range_db'  : np.log(4.0),
            'low_level_db'    : -200.0,
            'high_level_db'   : 200.0
        },
        True,
        False
    )
])
def test_tracking_agc(
    data: Any, 
    params: dict, 
    expected: bool | TypeError, 
    is_error: bool
    ) -> None:
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
            data = tracking_agc(
                data,
                initial_gain_db = params['initial_gain_db'],
                alpha_smooth    = params['alpha_smooth'],
                alpha_track     = params['alpha_track'],
                alpha_overflow  = params['alpha_overflow'],
                alpha_acquire   = params['alpha_acquire'],
                ref_level_db    = params['ref_level_db'],
                track_range_db  = params['track_range_db'],
                low_level_db    = params['low_level_db'],
                high_level_db   = params['high_level_db'],
            ) 
    else:    
        reference_level = params['ref_level']
        data_type = type(data)

        data = tracking_agc(
            data,
            initial_gain_db = params['initial_gain_db'],
            alpha_smooth    = params['alpha_smooth'],
            alpha_track     = params['alpha_track'],
            alpha_overflow  = params['alpha_overflow'],
            alpha_acquire   = params['alpha_acquire'],
            ref_level_db    = params['ref_level_db'],
            track_range_db  = params['track_range_db'],
            low_level_db    = params['low_level_db'],
            high_level_db   = params['high_level_db'],
        )    
        mean_level_est = np.round(np.mean(np.abs(data[-128:])))

        assert (abs(mean_level_est - reference_level) < 1E-1) == expected
        assert (type(data) == data_type) == expected
        assert (data.dtype == torchsig_complex_data_type) == expected
