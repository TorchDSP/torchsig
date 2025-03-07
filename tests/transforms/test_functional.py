"""Unit Tests: transforms/functional
"""

from torchsig.transforms.functional import (
    add_slope,
    agc,
    awgn,
    block_agc,
    channel_swap,
    cut_out,
    drop_samples,
    fading,
    intermodulation_products,
    iq_imbalance,
    mag_rescale,
    nonlinear_amplifier,
    normalize,
    patch_shuffle,
    phase_offset,
    quantize,
    spectral_inversion,
    spectrogram,
    spectrogram_drop_samples,
    time_reversal,
    time_varying_noise
)
from test_transforms_utils import (
    generate_test_signal
)
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    compute_spectrogram,
)

# Third Party
from typing import Any
import numpy as np
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
def test_agc(
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
            data = agc(
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
        data_dtype = data.dtype

        data = agc(
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
        mean_power_est = np.round(np.mean(np.abs(data[-128:])))

        assert (abs(mean_power_est - reference_level) < 1E-1) == expected
        assert (type(data) == data_type) == expected
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
    (
        0,
        {'gain_change_db': 1.0, 'start_idx': 13},
        TypeError, 
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'gain_change_db': 3.0, 'start_idx': 17}, 
        True,
        False
    )
])
def test_block_agc(
    data: Any, 
    params: dict, 
    expected: bool | TypeError, 
    is_error: bool
    ) -> None:
    """Test the block_agc functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | TypeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """    
    gain_change_db = params['gain_change_db']
    start_idx = params['start_idx']

    if is_error:
        with pytest.raises(expected): 
            data = block_agc(
                data, 
                gain_change_db = gain_change_db,
                start_idx   = start_idx
            )
    else:
        gain_change_linear = 10**(gain_change_db/10)
        data_test = deepcopy(data)
        data = block_agc(
            data, 
            gain_change_db = gain_change_db,
            start_idx   = start_idx
        )

        assert np.allclose(data[start_idx:], gain_change_linear * data_test[start_idx:], RTOL) == expected
        assert np.allclose(data[:start_idx], data_test[:start_idx], RTOL) == expected
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
    (deepcopy(TEST_DATA), {'coeffs': np.array([0.5])}, True, False),
    (deepcopy(TEST_DATA), {'coeffs': np.array([0.2, 1.0, 0.1])}, True, False)
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

        if len(coeffs) < 3:
            assert np.allclose(data, coeffs[0]*data_test, RTOL) == expected
        else: # assume first-order and third-order products dominate
            distorted_data = coeffs[0]*data_test + coeffs[2]*((np.abs(data_test) ** (2)) * data_test)
            assert np.allclose(data, distorted_data, RTOL) == expected
        
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
        0,
        {'start': 0.1, 'scale': -0.5}, 
        AttributeError,
        True
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 0.42, 'scale': 1.7}, 
        True,
        False
    ),
    (
        deepcopy(TEST_DATA),
        {'start': 5, 'scale': -2.2}, 
        True,
        False
    )    
])
def test_mag_rescale(
    data: Any,
    params: dict,
    expected: bool | AttributeError, 
    is_error: bool
    ) -> None:
    """Test the mag_rescale functional with pytest.

    Args:
        data (Any): Data input, nominally np.ndarray.
        params (dict): Function call parameters (see description).
        expected (bool | AttributeError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    start = params['start']
    scale = params['scale']
    
    if is_error:
        with pytest.raises(expected):   
            data = mag_rescale(data, start, scale)
    else:
        data_test = deepcopy(data)

        data = mag_rescale(data, start, scale)

        start_ind = int(data.shape[0] * start)
        assert (np.allclose(data[start_ind:], scale * data_test[start_ind:], RTOL)) == expected
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
def test_nonlinear_amplifier(
    data: Any, 
    params: dict, 
    expected: bool | ValueError, 
    is_error: bool
    ) -> None:
    """Test the nonlinear_amplifier functional with pytest.

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
            data = nonlinear_amplifier(
                data = data,
                Pin  = Pin,
                Pout = Pout,
                Phi  = Phi
            )
    else:
        data_test = deepcopy(data)

        data = nonlinear_amplifier(
            data = data,
            Pin  = Pin,
            Pout = Pout,
            Phi  = Phi
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
        0, 
        {'num_levels': 4, 'round_type': 'nearest'},
        TypeError,
        True
    ),
    (
        np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_levels': 4, 'round_type': 'invalid_round_type'},
        ValueError,
        True
    ),    
    (
        np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_levels': 4, 'round_type': 'nearest'},
        1.5 * (np.ones((16,)) + 1j*np.ones((16,))),
        False
    ),
    (
        2.0 * np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_levels': 4, 'round_type': 'floor'},
        2.0 * (np.ones((16,)) + 1j*np.ones((16,))),
        False
    ),
    (
        np.sqrt(2) * (np.ones((16,)) + 1j*np.ones((16,))), 
        {'num_levels': 4, 'round_type': 'ceiling'},
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
    num_levels = params['num_levels']
    round_type = params['round_type']

    if is_error:
        with pytest.raises(expected):      
            data = quantize(
                data,
                num_levels  = num_levels,
                round_type  = round_type
            )
    else:
        data = quantize(
            data,
            num_levels  = num_levels,
            round_type  = round_type
        )

        assert np.allclose(data, expected, rtol=RTOL)
        assert type(data) == type(expected)
        assert data.dtype == torchsig_complex_data_type


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

