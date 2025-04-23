"""Unit Tests: transforms/signal_transforms.py
"""
from torchsig.transforms.signal_transforms import (
    SignalTransform,
    AdditiveNoiseSignalTransform,
    AdjacentChannelInterference,
    CochannelInterference,    
    CarrierPhaseOffsetSignalTransform,
    DopplerSignalTransform,
    Fading,
    IntermodulationProductsSignalTransform,
    IQImbalanceSignalTransform,
    LocalOscillatorFrequencyDriftSignalTransform,
    LocalOscillatorPhaseNoiseSignalTransform,
    NonlinearAmplifierSignalTransform,
    PassbandRippleSignalTransform,  
    Shadowing,
    SpectralInversionSignalTransform
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    low_pass
)
from test_transforms_utils import (
    generate_test_signal
)

# Third Party
from copy import deepcopy
import numpy as np
import pytest


RTOL = 1E-6
TEST_SIGNAL = generate_test_signal(num_iq_samples = 64, scale = 1.0)


@pytest.mark.parametrize("is_error", [False])
def test_SignalTransform(is_error: bool) -> None:
    """Test the parent SignalTransform with pytest.

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


@pytest.mark.parametrize("signal, params, is_error", [
    (deepcopy(TEST_SIGNAL),{'power_range': (0.01, 10.0),'color': 'white','continuous': True, 'measure': False}, False),
    (deepcopy(TEST_SIGNAL),{'power_range': (0.5, 2.0),'color': 'pink','continuous': False, 'measure': False}, False),
    (deepcopy(TEST_SIGNAL),{'power_range': (2.0, 2.0),'color': 'white','continuous': True, 'measure': True}, False)
])
def test_AdditiveNoiseSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test AdditiveNoiseSignalTransform SignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    power_range = params['power_range']
    color = params['color']
    continuous = params['continuous']
    measure = params['measure']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = AdditiveNoiseSignalTransform(
                    power_range = power_range,
                    color = color,
                    continuous = continuous,
                    measure = measure,
                    seed = 42
                )
                signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = AdditiveNoiseSignalTransform(
            power_range = power_range,
            color = color,
            continuous = continuous,
            measure = measure,
            seed = 42
        )
        signal = T(signal)
        
        if measure:
            orig_power = np.sum(np.abs(signal_test.data)**2)/len(signal_test.data)
            orig_snr_linear = 10 ** (signal_test.metadata.snr_db / 10)
            out_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
            add_noise_power = out_power - orig_power
            sig_power = orig_power / (1 + 1/orig_snr_linear)
            noise_power = sig_power / orig_snr_linear
            new_snr_db = 10*np.log10(sig_power / (noise_power + add_noise_power))
            assert np.abs(signal.metadata.snr_db - new_snr_db) < 10**(1.0/10)
        else:
            assert signal.metadata.snr_db == signal_test.metadata.snr_db

        assert isinstance(T, AdditiveNoiseSignalTransform)
        assert isinstance(T.power_distribution(), float)
        assert isinstance(T.color, str)
        assert isinstance(T.continuous, bool)
        assert isinstance(T.measure, bool)
        assert isinstance(signal, Signal)
        assert len(signal.data) == len(signal_test.data)
        assert (signal.data.dtype == torchsig_complex_data_type)
    

@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'sample_rate': 1.0, 
            'power_range': (0.5, 2.0),
            'center_frequency_range': (0.25, 0.25),
            'phase_sigma_range': (0.0, 0.0),
            'time_sigma_range': (0.0, 0.0),
            'filter_weights': low_pass(0.125, 0.125, 1.0)
        },
        True,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'sample_rate': 2.4, 
            'power_range': (0.01, 100.0),
            'center_frequency_range': (-0.5, -0.07),
            'phase_sigma_range': (0.0, 1.0),
            'time_sigma_range': (0.0, 10.0),
            'filter_weights': low_pass(0.04, 0.16, 2.4)
        },
        True,
        False
    ),
])
def test_AdjacentChannelInterference(
    signal: Signal,
    params: dict, 
    expected: bool | ValueError, 
    is_error: bool
) -> None:
    """Test AdjacentChannelInterference SignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    sample_rate = params['sample_rate']
    power_range = params['power_range']
    center_frequency_range = params['center_frequency_range']
    phase_sigma_range = params['phase_sigma_range']
    time_sigma_range = params['time_sigma_range']
    filter_weights = params['filter_weights']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = AdjacentChannelInterference(
                    sample_rate = sample_rate,
                    power_range = power_range,
                    center_frequency_range = center_frequency_range,
                    phase_sigma_range = phase_sigma_range,
                    time_sigma_range = time_sigma_range,
                    filter_weights = filter_weights,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = AdjacentChannelInterference(
            sample_rate = sample_rate,
            power_range = power_range,
            center_frequency_range = center_frequency_range,
            phase_sigma_range = phase_sigma_range,
            time_sigma_range = time_sigma_range,
            filter_weights = filter_weights,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, AdjacentChannelInterference) == expected
        assert isinstance(T.sample_rate, float) == expected
        assert isinstance(T.power_distribution(), float) == expected
        assert isinstance(T.center_frequency_distribution(), float) == expected
        assert isinstance(T.phase_sigma_distribution(), float) == expected
        assert isinstance(T.time_sigma_distribution(), float) == expected
        assert isinstance(T.filter_weights, np.ndarray) == expected
        assert isinstance(signal, Signal) == expected
        assert (signal.data.dtype == torchsig_complex_data_type) == expected
        # no metadata impacts


@pytest.mark.parametrize("signal, is_error", [
    ( generate_test_signal(num_iq_samples = 64, scale = 1.0), False ),
    ( generate_test_signal(num_iq_samples = 256, scale = 1.0), False )
])
def test_CarrierPhaseOffsetSignalTransform(
    signal: Signal, 
    is_error: bool
) -> None:
    """Test CarrierPhaseOffsetSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """       
    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = CarrierPhaseOffsetSignalTransform(
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = CarrierPhaseOffsetSignalTransform(
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, CarrierPhaseOffsetSignalTransform)
        assert isinstance(T.phase_offset_distribution(), float)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'power_range': (0.5, 2.0),
            'filter_weights': low_pass(0.125, 0.125, 1.0),
            'color': 'white',
            'continuous': True,
            'measure': False,
        },
        True,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'power_range': (0.01, 100.0),
            'filter_weights': low_pass(0.04, 0.16, 2.4),
            'color': 'pink',
            'continuous': False,
            'measure': False,
        },
        True,
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'power_range': (0.1, 0.1),
            'filter_weights': low_pass(0.125, 0.125, 1.0),
            'color': 'white',
            'continuous': True,
            'measure': True,
        },
        True,
        False
    ),    
])
def test_CochannelInterference(
    signal: Signal,
    params: dict, 
    expected: bool, 
    is_error: bool
) -> None:
    """Test CochannelInterference SignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        expected (bool): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    power_range = params['power_range']
    filter_weights = params['filter_weights']
    color = params['color']
    continuous = params['continuous']
    measure = params['measure']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CochannelInterference(
                power_range = power_range,
                filter_weights = filter_weights,
                color = color,
                continuous = continuous,
                measure = measure,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        
        T = CochannelInterference(
            power_range = power_range,
            filter_weights = filter_weights,
            color = color,
            continuous = continuous,
            measure = measure,
            seed = 42
        )
        signal = T(signal)

        if measure:
            orig_power = np.sum(np.abs(signal_test.data)**2)/len(signal_test.data)
            orig_snr_linear = 10 ** (signal_test.metadata.snr_db / 10)
            out_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
            add_noise_power = out_power - orig_power
            sig_power = orig_power / (1 + 1/orig_snr_linear)
            noise_power = sig_power / orig_snr_linear
            new_snr_db = 10*np.log10(sig_power / (noise_power + add_noise_power))
            assert np.abs(signal.metadata.snr_db - new_snr_db) < 10**(1.0/10)
        else:
            assert signal.metadata.snr_db == signal_test.metadata.snr_db

        assert isinstance(T, CochannelInterference) == expected
        assert isinstance(T.power_distribution(), float) == expected
        assert isinstance(T.filter_weights, np.ndarray) == expected
        assert isinstance(T.color, str) == expected
        assert isinstance(T.continuous, bool) == expected
        assert isinstance(T.measure, bool) == expected
        assert isinstance(signal, Signal) == expected
        assert (signal.data.dtype == torchsig_complex_data_type) == expected


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'velocity_range': (0.0, 10.0), 
            'propagation_speed': 2.9979e8,
            'sampling_rate': 1.0
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'velocity_range': (-12.0, 12.0), 
            'propagation_speed': 343.0,
            'sampling_rate': 10e3
        },
        False
    ),    
])
def test_DopplerSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test DopplerSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    velocity_range = params['velocity_range']
    propagation_speed = params['propagation_speed']
    sampling_rate = params['sampling_rate']

    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = DopplerSignalTransform(
                velocity_range = velocity_range,
                propagation_speed = propagation_speed,
                sampling_rate = sampling_rate,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = DopplerSignalTransform(
            velocity_range = velocity_range,
            propagation_speed = propagation_speed,
            sampling_rate = sampling_rate,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, DopplerSignalTransform)
        assert isinstance(T.velocity_distribution(), float)
        assert isinstance(T.propagation_speed, float)
        assert isinstance(T.sampling_rate, float)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'coherence_bandwidth': (0.01, 0.1), 
            'power_delay_profile': [0.5, 0.25, 0.125]
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'coherence_bandwidth': (0.05, 0.2), 
            'power_delay_profile': (0.1, 0.4)
        },
        False
    )
])
def test_Fading(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test Fading SignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """    
    coherence_bandwidth = params['coherence_bandwidth']
    power_delay_profile = params['power_delay_profile']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = Fading(
                coherence_bandwidth = coherence_bandwidth,
                power_delay_profile = power_delay_profile,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = Fading(
            coherence_bandwidth = coherence_bandwidth,
            power_delay_profile = power_delay_profile,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, Fading)
        assert isinstance(T.coherence_bandwidth_distribution(), float)
        assert np.allclose(T.power_delay_profile, power_delay_profile)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (deepcopy(TEST_SIGNAL), 
        {
            'model_order': [3,5], 
            'coeffs_range': (1e-3, 1e-1),
        },
        False
    ) 
])
def test_IntermodulationProductsSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test IntermodulationProductsSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    model_order = params['model_order']
    coeffs_range = params['coeffs_range']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = IntermodulationProductsSignalTransform(
                    model_order = model_order,
                    coeffs_range = coeffs_range,
                    seed = 42
                )
                signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = IntermodulationProductsSignalTransform(
            model_order = model_order,
            coeffs_range = coeffs_range,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, IntermodulationProductsSignalTransform)
        assert isinstance(T.model_order_distribution(), np.int64)
        assert isinstance(T.coeffs_distribution(), float)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'amplitude_imbalance': (0.0, 6.0), 
            'phase_imbalance': (-np.pi, np.pi),
            'dc_offset': ((-0.2, 0.2),(-0.2, 0.2))
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'amplitude_imbalance': [0.2, 2.2], 
            'phase_imbalance': [-np.pi/8, np.pi/8],
            'dc_offset': ([-0.03, 0.03],[-0.03, 0.03])
        },
        False
    )    
])
def test_IQImbalanceSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test IQImbalanceSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    amplitude_imbalance = params['amplitude_imbalance']
    phase_imbalance = params['phase_imbalance']
    dc_offset = params['dc_offset']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = IQImbalanceSignalTransform(
                    amplitude_imbalance = amplitude_imbalance,
                    phase_imbalance = phase_imbalance,
                    dc_offset = dc_offset,
                    seed = 42
                )
                signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = IQImbalanceSignalTransform(
            amplitude_imbalance = amplitude_imbalance,
            phase_imbalance = phase_imbalance,
            dc_offset = dc_offset,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, IQImbalanceSignalTransform)
        assert isinstance(T.amplitude_imbalance_distribution(), float)
        assert isinstance(T.phase_imbalance_distribution(), float)
        assert isinstance(T.dc_offset_distribution(), np.ndarray)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'drift_std_range': (10,  100)
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'drift_std_range': (10, 100), 
        },
        False
    ),    
])
def test_LocalOscillatorFrequencyDriftSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test LocalOscillatorFrequencyDriftSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    drift_std_range = params['drift_std_range']

    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = LocalOscillatorFrequencyDriftSignalTransform(
                drift_std_range = drift_std_range, 
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = LocalOscillatorFrequencyDriftSignalTransform(
            drift_std_range = drift_std_range, 
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, LocalOscillatorFrequencyDriftSignalTransform)
        assert isinstance(T.drift_std_distribution(), float)
        assert isinstance(signal, Signal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'phase_noise_degrees': (0.25, 1), 
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'phase_noise_degrees': (0.25, 1), 
        },
        False
    ),    
])
def test_LocalOscillatorPhaseNoiseSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test LocalOscillatorPhaseNoiseSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    phase_noise_degrees = params['phase_noise_degrees']

    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = LocalOscillatorPhaseNoiseSignalTransform(
                phase_noise_degrees = phase_noise_degrees,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = LocalOscillatorPhaseNoiseSignalTransform(
            phase_noise_degrees = phase_noise_degrees,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, LocalOscillatorPhaseNoiseSignalTransform)
        assert isinstance(T.phase_noise_degrees, tuple)
        assert isinstance(T.phase_noise_degrees_distribution(), float)
        assert isinstance(signal, Signal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'gain_range': (1.0, 4.0),
            'psat_backoff_range': (5.0, 20.0),
            'phi_range': (0.0, 0.0),
            'auto_scale': True
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'gain_range': (0.5, 17.2),
            'psat_backoff_range': (0.0, 7.0),
            'phi_range': (-np.deg2rad(10.0), np.deg2rad(17.0)),
            'auto_scale': True
        },
        False
    )  
])
def test_NonlinearAmplifierSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test NonlinearAmplifierSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    gain_range = params['gain_range']
    psat_backoff_range = params['psat_backoff_range']
    phi_range = params['phi_range']
    auto_scale = params['auto_scale']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = NonlinearAmplifierSignalTransform(
                    gain_range  = gain_range,
                    psat_backoff_range = psat_backoff_range,
                    phi_range = phi_range,
                    auto_scale = auto_scale,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = NonlinearAmplifierSignalTransform(
            gain_range  = gain_range,
            psat_backoff_range = psat_backoff_range,
            phi_range = phi_range,
            auto_scale = auto_scale,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, NonlinearAmplifierSignalTransform)
        assert isinstance(T.gain_distribution(), float)
        assert isinstance(T.psat_backoff_distribution(), float)
        assert isinstance(T.phi_distribution(), float)
        assert isinstance(signal, Signal)
        assert (signal.data.dtype == torchsig_complex_data_type)
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (deepcopy(TEST_SIGNAL), {'mean_db_range': (0.0, 4.0), 'sigma_db_range': (2.0, 6.0)},False),
    (deepcopy(TEST_SIGNAL), {'mean_db_range': (0.0, 0.0), 'sigma_db_range': (3.0, 9.0)},False),    
])
def test_Shadowing(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test Shadowing SignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    mean_db_range = params['mean_db_range']
    sigma_db_range = params['sigma_db_range']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Shadowing(
                mean_db_range = mean_db_range,
                sigma_db_range = sigma_db_range
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = Shadowing(
            mean_db_range = mean_db_range,
            sigma_db_range = sigma_db_range
        )
        signal = T(signal)

        assert isinstance(T, Shadowing)
        assert isinstance(T.mean_db_distribution(), float)
        assert isinstance(T.sigma_db_distribution(), float)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type



@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'passband_ripple_db': 1.0, 
            'cutoff': 0.25, 
            'order': 5, 
            'numtaps': 63
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'passband_ripple_db': 4.2, 
            'cutoff': 0.12, 
            'order': 10, 
            'numtaps': 127
        },
        False
    ),    
])
def test_PassbandRippleSignalTransform(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test PassbandRippleSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """ 
    passband_ripple_db = params['passband_ripple_db']
    cutoff = params['cutoff']
    order = params['order']
    numtaps = params['numtaps']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = PassbandRippleSignalTransform(
                passband_ripple_db = passband_ripple_db,
                cutoff = cutoff,
                order = order,
                numtaps = numtaps
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = PassbandRippleSignalTransform(
            passband_ripple_db = passband_ripple_db,
            cutoff = cutoff,
            order = order,
            numtaps = numtaps
        )
        signal = T(signal)

        assert isinstance(T, PassbandRippleSignalTransform)
        assert isinstance(T.passband_ripple_db, float)
        assert isinstance(T.cutoff, float)
        assert isinstance(T.order, int)
        assert isinstance(T.numtaps, int)
        assert isinstance(T.fir_coeffs, np.ndarray)
        assert len(T.fir_coeffs) ==  numtaps
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        

@pytest.mark.parametrize("signal, is_error", [
    (generate_test_signal(num_iq_samples = 64, scale = 1.0), False),
    (generate_test_signal(num_iq_samples = 256, scale = 1.0), False)
])
def test_SpectralInversionSignalTransform(
    signal: Signal,
    is_error: bool
) -> None:
    """Test SpectralInversionSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """     
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = SpectralInversionSignalTransform()
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = SpectralInversionSignalTransform()
        signal = T(signal)

        assert isinstance(T, SpectralInversionSignalTransform)
        assert isinstance(signal, Signal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        
        assert signal.metadata.center_freq == -1 * signal_test.metadata.center_freq

    
