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
    IntermodulationProducts,
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
    (deepcopy(TEST_SIGNAL),{'power_range': (0.01, 10.0),'color': 'white','continuous': True}, False),
    (deepcopy(TEST_SIGNAL),{'power_range': (0.5, 2.0),'color': 'pink','continuous': False}, False)
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
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = AdditiveNoiseSignalTransform(
                    power_range = power_range,
                    color = color,
                    continuous = continuous,
                    seed = 42
                )
                signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = AdditiveNoiseSignalTransform(
            power_range = power_range,
            color = color,
            continuous = continuous,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, AdditiveNoiseSignalTransform)
        assert isinstance(T.power_distribution(), float)
        assert isinstance(T.color, str)
        assert isinstance(T.continuous, bool)
        assert isinstance(signal, Signal)
        assert len(signal.data) == len(signal_test.data)
        assert (signal.data.dtype == torchsig_complex_data_type)
        # no metadata impacts


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
            'continuous': True
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
            'continuous': False            
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
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = CochannelInterference(
                    power_range = power_range,
                    filter_weights = filter_weights,
                    color = color,
                    continuous = continuous,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = CochannelInterference(
            power_range = power_range,
            filter_weights = filter_weights,
            color = color,
            continuous = continuous,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, CochannelInterference) == expected
        assert isinstance(T.power_distribution(), float) == expected
        assert isinstance(T.filter_weights, np.ndarray) == expected
        assert isinstance(T.color, str) == expected
        assert isinstance(T.continuous, bool) == expected
        assert isinstance(signal, Signal) == expected
        assert (signal.data.dtype == torchsig_complex_data_type) == expected
        # no metadata impacts


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
        # no metadata impacts


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
            'model_order_range': (0, 5), 
            'coeffs_range': (0., 1.),
        },
        False
    ) 
])
def test_IntermodulationProducts(
    signal: Signal,
    params: dict, 
    is_error: bool
) -> None:
    """Test IntermodulationProducts with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    model_order_range = params['model_order_range']
    coeffs_range = params['coeffs_range']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = IntermodulationProducts(
                    model_order_range = model_order_range,
                    coeffs_range = coeffs_range,
                    seed = 42
                )
                signal = T(signal)
    else:
        signal_test = deepcopy(signal)

        T = IntermodulationProducts(
            model_order_range = model_order_range,
            coeffs_range = coeffs_range,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, IntermodulationProducts)
        assert isinstance(T.model_order_distribution(), float)
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
            'phase_noise_std_range': (0.0001, 0.001), 
        },
        False
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'phase_noise_std_range': (0.0001, 0.001), 
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
    phase_noise_std_range = params['phase_noise_std_range']

    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = LocalOscillatorPhaseNoiseSignalTransform(
                phase_noise_std_range = phase_noise_std_range,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = LocalOscillatorPhaseNoiseSignalTransform(
            phase_noise_std_range = phase_noise_std_range,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, LocalOscillatorPhaseNoiseSignalTransform)
        assert isinstance(T.phase_noise_std_range, tuple)
        assert isinstance(T.phase_noise_std_distribution(), float)
        assert isinstance(signal, Signal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, expected, is_error", [
    (
        deepcopy(TEST_SIGNAL), 
        {
            'Pin': np.zeros((1,)),
            'Pout': np.zeros((2,)),
            'Phi': np.zeros((3,))
        },
        ValueError,
        True
    ),
    (
        deepcopy(TEST_SIGNAL), 
        {
            'Pin': 10**((np.array([-100., -20., -10.,  0.,  5., 10. ]) / 10)),
            'Pout': 10**((np.array([ -90., -10.,   0.,  9., 9.9, 10. ]) / 10)),
            'Phi': np.deg2rad(np.array([0., -2.,  -4.,  7., 12., 23.]))
        },
        True,
        False
    )    
])
def test_NonlinearAmplifierSignalTransform(
    signal: Signal,
    params: dict, 
    expected: bool | ValueError, 
    is_error: bool
) -> None:
    """Test NonlinearAmplifierSignalTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        expected (bool | ValueError): Expected test result.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    Pin = params['Pin']
    Pout = params['Pout']
    Phi = params['Phi']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = NonlinearAmplifierSignalTransform(
                    Pin  = Pin,
                    Pout = Pout,
                    Phi  = Phi,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = NonlinearAmplifierSignalTransform(
            Pin  = Pin,
            Pout = Pout,
            Phi  = Phi,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, NonlinearAmplifierSignalTransform) == expected
        assert isinstance(T.Pin, np.ndarray) == expected
        assert isinstance(T.Pout, np.ndarray) == expected
        assert isinstance(T.Phi, np.ndarray) == expected
        assert isinstance(signal, Signal) == expected
        assert (signal.data.dtype == torchsig_complex_data_type) == expected
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

    
