"""Unit Tests: transforms/signal_transforms.py
"""
from torchsig.transforms.signal_transforms import (
    SignalTransform,
    # AdjacentChannelInterference,
    # AtmospherericDuct,
    # CochannelInterference,    
    CarrierPhaseOffsetSignalTransform,
    # Clock,   
    # Doppler,
    Fading,
    IntermodulationProducts,
    IQImbalanceSignalTransform,
    # LocalOscillatorPhaseNoiseSignalTransform,
    # LocalOscillatorFrequencyDriftSignalTransform,
    NonlinearAmplifierSignalTransform,
    # PassbandRippleSignalTransform,  
    # Shadowing,
    SpectralInversionSignalTransform
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import torchsig_complex_data_type
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
def test_NonlinearAmplifierSignalTransform(signal: Signal,
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

    