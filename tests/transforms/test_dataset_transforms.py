"""Unit Tests for transforms/dataset_transforms.py
"""
from torchsig.transforms.dataset_transforms import (
    DatasetTransform,
    AdditiveNoiseDatasetTransform,
    AGC,
    AWGN,
    BlockAGC,
    CarrierPhaseOffsetDatasetTransform,
    IQImbalanceDatasetTransform,
    LocalOscillatorFrequencyDriftDatasetTransform,
    LocalOscillatorPhaseNoiseDatasetTransform,
    NonlinearAmplifierDatasetTransform,
    PassbandRippleDatasetTransform,    
    Quantize,
    SpectralInversionDatasetTransform,
    Spectrogram,
    TimeVaryingNoise,

    # ML Transforms
    AddSlope,
    ChannelSwap,
    CutOut,
    PatchShuffle,
    RandomDropSamples,
    RandomMagRescale,
    SpectrogramDropSamples,
    TimeReversal
)
from torchsig.signals.signal_types import DatasetSignal
from test_transforms_utils import (
    generate_test_dataset_signal
)
from torchsig.utils.dsp import (
    compute_spectrogram,
    torchsig_complex_data_type
)

# Third Party
from copy import deepcopy
import numpy as np
import pytest

TEST_DS_SIGNAL = generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0)


@pytest.mark.parametrize("is_error", [False])
def test_DatasetTransform(is_error: bool) -> None:
    """Test DatasetTransform parent class with pytest.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = DatasetTransform()
    else:
        T = DatasetTransform()
    
        assert isinstance(T, DatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)


@pytest.mark.parametrize("signal, params, is_error", [
    (deepcopy(TEST_DS_SIGNAL),{'power_range': (0.01, 10.0), 'color': 'white', 'continuous': True},False),
    (deepcopy(TEST_DS_SIGNAL),{'power_range': (2.0, 4.0), 'color': 'red', 'continuous': False},False),
])
def test_AdditiveNoiseDatasetTransform(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the AdditiveNoiseDatasetTransform with pytest.

    Args:
        signal (is_error: bool) -> None:: input dataset.
        params (dict): Test parameters
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    power_range = params['power_range']
    color = params['color']
    continuous = params['continuous']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AdditiveNoiseDatasetTransform(
                power_range = power_range,
                color = color,
                continuous = continuous,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)         
        T = AdditiveNoiseDatasetTransform(
            power_range = power_range,
            color = color,
            continuous = continuous,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, AdditiveNoiseDatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)   
        assert isinstance(T.power_distribution(), float) 
        assert isinstance(T.color, str) 
        assert isinstance(T.continuous, bool) 
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert len(signal.data) == len(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 3.0}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 0.1}, False ),
])
def test_AWGN(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the AWGN DatasetTransform with pytest.

    Args:
        signal (is_error: bool) -> None:: input dataset.
        params (dict): AWGN parameters (see functional AWGN description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AWGN(
                noise_power_db = params['noise_power_db'],
                seed = 42
            )
            signal = T(signal)
    else:        
        T = AWGN(
            noise_power_db = params['noise_power_db'],
            seed = 42
        )
        signal_test = deepcopy(signal) 
        signal = T(signal)

        assert isinstance(T, AWGN)
        assert isinstance(T.random_generator, np.random.Generator)   
        assert isinstance(T.noise_power_db, float) 
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL),
        { 
            'rand_scale'      : (1.0, 10.0),
            'initial_gain_db' : 0.0,
            'alpha_smooth'    : 0.00004,
            'alpha_track'     : 0.0004,
            'alpha_overflow'  : 0.3,
            'alpha_acquire'   : 0.04,
            'ref_level'       : 1.0,
            'ref_level_db'    : np.log(1.0),
            'track_range_db'  : 1.0,
            'low_level_db'    : -80.0,
            'high_level_db'   : 6.0
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL),
        { 
            'rand_scale'      : (1.0, 8.0),
            'initial_gain_db' : 2.0,
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
        False
    )
])
def test_AGC(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the AGC DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): AGC parameters (see AGC DatasetTransform description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AGC(
                rand_scale      = params['rand_scale'],
                initial_gain_db = params['initial_gain_db'],
                alpha_smooth    = params['alpha_smooth'],
                alpha_track     = params['alpha_track'],
                alpha_overflow  = params['alpha_overflow'],
                alpha_acquire   = params['alpha_acquire'],
                ref_level_db    = params['ref_level_db'],
                track_range_db  = params['track_range_db'],
                low_level_db    = params['low_level_db'],
                high_level_db   = params['high_level_db'],
                seed = 42
            )
            signal = T(signal)
    else:
        T = AGC(
            rand_scale      = params['rand_scale'],
            initial_gain_db = params['initial_gain_db'],
            alpha_smooth    = params['alpha_smooth'],
            alpha_track     = params['alpha_track'],
            alpha_overflow  = params['alpha_overflow'],
            alpha_acquire   = params['alpha_acquire'],
            ref_level_db    = params['ref_level_db'],
            track_range_db  = params['track_range_db'],
            low_level_db    = params['low_level_db'],
            high_level_db   = params['high_level_db'],
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, AGC)
        assert isinstance(T.initial_gain_db, float)
        assert isinstance(T.alpha_smooth, float)
        assert isinstance(T.alpha_track, float)
        assert isinstance(T.alpha_overflow, float)
        assert isinstance(T.alpha_acquire, float)
        assert isinstance(T.ref_level_db, float)
        assert isinstance(T.track_range_db, float)
        assert isinstance(T.low_level_db, float)
        assert isinstance(T.high_level_db, float)   
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 3.0}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 0.1}, False ),
])
def test_AWGN(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the AWGN DatasetTransform with pytest.

    Args:
        signal (is_error: bool) -> None:: input dataset.
        params (dict): AWGN parameters (see functional AWGN description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AWGN(
                noise_power_db = params['noise_power_db'],
                seed = 42
            )
            signal = T(signal)
    else:        
        T = AWGN(
            noise_power_db = params['noise_power_db'],
            seed = 42
        )
        signal_test = deepcopy(signal) 
        signal = T(signal)

        assert isinstance(T, AWGN)
        assert isinstance(T.random_generator, np.random.Generator)   
        assert isinstance(T.noise_power_db, float) 
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    ( deepcopy(TEST_DS_SIGNAL), {'max_gain_change_db': 3.0}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'max_gain_change_db': 10.0}, False ),
])
def test_BlockAGC(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the BlockAGC DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): BlockAGC parameters (see functional BlockAGC description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):    
            T = BlockAGC(
                max_gain_change_db = params['max_gain_change_db'],
            )
            signal = T(signal)
    else:
        T = BlockAGC(
            max_gain_change_db = params['max_gain_change_db'],
        )
        signal_test = deepcopy(signal) 
        signal = T(signal)

        assert isinstance(T, BlockAGC)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.gain_change_db_distribution(), float) 
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, is_error", [
    ( generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0), False ),
    ( generate_test_dataset_signal(num_iq_samples = 128, scale = 10.0), False )
])
def test_CarrierPhaseOffsetDatasetTransform(signal: DatasetSignal, is_error: bool) -> None:
    """Test the CarrierPhaseOffset DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CarrierPhaseOffsetDatasetTransform(
                seed = 42
            )
            signal = T(signal)
    else:
        T = CarrierPhaseOffsetDatasetTransform(
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, CarrierPhaseOffsetDatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.phase_offset_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'amplitude_imbalance': (0.0, 6.0), 
            'phase_imbalance': (-np.pi, np.pi),
            'dc_offset' : ((-0.2, 0.2),(-0.2, 0.2))
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'amplitude_imbalance': (0.0, 1.0), 
            'phase_imbalance': (-np.pi/4, np.pi/4),
            'dc_offset' : ((-0.01, 0.1),(-0.01, 0.1))
        },
        False
    )
])
def test_IQImbalanceDatasetTransform(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the IQImbalance DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): IQImbalance parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    amplitude_imbalance = params['amplitude_imbalance']
    phase_imbalance = params['phase_imbalance']
    dc_offset = params['dc_offset']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = IQImbalanceDatasetTransform(
                amplitude_imbalance = amplitude_imbalance,
                phase_imbalance = phase_imbalance,
                dc_offset = dc_offset,
                seed = 42
            )
            signal = T(signal)
    else:
        T = IQImbalanceDatasetTransform(
            amplitude_imbalance = amplitude_imbalance,
            phase_imbalance = phase_imbalance,
            dc_offset = dc_offset,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, IQImbalanceDatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.amplitude_imbalance_distribution(), float)
        assert isinstance(T.phase_imbalance_distribution(), float)
        assert isinstance(T.dc_offset_distribution(), np.ndarray)
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'drift_std_range': (10, 100), 
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'drift_std_range': (10, 100), 
        },
        False
    ),    
])
def test_LocalOscillatorFrequencyDriftDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test LocalOscillatorFrequencyDriftDatasetTransform with pytest.

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
            T = LocalOscillatorFrequencyDriftDatasetTransform(
                drift_std_range = drift_std_range,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = LocalOscillatorFrequencyDriftDatasetTransform(
            drift_std_range = drift_std_range, 
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, LocalOscillatorFrequencyDriftDatasetTransform)
        assert isinstance(T.drift_std_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'phase_noise_std_range': (10, 100)
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'phase_noise_std_range': (10, 100)
        },
        False
    ),    
])
def test_LocalOscillatorPhaseNoiseDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test LocalOscillatorPhaseNoiseDatasetTransform with pytest.

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
            T = LocalOscillatorPhaseNoiseDatasetTransform(
                phase_noise_std_range = phase_noise_std_range,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = LocalOscillatorPhaseNoiseDatasetTransform(
            phase_noise_std_range = phase_noise_std_range,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, LocalOscillatorPhaseNoiseDatasetTransform)
        assert isinstance(T.phase_noise_std_range, tuple)
        assert isinstance(T.phase_noise_std_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'Pin': np.zeros((1,)),
            'Pout': np.zeros((2,)),
            'Phi': np.zeros((3,))
        },
        True
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'Pin': 10**((np.array([-100., -20., -10.,  0.,  5., 10. ]) / 10)),
            'Pout': 10**((np.array([ -90., -10.,   0.,  9., 9.9, 10. ]) / 10)),
            'Phi': np.deg2rad(np.array([0., -2.,  -4.,  7., 12., 23.]))
        },
        False
    )    
])
def test_NonlinearAmplifierDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test NonlinearAmplifierDatasetTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    Pin = params['Pin']
    Pout = params['Pout']
    Phi = params['Phi']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = NonlinearAmplifierDatasetTransform(
                    Pin  = Pin,
                    Pout = Pout,
                    Phi  = Phi,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = NonlinearAmplifierDatasetTransform(
            Pin  = Pin,
            Pout = Pout,
            Phi  = Phi,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, NonlinearAmplifierDatasetTransform)
        assert isinstance(T.Pin, np.ndarray)
        assert isinstance(T.Pout, np.ndarray)
        assert isinstance(T.Phi, np.ndarray)
        assert isinstance(signal, DatasetSignal)
        assert (signal.data.dtype == torchsig_complex_data_type)
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'passband_ripple_db': 1.0, 
            'cutoff': 0.25, 
            'order': 5, 
            'numtaps': 63
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'passband_ripple_db': 4.2, 
            'cutoff': 0.12, 
            'order': 10, 
            'numtaps': 127
        },
        False
    ),    
])
def PassbandRippleDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test PassbandRippleDatasetTransform with pytest.

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
            T = PassbandRippleDatasetTransform(
                passband_ripple_db = passband_ripple_db,
                cutoff = cutoff,
                order = order,
                numtaps = numtaps
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = PassbandRippleDatasetTransform(
            passband_ripple_db = passband_ripple_db,
            cutoff = cutoff,
            order = order,
            numtaps = numtaps
        )
        signal = T(signal)

        assert isinstance(T, PassbandRippleDatasetTransform)
        assert isinstance(T.passband_ripple_db, float)
        assert isinstance(T.cutoff, float)
        assert isinstance(T.order, int)
        assert isinstance(T.numtaps, int)
        assert isinstance(T.fir_coeffs, np.ndarray)
        assert len(T.fir_coeffs) ==  numtaps
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'num_levels': [16], 
            'round_type': ["floor"]
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'num_levels': [4], 
            'round_type': ["nearest"]
        },
        False
    )
])
def test_Quantize(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the Quantize DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): Quantize parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    num_levels = params['num_levels']
    round_type = params['round_type']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Quantize(
                num_levels = num_levels,
                round_type = round_type,
                seed = 42
            )
            signal = T(signal)
    else:
        T = Quantize(
            num_levels = num_levels,
            round_type = round_type,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, Quantize)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.num_levels_distribution(), np.int_)
        assert isinstance(T.round_type_distribution(), str)
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, is_error", [
    ( generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0), False ),
    ( generate_test_dataset_signal(num_iq_samples = 128, scale = 10.0), False )
])
def test_SpectralInversionSignalTransform(signal: DatasetSignal, is_error: bool) -> None:
    """Test the SpectralInversion DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = SpectralInversionDatasetTransform()
            signal = T(signal)
    else:            
        T = SpectralInversionDatasetTransform()
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, SpectralInversionDatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()
        
        for idx, m in enumerate(signal.metadata):
            assert m.center_freq == -1 * signal_test.metadata[idx].center_freq


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL),
        {
            'noise_power_low' : (2.0, 3.0), 
            'noise_power_high': (3.0, 4.0),
            'inflections' : [int(0), int(10)],
            'random_regions' : False
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL),
        {
            'noise_power_low' : (0.0, 0.0), 
            'noise_power_high': (6.0, 12.0),
            'inflections' : [int(4), int(17)],
            'random_regions' : True
        },
        False
    )       
])
def test_TimeVaryingNoise(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the TimeVaryingNoise DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): TimeVaryingNoise parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    rng = np.random.default_rng(42)
    
    noise_power_low = params['noise_power_low']
    noise_power_high = params['noise_power_high']
    inflections = params['inflections']
    random_regions = params['random_regions']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = TimeVaryingNoise(
                noise_power_low  = noise_power_low, 
                noise_power_high = noise_power_high, 
                inflections      = inflections, 
                random_regions   = random_regions,
                seed = 42
            )
            signal = T(signal)
    else:
        T = TimeVaryingNoise(
            noise_power_low  = noise_power_low, 
            noise_power_high = noise_power_high, 
            inflections      = inflections, 
            random_regions   = random_regions,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, TimeVaryingNoise)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.noise_power_low_distribution(), float)
        assert isinstance(T.noise_power_high_distribution(), float)
        assert isinstance(T.inflections_distribution(), np.int_)
        assert isinstance(T.random_regions_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


### ML Tests ----------------


@pytest.mark.parametrize("signal, is_error", [
    ( generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0), False ),
    ( generate_test_dataset_signal(num_iq_samples = 128, scale = 10.0), False )
])
def test_AddSlope(signal: DatasetSignal, is_error: bool) -> None:
    """Test the AddSlope DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AddSlope()
    else:
        T = AddSlope()
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, AddSlope)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()


@pytest.mark.parametrize("signal, is_error", [
    ( generate_test_dataset_signal(num_iq_samples = 64, scale = 1.0), False ),
    ( generate_test_dataset_signal(num_iq_samples = 128, scale = 10.0), False )
])
def test_ChannelSwap(signal: DatasetSignal, is_error: bool) -> None:
    """Test the ChannelSwap DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):    
            T = ChannelSwap()
            signal = T(signal)
    else:
        T = ChannelSwap()
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, ChannelSwap)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        for idx, m in enumerate(signal.metadata):
            assert m.center_freq ==  -1 * signal_test.metadata[idx].center_freq


@pytest.mark.parametrize("signal, params, is_error", [
    ( 
        deepcopy(TEST_DS_SIGNAL), 
        {'duration': [0.17, 0.17], 'cut_type': ['zeros']},
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'duration': [0.12], 
         'cut_type': ['low_noise', 'avg_noise', 'high_noise']},
        False     
    )
])
def test_CutOut(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the CutOut DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): CutOut parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    duration = params['duration']
    cut_type = params['cut_type']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = CutOut(
                duration = duration, 
                cut_type = cut_type,
                seed = 42
            )
            signal = T(signal)
    else:
        T = CutOut(
            duration = duration, 
            cut_type = cut_type,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)            

        assert isinstance(T, CutOut)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.duration_distribution(), float)
        assert isinstance(T.cut_type_distribution(), str)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(duration, list): # assume constant value list
            duration_samples = int(duration[0] * signal.data.size)
        else:
            duration_samples = int(duration * signal.data.size)

        cut_inds = np.where(signal.data != signal_test.data)[0]
        assert (duration_samples == (cut_inds[-1] - cut_inds[0] + 1))


@pytest.mark.parametrize("signal, params, is_error", [
    ( 
        deepcopy(TEST_DS_SIGNAL), 
        {'patch_size': [2, 2], 'shuffle_ratio': 0.5},
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'patch_size': [2, 2], 'shuffle_ratio': [0.5, 0.5]},
        False
    )
])
def test_PatchShuffle(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the PatchShuffle DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): PatchShuffle parameters (see functional PatchShuffle description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    patch_size = params['patch_size']
    shuffle_ratio = params['shuffle_ratio']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = PatchShuffle(
                patch_size = patch_size, 
                shuffle_ratio = shuffle_ratio,
                seed = 42
            )
            signal = T(signal)
    else:
        T = PatchShuffle(
            patch_size = patch_size, 
            shuffle_ratio = shuffle_ratio,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, PatchShuffle)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.patch_size_distribution(), np.int_)
        assert isinstance(T.shuffle_ratio_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(patch_size, list): # assume constant value list
            size_samples = patch_size[0]
        else:
            size_samples = patch_size

        patch_inds = np.where(signal.data != signal_test.data)[0]
        assert ((patch_inds[0] + size_samples - 1) in patch_inds)


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'drop_rate': (0.01, 0.02), 'size': (5, 7), 'fill': ['zero']},
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'drop_rate': (0.01, 0.02), 'size': (5, 7), 'fill': ['mean', 'bfill', 'ffill']},
        False
    )
])
def test_RandomDropSamples(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the RandomDropSamples DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): RandomDropSamples parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    drop_rate = params['drop_rate']
    size = params['size']
    fill = params['fill']

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = RandomDropSamples(
                drop_rate = drop_rate, 
                size = size,
                fill = fill,
                seed = 42
            )
            signal = T(signal)
    else:
        T = RandomDropSamples(
            drop_rate = drop_rate, 
            size = size,
            fill = fill,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, RandomDropSamples)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.drop_rate_distribution(), float)
        assert isinstance(T.size_distribution(), float)
        assert isinstance(T.fill_distribution(), str)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(size, list): # assume constant value list
            size_samples = size[0]
        else:
            size_samples = size

        fill_inds = np.where(signal.data != signal_test.data)[0]
        assert ((fill_inds[0] + size_samples[0] - 1) in fill_inds)   


@pytest.mark.parametrize("signal, params, is_error", [
    ( deepcopy(TEST_DS_SIGNAL), {'start': [0.25], 'scale': [0.4]}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'start': [0.42, 0.42],'scale': [0.17, 0.17, 0.17]}, False )
])
def test_RandomMagRescale(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the RandomMagRescale DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): RandomMagRescale parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    start = params['start']
    scale = params['scale']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = RandomMagRescale(
                start = start, 
                scale = scale,
                seed = 42
            )
            signal = T(signal)
    else:
        T = RandomMagRescale(
            start = start, 
            scale = scale,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)            

        assert isinstance(T, RandomMagRescale)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.start_distribution(), float)
        assert isinstance(T.scale_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(start, list): # assume constant value list
            start_ind = int(signal.data.shape[0] * start[0])
        else:
            start_ind = int(signal.data.shape[0] * start)

        if isinstance(scale, list): # assume constant value list
            sc = scale[0]
        else:
            sc = scale
        
        assert np.allclose(signal.data[start_ind:], sc * signal_test.data[start_ind:])


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'fft_size': 16, 'fft_stride': 4},
        False
    )
])
def test_Spectrogram(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the Spectrogram DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): Spectrogram parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    fft_size = params['fft_size']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            signal.data = np.tile(signal.data, (16, 1, 1))
            T = Spectrogram(
                fft_size = fft_size
            )
            signal = T(signal)
    else:
        T = Spectrogram(
            fft_size = fft_size
        )

        signal_test = deepcopy(signal)
        signal = T(signal)
    
        assert isinstance(T, Spectrogram)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.fft_size, int)
        assert isinstance(signal, DatasetSignal)
        assert signal.data.dtype == np.float32


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'drop_rate': [0.1], 'size': [5], 'fill': ['zero']},
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {'drop_rate': [0.11], 'size': [7, 7], 'fill': ['mean', 'bfill', 'ffill']},
        False
    )
])
def test_SpectrogramDropSamples(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the SpectrogramDropSamples DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): SpectrogramDropSamples parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    drop_rate = params['drop_rate']
    size = params['size']
    fill = params['fill']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            spec_data = compute_spectrogram(
                iq_samples = signal.data, 
                fft_size = 16, 
                fft_stride = 4
            )
            signal.data = np.tile(spec_data, (16, 1, 1))
            
            T = SpectrogramDropSamples(
                drop_rate = drop_rate, 
                size = size,
                fill = fill,
                seed = 42
            )
            signal = T(signal)
    else:        
        spec_data = compute_spectrogram(
            iq_samples = signal.data, 
            fft_size = 16, 
            fft_stride = 4
        )
        signal.data = np.tile(spec_data, (16, 1, 1))
        signal_test = deepcopy(signal)

        T = SpectrogramDropSamples(
            drop_rate = drop_rate, 
            size = size,
            fill = fill,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, SpectrogramDropSamples)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.drop_rate_distribution(), float)
        assert isinstance(T.size_distribution(), np.int_) 
        assert isinstance(T.fill_distribution(), str)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        if isinstance(size, list): # assume constant value list
            size_samples = size[0]
        else:
            size_samples = size

        fill_inds = np.where(signal.data != signal_test.data)
        assert ((fill_inds[2][0] + size_samples - 1) in fill_inds[2]) 


@pytest.mark.parametrize("signal, params, is_error", [
    ( deepcopy(TEST_DS_SIGNAL), {'allow_spectral_inversion': False}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'allow_spectral_inversion': True}, False )
])
def test_TimeReversal(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the TimeReversal DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): TimeReversal parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = TimeReversal(
                allow_spectral_inversion = params['allow_spectral_inversion'],
                seed = 42
            )
            signal = T(signal)
    else:
        T = TimeReversal(
            allow_spectral_inversion = params['allow_spectral_inversion'],
            seed = 42
        )        
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, TimeReversal)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(signal, DatasetSignal)
        assert type(signal) == type(signal_test)
        assert signal.data.dtype == signal_test.data.dtype
        assert np.not_equal(signal.data, signal_test.data).any()

        num_data_samples = len(signal.data)

        for idx, m in enumerate(signal.metadata):
            assert m.start_in_samples == num_data_samples - signal_test.metadata[idx].stop_in_samples
            assert m.center_freq == signal_test.metadata[idx].center_freq
