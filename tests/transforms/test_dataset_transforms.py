"""Unit Tests for transforms/dataset_transforms.py
"""
from torchsig.transforms.dataset_transforms import (
    DatasetTransform,
    AdditiveNoiseDatasetTransform,
    AWGN,
    CarrierFrequencyDriftDatasetTransform,
    CarrierPhaseNoiseDatasetTransform,
    CarrierPhaseOffsetDatasetTransform,
    CoarseGainChange,
    IQImbalanceDatasetTransform,
    NonlinearAmplifierDatasetTransform,
    PassbandRippleDatasetTransform,    
    QuantizeDatasetTransform,
    SpectralInversionDatasetTransform,
    Spectrogram,
    TimeVaryingNoise,
    TrackingAGC,

    # ML Transforms
    AddSlope,
    ChannelSwap,
    CutOut,
    PatchShuffle,
    RandomDropSamples,
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
    # (deepcopy(TEST_DS_SIGNAL),{'power_range': (0.01, 10.0), 'color': 'white', 'continuous': True,'measure': False},False),
    # (deepcopy(TEST_DS_SIGNAL),{'power_range': (2.0, 4.0), 'color': 'red', 'continuous': False,'measure': False},False),
    (deepcopy(TEST_DS_SIGNAL),{'power_range': (2.0, 2.0),'color': 'white','continuous': True,'measure': True},False)    
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
    measure = params['measure']    

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AdditiveNoiseDatasetTransform(
                power_range = power_range,
                color = color,
                continuous = continuous,
                measure = measure,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)         
        T = AdditiveNoiseDatasetTransform(
            power_range = power_range,
            color = color,
            continuous = continuous,
            measure = measure,
            seed = 42
        )
        signal = T(signal)

        for i, m in enumerate(signal.metadata):
            start = m.start_in_samples
            duration = m.duration_in_samples
            stop = start + duration
            orig_snr_linear = 10 ** (signal_test.metadata[i].snr_db / 10)
            orig_power = np.sum(np.abs(signal_test.data[start:stop])**2)/duration
            out_power = np.sum(np.abs(signal.data[start:stop])**2)/duration
            add_noise_power = out_power - orig_power
            sig_power = orig_power / (1 + 1/orig_snr_linear)
            noise_power = sig_power / orig_snr_linear
            new_snr_db = 10*np.log10(sig_power / (noise_power + add_noise_power))
            
            if measure:
                assert np.abs(signal.metadata[i].snr_db - new_snr_db) < 10**(1.0/10)
            else:
                assert signal.metadata[i].snr_db == signal_test.metadata[i].snr_db

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
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 3.0, 'measure': False}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 0.1, 'measure': False}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'noise_power_db': 0.5, 'measure': True}, False )
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
    noise_power_db = params['noise_power_db']
    add_noise_power_linear = 10**(noise_power_db / 10)
    measure = params['measure']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = AWGN(
                noise_power_db = noise_power_db,
                measure = measure,
                seed = 42
            )
            signal = T(signal)
    else:        
        signal_test = deepcopy(signal)
        T = AWGN(
            noise_power_db = noise_power_db,
            measure = measure,
            seed = 42
        )
        signal = T(signal)

        for i, m in enumerate(signal.metadata):
            if measure:
                start = m.start_in_samples
                duration = m.duration_in_samples
                stop = start + duration
                orig_snr_linear = 10 ** (signal_test.metadata[i].snr_db / 10)
                orig_power = np.sum(np.abs(signal_test.data[start:stop])**2)/duration
                sig_power = orig_power / (1 + 1/orig_snr_linear)
                noise_power = sig_power / orig_snr_linear
                new_snr_db = 10*np.log10(sig_power / (noise_power + add_noise_power_linear))
                assert np.abs(signal.metadata[i].snr_db - new_snr_db) < 10**(1.0/10)
            else:
                assert signal.metadata[i].snr_db == signal_test.metadata[i].snr_db
                
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
            'drift_ppm': (0.1, 1), 
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'drift_ppm': (0.1, 1), 
        },
        False
    ),    
])
def test_CarrierFrequencyDriftDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test CarrierFrequencyDriftDatasetTransform with pytest.

    Args:
        signal (Signal): Input signal to transform.
        params (dict): Transform call parameters (see description).
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """      
    drift_ppm = params['drift_ppm']

    if is_error:
        with pytest.raises(Exception, match=r".*"):   
            T = CarrierFrequencyDriftDatasetTransform(
                drift_ppm = drift_ppm,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = CarrierFrequencyDriftDatasetTransform(
            drift_ppm = drift_ppm, 
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, CarrierFrequencyDriftDatasetTransform)
        assert isinstance(T.drift_ppm_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


@pytest.mark.parametrize("signal, params, is_error", [
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'phase_noise_degrees': (0.25, 1)
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'phase_noise_degrees': (0.25, 1)
        },
        False
    ),    
])
def test_CarrierPhaseNoiseDatasetTransform(
    signal: DatasetSignal,
    params: dict, 
    is_error: bool
) -> None:
    """Test CarrierPhaseNoiseDatasetTransform with pytest.

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
            T = CarrierPhaseNoiseDatasetTransform(
                phase_noise_degrees = phase_noise_degrees,
                seed = 42
            )
            signal = T(signal)
    else:
        signal_test = deepcopy(signal)
        T = CarrierPhaseNoiseDatasetTransform(
            phase_noise_degrees = phase_noise_degrees,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, CarrierPhaseNoiseDatasetTransform)
        assert isinstance(T.phase_noise_degrees, tuple)
        assert isinstance(T.phase_noise_degrees_distribution(), float)
        assert isinstance(signal, DatasetSignal)
        assert len(signal.data) == len(signal_test.data)
        assert type(signal.data) == type(signal_test.data)
        assert signal.data.dtype == torchsig_complex_data_type
        # no metadata impacts


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
    ( deepcopy(TEST_DS_SIGNAL), {'max_gain_change_db': 3.0}, False ),
    ( deepcopy(TEST_DS_SIGNAL), {'max_gain_change_db': 10.0}, False ),
])
def test_CoarseGainChange(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the CoarseGainChange DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): CoarseGainChange parameters (see functional BlockAGC description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):    
            T = CoarseGainChange(
                max_gain_change_db = params['max_gain_change_db'],
            )
            signal = T(signal)
    else:
        T = CoarseGainChange(
            max_gain_change_db = params['max_gain_change_db'],
        )
        signal_test = deepcopy(signal) 
        signal = T(signal)

        assert isinstance(T, CoarseGainChange)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.gain_change_db_distribution(), float) 
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
            'gain_range': (1.0, 4.0),
            'psat_backoff_range': (5.0, 20.0),
            'phi_range': (0.0, 0.0),
            'auto_scale': True
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'gain_range': (0.25, 4.2),
            'psat_backoff_range': (0.0, 2.2),
            'phi_range': (np.deg2rad(4.0), np.deg2rad(7.0)),
            'auto_scale': True
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
    gain_range = params['gain_range']
    psat_backoff_range = params['psat_backoff_range']
    phi_range = params['phi_range']
    auto_scale = params['auto_scale']
    
    if is_error:
        with pytest.raises(Exception, match=r".*"):
                T = NonlinearAmplifierDatasetTransform(
                    gain_range  = gain_range,
                    psat_backoff_range = psat_backoff_range,
                    phi_range = phi_range,
                    auto_scale = auto_scale,
                    seed = 42
                )
                signal = T(signal)
    else:
        T = NonlinearAmplifierDatasetTransform(
            gain_range  = gain_range,
            psat_backoff_range = psat_backoff_range,
            phi_range = phi_range,
            auto_scale = auto_scale,
            seed = 42
        )
        signal = T(signal)

        assert isinstance(T, NonlinearAmplifierDatasetTransform)
        assert isinstance(T.gain_distribution(), float)
        assert isinstance(T.psat_backoff_distribution(), float)
        assert isinstance(T.phi_distribution(), float)
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
            'num_bits': [4]
        },
        False
    ),
    (
        deepcopy(TEST_DS_SIGNAL), 
        {
            'num_bits': [16]
        },
        False
    )
])
def test_QuantizeDatasetTransform(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the Quantize DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): Quantize parameters (see functional description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test output.

    """
    num_bits = params['num_bits']
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = QuantizeDatasetTransform(
                num_bits = num_bits,
                seed = 42
            )
            signal = T(signal)
    else:
        T = QuantizeDatasetTransform(
            num_bits = num_bits,
            seed = 42
        )
        signal_test = deepcopy(signal)
        signal = T(signal)

        assert isinstance(T, QuantizeDatasetTransform)
        assert isinstance(T.random_generator, np.random.Generator)
        assert isinstance(T.num_bits_distribution(), np.int_)
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
def test_TrackingAGC(signal: DatasetSignal, params: dict, is_error: bool) -> None:
    """Test the TrackingAGC DatasetTransform with pytest.

    Args:
        signal (DatasetSignal): input dataset.
        params (dict): AGC parameters (see TrackingAGC DatasetTransform description).
        is_error (bool): Is a test error expected. 

    Raises:
        AssertionError: If unexpected test outcome.

    """
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = TrackingAGC(
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
        T = TrackingAGC(
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

        assert isinstance(T, TrackingAGC)
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
