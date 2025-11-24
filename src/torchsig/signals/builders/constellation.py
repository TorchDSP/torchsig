"""Constellation Signal Builder and Modulator
"""
# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    estimate_filter_length,
    srrc_taps, 
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
    TorchSigComplexDataType
)
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.signals.builders.constellation_maps import all_symbol_maps

# Third Party
import numpy as np
import scipy.signal as sp

# Built-In
from copy import copy


# Modulator

def constellation_modulator_baseband ( class_name:str, pulse_shape_name:str, max_num_samples:int, oversampling_rate_nominal:int, alpha_rolloff:float=None, rng=np.random.default_rng() ) -> np.ndarray:
    """Modulates constellation based signals (QAM/PSK/ASK/OOK) at complex baseband.

    Args:
        class_name (str): Name of the signal to modulate, ex: 'qpsk'.
        pulse_shape_name (str): Pulse shaping filter selection, 'rectangular' or 'srrc' (square-root raised cosine).
        max_num_samples (int): Maximum number of samples to be produced. The length of
            the output signal must be less than or equal to this number.
        oversampling_rate_nominal (int): The amount of oversampling, which is equal to
            the ratio of the ratio of the sampling rate and bandwidth.
        alpha_rolloff (float, optional): The alpha-rolloff value for the SRRC filter. If pulse_shape_name == 'recantangular'
            then this value is ignored. If pulse_shape_name == 'srrc' then this value must be defined, and within the range
            0 < alpha_rolloff < 1. Defaults to None.
        rng (optional): Seedable random number generator for reproducibility.

    Raises:
        ValueError: Raises ValueError if pulse_shape_name is neither 'rectangular' or 'srrc'.
        ValueError: Raises ValueError if alpha_rolloff is not defined when selecting 'srrc'.

    Returns:
        np.ndarray: IQ samples of the constellation-modulated complex baseband signal.
    """

    # also the samples per symbol
    samples_per_symbol = oversampling_rate_nominal

    # get symbol map
    symbol_map = all_symbol_maps[class_name]

    # ensure symbol map is avg unit power
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map)**2))

    # pulse shape
    if pulse_shape_name == 'rectangular':
        pulse_shape = np.ones(samples_per_symbol)
        pulse_shape_filter_span = 0
    elif pulse_shape_name == 'srrc':
        if alpha_rolloff is None:
            raise ValueError('must define an alpha rolloff for SRRC filter')
        # design the pulse shaping filter
        attenuation_db = 120
        pulse_shape_filter_length = estimate_filter_length(alpha_rolloff,attenuation_db,1)
        pulse_shape_filter_span = int(np.ceil((pulse_shape_filter_length - 1) / (2*samples_per_symbol)))  # convert filter length into the span
        pulse_shape = srrc_taps(samples_per_symbol, pulse_shape_filter_span, alpha_rolloff)
    else:
        raise ValueError('pulse shape ' + str(pulse_shape_name) + ' not supported')

    # number of symbols to subtract off from generation in order to produce
    # a signal that is less than the desired length to avoid slicing a symbol
    # 
    # filter span = (number of symbols in pulse shape - 1)/2, therefore the span
    # for a rectangular pulse shape is zero
    subtract_off_symbols = 2*pulse_shape_filter_span

    # number of symbols to create. use floor() and subtract off based
    # on filter span to ensure that a smaller number of samples is created
    # and does not equal or exceed the max_num_samples. the idea is to avoid
    # slicing a symbol, rather all samples from the transition periods are 
    # to be retained
    num_symbols = int(np.floor(max_num_samples/samples_per_symbol))-subtract_off_symbols

    # enforce that the minimum number cannot be less than 1
    num_symbols = 1 if num_symbols < 1 else num_symbols

    # create symbols. because OOK has symbols which are zeros this needs to run
    # a loop until 1 or more symbols are non-zero
    symbols = np.zeros(1)
    while np.equal(np.sum(np.abs(symbols)), 0):
        # index into the symbol map
        map_index = rng.integers(low=0,high=len(symbol_map),size=num_symbols)

        # randomly generate symbols
        symbols = symbol_map[map_index]

    # interplate using pulse shaping filter
    constellation_signal_baseband = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)

    # zero-pad if signal is too short
    if len(constellation_signal_baseband) < max_num_samples:
        constellation_signal_baseband = pad_head_tail_to_length ( constellation_signal_baseband, max_num_samples )
    # slice if signal is too long
    elif len(constellation_signal_baseband) > max_num_samples:
        constellation_signal_baseband = slice_tail_to_length ( constellation_signal_baseband, max_num_samples )
    # else: signal correct length, do nothing

    # ensure proper data type
    constellation_signal_baseband = constellation_signal_baseband.astype(TorchSigComplexDataType)

    return constellation_signal_baseband

def constellation_modulator ( class_name:str, pulse_shape_name:str, bandwidth:float, sample_rate:float, num_samples:int, alpha_rolloff:float=None, rng=np.random.default_rng() ) -> np.ndarray:
    """Modulator for constellation-based signals (QAM/PSK/ASK/OOK).

    Args:
        class_name (str): The modulation to create, ex: 'qpsk'.
        pulse_shape_name (str): Pulse shaping filter selection, 'rectangular' or 'srrc' (square-root raised cosine).
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        alpha_rolloff (float, optional): The alpha-rolloff value for the SRRC filter. This is a pass through to
            constellation_baseband_modulator(). If pulse_shape_name == 'recantangular' then this value is ignored. If
            pulse_shape_name == 'srrc' then this value must be defined, and within the range 0 < alpha_rolloff < 1. Defaults
            to None.
        rng (optional): Seedable random number generator for reproducibility.

    Raises:
        ValueError: Raises ValueError if the number of samples produced is incorrect.

    Returns:
        np.ndarray: Returns the constellation-modulated IQ samples at the appropriate center frequency and bandwidth.
    """


    # calculate final oversampling rate
    oversampling_rate = sample_rate/bandwidth

    # modulate at a nominal oversampling rate. a resampling will be applied
    # after the baseband modulation to bring it to the appropriate bandwidth.
    oversampling_rate_baseband = 4

    # calculate the resampling rate needed to convert the baseband signal into proper bandwidth
    resample_rate_ideal = oversampling_rate/oversampling_rate_baseband

    # determine how many samples baseband modulator needs to implement.
    # use floor() to ensure that generated sequence is slightly less in
    # order to avoid slicing any portion of the burst, and instead 
    # zero-pad with a small number of samples at the end to bring up to 
    # appropriate length
    num_samples_baseband_init = int(np.floor(num_samples/resample_rate_ideal))
    if num_samples_baseband_init <= 0:
        num_samples_baseband = oversampling_rate_baseband
    else:
        num_samples_baseband = num_samples_baseband_init
    
    # modulate at baseband
    constellation_signal_baseband = constellation_modulator_baseband ( class_name, pulse_shape_name, num_samples_baseband, oversampling_rate_baseband, alpha_rolloff, rng )

    # apply resampling
    constellation_mod_correct_bw = multistage_polyphase_resampler ( constellation_signal_baseband, resample_rate_ideal )

    # either slice or pad the signal to the proper length
    if len(constellation_mod_correct_bw) > num_samples:
        constellation_mod_signal = slice_head_tail_to_length ( constellation_mod_correct_bw, num_samples )
    else:
        constellation_mod_signal = pad_head_tail_to_length ( constellation_mod_correct_bw, num_samples )

    if len(constellation_mod_signal) != num_samples:
        raise ValueError('constellation mod producing incorrect number of samples: ' + str(len(constellation_mod_signal)) + ' but requested: ' + str(num_samples))

    # convert to appropriate type
    constellation_mod_signal = constellation_mod_signal.astype(TorchSigComplexDataType)

    return constellation_mod_signal



# Builder
class ConstellationSignalBuilder(SignalBuilder):
    """Implements the Constellation family signal generator.

    Implements SignalBuilder() for the linearly modulated constellation-based families: QAM, PSK, PAM, ASK, OOK.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `TorchSigSignalLists.constellation_signals`.
    """
    supported_classes = TorchSigSignalLists.constellation_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name: str, **kwargs):
        """Initializes Constellation Signal Builder.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name. 
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)


    def _update_data(self) -> None:
        """Creates the IQ samples for the constellation waveform based on the signal metadata fields.
        """        
        # dataset params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        class_name = self._signal.metadata.class_name
        bandwidth = self._signal.metadata.bandwidth
        num_iq_samples_signal = self._signal.metadata.duration_in_samples

        # randomize pulse shape selection
        if np.equal(self.random_generator.integers(0,2),0):
            pulse_shape_name = 'srrc'
            # randomize alpha_rolloff
            alpha_rolloff = self.random_generator.uniform(0.1,0.5)
        else:
            pulse_shape_name = 'rectangular'
            alpha_rolloff = None

        # modulate waveform to complex baseband
        self._signal.data = constellation_modulator(
            class_name,
            pulse_shape_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            alpha_rolloff,
            self.random_generator
        )

    def _update_metadata(self) -> None:
        """Performs a signals-specific update of signal metadata.

        The signal duration for a constellation waveform must be at least
        1 symbol, which is greater than the default in dataset metadata which
        is 1 sample. Therefore the duration needs to be recalculated based
        on the other signal metadata fields, since a symbol period is dependent
        on the oversampling rate (and therefore bandwidth).
        """        
        # the duration self._signal.metadata.duration_in_samples cannot be used for Constellation 
        # waveform. the base class Builder() uses a minimum duration of 1 sample which is too small 
        # for constellation and other modulated waveforms. instead, it has to be calculated to be at 
        # least 1 symbol long which is based on the bandwidth and oversampling rate
        minimum_duration_in_symbols = 1
        oversampling_rate = int(np.ceil(self.dataset_metadata.sample_rate/self._signal.metadata.bandwidth))
        minimum_duration_for_one_symbol = np.clip(oversampling_rate*minimum_duration_in_symbols, a_min=None, a_max=self.dataset_metadata.num_iq_samples_dataset)

        # choose the larger of the two minimums
        minimum_duration_in_samples = np.max((minimum_duration_for_one_symbol,self.dataset_metadata.signal_duration_in_samples_min))

        # is duration parameter to be randomized?
        if minimum_duration_in_samples == self.dataset_metadata.signal_duration_in_samples_max:
            # the min and max fields are the same, so just use one of the fields
            self._signal.metadata.duration_in_samples = copy(self.dataset_metadata.signal_duration_in_samples_min)
        else:
            # randomize the duration
            self._signal.metadata.duration_in_samples = self.random_generator.integers(low=minimum_duration_in_samples, high=self.dataset_metadata.signal_duration_in_samples_max,dtype=int)

        # is start parameter to be randomized?
        if self._signal.metadata.duration_in_samples == self.dataset_metadata.num_iq_samples_dataset:
            # duration is equal to the total dataset length, therefore start must be zero
            self._signal.metadata.start_in_samples = 0
        else:
            # given duration, start is randomly set from 0 to rightmost time that the duration still fits inside the dataset iq samples
            self._signal.metadata.start_in_samples = self.random_generator.integers(low=0, high=self.dataset_metadata.num_iq_samples_dataset - self._signal.metadata.duration_in_samples,dtype=int)


