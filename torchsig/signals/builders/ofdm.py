"""OFDM Signal Builder and Modulator
"""
# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
    torchsig_complex_data_type
)
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.signals.builders.constellation_maps import all_symbol_maps

# Third Party
import numpy as np
from copy import copy


# Modulator
def ofdm_modulator_baseband ( class_name:str, max_num_samples:int, oversampling_rate_nominal:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Modulates OFDM signal at baseband

    Args:
        class_name (str): Name of the signal to modulate, ex: 'ofdm-1024'.
        max_num_samples (int): Maximum number of samples to be produced. The length of
            the output signal must be less than or equal to this number.
        oversampling_rate_nominal (int): The amount of oversampling, which is equal to
            the ratio of the ratio of the sampling rate and bandwidth.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: OFDM modulated signal
    """

    # split the class name to determine how many subcarriers
    num_subcarriers = int(class_name.split('-')[1])

    # define the oversampling rate for the OFDM signal to be generated
    oversampling_rate_nominal = 4

    # calculate IFFT size
    ifft_size = int(oversampling_rate_nominal*num_subcarriers)

    # 50% chance to use cyclic prefix or not
    if rng.uniform(0,1) < 0.5:
        # no cyclic prefix
        cp_len = 0
    else:
        # randomize the cyclic prefix length
        cp_len = rng.integers(2,int(num_subcarriers/2))

    # update cyclic prefix length to account for oversampling
    cp_len_oversampled = cp_len*oversampling_rate_nominal

    # length of ofdm symbol
    ofdm_symbol_length = num_subcarriers+cp_len
    # length of ofdm symbol to account for oversampling
    ofdm_symbol_length_oversampled = ofdm_symbol_length*oversampling_rate_nominal

    # how many OFDM symbols to generate? always produce a number of samples slightly
    # larger than the maximum, this is to avoid returning a long string of zeros
    # which makes the FFT produce a log10(0) which produces an error
    num_ofdm_symbols = int(np.ceil(max_num_samples/ofdm_symbol_length_oversampled))

    # randomize which modulation is to be used as data on subcarriers
    potential_subcarrier_modulations = TorchSigSignalLists.ofdm_subcarrier_modulations
    random_index = rng.integers(0,len(potential_subcarrier_modulations))
    constellation_name = potential_subcarrier_modulations[random_index]

    # get the constellation map
    symbol_map = all_symbol_maps[constellation_name]
    # normalize to unit avg power
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map)**2))

    # generate symbols for active subcarriers across time/frequency grid
    map_index_grid = rng.integers(0,len(symbol_map),(num_subcarriers,num_ofdm_symbols))
    symbol_grid = symbol_map[map_index_grid]

    # create the full time/frequency grid
    time_frequency_grid = np.zeros((ifft_size,num_ofdm_symbols),dtype=torchsig_complex_data_type)

    # fill in the active subcarriers, ignoring index 0 in order to notch DC subcarrier
    half_num_subcarriers = int(num_subcarriers/2)
    time_frequency_grid[1:half_num_subcarriers+1,:] = symbol_grid[0:half_num_subcarriers,:]
    time_frequency_grid[ifft_size-half_num_subcarriers:,:] = symbol_grid[half_num_subcarriers:,:]

    # perform IFFT to get to time-series, but still in grid
    modulated_grid = np.fft.ifft(time_frequency_grid,axis=0)
    # prepend the cyclic prefix
    cp_grid = modulated_grid[ifft_size-cp_len_oversampled:,:]
    modulated_with_cp_grid = np.concatenate((cp_grid,modulated_grid),axis=0)
    # serialize the time series
    ofdm_signal = np.ravel(np.transpose(modulated_with_cp_grid))

    # enforce that the signal has the proper length. the signal here should always
    # be the exact proper length or a little longer, so use this function to slice
    # it down to proper length. discard off the tail to retain the appropriate
    # cyclic prefix and framing samples
    ofdm_signal = slice_tail_to_length(ofdm_signal,max_num_samples)

    return ofdm_signal


def ofdm_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Modulator for OFDM signals

    Args:
        class_name (str): The modulation to create, ex: 'ofdm-1024'.
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: OFDM modulated signal at the appropriate bandwidth.
    """

    # calculate final oversampling rate
    oversampling_rate = sample_rate/bandwidth

    # the oversampling rate in the baseband modulator
    oversampling_rate_baseband = 4

    # calculate the resampling rate
    resample_rate_ideal = oversampling_rate/oversampling_rate_baseband

    # determine how many samples baseband modulator needs to implement.
    num_samples_baseband = int(np.ceil(num_samples/resample_rate_ideal))

    # baseband modulator
    ofdm_signal_baseband = ofdm_modulator_baseband ( class_name, num_samples_baseband, oversampling_rate_baseband, rng )

    # apply resampling to get to the proper bandwidth
    ofdm_signal_correct_bw = multistage_polyphase_resampler ( ofdm_signal_baseband, resample_rate_ideal )

    # either slice or pad the signal to the proper length
    if len(ofdm_signal_correct_bw) > num_samples:
        ofdm_signal_correct_bw = slice_head_tail_to_length ( ofdm_signal_correct_bw, num_samples )
    else:
        ofdm_signal_correct_bw = pad_head_tail_to_length ( ofdm_signal_correct_bw, num_samples )
    # else: correct length, do nothing

    # convert to appropriate type
    ofdm_signal_correct_bw = ofdm_signal_correct_bw.astype(torchsig_complex_data_type)

    return ofdm_signal_correct_bw

# Builder
class OFDMSignalBuilder(SignalBuilder):
    """Implements the OFDM family signal generator.

    Implements SignalBuilder() for the OFDM signals.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `TorchSigSignalLists.ofdm_signals`.
    """
    supported_classes = TorchSigSignalLists.ofdm_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name: str, **kwargs):
        """Initializes OFDM Signal Builder.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name. 
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the OFDM waveform based on the signal metadata fields
        """        
        # wideband params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        class_name = self._signal.metadata.class_name
        bandwidth = self._signal.metadata.bandwidth
        num_iq_samples_signal = self._signal.metadata.duration_in_samples

        # modulate waveform to complex baseband
        self._signal.data = ofdm_modulator(
            class_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator
        )

    def _update_metadata(self) -> None:
        """Performs a signal-specific update of signal metadata.

        Properly defines the minimum duration such that OFDM will
        generate at least 1 symbol.
        """

        # split the class name to determine how many subcarriers
        num_subcarriers = int(self._signal.metadata.class_name.split('-')[1])

        # calculate final oversampling rate
        oversampling_rate = self.dataset_metadata.sample_rate/self._signal.metadata.bandwidth

        # calculate the minimum samples to generate
        minimum_duration_in_samples_for_ofdm = int(np.round(num_subcarriers*oversampling_rate))

        # select the appropriate value against the signal minimum and the dataset minimum
        minimum_duration_in_samples = np.max((minimum_duration_in_samples_for_ofdm,self.dataset_metadata.signal_duration_in_samples_min))

        if (minimum_duration_in_samples >= self.dataset_metadata.signal_duration_in_samples_max):
            # the estimated minimum is too large, use the max instead
            self._signal.metadata.duration_in_samples = copy(self.dataset_metadata.signal_duration_in_samples_max)
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



