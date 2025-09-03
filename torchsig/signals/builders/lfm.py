"""LFM Signal Builder and Modulator
"""
# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.signals.builders.chirp import chirp
from torchsig.signals.signal_utils import random_limiting_filter_design
from torchsig.signals.signal_lists import TorchSigSignalLists

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    convolve,
    multistage_polyphase_resampler,
    slice_head_tail_to_length,
    pad_head_tail_to_length
)

# Third Party
import numpy as np

# Built-In
from collections import OrderedDict



def get_symbol_map ( ):
    """Symbol maps for LFM

    Returns:
        OrderedDict: Symbol maps
    """
    lfm_symbol_map = OrderedDict(
        {
            'lfm_data': np.array([-1.,1.]),
            'lfm_radar': np.array([1.]),
        })
    return lfm_symbol_map

def lfm_modulator_baseband ( class_name:str, max_num_samples:int, oversampling_rate_nominal:int, rng=np.random.default_rng() ) -> np.ndarray:
    """LFM modulator at baseband.

    Args:
        class_name (str): Name of the signal to modulate, ex: 'lfm_data'
        max_num_samples (int): Maximum number of samples to be produced. The length of
            the output signal must be less than or equal to this number.
        oversampling_rate_nominal (int): The amount of oversampling, which is equal to
            the ratio of the ratio of the sampling rate and bandwidth.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: LFM modulated signal.
    """

    # modulator has implied sampling rate of 1.0
    sample_rate = 1.0

    # randomize the number of samples per symbol
    samples_per_symbol = rng.integers(low=128,high=4096)

    # calculate bandwidth
    bandwidth = sample_rate/oversampling_rate_nominal

    # calculate the two bounds for the start/stop of the chirp
    f0 = -bandwidth / 2
    f1 = bandwidth / 2

    # get all LFM-related symbol maps
    symbol_map = get_symbol_map()

    # select specific symbol map
    const = symbol_map[class_name] 

    # create the index into the symbol map
    symbol_nums = rng.integers(
        0, len(const), int(np.ceil(max_num_samples / samples_per_symbol))
    )

    # get the foundational symbols
    symbols = const[symbol_nums]

    # define the upchirp and downchirp
    upchirp = chirp(f0,f1,samples_per_symbol)
    downchirp = chirp(f1,f0,samples_per_symbol)

    # pre-allocate memory for the output modulated signal
    modulated = np.zeros((max_num_samples,), dtype=TorchSigComplexDataType)

    # initialize time pointer to the first output index of each symbol
    sym_start_index = 0

    # iterate over each symbol to be modulated
    for s in symbols:

        # calculate the time indexing to store the symbol in the output time series
        time_index = np.arange(sym_start_index,sym_start_index+samples_per_symbol)

        # what if the time index is longer than what is available to store?
        if time_index[-1] >= len(modulated):
            # slice down the time series to the length of what can be stored.
            # once this condition is hit the outer for loop will exit
            time_index = time_index[np.where(time_index < len(modulated))[0]]

        # store the symbols. index into upchirp and downchirp by what proportion
        # of the symbol is available to be stored in the output time series
        if s > 0:
            modulated[time_index] = upchirp[0:len(time_index)]
        else:
            modulated[time_index] = downchirp[0:len(time_index)]

        # increment the pointer into the output time series for the next symbol
        sym_start_index = sym_start_index + samples_per_symbol


    if rng.uniform(0,1) < 0.5: # 50% chance to enable BW limiting filter
        filter_taps = random_limiting_filter_design(bandwidth, sample_rate, rng)
        modulated = convolve(modulated, filter_taps)

    return modulated


def lfm_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """LFM modulator.

    Args:
        class_name (str): The modulation to create, ex: 'lfm_data'.
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: LFM modulated signal at the appropriate bandwidth.
    """

    # calculate final oversampling rate
    oversampling_rate = sample_rate/bandwidth

    # modulate at a nominal oversampling rate. a resampling will be applied
    # after the baseband modulation to bring it to the appropriate bandwidth.
    oversampling_rate_baseband = 4

    # calculate the resampling rate needed to convert the baseband signal into proper bandwidth
    resample_rate_ideal = oversampling_rate/oversampling_rate_baseband

    # determine how many samples baseband modulator needs to implement.
    num_samples_baseband = int(np.ceil(num_samples/resample_rate_ideal))

    num_samples_baseband = max(num_samples_baseband, 1)

    # modulate at baseband
    lfm_signal_baseband = lfm_modulator_baseband ( class_name, num_samples_baseband, oversampling_rate_baseband, rng )

    # apply resampling
    lfm_mod_correct_bw = multistage_polyphase_resampler ( lfm_signal_baseband, resample_rate_ideal )

    # either slice or pad the signal to the proper length
    if len(lfm_mod_correct_bw) > num_samples:
        lfm_mod_correct_bw = slice_head_tail_to_length ( lfm_mod_correct_bw, num_samples )
    else:
        lfm_mod_correct_bw = pad_head_tail_to_length ( lfm_mod_correct_bw, num_samples )

    # convert to appropriate type
    lfm_mod_correct_bw = lfm_mod_correct_bw.astype(TorchSigComplexDataType)

    return lfm_mod_correct_bw

# Builder
class LFMSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for linear frequency modulation (LFM) waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["lfm_data"]`.
    """
    
    supported_classes = TorchSigSignalLists.lfm_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = 'lfm_data', **kwargs):
        """Initializes LFM Signal Builder. Sets `class_name= "lfm_data"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the LFM waveform based on the signal metadata fields
        """        
        # dataset params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth
        class_name = self._signal.metadata.class_name

        # LFM modulator at complex baseband
        self._signal.data = lfm_modulator(
            class_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator
        )

    def _update_metadata(self) -> None:
        """Performs a signals-specific update of signal metadata

        This does nothing because the signal does not need any 
        fields to be updated. This `_update_metadata()` must be
        implemented but is not required to create or modify any data
        or fields for this particular signal case.
        """


