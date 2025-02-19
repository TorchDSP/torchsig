""" Chirp Spread Signal
"""

# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.signals.builders.chirp import chirp
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    low_pass_iterative_design,
    convolve,
    torchsig_complex_data_type,
    multistage_polyphase_resampler,
    slice_head_tail_to_length,
    pad_head_tail_to_length,
)
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import numpy as np

# Built-In
from collections import OrderedDict


def get_symbol_map ( ):
    """Symbol map for Chirp SS.

    Returns:
        OrderedDict: Chirp SS symbol map.
    """
    chirpss_symbol_map = OrderedDict(
        {
            'chirpss': np.linspace(0,2**7-1,2**7),
        })
    return chirpss_symbol_map

def chirpss_modulator_baseband ( class_name:str, max_num_samples:int, oversampling_rate_nominal:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Chirp Spread Spectrum modulator at baseband.

    Args:
        class_name (str): Name of the signal to modulate, ex: 'chirpss'.
        max_num_samples (int): Maximum number of samples to be produced. The length of
            the output signal must be less than or equal to this number.
        oversampling_rate_nominal (int): The amount of oversampling, which is equal to
            the ratio of the ratio of the sampling rate and bandwidth.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: Chirp SS modulated signal
    """

    # modulator has implied sampling rate of 1.0
    sample_rate = 1.0

    # randomize the number of samples per symbol
    samples_per_symbol = rng.integers(low=128,high=4096)

    # calculate bandwidth
    bandwidth = sample_rate/oversampling_rate_nominal

    # get a list of possible symbol maps
    symbol_map = get_symbol_map()

    # select the chirp-ss specific symbol map
    const = symbol_map[class_name]

    # create the index into the symbol map
    symbol_nums = rng.integers(
        0, len(const), int(np.ceil(max_num_samples / samples_per_symbol))
    )

    # get the foundational symbols
    symbols = const[symbol_nums]

    # construct template symbols
    upchirp = chirp(-bandwidth,bandwidth,samples_per_symbol)
    double_upchirp = np.concatenate((upchirp, upchirp), axis=0)

    # pre-allocate memory for the output modulated signal
    modulated = np.zeros((max_num_samples,), dtype=torchsig_complex_data_type)

    # create the modulated signal by selecting the appropriate symbol and inserting into the IQ array
    sym_start_index = 0
    m = const.size
    for s in symbols:
        # calculate the output time for the symbol
        output_time = np.arange(sym_start_index,sym_start_index+samples_per_symbol)

        # what if the time index is longer than what is available to store?
        if output_time[-1] >= len(modulated):
            # slice down the time series to the length of what can be stored.
            # once this condition is hit the outer for loop will exit
            output_time = output_time[np.where(output_time < len(modulated))[0]]


        # calculate the time index into the chirp
        chirp_start_index = int((s/m)*samples_per_symbol)
        input_time = np.arange(chirp_start_index,chirp_start_index+len(output_time))
        modulated[output_time] = double_upchirp[input_time] 
        # increment the time next for next symbol
        sym_start_index = sym_start_index + samples_per_symbol

    if rng.uniform(0,1) < 0.5: # 50% chance to turn on BW limiting filter
        # randomize the cutoff
        cutoff = rng.uniform(0.8*bandwidth/2,0.95*sample_rate/2)
        # calculate maximum transition bandwidth
        max_transition_bandwidth = sample_rate/2 - cutoff
        # transition bandwidth is randomized value less than max transition bandwidth
        transition_bandwidth = rng.uniform(0.5,1.5)*max_transition_bandwidth
        # design bandwidth-limiting filter
        lpf = low_pass_iterative_design(cutoff=cutoff,transition_bandwidth=transition_bandwidth,sample_rate=sample_rate)
        # apply bandwidth-limiting LPF to reduce sidelobes
        modulated = convolve(modulated,lpf)

    return modulated


def chirpss_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Chirp Spread Spectrum modulator.

    Args:
        class_name (str): The modulation to create, ex: 'chirpss'.
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: Chirp SS modulated signal at the appropriate bandwidth.
    """

    # calculate final oversampling rate
    oversampling_rate = sample_rate/bandwidth

    # modulate at a nominal oversampling rate. a resampling will be applied
    # after the baseband modulation to bring it to the appropriate bandwidth.
    oversampling_rate_baseband = 4

    # calculate the resampling rate needed to convert the baseband signal into proper bandwidth
    resample_rate_ideal = oversampling_rate/oversampling_rate_baseband

    # determine how many samples baseband modulator needs to implement.
    num_samples_baseband = int(np.floor(num_samples/resample_rate_ideal))

    # modulate at baseband
    chirpss_signal_baseband = chirpss_modulator_baseband ( class_name, num_samples_baseband, oversampling_rate_baseband, rng )

    # apply resampling
    chirpss_mod_correct_bw = multistage_polyphase_resampler ( chirpss_signal_baseband, resample_rate_ideal )

    # either slice or pad the signal to the proper length
    if len(chirpss_mod_correct_bw) > num_samples:
        chirpss_mod_correct_bw = slice_head_tail_to_length ( chirpss_mod_correct_bw, num_samples )
    else:
        chirpss_mod_correct_bw = pad_head_tail_to_length ( chirpss_mod_correct_bw, num_samples )

    # convert to appropriate type
    chirpss_mod_correct_bw = chirpss_mod_correct_bw.astype(torchsig_complex_data_type)

    return chirpss_mod_correct_bw

# Builder
class ChirpSSSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for chirp spread spectrum waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["chirpss_data"]`.
    """
    
    supported_classes = TorchSigSignalLists.chirpss_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = 'chirpss_data', **kwargs):
        """Initializes Chirp Spread Spectrum Signal Builder. Sets `class_name= "chirpss_data"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the Chirp SS waveform based on the signal metadata fields.
        """        
        # wideband params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth
        class_name = self._signal.metadata.class_name

        # chirp SS modulator at complex baseband
        self._signal.data = chirpss_modulator(
            class_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator
        )

    def _update_metadata(self) -> None:
        """Performs a signals-specific update of signal metadata.

        This does nothing because the signal does not need any 
        fields to be updated. This `_update_metadata()` must be
        implemented but is not required to create or modify any data
        or fields for this particular signal case.
        """



