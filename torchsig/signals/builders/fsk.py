"""Frequency Shift Keying (FSK) and related Signal Builder and Modulator
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

# Third Party
import numpy as np
import scipy.signal as sp
from copy import copy

# Built-In
from collections import OrderedDict

def get_fsk_freq_map ( ):
    """Contains symbol maps for FSK and MSK variants.

    Returns:
        OrderedDict: a dictionary of all symbol maps.
    """
    freq_map = OrderedDict(
        {
            "2fsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2gfsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2msk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2gmsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "4fsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4gfsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4msk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4gmsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "8fsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8gfsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8msk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8gmsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "16fsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16gfsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16msk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16gmsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
        }
    )
    return freq_map


def get_fsk_mod_index( class_name:str, rng=np.random.default_rng() ) -> float:
    """FSK modulation index.

    The modulation index is a parameter that is derived from the symbol
    spacing in the frequency domain. Orthogonal FSK has a modulation index of
    1.0, and MSK and GMSK have a modulation index of 0.5. The modulation index
    is randomized over a wide range for both GFSK and FSK.

    Args:
        class_name (str): Class name to return modulation index, ex: '2fsk'.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        float: Modulation index.
    """
    # returns the modulation index based on the modulation
    if "gfsk" in class_name:
        # bluetooth
        mod_idx = rng.uniform(0.1,0.5)
    elif "msk" in class_name:
        # MSK, GMSK
        mod_idx = 0.5
    else: # FSK
        # 50% chance to use mod index of 1 (orthogonal) ...
        if rng.uniform(0,1) < 0.5:
            mod_idx = 1
        else:
            # ... or something else (non-orthogonal). include
            # a modulation index both less than 1 and greater
            # than 1 to train over a variety of parameters
            mod_idx = rng.uniform(0.7,1.1)
    return mod_idx

def gaussian_taps(samples_per_symbol: int, bt:float, rng=np.random.default_rng()) -> np.ndarray:
    """Designs a gaussian pulse shape for GMSK and GFSK.

    Args:
        samples_per_symbol (int): Number of samples per symbol.
        bt (float): The time-bandwidth product for the Gaussian pulse shape. On the range 0.0 to 1.0.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: Filter weights for the gaussian pulse shape.
    """
    # pre-modulation Bb*T product which sets the bandwidth of the Gaussian lowpass filter
    m = rng.integers(1,5) # randomize the filter span
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / float(samples_per_symbol)) ** 2)
    p = p / np.sum(p)
    return p


def fsk_modulator_baseband ( class_name:str, max_num_samples:int, oversampling_rate_nominal:int, rng=np.random.default_rng() ) -> np.ndarray:
    """FSK modulator at baseband.

    Args:
        class_name (str): Name of the signal to modulate, ex: '2fsk'.
        max_num_samples (int): Maximum number of samples to be produced. The length of
            the output signal must be less than or equal to this number.
        oversampling_rate_nominal (int): The amount of oversampling, which is equal to
            the ratio of the ratio of the sampling rate and bandwidth.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: FSK modulated signal at baseband.
    """

    # determine modulation index
    mod_idx = get_fsk_mod_index(class_name,rng)

    # get the FSK frequency symbol map
    freq_map = get_fsk_freq_map()

    # get the constellation to modulate
    const = freq_map[class_name]

    # scale the frequency map by the oversampling rate such that the tones
    # are packed tighter around f=0 the larger the oversampling rate
    const_oversampled = const / oversampling_rate_nominal

    # calculate the modulation order, ex: the "4" in "4-FSK"
    mod_order = len(const)

    # determine how many samples are in each symbol
    samples_per_symbol = int(mod_order * oversampling_rate_nominal)

    # rectangular pulse shape
    pulse_shape = np.ones(samples_per_symbol)

    if "g" in class_name: # GMSK, GFSK
        # design the gaussian pulse shape with the bandwidth as dictated by the oversampling rate
        #preresample_bandwidth = 1/oversampling_rate_nominal
        bt = rng.uniform(0.1,0.5) # randomize the time-bandwdith product
        taps = gaussian_taps(samples_per_symbol, bt, rng)
        pulse_shape = sp.convolve(taps,pulse_shape)

    # account for the increase in samples due to convolution of pulse shaping filter
    max_num_samples_minus_pulse_shape = max_num_samples - len(pulse_shape) + 1

    # convert number of samples into number of symbols
    num_symbols = int(np.floor(max_num_samples_minus_pulse_shape/samples_per_symbol))

    num_symbols = 1 if num_symbols <= 0 else num_symbols

    # calculate the indexes into symbol table
    symbol_nums = rng.integers(0, len(const_oversampled), num_symbols)

    # produce data symbols
    symbols = const_oversampled[symbol_nums]

    # upsample symbols and apply pulse shaping
    filtered = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)

    # phase accumulator and scaling
    phase = np.cumsum(np.array(filtered) * 1j * mod_idx * np.pi)

    # apply frequency modulation
    modulated = np.exp(phase)

    # pad if signal is too long
    if len(modulated) > max_num_samples:
        # slice to max length
        modulated = slice_tail_to_length ( modulated, max_num_samples )
    elif len(modulated) < max_num_samples:
        # pad to full length
        modulated = pad_head_tail_to_length ( modulated, max_num_samples )
    # else: correct length do nothing

    return modulated

def fsk_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """FSK modulator.

    Args:
        class_name (str): The modulation to create, ex: '2fsk'.
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: FSK modulated at the appropriate bandwidth.
    """

    # define the baseband oversampling rate
    oversampling_rate_nominal = 4

    # calculate the output resampling rate
    oversampling_rate = sample_rate/bandwidth

    # how much to resample the baseband signal to match the bandwidth
    resample_rate_ideal = oversampling_rate/oversampling_rate_nominal

    # calculate the maximum number of samples to be produced by the baseband modulator
    max_num_samples = int(np.floor(num_samples/resample_rate_ideal))

    # ensures a minimum number of samples
    if (max_num_samples < oversampling_rate_nominal):
        max_num_samples = copy(oversampling_rate_nominal)

    # modulate the baseband signal
    baseband_signal = fsk_modulator_baseband ( class_name, max_num_samples, oversampling_rate_nominal, rng )

    # apply resampling
    fsk_correct_bw = multistage_polyphase_resampler ( baseband_signal, resample_rate_ideal )

    # scale to account for the resampling
    fsk_correct_bw *= 1/resample_rate_ideal

    # either slice or pad the signal to the proper length
    if len(fsk_correct_bw) > num_samples:
        fsk_correct_bw = slice_head_tail_to_length ( fsk_correct_bw, num_samples )
    else:
        fsk_correct_bw = pad_head_tail_to_length ( fsk_correct_bw, num_samples )

    # convert into the appropriate data type
    fsk_correct_bw = fsk_correct_bw.astype(torchsig_complex_data_type)

    return fsk_correct_bw


# Builder
class FSKSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for frequency shift keying modulation (FSK) waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["2fsk"]`.
    """
    
    supported_classes = TorchSigSignalLists.fsk_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = '2fsk', **kwargs):
        """Initializes FSK Signal Builder. Sets `class_name= "2fsk"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the FSK waveform based on the signal metadata fields.
        """        
        # dataset params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth
        class_name = self._signal.metadata.class_name

        # FSK modulator at complex baseband
        self._signal.data = fsk_modulator(
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



