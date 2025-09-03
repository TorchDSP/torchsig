"""FM Signal Builder and Modulator
"""

# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    low_pass_iterative_design,
    convolve,
    TorchSigComplexDataType
)
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import numpy as np

def fm_modulator ( bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Frequency Modulator (FM).

    Args:
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: FM modulated signal at the appropriate bandwidth.
    """
    # randomly determine modulation index
    mod_index = rng.uniform(1,10)
    # calculate the frequency deviation using Carson's Rule
    fdev = (bandwidth/2)/(1 + (1/mod_index))
    # calculate the maximum deviation
    fmax = fdev/mod_index
    # compute input message
    message = rng.normal(0,1,num_samples)
    # scale to unit power
    message = message/np.sqrt(np.mean(np.abs(message)**2))
    # design LPF to limit frequencies based on fmax
    lpf = low_pass_iterative_design(cutoff=fmax,transition_bandwidth=fmax,sample_rate=sample_rate)
    # apply the LPF to noise to limit the bandwidth prior to modulation
    source = convolve(message,lpf)
    # apply FM modulation
    modulated = np.exp(2j * np.pi * np.cumsum(source) * fdev/sample_rate)
    # convert to appropriate data type
    modulated = modulated.astype(TorchSigComplexDataType)
    return modulated


# Builder
class FMSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for frequency modulation (FM) waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["fm"]`.
    """
    
    supported_classes = TorchSigSignalLists.fm_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = 'fm', **kwargs):
        """Initializes FM Signal Builder. Sets `class_name= "fm"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the FM waveform based on the signal metadata fields.
        """        
        # dataset params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth

        # FM modulator at complex baseband
        self._signal.data = fm_modulator(
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



