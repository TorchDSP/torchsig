"""Test Signal Builder and Modulator
"""
# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    low_pass_iterative_design,
    convolve
)

# Third Party
import numpy as np


# Modulator
def shaped_noise_modulator ( bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Implements a shaped noise modulator for testing and development.

    Args:
        bandwidth (float): the desired 3 dB bandwidth of the signal
        signal_power_db (float): the desired power of the signal in dB
        sample_rate (float): the sample rate (must align with CF and BW values)
        num_samples (float): number of samples of shaped noise to create
        rng (optional): Seedable random number generator for reproducibility.
    Returns:
        np.ndarray: Shaped noise IQ samples with proper bandwidth and center frequency.
    """
    # compute noise ("message") at 0 dB gain
    noise = rng.normal(0,np.sqrt(1/2),num_samples) + 1j*rng.normal(0,np.sqrt(1/2),num_samples)
    # apply LPF to get down to bandwidth
    lpf = low_pass_iterative_design(cutoff=bandwidth/2, transition_bandwidth=bandwidth/16, sample_rate=sample_rate)
    shaped_noise_iq = convolve(signal=noise,taps=lpf)
    
    #return passband_signal
    return shaped_noise_iq


# Builder
class TestSignalBuilder(SignalBuilder):
    """Implements the test signal generator.

    Implements SignalBuilder() for a test waveform which is shaped noise.
    The waveform is useful in development but is not a practical signal.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["testsignal"]`.
    """

    supported_classes = ["testsignal"]

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = 'testsignal', **kwargs):
        """Initializes Test Signal Builder. Sets `class_name= "testsignal"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """    
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the test signal waveform based on the signal metadata fields.
        """        
        # dataset params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        bandwidth = self._signal.metadata.bandwidth
        num_iq_samples_signal = self._signal.metadata.duration_in_samples

        # modulate waveform at complex baseband
        self._signal.data = shaped_noise_modulator(
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

