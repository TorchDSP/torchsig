"""Tone Signal Builder and Modulator
"""

# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import numpy as np


# Modulator
def tone_modulator ( num_samples:int ) -> np.ndarray:
    """Implements a tone modulator.

    Args:
        num_samples (float): number of samples of shaped noise to create.

    Returns:
        np.ndarray: Modulated tone IQ samples with proper center frequency.
    """
    # the tone at baseband is all ones
    iq_samples = np.ones(num_samples,dtype=TorchSigComplexDataType)
    return iq_samples



# Builder
class ToneSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for tone waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["tone"]`.
    """
    
    supported_classes = TorchSigSignalLists.tone_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str = 'tone', **kwargs):
        """Initializes Tone Signal Builder. Sets `class_name= "tone"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    
    def _update_data(self) -> None:
        """Creates the IQ samples for the tone waveform based on the signal metadata fields.
        """        

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples

        # tone modulator at complex baseband
        self._signal.data = tone_modulator(
            num_iq_samples_signal
        )

    def _update_metadata(self) -> None:
        """Performs a signals-specific update of signal metadata.

        This does nothing because the signal does not need any 
        fields to be updated. This `_update_metadata()` must be
        implemented but is not required to create or modify any data
        or fields for this particular signal case.
        """

