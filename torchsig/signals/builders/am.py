""" AM Signal
"""

# TorchSig
from torchsig.signals.builder import SignalBuilder
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.dsp import (
    low_pass_iterative_design,
    convolve,
    frequency_shift,
    polyphase_decimator,
    torchsig_complex_data_type
)
from torchsig.signals.signal_lists import TorchSigSignalLists

# Third Party
import numpy as np


def am_modulator ( class_name:str, bandwidth:float, sample_rate:float, num_samples:int, rng=np.random.default_rng() ) -> np.ndarray:
    """Amplitude Modulator (AM).

    Args:
        class_name (str): Name of the signal to modulate, ex: 'am-dsb'.
        bandwidth (float): The desired 3 dB bandwidth of the signal. Must be in the same
            units as `sample_rate` and within the bounds 0 < `bandwidth` < `sample_rate`.
        sample_rate (float): The sampling rate for the IQ signal. The sample rate can use a normalized value of 1, or it
            can use a practical sample rate such as 10 MHz. However, it must use the same units as the bandwidth parameter.
        num_samples (int): The number of IQ samples to produce.
        rng (optional): Seedable random number generator for reproducibility.

    Returns:
        np.ndarray: AM modulated signal at the appropriate bandwidth.
        
    """

    if ("lsb" in class_name or "usb" in class_name):
        # must generate 2x the number of samples to account for the decimation by 2
        # with the LSB and USB modulations
        num_samples_mod = 2*num_samples
    else:
        # no rate change, so no change to number of samples
        num_samples_mod = num_samples

    # generate the random message, and make complex data type
    message = rng.normal(0,1,num_samples_mod).astype(torchsig_complex_data_type)
    # scale to unit power
    message = message/np.sqrt(np.mean(np.abs(message)**2))
    # calculate filter cutoff
    cutoff = bandwidth/2

    # calculate maximum transition bandwidth
    max_transition_bandwidth = (sample_rate/2) - cutoff
    # derive actual transition bandwidth
    transition_bandwidth = max_transition_bandwidth/2

    # generate bandwidth-limiting LPF
    lpf = low_pass_iterative_design(cutoff=cutoff, transition_bandwidth=transition_bandwidth, sample_rate=sample_rate)
    # apply bandwidth-limiting filter
    shaped_message = convolve(message, lpf)
    if class_name == "am-dsb-sc":
        baseband_signal = shaped_message
    elif class_name == "am-dsb":
        # build carrier
        carrier = np.ones(len(shaped_message))
        # scale to unit power
        carrier = carrier/np.sqrt(np.mean(np.abs(carrier)**2))
        # randomly determine modulation index
        modulation_index = rng.uniform(0.1,1)
        # add in the carrier
        baseband_signal = (modulation_index*shaped_message) + carrier
    elif class_name == "am-lsb":
        # upconvert signal to center frequency = bandwidth/2
        dsb_upconverted = frequency_shift(shaped_message,bandwidth/2,sample_rate)
        # the existing BW limiting filter can be be repurposed to discard upper band
        lsb_signal_at_if = convolve(dsb_upconverted,lpf)
        # mix LSB back down to baseband from center frequency = bandwidth/4
        baseband_signal_oversampled = frequency_shift(lsb_signal_at_if,-bandwidth/4,sample_rate)
        # since threw away 1/2 the bandwidth to only retain LSB, then downsample by 2 in order to match
        # the requested self.bandwidth
        baseband_signal = polyphase_decimator ( baseband_signal_oversampled, 2 )
        # scale by 2
        baseband_signal *= 2
    elif class_name == "am-usb":
        # downconvert signal to -bandwidth/2
        dsb_downconverted = frequency_shift(shaped_message,-bandwidth/2,sample_rate)
        # the existing BW limiting filter can be be repurposed to discard upper band
        usb_signal_atif = convolve(dsb_downconverted,lpf)
        # mix USB back up to baseband
        baseband_signal_oversampled = frequency_shift(usb_signal_atif,bandwidth/4,sample_rate)
        # since threw away 1/2 the bandwidth to only retain USB, then downsample by 2 in order to match
        # the requested bandwidth
        baseband_signal = polyphase_decimator ( baseband_signal_oversampled, 2 )
        # scale by 2
        baseband_signal *= 2

    # convert to appropriate type
    baseband_signal = baseband_signal.astype(torchsig_complex_data_type)

    return baseband_signal


# Builder
class AMSignalBuilder(SignalBuilder):
    """Implements SignalBuilder() for amplitude modulation (AM) waveform.

    Attributes:
        dataset_metadata (DatasetMetadata): Parameters describing the dataset required for signal generation. 
        supported_classes (List[str]): List of supported signal classes. Set to `["am-dsb-sc"]`.

    """
    
    supported_classes = TorchSigSignalLists.am_signals

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name='am-dsb-sc', **kwargs):
        """Initializes AM Signal Builder. Sets `class_name= "am-dsb-sc"`.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str, optional): Class name.
        """        
        super().__init__(dataset_metadata=dataset_metadata, class_name=class_name, **kwargs)

    def _update_data(self) -> None:
        """Creates the IQ samples for the AM waveform based on the signal metadata fields.
        """        
        # wideband params
        sample_rate = self.dataset_metadata.sample_rate

        # signal params
        num_iq_samples_signal = self._signal.metadata.duration_in_samples
        bandwidth = self._signal.metadata.bandwidth
        class_name = self._signal.metadata.class_name

        # AM modulator at complex baseband
        self._signal.data = am_modulator(
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



