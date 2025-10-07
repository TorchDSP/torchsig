"""Signal and Composite Signal Builders

Examples
    Signal Builder
        >>> from torchsig.signals import SignalBuilder
        >>> sb = SignalBuilder()
        >>> sb.data = np.array([1.0, 2.0])
        >>> sb.sample_rate = 1.5
        >>> ...
        >>> new_signal = sb.build()
    Composite Signal Builder
        >>> from torchsig.signals import CompositeSignalBuilder, SignalBuilder
        >>> builder1 = SignalBuilder()
        >>> builder1.data = [1.0, 2.0]
        >>> ...
        >>> csb = CompositeSignalBuilder()
        >>> csb.builders.append(builder1)
        >>> csb.sample_rate = 2.0
        >>> ...
        >>> new_composite_signal = csb.build()
"""

from __future__ import annotations

# TorchSig
from torchsig.signals.signal_types import Signal, SignalMetadata
from torchsig.utils.random import Seedable
from torchsig.utils.dsp import compute_spectrogram

# Third Party
import numpy as np

# Built-In
from abc import ABC, abstractmethod
from copy import copy
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchsig.datasets.dataset_metadata import DatasetMetadata



class Builder(ABC):
    """Abstract builder class for signals

    Attributes:
        name (str): Builder name. Defaults to "Builder".
    """
    def __init__(self, name: str = "Builder"):
        """Initialize builder, reset.
        """
        self.name = name
        self._signal = None

    @abstractmethod
    def build(self) -> Signal:
        """Build and Return Signal() object. To be implemented by subclasses.

        Returns:
            Signal: signal being built.
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets builder. To be implemented by subclasses.
        """
        self._signal = None
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

class SignalBuilder(Builder, Seedable):
    """ Signal Builder. Creates a Signal.

    Attributes:
        dataset_metadata (DatasetMetadata): Dataset metadata for signal.
        class_name (str): Name of the specific waveform to build, ex: 2fsk, qpsk, ofdm-1024
        name (str): Signal builder name. Defaults to "class_name Signal Builder"
        supported_classes (List[str]): Defines what classes builder supports. Defaults to [].
        supported_classes (List[str]): List of signal class names that the builder supports. Set to `[]`.
    """

    supported_classes = []

    
    def __init__(self, dataset_metadata: DatasetMetadata, class_name:str, **kwargs):
        """Initializes Signal Builder.

        Args:
            dataset_metadata (DatasetMetadata): Dataset metadata.
            class_name (str): Class name.

        Raises:
            ValueError: Signal builder does not support class_name signal.
        """
        self.class_name = class_name
        self._signal = None
        Builder.__init__(self, name=f" {self.class_name} Signal Builder")
        
        Seedable.__init__(self, **kwargs)
        # retains dataset metadata info
        self.dataset_metadata = dataset_metadata
        
        

        if not self.class_name in self.supported_classes:
            raise ValueError(f"{self.class_name} + ' not supported by {self.__class__.__name__}. List of supported waveforms: {self.supported_classes}")

    def __repr__(self):
        return f"{self.__class__.__name__}(class_name={self.class_name})"

    def _update_data(self) -> None:
        """Creates the IQ samples and sets them to `self._signal.data`.

        Creates the IQ samples for the waveform modulated with the 
        center frequency, SNR, and other fields as described by `self._signal.metadata`.
        `_update_data()` builds the waveform from metadata defined by `_update_metadata()`.

        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError("_update_data() not implemented.")
    
    def _update_metadata(self) -> None:
        """Updates `self._signal.metadata`.

        Signal metadata such as center frequency, SNR and others are randomly
        created and set to default values as defined by dataset metadata inside 
        `reset()`. This `_update_metadata()` function is only used to override 
        those default values for signal-specific cases. The metadata will inform 
        the creation of the IQ samples within the `_update_data()` function.
        
        Raises:
            NotImplementedError: Inherited classes must implement this method.
        """        
        raise NotImplementedError("_update_metadata() not implemented.")

            


    
    def build(self) -> Signal:     
        """Builds and returns Signal

        The Builder() __init__ calls `reset()` which initializes the signal
        metadata according to default values defined within dataset metadata.
        `_update_metadata()` then updates signal-specific metadata fields
        as needed. `_update_data()` creates the IQ samples and assigns them
        to the Signal() data field. `_correct_bandwidth_and_snr()` sets the
        power of the signal based on the PSD estimate to have the appropriate
        SNR and then estimates the total occupied bandwidth of the signal,
        including sidelobes, to develop a more accurate bounding box.

        In order, calls:
        - `reset()`
        - `_update_metadata()`
        - `_update_data()`
        - `_correct_bandwidth_and_snr()`


        Returns:
            Signal: Signal being built.
        """

        # performs a reset of the signal metadata to default values as defined
        # by dataset metadata, as was done in the Builder() parent class __init__
        self.reset()

        # the __init__ within the parent class Builder() has called .reset()
        # to establish the signal metadata according to the dataset metadata
        # defaults. this function is used to update signal-specific metadata
        # fields
        self._update_metadata()

        # generate IQ with appropriate modulator based on signal metadata
        self._update_data()

        # reestimate bandwidth in order to better fit the bounding box
        # in the frequency domain
        self._correct_bandwith_and_snr()

        # signal object to be returned
        new_signal = self._signal

        # ensures IQ data is in complex64
        new_signal.data = new_signal.data.astype(np.complex64)

        return new_signal
    
    
    def reset(self) -> None:
        """Resets `_signal` according to defaults defined by dataset metadata.

        Signal metadata is generated according to the default values as defined
        by the dataset metadata. The signal data is set to noise samples.

        These metadata value can be overridden if a particular modulator or 
        special signal requires it. For example, the bandwidth of a tone is 
        dependent on the signal duration and therefore requires recalculation 
        and an update to the signal metadata inside `_update_metadata()` in 
        the tone builder class. Similarly, the minimum duration for a 
        constellation based signal must be at least 1 symbol, which requires 
        recomputation in the `_update_metadata()` field for the constellation
        builder class.
        
        This method is called by the parent Builder() `__init__` and after `build()`.
        """

        # is duration parameter to be randomized?
        if self.dataset_metadata.signal_duration_in_samples_min == self.dataset_metadata.signal_duration_in_samples_max:
            # the min and max fields are the same, so just use one of the fields
            duration = copy(self.dataset_metadata.signal_duration_in_samples_min)
        else:
            # sets duration randomly between min duration and max duration as defined by dataset metadata
            duration = self.random_generator.integers(low=self.dataset_metadata.signal_duration_in_samples_min, high=self.dataset_metadata.signal_duration_in_samples_max,dtype=int)

        # is start parameter to be randomized?
        if duration == self.dataset_metadata.num_iq_samples_dataset:
            # duration is equal to the total dataset length, therefore start must be zero
            start = 0
        else:
            # given duration, start is randomly set from 0 to rightmost time that the duration still fits inside the dataset iq samples
            start = self.random_generator.integers(low=0, high=self.dataset_metadata.num_iq_samples_dataset-duration,dtype=int)

        # randomly set bandwidth between a minimum and max. use a log10-scaling on the
        # randomization there can be a large difference between the min and max bandwidth
        bw_min_log10 = np.log10(self.dataset_metadata.signal_bandwidth_min)
        bw_max_log10 = np.log10(self.dataset_metadata.signal_bandwidth_max)
        bw = 10**(self.random_generator.uniform(bw_min_log10,bw_max_log10))

        # center frequency always zero, will be randomized within dataset
        # due to the need to apply impairments at complex baseband first 
        # before upconversion to the IF
        center_freq = 0
        # randomly select SNR
        snr_db = np.round(self.random_generator.uniform(self.dataset_metadata.snr_db_min,self.dataset_metadata.snr_db_max),1)

        # define SignalMetadata
        default_metadata = SignalMetadata(
            dataset_metadata = self.dataset_metadata,
            start_in_samples = start,
            duration_in_samples = duration,
            bandwidth = bw,
            center_freq = center_freq,
            snr_db = snr_db,
            class_name = self.class_name,
            class_index = self.dataset_metadata.class_list.index(self.class_name)
        )

        # set signal data to be zeros; the proper IQ samples will be
        # recreated as part of the _update_data() call
        default_data = np.zeros(self.dataset_metadata.num_iq_samples_dataset,dtype=np.complex64)

        # reset _signal
        self._signal = Signal(data=default_data, metadata=default_metadata)

    def _correct_bandwith_and_snr(self) -> None:
        """Corrects SNR of time-series, and metadata bandwidth to produce more 
        accurate bounding box

        The SNR is set by estimating the signal PSD and scaling the signal to
        produce an accurate SNR estimate. The SNR of the signal is calculated by
        the maximum of the PSD estimate subtracted by the noise power.

        The tone signal bandwidth is often larger than the predicted bounding box,
        in particular for high SNR signals. A spectral estimation 
        method is used to estimate the 99% bandwidth of the signal and therefore
        marginally increase the bounding box size to fit. Note that the tone signal
        itself cannot be resampled to fit into the bounding box because the
        bandwidth of the tone is soley derived from the number of samples, which
        would force a change to the time duration which is not desirable. This same
        method is applied for all signals to produce a more accurate bounding box.
        """

        # compute spectral estimate of signal. use a large stride to process only a
        # subset of the data to reduce computation
        signal_spectrogram_db = compute_spectrogram(
            self._signal.data, 
            self.dataset_metadata.fft_size, 
            self.dataset_metadata.fft_stride
        )

        # average over time, used in PSD estimate for SNR calculation
        signal_avg_fft_db = np.mean(signal_spectrogram_db,axis=1)

        # estimate the frequency response maximum value
        max_value_db = np.max(signal_avg_fft_db)

        # estimate SNR
        snr_estimate_db = max_value_db - self.dataset_metadata.noise_power_db

        # calculate the appropriate correction to set SNR
        correction_db = self._signal.metadata.snr_db - snr_estimate_db

        # convert correction value to linear
        correction = 10**(correction_db/10)
        
        # apply correction value to signal
        self._signal.data *= np.sqrt(correction)

        # also apply correction to avg FFT
        signal_avg_fft_db += correction_db

        # apply correction to spectrogram
        signal_spectrogram_db += correction_db

        # compute max hold for bandwidth estimation
        signal_max_fft_db = np.max(signal_spectrogram_db,axis=1)

        # find edges where the signal is above noise floor by the amount dictated by the relative threshold
        noise_floor_db = self.dataset_metadata.noise_power_db
        relative_threshold_db = 3
        bandwidth_estimation_threshold_db = noise_floor_db + relative_threshold_db
        exceedance_indices = np.where(signal_max_fft_db > bandwidth_estimation_threshold_db)[0]

        upper_edge_index = 0
        lower_edge_index = 0

        if len(exceedance_indices) == 1:
            # single threshold exceedance, measured bandwidth is equal to 1 FFT bin width
            lower_edge_index = copy(exceedance_indices[0])
            upper_edge_index = copy(exceedance_indices[0])
            # set flag that bandwidth field needs updating
            update_bandwidth = True
        elif len(exceedance_indices) > 1:
            # multiple exceedances, must compute bandwidth
            lower_edge_index = exceedance_indices[0]
            upper_edge_index = exceedance_indices[-1]
            # set flag that bandwidth field needs updating
            update_bandwidth = True
        else:
            # no exceedances, signal power is too low to be detected, current
            # bandwidth is fine
            update_bandwidth = False

        if update_bandwidth:

            # create frequency vector for the FFT
            f = np.linspace(-0.5,0.5-(1/self.dataset_metadata.fft_size),self.dataset_metadata.fft_size)*self.dataset_metadata.sample_rate

            # determine estimated upper and lower freq bounds
            upper_freq = f[upper_edge_index]
            lower_freq = f[lower_edge_index]

            # widen bandwidth by a small proportion
            widen_bandwidth_value = self.dataset_metadata.fft_frequency_resolution/2

            # logic to widen bandwidth on upper and lower freq edge, and avoid running
            # past the boundary
            lower_freq -= widen_bandwidth_value
            if lower_freq < self.dataset_metadata.frequency_min:
                lower_freq = copy(self.dataset_metadata.frequency_min)

            upper_freq += widen_bandwidth_value
            if upper_freq > self.dataset_metadata.frequency_max:
                upper_freq = copy(self.dataset_metadata.frequency_max)

            # compute 99% bandwidth
            bandwidth99 =  upper_freq - lower_freq
            # because the tone's bandwidth cannot be resampled we must instead change
            # the bounding box by changing the bandwidth metadata field
            self._signal.metadata.bandwidth = bandwidth99

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(name={self.name})\n\t_signal={self._signal}"




###
# class CompositeSignalBuilder(SignalBuilder):
#     """CompositeSignal Builder. Creates a complex signal with multiple Signals inside it.

#     Attributes:
#         dataset_metadata (DatasetMetadata): Dataset metadata for signal.
#         name (str): Composite signal builder name. Defaults to "Composite Signal Builder"
#         builders (List[SignalBuilder]): List of SignalBuilders. Defaults to [].
#     """
    
#     def __init__(self, dataset_metadata: DatasetMetadata, seed:int = None):
#         super().__init__(dataset_metadata = dataset_metadata, class_name = "Composite", seed=seed)
#         self.builders = None

    
#     def build(self) -> CompositeSignal:  
#         """Builds and returns CompositeSignal.

#         Returns:
#             CompositeSignal: CompositeSignal being built.
#         """
#         for b in self.builders:
#             self._signal.signals.append(b.build())

#         return self._signal

    
#     def reset(self) -> None:
#         """Resets CompositeSignalBuilder
#         """
#         self.builders = []
#         self._signal = CompositeSignal(self.dataset_metadata)

#     def add_builder(self, b: SignalBuilder) -> None:
#         self.builders.append(b)

#     def __len__(self):
#         return len(self.builders)

    
#     def __repr__(self):
#         return f"{self.__class__.__name__}(name={self.name})\n\tbuilders={self.builders}"
