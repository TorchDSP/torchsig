"""Transforms on Signal and DatasetSignal objects.
"""

__all__ = [
    "SignalTransform",
    "AWGN",
    "AddSlope",
    "AdditiveNoise",
    "AdjacentChannelInterference",
    "CarrierFrequencyDrift",
    "CarrierPhaseNoise",
    "CarrierPhaseOffset",
    "ChannelSwap",
    "CoarseGainChange",
    "CochannelInterference",
    "ComplexTo2D",
    "CutOut",
    "DigitalAGC"
    "Doppler",
    "Fading",
    "IntermodulationProducts",
    "IQImbalance",
    "InterleaveComplex",
    "NonlinearAmplifier",
    "PassbandRipple",
    "PatchShuffle",
    "Quantize",
    "RandomDropSamples",
    "Shadowing",
    "SpectralInversion",
    "Spectrogram",
    "SpectrogramDropSamples",
    "SpectrogramImage",
    "Spurs",
    "TimeReversal",
    "TimeVaryingNoise",
]

# TorchSig
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_types import Signal, DatasetSignal, SignalMetadata
import torchsig.transforms.functional as F
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    torchsig_real_data_type,
    low_pass
)

# Third Party
import numpy as np
import scipy as sp
from copy import copy

# Built-In
from typing import Tuple, List, Union



class SignalTransform(Transform):
    """SignalTransform parent class.
    """

    def update(self, signal: Union[Signal, DatasetSignal]) -> None:
        """Updates bookkeeping to Transforms metadata for Signals and DatsetSignals and checks signal valididty.
        Inherited classes should always call self.update() after performing transform operation (inside __call__).

        Args:
            signal (Union[Signal, DatasetSignal]): Transformed signal.

        """
        if isinstance(signal, DatasetSignal):
            for m in signal.metadata:
                m.applied_transforms.append(self)
        else:
            signal.metadata.applied_transforms.append(self)
        # signal.verify()

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        """Performs transforms.

        Args:
            signal (Signal): Signal to be transformed.

        Raises:
            NotImplementedError: Inherited classes must override this method.

        Returns:
            Signal: Transformed Signal.

        """
        raise NotImplementedError


class AWGN(SignalTransform):
    """Apply Additive White Gaussian Noise to signal.

    Attributes:
        noise_power_db (float): noise AWGN power in dB (absolute).
        measure (bool): Measure and update SNR metadata. Default to False.
    
    """
    def __init__(
        self,
        noise_power_db: float,
        measure: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_power_db = noise_power_db
        self.noise_power_linear = 10**(self.noise_power_db / 10)
        self.measure = measure

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        if self.measure:
            if isinstance(signal, DatasetSignal):
                for i, m in enumerate(signal.metadata):
                    start = m.start_in_samples
                    duration = m.duration_in_samples
                    stop = start + duration
                    snr_linear = 10 ** (m.snr_db / 10) 
                    
                    # update SNR assuming independent noise
                    total_power = np.sum(np.abs(signal.data[start:stop])**2)/duration
                    sig_power = total_power / (1 + 1/snr_linear)
                    noise_power = sig_power / snr_linear
                    new_snr = sig_power / (noise_power + self.noise_power_linear)
                    signal.metadata[i].snr_db = 10*np.log10(new_snr)
            else:
                # update SNR for full sampled band, assuming independent noise
                snr_linear = 10 ** (signal.metadata.snr_db / 10)
                total_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
                sig_power = total_power / (1 + 1/snr_linear)
                noise_power = sig_power / snr_linear
                new_snr = sig_power / (noise_power + self.noise_power_linear)
                signal.metadata.snr_db = 10*np.log10(new_snr)                

        signal.data = F.awgn(
            signal.data,
            noise_power_db = self.noise_power_db,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class AddSlope(SignalTransform):
    """Add the slope of each sample with its preceeding sample to itself.
    Creates a weak 0 Hz IF notch filtering effect.

    """
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.add_slope(signal.data) 
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class AdditiveNoise(SignalTransform):
    """Adds noise with specified properties to signal.

    Attributes:  
        power_range (Tuple[float, float]): Range bounds for interference power level (W). 
            Defaults to (0.01, 10.0).
        power_distribution (Callable[[], float]): Random draw of interference power.
        color (str): Noise color, supports 'white', 'pink', or 'red' noise frequency spectrum types. 
            Defaults to 'white'.
        continuous (bool): Sets noise to continuous (True) or impulsive (False). Defaults to True.
        measure (bool): Measure and update SNR metadata. Default to False.
    
    """
    def __init__(
        self,
        power_range: Tuple = (0.01, 10.0),
        color: str = 'white',
        continuous: bool = True,
        measure: bool = False,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.power_range = power_range
        self.power_distribution = self.get_distribution(self.power_range)
        self.color = color
        self.continuous = continuous
        self.measure = measure
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        add_noise_power = self.power_distribution()
        
        if self.measure:
            if isinstance(signal, DatasetSignal):
                for i, m in enumerate(signal.metadata):            
                    start = m.start_in_samples
                    duration = m.duration_in_samples
                    stop = start + duration
                    snr_linear = 10 ** (m.snr_db / 10) 
                    
                    # update SNR assuming independent noise
                    total_power = np.sum(np.abs(signal.data[start:stop])**2)/duration
                    sig_power = total_power / (1 + 1/snr_linear)
                    noise_power = sig_power / snr_linear
                    new_snr = sig_power / (noise_power + add_noise_power)
                    signal.metadata[i].snr_db = 10*np.log10(new_snr)
            else:
                # update SNR for full sampled band, assuming independent noise
                snr_linear = 10 ** (signal.metadata.snr_db / 10)
                total_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
                sig_power = total_power / (1 + 1/snr_linear)
                noise_power = sig_power / snr_linear
                new_snr = sig_power / (noise_power + add_noise_power)
                signal.metadata.snr_db = 10*np.log10(new_snr)           
            
        signal.data = F.additive_noise(
            data = signal.data,
            power = add_noise_power,
            color = self.color,
            continuous = self.continuous,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class AdjacentChannelInterference(SignalTransform):
    """Apply adjacent channel interference to signal.

    Attributes:  
        sample_rate (float): Sample rate (normalized). Defaults to 1.0.
        power_range (Tuple[float, float]): Range bounds for interference power level (W). 
            Defaults to (0.01, 10.0).
        power_distribution (Callable[[], float]): Random draw of interference power.
        center_frequency_range (Tuple[float, float]): Range bounds for interference center
            frequency (normalized). Defaults to (0.2, 0.3).
        center_frequency_distribution (Callable[[], float]): Random draw of interference power.        
        phase_sigma_range (Tuple[float, float]): Range bounds for interference phase sigma. 
            Defaults to (0.0, 1.0).
        phase_sigma_distribution (Callable[[], float]): Random draw of phase sigma. 
        time_sigma_range (Tuple[float, float]): Range bounds for interference time sigma. 
            Defaults to (0.0, 10.0).
        time_sigma_distribution (Callable[[], float]): Random draw of time sigma.      
        filter_weights (np.ndarray): Predefined baseband lowpass filter, fixed for all calls.
            Defaults to low_pass(0.125, 0.125, 1.0).
    
    """
    def __init__(
        self,
        sample_rate: float = 1.0,
        power_range: Tuple = (0.01, 10.0),
        center_frequency_range: Tuple = (0.2, 0.3),
        phase_sigma_range: Tuple = (0.0, 1.0),
        time_sigma_range: Tuple = (0.0, 10.0),
        filter_weights: np.ndarray = low_pass(0.125, 0.125, 1.0),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.power_range = power_range
        self.power_distribution = self.get_distribution(self.power_range)
        self.center_frequency_range = center_frequency_range
        self.center_frequency_distribution = self.get_distribution(self.center_frequency_range)
        self.phase_sigma_range = phase_sigma_range
        self.phase_sigma_distribution = self.get_distribution(self.phase_sigma_range)
        self.time_sigma_range = time_sigma_range
        self.time_sigma_distribution = self.get_distribution(self.time_sigma_range)
        self.filter_weights = filter_weights # predefined, fixed filter     
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.adjacent_channel_interference(
            data = signal.data,
            sample_rate = self.sample_rate,
            power = self.power_distribution(),
            center_frequency = self.center_frequency_distribution(),
            phase_sigma = self.phase_sigma_distribution(),
            time_sigma = self.time_sigma_distribution(),
            filter_weights = self.filter_weights,
            rng = self.random_generator
        )        
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class CarrierFrequencyDrift(SignalTransform):
    """Apply carrier frequency drift to signal.

    Attributes:
        drift_ppm_range (Tuple[float, float]): Drift in parts per million (ppm). Default (0.1,1).
        drift_ppm_distribution (Callable[[], float]): Random draw from drift_ppm_range distribution.
        
    """
    def __init__(
        self, 
        drift_ppm: Tuple[float, float] = (0.1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.drift_ppm = drift_ppm
        self.drift_ppm_distribution = self.get_distribution(self.drift_ppm,'log10')
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        drift_ppm = self.drift_ppm_distribution()

        signal.data = F.carrier_frequency_drift(
            data = signal.data, 
            drift_ppm = drift_ppm, 
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)       
        self.update(signal)
        return signal


class CarrierPhaseNoise(SignalTransform):
    """Apply Carrier phase noise to signal.

    Attributes:
       phase_noise_degrees (Tuple[float, float]): Range for phase noise (in degrees). Defaults to (0.25, 1).
       phase_noise_degrees_distribution (Callable[[], float]): Random draw from phase_noise_degrees distribution.
        
    """
    def __init__(
        self, 
        phase_noise_degrees: Tuple[float, float] = (0.25, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.phase_noise_degrees = phase_noise_degrees
        self.phase_noise_degrees_distribution = self.get_distribution(self.phase_noise_degrees)
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        phase_noise_degrees = self.phase_noise_degrees_distribution()

        signal.data = F.carrier_phase_noise(
            data = signal.data,
            phase_noise_degrees = phase_noise_degrees,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class CarrierPhaseOffset(SignalTransform):
    """Apply a randomized carrier phase offset to signal.

    The randomized phase offset is of the form exp(j * phi) where
    phi is in the range of 0 to 2pi radians. Real world effects such as
    time delays as a signal transits the air and others can cause
    such randomized phase offsets.

    The transform does not usually require any arguments due to its simplicity. 
    It is generally unrealistic to have a randomized phase  offset of a range less than 0 to 2pi.

    Attributes:
        phase_offset_range (Tuple[float, float]): Range bounds for phase offset (radians).
        phase_offset_distribution (Callable[[], float]): Random draw from phase offset distribution.
        
    """
    def __init__(
        self, 
        phase_offset_range: Tuple[float, float] = (0, 2*np.pi),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.phase_offset_range = phase_offset_range
        self.phase_offset_distribution = self.get_distribution(self.phase_offset_range)
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        phase_offset = self.phase_offset_distribution()

        signal.data = F.phase_offset(signal.data, phase_offset)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class ChannelSwap(SignalTransform):
    """Swaps the I and Q channels of complex input data.
    
    """
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.channel_swap(signal.data)
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # metadata: swapping I/Q channels creates a frequency mirroring
        if isinstance(signal, DatasetSignal):
            for m in signal.metadata:
                m.center_freq *= -1
        else:
            signal.metadata.center_freq *= -1

        self.update(signal)
        return signal


class CoarseGainChange(SignalTransform):
    """Apply a randomized instantaneous jump in signal magnitude to model an abrupt receiver gain change.

    Attributes:
        gain_change_db_range (Tuple): Sets the (min, max) gain change in dB.
        gain_change_db_distribution (Callable[[], float]): Random draw from gain_change_db distribution.
        
    """
    def __init__(
        self, 
        gain_change_db: Tuple[float, float] = (-20, 20),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gain_change_db_distribution = self.get_distribution(gain_change_db)
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        # select a gain value change from distribution
        gain_change_db = self.gain_change_db_distribution()
        # determine which samples gain change will be applied to. minimum index is 1, and maximum
        # index is second to last sample, such that at minimum the gain will be applied to one
        # sample or at maximum it will be applied to all less 1 samples. applying to zero samples
        # or to all samples does not have a practical effect for this specific transform.
        start_index = self.random_generator.integers(1,len(signal.data)-1)
        
        signal.data = F.coarse_gain_change(signal.data, gain_change_db, start_index)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class CochannelInterference(SignalTransform):
    """Apply cochannel interference to signal.

    Attributes:  
        power_range (Tuple[float, float]): Range bounds for interference power level (W). 
            Default (0.01, 10.0).
        power_distribution (float): Random draw of interference power.
        filter_weights (np.ndarray): Predefined baseband lowpass filter, fixed for all calls.
            Default low_pass(0.125, 0.125, 1.0).
        noise_color (str): Base noise color, supports 'white', 'pink', or 'red' noise 
            frequency spectrum types. Default 'white'.
        continuous (bool): Sets noise to continuous (True) or impulsive (False). Default True.
        measure (bool): Measure and update SNR metadata. Default to False.
    
    """
    def __init__(
        self,
        power_range: Tuple = (0.01, 10.0),
        filter_weights: np.ndarray = low_pass(0.125, 0.125, 1.0),
        color: str = 'white',
        continuous: bool = True,
        measure: bool = False,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.power_range = power_range
        self.power_distribution = self.get_distribution(self.power_range)
        self.filter_weights = filter_weights # predefined, fixed band limiting filter
        self.color = color
        self.continuous = continuous
        self.measure = measure
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        cochan_noise_power = self.power_distribution()

        if self.measure:
            if isinstance(signal, DatasetSignal):  
                for i, m in enumerate(signal.metadata):
                    snr_linear = 10 ** (m.snr_db / 10)
                    total_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
                    sig_power = total_power / (1 + 1/snr_linear)
                    noise_power = sig_power / snr_linear
                    new_snr = sig_power / (noise_power + cochan_noise_power)
                    signal.metadata[i].snr_db = 10*np.log10(new_snr)
            else:
                snr_linear = 10 ** (signal.metadata.snr_db / 10)
                total_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
                sig_power = total_power / (1 + 1/snr_linear)
                noise_power = sig_power / snr_linear
                new_snr = sig_power / (noise_power + self.noise_power_linear)
                signal.metadata.snr_db = 10*np.log10(new_snr)
       
        signal.data = F.cochannel_interference(
            data = signal.data,
            power = cochan_noise_power,
            filter_weights = self.filter_weights,
            color = self.color,
            continuous = self.continuous,
            rng = self.random_generator
        )        
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class ComplexTo2D(SignalTransform):
    """Converts IQ data to two channels (real and imaginary parts).
    """
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.complex_to_2d(signal.data)
        signal.data = signal.data.astype(torchsig_real_data_type)
        self.update(signal)
        return signal


class CutOut(SignalTransform):
    """Applies the CutOut transform operation in the time domain. The
    `cut_dur` input specifies how long the cut region should be, and the
    `cut_fill` input specifies what the cut region should be filled in with.
    Options for the cut type include: zeros, ones, low_noise, avg_noise, and
    high_noise. Zeros fills in the region with zeros; ones fills in the region
    with 1+1j samples; low_noise fills in the region with noise with -100dB
    power; avg_noise adds noise at power average of input data, effectively
    slicing/removing existing signals in the most RF realistic way of the
    options; and high_noise adds noise with 40dB power. If a list of multiple
    options are passed in, they are randomly sampled from.

    This transform is loosely based on
    `"Improved Regularization of Convolutional Neural Networks with Cutout" <https://arxiv.org/pdf/1708.04552v2.pdf>`_.

    Attributes:
        duration (float, list, tuple):
            cut_dur sets the duration of the region to cut out
            * If float, cut_dur is fixed at the value provided.
            * If list, cut_dur is any element in the list.
            * If tuple, cut_dur is in range of (tuple[0], tuple[1]).
        duration_distribution (Callable[[], float]): Random draw from duration distribution.
        cut_type (float, list, tuple):
            cut_fill sets the type of data to fill in the cut region with from
            the options: `zeros`, `ones`, `low_noise`, `avg_noise`, and `high_noise`
            * If list, cut_fill is any element in the list.
            * If str, cut_fill is fixed at the method provided.
        cut_type_distribution (Callable[[], str]): Random draw from cut_type distribution.

    """
    def __init__(
        self,
        duration = (0.01, 0.2),
        cut_type: List[str] = (["zeros", "ones", "low_noise", "avg_noise", "high_noise"]),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.duration = duration
        self.cut_type = cut_type

        self.duration_distribution = self.get_distribution(self.duration)
        self.cut_type_distribution = self.get_distribution(self.cut_type)
        
    def _determine_overlap(self, metadata: SignalMetadata, cut_start: float, cut_duration: float) -> str:
        signal_start = metadata.start
        signal_stop = metadata.stop

        cut_stop = cut_start + cut_duration
        
        # inside
        if signal_start > cut_start and signal_stop < cut_stop:
            return "inside"
        # left
        if signal_start < cut_start and signal_stop < cut_stop:
            return "left"
        # right
        if signal_start > cut_start and signal_stop > cut_stop:
            return "right"
        # split
        if signal_start < cut_start and signal_stop > cut_stop:
            return "split"
        
        # only remaining type
        return "outside"

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        cut_duration = self.duration_distribution()  
        cut_type = self.cut_type_distribution()
        cut_start = self.random_generator.uniform(low = 0.0, high = 1.0 - cut_duration)

        signal.data = F.cut_out(signal.data, cut_start, cut_duration, cut_type )
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # metadata 
        # CutOut can have complicated signal feature effects in practice.
        # Any other desired metadata updates should be made manually.

        # update start, duration
        cut_stop = cut_start + cut_duration

        if isinstance(signal, DatasetSignal):
            new_metadata = []
            for m in signal.metadata:
                overlap = self._determine_overlap(m, cut_start, cut_duration)
                if overlap == "left":
                    m.stop = cut_start
                elif overlap == "right":
                    m.start = cut_stop
                elif overlap == "split":
                    # left half = update current metadata
                    m.stop = cut_start
                    # right half = create new metadata
                    right_half_metadata = m.deepcopy()
                    right_half_metadata.start = cut_stop
                    new_metadata.append(right_half_metadata)
                elif overlap == "inside":
                    continue
                # else: signal outside of cut region
                new_metadata.append(m)
            
            signal.metadata = new_metadata
        else:
            overlap = self._determine_overlap(signal.metadata, cut_start, cut_duration)
            if overlap == "left":
                signal.metadata.stop = cut_start
            elif overlap == "right":
                signal.metadata.start = cut_stop
            elif overlap == "split":
                # left half = update current metadata
                signal.metadata.stop = cut_start
                # right half = create new metadata
                right_half_metadata = signal.metadata.deepcopy()
                right_half_metadata.start = cut_stop
                signal.metadata.start = right_half_metadata.start
            #elif overlap == "inside":
            # else: signal outside of cut region

        self.update(signal)
        return signal

class DigitalAGC(SignalTransform):
    """Automatic Gain Control performing sample-by-sample AGC algorithm.

    Attributes:
        initial_gain_db (float): Inital gain value in dB.
        alpha_smooth (float): Alpha for avergaing the measure signal level `level_n = level_n * alpha + level_n-1(1-alpha)`
        alpha_track (float): Amount to adjust gain when in tracking state.
        alpha_overflow (float): Amount to adjust gain when in overflow state `[level_db + gain_db] >= max_level`.
        alpha_acquire (float): Amount to adjust gain when in acquire state.
        track_range_db (float): dB range for operating in tracking state.

    """
    def __init__(
        self,
        initial_gain_db: Tuple[float] = (0,0),
        alpha_smooth: Tuple[float] = (1e-7, 1e-6),
        alpha_track: Tuple[float] = (1e-6, 1e-5),
        alpha_overflow: Tuple[float] = (1e-1, 3e-1),
        alpha_acquire: Tuple[float] = (1e-6, 1e-5),
        track_range_db: Tuple[float] = (0.5, 2),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_gain_db = initial_gain_db
        self.alpha_smooth = alpha_smooth
        self.alpha_track = alpha_track
        self.alpha_overflow = alpha_overflow
        self.alpha_acquire = alpha_acquire
        self.track_range_db = track_range_db

        self.initial_gain_db_distribution = self.get_distribution(self.initial_gain_db)
        self.alpha_smooth_distribution = self.get_distribution(self.alpha_smooth,'log10')
        self.alpha_track_distribution = self.get_distribution(self.alpha_track,'log10')
        self.alpha_overflow_distribution = self.get_distribution(self.alpha_track,'log10')
        self.alpha_acquire_distribution = self.get_distribution(self.alpha_acquire,'log10')
        self.track_range_db_distribution = self.get_distribution(self.track_range_db)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:

        initial_gain_db = self.initial_gain_db_distribution()
        alpha_smooth = self.alpha_smooth_distribution()
        alpha_track = self.alpha_track_distribution()
        alpha_overflow = self.alpha_overflow_distribution()
        alpha_acquire = self.alpha_acquire_distribution()
        track_range_db = self.track_range_db_distribution()

        # calculate derived parameters for AGC

        # create a copy of the input data since it may need to be
        # modified in order to avoid a log10(0)
        receive_signal = copy(signal.data)

        # get linear magnitude
        receive_signal_mag = np.abs(receive_signal)

        # find and replace all zeros
        zero_sample_index = np.where(receive_signal_mag == 0)[0]

        # calculate all other values
        non_zero_sample_index = np.setdiff1d(np.arange(0,len(receive_signal)),zero_sample_index)

        # calculate the non-zero minimum
        smallest_non_zero_value = np.min(receive_signal_mag[non_zero_sample_index])

        # scale to get the "epsilon" to replace the zero values
        epsilon = smallest_non_zero_value * 1e-6

        # replace zero values
        receive_signal[zero_sample_index] = epsilon

        # determine average range for input in dB
        receive_signal_db = np.log(np.abs(receive_signal))
        receive_signal_mean_db = np.mean(receive_signal_db)

        # calculate ranges for how to set AGC reference level.
        # it is set (roughly) within range of data to provide
        # a slight AGC effect
        ref_level_max_db = receive_signal_mean_db+5
        ref_level_min_db = receive_signal_mean_db-5

        # randomly select the reference level the AGC will set
        ref_level_db = self.random_generator.uniform(ref_level_min_db,ref_level_max_db)

        # define the operating bounds of the AGC
        low_level_db = ref_level_min_db-10
        high_level_db = ref_level_max_db+10

        signal.data = F.digital_agc(
            np.ascontiguousarray(signal.data, dtype=np.complex64),
            np.float64(initial_gain_db),
            np.float64(alpha_smooth),
            np.float64(alpha_track),
            np.float64(alpha_overflow),
            np.float64(alpha_acquire),
            np.float64(ref_level_db),
            np.float64(track_range_db),
            np.float64(low_level_db),
            np.float64(high_level_db)
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class Doppler(SignalTransform):
    """Apply a wideband Doppler effect to signal.

    Attributes:
        velocity_range (Tuple[float, float]): Relative velocity bounds in m/s. Default (0.0, 10.0)
        velocity_distribution (Callable[[], float]): Random draw from velocity distribution.
        propagation_speed (float): Wave speed in medium. Default 2.9979e8 m/s.
        sampling_rate (float): Data sampling rate. Default 1.0.
        
    """
    def __init__(
        self, 
        velocity_range: Tuple[float, float] = (0.0, 10.0),
        propagation_speed: float = 2.9979e8,
        sampling_rate: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.velocity_range = velocity_range
        self.velocity_distribution = self.get_distribution(self.velocity_range)
        self.propagation_speed = propagation_speed
        self.sampling_rate = sampling_rate
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        velocity = self.velocity_distribution()
        alpha = self.propagation_speed / (self.propagation_speed - velocity) # scaling factor

        signal.data = F.doppler(
            data = signal.data, 
            velocity = velocity, 
            propagation_speed = self.propagation_speed, 
            sampling_rate = self.sampling_rate
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        
        # adjust metadata by scaling factor
        if isinstance(signal, DatasetSignal):
            for m in signal.metadata:
                m.center_freq *= alpha
                m.bandwidth *= alpha               
        else:
            signal.metadata.center_freq *= alpha
            signal.metadata.bandwidth *= alpha

        self.update(signal)
        return signal


class Fading(SignalTransform): # slow, fast, block fading
    """Apply a channel fading model to signal.

    Note, currently only performs Rayleigh fading:
        A Rayleigh fading channel can be modeled as an FIR filter with Gaussian distributed
        taps which vary over time. The length of the filter determines the coherence bandwidth
        of the channel and is inversely proportional to the delay spread. The rate at which 
        the channel taps vary over time is related to the coherence time and this is inversely
        proportional to the maximum Doppler spread. This time variance is not included in this model.
    
    Attributes:
        coherence_bandwidth (optional): Coherence bandwidth sampling parameters. 
                Defaults to (0.01, 0.1).
        coherence_bandwidth_distribution (Callable[[], float]): Random draw from coherence bandwidth distribution.
        power_delay_profile (Tuple | List | np.ndarray, optional): A list of positive values 
            assigning power to taps of the channel model. When the number of taps exceeds the number 
            of items in the provided power_delay_profile, the list is linearly interpolated to 
            provide values for each tap of the channel. Defaults to (1, 1).

    """    
    def __init__(
        self, 
        coherence_bandwidth = (0.01, 0.1),
        power_delay_profile: Tuple | List | np.ndarray = (1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.coherence_bandwidth = coherence_bandwidth
        self.power_delay_profile = np.asarray(power_delay_profile)
        self.coherence_bandwidth_distribution = self.get_distribution(self.coherence_bandwidth)
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        coherence_bandwidth = self.coherence_bandwidth_distribution()

        signal.data = F.fading(
            data = signal.data, 
            coherence_bandwidth = coherence_bandwidth, 
            power_delay_profile = self.power_delay_profile,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class IntermodulationProducts(SignalTransform):
    """Apply simulated basebanded intermodulation products to a signal.

    Attributes:
        model_order (List[int]): The choices model order, 3rd or 5th order. Defaults to [3,5].
        coeffs_range (Tuple[float, float]): Range bounds for each intermodulation coefficient. 
            Defaults to (0., 1.).
        
    """
    def __init__(
        self,
        model_order: List[int] = [3, 5],
        coeffs_range: Tuple[float, float] = (1e-4, 1e-1),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.model_order = model_order
        self.model_order_distribution = self.get_distribution(self.model_order)
        self.coeffs_range = coeffs_range
        self.coeffs_distribution = self.get_distribution(self.coeffs_range,'log10')
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        # get randomized choice for model order
        model_order = self.model_order_distribution()

        # determine how many non-zero coefficients
        num_coefficients = len(np.arange(0,model_order,2))
        # pre-allocate with all zeros
        non_zero_coeffs = np.zeros(num_coefficients,dtype=torchsig_complex_data_type)
        # randomize each coefficient
        for index in range(num_coefficients):
            if (index == 0):
                non_zero_coeffs[index] = 1
            else:
                # calculate coefficient
                non_zero_coeffs[index] = self.coeffs_distribution()
                # run loop to ensure each coefficient must be smaller than the previous
                while (non_zero_coeffs[index] > non_zero_coeffs[index-1]):
                    non_zero_coeffs[index] = self.coeffs_distribution()

        # form the coeff array with appropriate zero-based weights
        coeffs = np.zeros(model_order,dtype=torchsig_complex_data_type)
        inner_index = 0
        for outer_index in range(model_order):
            if (np.mod(outer_index,2) == 0):
                coeffs[outer_index] = non_zero_coeffs[inner_index]
                inner_index += 1

        signal.data = F.intermodulation_products(
            data = signal.data,
            coeffs = coeffs      
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class IQImbalance(SignalTransform):
    """Apply a set of I/Q imbalance effects to a signal: amplitude, phase, and DC offset.

    Attributes:
        amplitude_imbalance (optional): Range bounds of IQ amplitude imbalance (dB).    
        amplitude_imbalance_distribution (Callable[[], float]): Random draw from amplitude imbalance distribution.
        phase_imbalance (optional): Range bounds of IQ phase imbalance (radians).        
        phase_imbalance (Callable[[], float]): Random draw from phase imbalance distribution.
        dc_offset_db (Tuple, optional): Range bounds for DC offset in relative power
        dc_offset_db_distribution (Callable[[], (float, float)]): Random draw from dc_offset_db distribution.
        dc_offset_phase_rads (Tuple, optional): Range bounds for phase of DC offset
        dc_offset_phase_rads_distribution (Callable[[], (float, float)]): Random draw from dc_offset_phase_rads distribution.
        
    """
    def __init__(
        self,
        amplitude_imbalance = (-1., 1.),
        phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0),
        dc_offset_db = (0,20),
        dc_offset_rads = (0, 2*np.pi),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.amplitude_imbalance = amplitude_imbalance
        self.phase_imbalance = phase_imbalance
        self.dc_offset_db = dc_offset_db
        self.dc_offset_rads = dc_offset_rads
        
        self.amplitude_imbalance_distribution = self.get_distribution(self.amplitude_imbalance)
        self.phase_imbalance_distribution = self.get_distribution(self.phase_imbalance)
        self.dc_offset_db_distribution = self.get_distribution(self.dc_offset_db)
        self.dc_offset_phase_rads_distribution = self.get_distribution(self.dc_offset_rads)

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:

        amplitude_imbalance = self.amplitude_imbalance_distribution()
        phase_imbalance = self.phase_imbalance_distribution()
        dc_offset_db = self.dc_offset_db_distribution()
        dc_offset_rads = self.dc_offset_phase_rads_distribution()

        signal.data = F.iq_imbalance(
            signal.data,
            amplitude_imbalance,
            phase_imbalance,
            dc_offset_db,
            dc_offset_rads
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class InterleaveComplex(SignalTransform):
    """Transforms a complex-valued array into a real-valued array of interleaved IQ values.
    """
    def __init__(
        self,
        **kwargs
    ):  
        super().__init__(**kwargs)

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:

        signal.data = F.interleave_complex(signal.data)
        signal.data = signal.data.astype(torchsig_real_data_type)
        self.update(signal)
        return signal


class NonlinearAmplifier(SignalTransform):
    """Apply a memoryless nonlinear amplifier model to a signal.

    Attributes:
        gain_range (Tuple[float, float]): Small-signal gain range (linear). Defaults to (1.0, 4.0).
        gain_distribution (Callable[[], float]): Random draw from gain distribution.
        psat_backoff_range (Tuple[float, float]): Psat backoff factor (linear) reflecting saturated
            power level (Psat) relative to input signal mean power. Defaults to (5.0, 20.0).
        psat_backoff_distribution (Callable[[], float]): Random draw from psat_backoff distribution.   
        phi_max_range (Tuple[float, float]): Maximum signal relative phase shift at 
            saturation power level (radians). Defaults to (-0.05, 0.05).
        phi_max_distribution (Callable[[], float]): Random draw from phi_max distribution. 
        phi_slope_range (Tuple[float, float]): Slope of relative phase shift response 
            (W/radians). Defaults to (-0.1, 0.01).
        phi_slope_distribution (Callable[[], float]): Random draw from phi_max distribution.
        auto_scale (bool): Automatically rescale output power to match full-scale peak 
            input power prior to transform, based on peak estimates. Default True.
        
    """
    def __init__(
        self,
        gain_range: Tuple[float, float] = (1.0, 1.0),
        psat_backoff_range: Tuple[float, float] = (5.0, 20.0),
        phi_max_range: Tuple[float, float] = (-0.05, 0.05),
        phi_slope_range: Tuple[float, float] = (-0.1, 0.1),
        auto_scale: bool = True,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.gain_range = gain_range
        self.gain_distribution = self.get_distribution(self.gain_range)
        self.psat_backoff_range = psat_backoff_range
        self.psat_backoff_distribution = self.get_distribution(self.psat_backoff_range)        
        self.phi_max_range = phi_max_range
        self.phi_max_distribution = self.get_distribution(self.phi_max_range)
        self.phi_slope_range = phi_slope_range
        self.phi_slope_distribution = self.get_distribution(self.phi_slope_range)
        self.auto_scale = auto_scale
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        gain = self.gain_distribution()
        psat_backoff = self.psat_backoff_distribution()
        phi_max = self.phi_max_distribution()
        phi_slope = self.phi_slope_distribution()

        signal.data = F.nonlinear_amplifier(
            data = signal.data,
            gain = gain,
            psat_backoff = psat_backoff,
            phi_max = phi_max,
            phi_slope = phi_slope,
            auto_scale = self.auto_scale
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class PassbandRipple(SignalTransform):
    """Models analog filter passband ripple response for a signal.

    Attributes:
        passband_ripple_db (float): Desired passband ripple in dB. Default 1.0 dB.
        cutoff (float): Passband cutoff frequency relative to Fs=1.0 sample rate. Default 0.25.
        order (int): Desired filter order, which drives number of ripples present within
            the passband. Default 5.
        numtaps (int): Number of taps in filter. Default 63.
        
    """
    def __init__(
        self, 
        passband_ripple_db: float = 1.0,
        cutoff: float = 0.25,
        order: int = 5,
        numtaps: int = 63,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.passband_ripple_db = passband_ripple_db
        self.cutoff = cutoff
        self.order = order
        self.numtaps = numtaps

        # design filter 
        b, a = sp.signal.cheby1(
            self.order, 
            self.passband_ripple_db, 
            self.cutoff, 
            fs=1.0, 
            btype='low'
        )
        _, h = sp.signal.dimpulse((b, a, 1/1.0), n=numtaps)
        self.fir_coeffs = h[0].squeeze()
    
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.passband_ripple(
            data = signal.data,
            filter_coeffs = self.fir_coeffs,
            normalize = True
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class PatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `"PatchShuffle Regularization" <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Attributes:
         patch_size (int, float, list, tuple):
            patch_size sets the size of each patch to shuffle
            * If int or float, patch_size is fixed at the value provided.
            * If list, patch_size is any element in the list.
            * If tuple, patch_size is in range of (tuple[0], tuple[1]).
        patch_size_distribution (Callable[[], int]): Random draw from patch_size distribution.
        shuffle_ratio (int, float, list, tuple):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If int or float, shuffle_ratio is fixed at the value provided.
            * If list, shuffle_ratio is any element in the list.
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1]).
        shuffle_ratio_distribution (Callable[[], float]): Random draw from shuffle_ratio distribution.

    """

    def __init__(
        self,
        patch_size = (3, 10),
        shuffle_ratio = (0.01, 0.05),
        **kwargs

    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.shuffle_ratio = shuffle_ratio
        self.patch_size_distribution = self.get_distribution(self.patch_size )
        self.shuffle_ratio_distribution = self.get_distribution(self.shuffle_ratio )
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        patch_size = self.patch_size_distribution()
        shuffle_ratio = self.shuffle_ratio_distribution()

        num_patches = int(signal.data.shape[0] / patch_size)
        num_to_shuffle = int(num_patches * shuffle_ratio)
        patches_to_shuffle = self.random_generator.choice(
            num_patches,
            replace=False,
            size=num_to_shuffle,
        )

        signal.data = F.patch_shuffle(
            signal.data,
            patch_size,
            patches_to_shuffle,
            self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # PatchShuffle can have complicated signal feature effects in practice.
        # Any desired metadata updates should be made manually.
        
        self.update(signal)
        return signal


class Quantize(SignalTransform):
    """Quantize signal I/Q samples into specified levels with a rounding method.

    Attributes:
        num_levels: Number of quantization levels.
        num_levels_distribution (Callable[[], int]): Random draw from num_levels distribution.
        rounding_mode (str, List[str]): Quantization rounding method. Must be 'floor'
                or 'ceiling'.
        rounding_mode_distribution (Callable[[], str]): Random draw from rounding_mode distribution.
    
    """
    def __init__(
        self,
        num_bits:  Tuple[int, int] = (6, 18),
        ref_level_adjustment_db:  Tuple[float, float] = (-10, 3),
        rounding_mode: List[str] = ['floor', 'ceiling'],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_bits = num_bits
        self.num_bits_distribution = self.get_distribution(self.num_bits)
        self.ref_level_adjustment_db = ref_level_adjustment_db
        self.ref_level_adjustment_db_distribution = self.get_distribution(self.ref_level_adjustment_db)
        self.rounding_mode = rounding_mode
        self.rounding_mode_distribution = self.get_distribution(self.rounding_mode)
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        num_bits = int(np.round(self.num_bits_distribution()))
        ref_level_adjustment_db = self.ref_level_adjustment_db_distribution()
        rounding_mode = self.rounding_mode_distribution()

        # apply quantization
        signal.data = F.quantize(
            data = signal.data,
            num_bits = num_bits,
            ref_level_adjustment_db = ref_level_adjustment_db,
            rounding_mode = rounding_mode,
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class RandomDropSamples(SignalTransform):
    """Randomly drop IQ samples from the input data of specified durations and
    with specified fill techniques:
        * `ffill` (front fill): replace drop samples with the last previous value.
        * `bfill` (back fill): replace drop samples with the next value.
        * `mean`: replace drop samples with the mean value of the full data.
        * `zero`: replace drop samples with zeros.

    Transform is based off of the
    `TSAug Dropout Transform <https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/dropout.py>`_.

    Attributes:
        drop_rate (int, float, list, tuple):
            drop_rate sets the rate at which to drop samples
            * If int or float, drop_rate is fixed at the value provided.
            * If list, drop_rate is any element in the list.
            * If tuple, drop_rate is in range of (tuple[0], tuple[1]).
        drop_rate_distribution (Callable[[], float]): Random draw from drop_rate distribution.
        size (int, float, list, tuple):
            size sets the size of each instance of dropped samples
            * If int or float, size is fixed at the value provided.
            * If list, size is any element in the list.
            * If tuple, size is in range of (tuple[0], tuple[1]).
        size_distribution (Callable[[], int]): Random draw from size distribution.
        fill (list, str):
            fill sets the method of how the dropped samples should be filled
            * If list, fill is any element in the list.
            * If str, fill is fixed at the method provided.
        fill_distribution (Callable[[], str]): Random draw from fill distribution.

    """
    def __init__(
        self,
        drop_rate = (0.01, 0.05),
        size = (1, 10),
        fill: List[str] = (["ffill", "bfill", "mean", "zero"]),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.size = size
        self.fill = fill

        self.drop_rate_distribution = self.get_distribution(self.drop_rate )
        self.size_distribution = self.get_distribution(self.size )
        self.fill_distribution = self.get_distribution(self.fill )
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        drop_rate = self.drop_rate_distribution()
        fill = self.fill_distribution()

        drop_instances = int(signal.data.shape[0] * drop_rate)
        if drop_instances < 1:
            return signal # drop no samples and return the input signal if we have randomly selected to drop zero samples
        drop_sizes = self.size_distribution(size=drop_instances).astype(int)
        drop_starts = self.random_generator.uniform(
            1,
            signal.data.shape[0] - max(drop_sizes) - 1,
            drop_instances
        ).astype(int)
        signal.data = F.drop_samples(signal.data, drop_starts, drop_sizes, fill)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class Shadowing(SignalTransform):
    """Apply channel shadowing effect across entire signal.

    Attributes:
        mean_db_range (Tuple[float, float]): Mean value range in dB. Defaults to (0.0, 4.0).
        mean_db_distribution (Callable[[], float]): Random draw from mean_db distribution.
        sigma_db_range (Tuple[float, float]): Sigma value range in dB. Defaults to (2.0, 6.0).
        sigma_db_distribution (Callable[[], float]): Random draw from sigma_db distribution.
        
    """
    def __init__(
        self, 
        mean_db_range:  Tuple[float, float] = (0.0, 4.0),
        sigma_db_range: Tuple[float, float] = (2.0, 6.0),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mean_db_range = mean_db_range
        self.mean_db_distribution = self.get_distribution(self.mean_db_range)
        self.sigma_db_range = sigma_db_range
        self.sigma_db_distribution = self.get_distribution(self.sigma_db_range)

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        mean_db = self.mean_db_distribution()
        sigma_db = self.sigma_db_distribution()

        signal.data = F.shadowing(
            data = signal.data,
            mean_db = mean_db,
            sigma_db = sigma_db,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class SpectralInversion(SignalTransform):
    """Inverts spectrum of complex signal data.
    """
    def __call__(self, signal: Signal) -> Signal:
        signal.data = F.spectral_inversion(signal.data)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        
        # metadata
        if isinstance(signal, DatasetSignal):
            for m in signal.metadata:
                m.center_freq *= -1
        else:
            signal.metadata.center_freq *= -1
        
        self.update(signal)
        return signal


class Spectrogram(SignalTransform):
    """Computes the spectogram of I/Q data.

    Attributes:
        fft_size (int): The FFT size (number of bins) in the spectrogram
    
    """
    def __init__(
        self,
        fft_size: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fft_size = fft_size
        # fft_stride is the number of data points to move or "hop" over when computing the next FF
        self.fft_stride = copy(fft_size)

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.spectrogram(
            signal.data, 
            self.fft_size, 
            self.fft_stride, 
        )
        signal.data = signal.data.astype(torchsig_real_data_type)
        self.update(signal)
        return signal


class SpectrogramDropSamples(SignalTransform):
    """Randomly drop samples from the input data of specified durations and
    with specified fill techniques:
        * `ffill` (front fill): replace drop samples with the last previous value
        * `bfill` (back fill): replace drop samples with the next value
        * `mean`: replace drop samples with the mean value of the full data
        * `zero`: replace drop samples with zeros
        * `low`: replace drop samples with low power samples
        * `min`: replace drop samples with the minimum of the absolute power
        * `max`: replace drop samples with the maximum of the absolute power
        * `ones`: replace drop samples with ones

    Transform is based off of the
    `TSAug Dropout Transform <https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/dropout.py>`_.

    Attributes:
        drop_rate (int, float, list, tuple):
            drop_rate sets the rate at which to drop samples
            * If int or float, drop_rate is fixed at the value provided.
            * If list, drop_rate is any element in the list.
            * If tuple, drop_rate is in range of (tuple[0], tuple[1]).
        drop_rate_distribution (Callable[[], float]): Random draw from drop_rate distribution.
        size (int, float, list, tuple)::
            size sets the size of each instance of dropped samples
            * If int or float, size is fixed at the value provided.
            * If list, size is any element in the list.
            * If tuple, size is in range of (tuple[0], tuple[1]).
        size_distribution (Callable[[], int]): Random draw from size distribution.
        fill (list, str):
            fill sets the method of how the dropped samples should be filled
            * If list, fill is any element in the list.
            * If str, fill is fixed at the method provided.
        fill_distribution (Callable[[], float]): Random draw from fill distribution.

    """
    def __init__(
        self,
        drop_rate = (0.001, 0.005),
        size = (1, 10),
        fill: List[str] = (
            ["ffill", "bfill", "mean", "zero", "low", "min", "max", "ones"]
        ),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.size = size
        self.fill = fill

        self.drop_rate_distribution = self.get_distribution(self.drop_rate )
        self.size_distribution = self.get_distribution(self.size )
        self.fill_distribution = self.get_distribution(self.fill )
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        drop_rate = self.drop_rate_distribution()
        fill = self.fill_distribution()
        drop_instances = int(signal.data.shape[0] * drop_rate)        
        drop_sizes = self.size_distribution(drop_instances).astype(int)
        if drop_instances < 1:
            return signal # if drop sizes is empty, just return signal
        if len(drop_sizes) > 0:
            drop_starts = self.random_generator.uniform(
                0, 
                signal.data.shape[0] - max(drop_sizes), 
                drop_instances
            ).astype(int)

            signal.data = F.spectrogram_drop_samples(
                signal.data,
                drop_starts,
                drop_sizes,
                fill,
            )
            signal.data = signal.data.astype(torchsig_real_data_type)
            
            # SpectrogramDropSamples can have complicated signal feature effects in practice.
            # Any desired metadata updates should be made manually.
            
            self.update(signal)
        
        return signal


class SpectrogramImage(SignalTransform):
    """Transforms signal to a spectrogram image.
    """

    def __init__(
        self,
        fft_size: int,
        black_hot: bool=True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs) 
        self.fft_size = fft_size
        self.fft_stride = fft_size #note: size = stride
        self.black_hot = black_hot

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.spectrogram_image(
            data = signal.data,
            fft_size = self.fft_size,
            fft_stride = self.fft_stride,
            black_hot = self.black_hot
        )
        self.update(signal)
        return signal


class TimeReversal(SignalTransform):
    """Apply a time reversal to the input. 
    
    Note that applying a time reversal inherently also applies a spectral inversion. 
    If a time-reversal without spectral inversion is desired, the `undo_spectral_inversion` 
    argument can be set to True. By setting this value to True, an additional, manual
    spectral inversion is applied to revert the time-reversal's inversion effect.

    Attributes:
        allow_spectral_inversion (bool | float, optional): Whether to allow spectral inversion. 
        as a time reversal side effect (True) or not (False). Defaults to True.
        * If bool, applied to all signals.
        * If float, applied as probability to add signals.

    """
    def __init__(
        self, 
        allow_spectral_inversion: bool | float = True,
        **kwargs
    ) -> None:
        if isinstance(allow_spectral_inversion, bool):
            self.allow_spectral_inversion = 1.0 if allow_spectral_inversion else 0.0
        elif isinstance(allow_spectral_inversion, float):
            self.allow_spectral_inversion = allow_spectral_inversion
        else:
            raise ValueError(f"Invalid type for allow_spectral_inversion {type(allow_spectral_inversion)}. Must be bool or float.")

        super().__init__(**kwargs) 

    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        signal.data = F.time_reversal(signal.data)
        signal.data = signal.data.astype(torchsig_complex_data_type)

        do_si = self.random_generator.random() > self.allow_spectral_inversion
        if do_si:
            signal.data = F.spectral_inversion(signal.data)

        # metadata
        num_data_samples = len(signal.data)
        if isinstance(signal, DatasetSignal):
            for i, m in enumerate(signal.metadata):
                original_stop = m.stop_in_samples
                signal.metadata[i].start_in_samples = num_data_samples - original_stop
                if not do_si:
                    signal.metadata[i].center_freq *= -1
        else:
            original_stop = signal.metadata.stop_in_samples
            signal.metadata.start_in_samples = num_data_samples - original_stop
            if not do_si:
                signal.metadata.center_freq *= -1
        
        self.update(signal)
        return signal
    

class TimeVaryingNoise(SignalTransform):
    """Add time-varying noise to signal regions.

    Attributes:
        noise_power_low: Range bounds for minimum noise power in dB.
        noise_power_low_distribution (Callable[[], float]): Random draw from noise_power_low distribution.
        noise_power_high: Range bounds for maximum noise power in dB.
        noise_power_high_distribution (Callable[[], float]): Random draw from noise_power_high distribution.
        inflections: Number of inflection points over IQ data.
        inflections_distribution (Callable[[], float]): Random draw from inflections distribution.
        random_regions (List | bool): Inflections points spread randomly (True) or not (False).
        random_regions_distribution (Callable[[], bool]): Random draw from random_regions distribution.
        
    """
    def __init__(
        self, 
        noise_power_low = (-80., -60.),
        noise_power_high = (-40., -20.),
        inflections = [int(0), int(10)],
        random_regions: List | bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_power_low = noise_power_low
        self.noise_power_high = noise_power_high
        self.inflections = inflections
        self.random_regions = random_regions

        self.noise_power_low_distribution = self.get_distribution(self.noise_power_low )
        self.noise_power_high_distribution = self.get_distribution(self.noise_power_high )
        self.inflections_distribution = self.get_distribution(self.inflections )
        self.random_regions_distribution = self.get_distribution(self.random_regions )
        
    def __call__(self, signal: Union[Signal, DatasetSignal]) -> Union[Signal, DatasetSignal]:
        noise_power_low = self.noise_power_low_distribution()
        noise_power_high = self.noise_power_high_distribution()
        inflections = self.inflections_distribution()
        random_regions = self.random_regions_distribution

        signal.data = F.time_varying_noise(
            signal.data,
            noise_power_low,
            noise_power_high,
            inflections,
            random_regions,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal




class Spurs(SignalTransform):
    """Simulates spurs by adding tones into the receive signal

    Attributes:
        num_spurs (int): The range of numbers of spurs to add. Defaults to (1,4).
        relative_power_db (Tuple[float]): The range of relative power for  the spurs. The power is
            relative to the noise floor. Defaults to (5,15).
    """
    def __init__(
        self,
        num_spurs: Tuple[int] = (1,4),
        relative_power_db: Tuple[float] = (5,15),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_spurs = num_spurs
        self.num_spurs_distribution = self.get_distribution(self.num_spurs)

        self.relative_power_db = relative_power_db
        self.relative_power_db_distribution = self.get_distribution(self.relative_power_db)
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        num_spurs = int(np.round(self.num_spurs_distribution()))

        sample_rate = 1

        # randomize the parameters for each spur
        relative_power_db = []
        center_freqs = []
        for index in range(num_spurs):
            # randomize the relative power in dB
            relative_power_db.append(  self.relative_power_db_distribution() )
            # determine the corresponding center frequency
            low_freq = -sample_rate/2
            high_freq = sample_rate/2
            center_freqs.append( self.random_generator.uniform(low_freq,high_freq) )

        # apply spurs
        signal.data = F.spurs(
            data = signal.data,
            sample_rate = sample_rate,
            center_freqs = center_freqs,
            relative_power_db = relative_power_db
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal

