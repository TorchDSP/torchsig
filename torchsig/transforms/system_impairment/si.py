import numpy as np
from copy import deepcopy
from scipy import signal as sp
from typing import Optional, Any, Union, List

from torchsig.utils.types import SignalData, SignalDescription
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.system_impairment import functional
from torchsig.transforms.functional import NumericParameter, IntParameter, FloatParameter
from torchsig.transforms.functional import to_distribution, uniform_continuous_distribution, uniform_discrete_distribution


class RandomTimeShift(SignalTransform):
    """Shifts tensor in the time dimension by shift samples. Zero-padding is applied to maintain input size.

    Args:
        shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling shift()
            * If int or float, shift is fixed at the value provided
            * If list, shift is any element in the list
            * If tuple, shift is in range of (tuple[0], tuple[1])

        interp_rate (:obj:`int`):
            Interpolation rate used by internal interpolation filter

        taps_per_arm (:obj:`int`):
            Number of taps per arm used in filter. More is slower, but more accurate.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Shift inputs by range of (-10, 20) samples with uniform distribution
        >>> transform = ST.RandomTimeShift(lambda size: np.random.uniform(-10, 20, size))
        >>> # Shift inputs by normally distributed time shifts
        >>> transform = ST.RandomTimeShift(lambda size: np.random.normal(0, 10, size))
        >>> # Shift by discrete set of values
        >>> transform = ST.RandomTimeShift(lambda size: np.random.choice([-10, 5, 10], size))
        >>> # Shift by 5 or 10
        >>> transform = ST.RandomTimeShift([5, 10])
        >>> # Shift by random amount between 5 and 10 with uniform probability
        >>> transform = ST.RandomTimeShift((5, 10))
        >>> # Shift fixed at 5 samples
        >>> transform = ST.RandomTimeShift(5)

    """
    def __init__(
        self,
        shift: NumericParameter = uniform_continuous_distribution(-10, 10),
        interp_rate: Optional[float] = 100,
        taps_per_arm: Optional[int] = 24
    ):
        super(RandomTimeShift, self).__init__()
        self.shift = to_distribution(shift, self.random_generator)
        self.interp_rate = interp_rate
        num_taps = int(taps_per_arm * interp_rate)
        self.taps = sp.firwin(num_taps, 1.0 / interp_rate, 1.0 / interp_rate / 4.0, scale=True) * interp_rate

    def __call__(self, data: Any) -> Any:
        shift = self.shift()
        integer_part, decimal_part = divmod(shift, 1)
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )            
            
            # Apply data transformation
            new_data.iq_data = functional.fractional_shift(
                data.iq_data,
                self.taps,
                self.interp_rate,
                -decimal_part  # this needed to be negated to be consistent with the previous implementation
            )
            new_data.iq_data = functional.time_shift(new_data.iq_data, int(integer_part))
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                new_signal_desc.start += (shift / data.iq_data.shape[0])
                new_signal_desc.stop += (shift / data.iq_data.shape[0])
                new_signal_desc.start = 0.0 if new_signal_desc.start < 0.0 else new_signal_desc.start
                new_signal_desc.stop = 1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            
        else:
            new_data = functional.fractional_shift(
                data,
                self.taps,
                self.interp_rate,
                -decimal_part  # this needed to be negated to be consistent with the previous implementation
            )
            new_data = functional.time_shift(new_data, int(integer_part))
        return new_data


class TimeCrop(SignalTransform):
    """Crops a tensor in the time dimension to the specified length. Optional
    crop techniques include: start, center, end, & random

    Args:
        crop_type (:obj:`str`):
            Type of cropping to perform. Options are: `start`, `center`, `end`,
            and `random`. `start` crops the input tensor such that the first 
            `length` samples are returned. `center` crops the input tensor such
            that the center `length` samples are returned. `end` crops the 
            input tensor such that the last `length` samples are returned.
            `random` crops randomly in the range `[0,length-1]`.

        length (:obj:`int`):
            Number of samples to include.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Crop inputs to first 256 samples
        >>> transform = ST.TimeCrop(crop_type='start', length=256)
        >>> # Crop inputs to center 512 samples
        >>> transform = ST.TimeCrop(crop_type='center', length=512)
        >>> # Crop inputs to last 1024 samples
        >>> transform = ST.TimeCrop(crop_type='end', length=1024)
        >>> # Randomly crop any 2048 samples from input
        >>> transform = ST.TimeCrop(crop_type='random', length=2048)

    """
    def __init__(
        self,
        crop_type: str = 'random',
        length: Optional[int] = 256
    ):
        super(TimeCrop, self).__init__()
        self.crop_type = crop_type
        self.length = length

    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        
        if iq_data.shape[0] == self.length:
            return data
        elif iq_data.shape[0] < self.length:
            raise ValueError('Input data length {} is less than requested length {}'.format(iq_data.shape[0], self.length))

        if self.crop_type == 'start':
            start = 0
        elif self.crop_type == 'end':
            start = iq_data.shape[0] - self.length
        elif self.crop_type == 'center':
            start = (iq_data.shape[0] - self.length) // 2
        elif self.crop_type == 'random':
            start = np.random.randint(0, iq_data.shape[0] - self.length)
        else:
            raise ValueError('Crop type must be: `start`, `center`, `end`, or `random`')
    
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )   
            
            # Perform data augmentation
            new_data.iq_data = functional.time_crop(iq_data, start, self.length)
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                original_start_sample = signal_desc.start * iq_data.shape[0]
                original_stop_sample = signal_desc.stop * iq_data.shape[0]
                new_start_sample = original_start_sample - start
                new_stop_sample = original_stop_sample - start
                new_signal_desc.start = new_start_sample / self.length
                new_signal_desc.stop = new_stop_sample / self.length
                new_signal_desc.start = 0.0 if new_signal_desc.start < 0.0 else new_signal_desc.start
                new_signal_desc.stop = 1.0 if new_signal_desc.stop > 1.0 else new_signal_desc.stop
                new_signal_desc.duration = new_signal_desc.stop - new_signal_desc.start
                new_signal_desc.num_iq_samples = self.length
                if new_signal_desc.start > 1.0 or new_signal_desc.stop < 0.0:
                    continue
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            
        else:
            new_data = functional.time_crop(data, start, self.length)
        return new_data
    
    
class TimeReversal(SignalTransform):
    """Applies a time reversal to the input. Note that applying a time reversal
    inherently also applies a spectral inversion. If a time-reversal without
    spectral inversion is desired, the `undo_spectral_inversion` argument 
    can be set to True. By setting this value to True, an additional, manual
    spectral inversion is applied to revert the time-reversal's inversion 
    effect.
    
    Args:
        undo_spectral_inversion (:obj:`bool`, :obj:`float`):
            * If bool, undo_spectral_inversion is always/never applied
            * If float, undo_spectral_inversion is a probability
    
    """
    def __init__(self, undo_spectral_inversion: Union[bool,float] = True):
        super(TimeReversal, self).__init__()
        if isinstance(undo_spectral_inversion, bool):
            self.undo_spectral_inversion = 1.0 if undo_spectral_inversion else 0.0
        else:
            self.undo_spectral_inversion = undo_spectral_inversion
        
    def __call__(self, data: Any) -> Any:
        spec_inversion_prob = np.random.rand()
        undo_spec_inversion = spec_inversion_prob <= self.undo_spectral_inversion
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Perform data augmentation
            new_data.iq_data = functional.time_reversal(data.iq_data)
            if undo_spec_inversion:
                # If spectral inversion not desired, reverse effect
                new_data.iq_data = functional.spectral_inversion(new_data.iq_data)
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                
                # Invert time labels
                original_start = new_signal_desc.start
                original_stop = new_signal_desc.stop
                new_signal_desc.start = original_stop * -1 + 1.0
                new_signal_desc.stop = original_start * -1 + 1.0
                
                if not undo_spec_inversion:
                    # Invert freq labels
                    original_lower = new_signal_desc.lower_frequency
                    original_upper = new_signal_desc.upper_frequency
                    new_signal_desc.lower_frequency = original_upper * -1
                    new_signal_desc.upper_frequency = original_lower * -1
                    new_signal_desc.center_frequency *= -1
                
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
                
        else:
            new_data = functional.time_reversal(data)
            if undo_spec_inversion:
                # If spectral inversion not desired, reverse effect
                new_data = functional.spectral_inversion(new_data)
        return new_data


class AmplitudeReversal(SignalTransform):
    """Applies an amplitude reversal to the input tensor by applying a value of
    -1 to each sample. Effectively the same as a static phase shift of pi
    
    """
    def __init__(self):
        super(AmplitudeReversal, self).__init__()
        
    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Perform data augmentation
            new_data.iq_data = functional.amplitude_reversal(data.iq_data)
                
        else:
            new_data = functional.amplitude_reversal(data)
        return new_data
    
    
class RandomFrequencyShift(SignalTransform):
    """Shifts each tensor in freq by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling freq_shift()
            * If int or float, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = ST.RandomFrequencyShift(lambda size: np.random.uniform(-.25, .25, size))
        >>> # Frequency shift inputs always fs/10
        >>> transform = ST.RandomFrequencyShift(lambda size: np.random.choice([.1], size))
        >>> # Frequency shift inputs with normal distribution with stdev .1
        >>> transform = ST.RandomFrequencyShift(lambda size: np.random.normal(0, .1, size))
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = ST.RandomFrequencyShift((-.25, .25))
        >>> # Frequency shift all inputs by fs/10
        >>> transform = ST.RandomFrequencyShift(.1)
        >>> # Frequency shift inputs with either -fs/4 or fs/4 (discrete)
        >>> transform = ST.RandomFrequencyShift([-.25, .25])

    """
    def __init__(
        self,
        freq_shift: NumericParameter = uniform_continuous_distribution(-.5, .5)
    ):
        super(RandomFrequencyShift, self).__init__()
        self.freq_shift = to_distribution(freq_shift, self.random_generator)

    def __call__(self, data: Any) -> Any:
        freq_shift = self.freq_shift()
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            avoid_aliasing = False
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                # Check bounds for partial signals
                new_signal_desc.lower_frequency = -0.5 if new_signal_desc.lower_frequency < -0.5 else new_signal_desc.lower_frequency
                new_signal_desc.upper_frequency = 0.5 if new_signal_desc.upper_frequency > 0.5 else new_signal_desc.upper_frequency
                new_signal_desc.bandwidth = new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                new_signal_desc.center_frequency = new_signal_desc.lower_frequency + new_signal_desc.bandwidth * 0.5
                
                # Shift freq descriptions
                new_signal_desc.lower_frequency += freq_shift
                new_signal_desc.upper_frequency += freq_shift
                new_signal_desc.center_frequency += freq_shift
                
                # Check bounds for aliasing
                if new_signal_desc.lower_frequency >= 0.5 or new_signal_desc.upper_frequency <= -0.5:
                    avoid_aliasing = True
                    continue
                if new_signal_desc.lower_frequency < -0.45 or new_signal_desc.upper_frequency > 0.45:
                    avoid_aliasing = True
                new_signal_desc.lower_frequency = -0.5 if new_signal_desc.lower_frequency < -0.5 else new_signal_desc.lower_frequency
                new_signal_desc.upper_frequency = 0.5 if new_signal_desc.upper_frequency > 0.5 else new_signal_desc.upper_frequency
                
                # Update bw & fc
                new_signal_desc.bandwidth = new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                new_signal_desc.center_frequency = new_signal_desc.lower_frequency + new_signal_desc.bandwidth * 0.5
                
                # Append SignalDescription to list
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
            
            # Apply data augmentation
            if avoid_aliasing:
                # If any potential aliasing detected, perform shifting at higher sample rate
                new_data.iq_data = functional.freq_shift_avoid_aliasing(data.iq_data, freq_shift)
            else:
                # Otherwise, use faster freq shifter
                new_data.iq_data = functional.freq_shift(data.iq_data, freq_shift)
            
        else:
            new_data = functional.freq_shift(data, freq_shift)
        return new_data


class RandomDelayedFrequencyShift(SignalTransform):
    """Apply a delayed frequency shift to the input data
    
    Args:
         start_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            start_shift sets the start time of the delayed shift
            * If Callable, produces a sample by calling start_shift()
            * If int, start_shift is fixed at the value provided
            * If list, start_shift is any element in the list
            * If tuple, start_shift is in range of (tuple[0], tuple[1])
            
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            freq_shift sets the translation along the freq-axis
            * If Callable, produces a sample by calling freq_shift()
            * If int, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])
    
    """
    def __init__(
        self,
        start_shift: IntParameter = uniform_continuous_distribution(0.1,0.9),
        freq_shift: IntParameter = uniform_continuous_distribution(-0.2,0.2),
    ):
        super(RandomDelayedFrequencyShift, self).__init__()
        self.start_shift = to_distribution(start_shift, self.random_generator)
        self.freq_shift = to_distribution(freq_shift, self.random_generator)

    def __call__(self, data: Any) -> Any:
        start_shift = self.start_shift()
        # Randomly generate a freq shift that is not near the original fc
        freq_shift = 0
        while freq_shift < 0.05 and freq_shift > -0.05:
            freq_shift = self.freq_shift()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            new_data.iq_data = data.iq_data
            num_iq_samples = data.iq_data.shape[0]

            # Setup new SignalDescription object
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            avoid_aliasing = False
            for signal_desc in signal_description:
                new_signal_desc_first_seg = deepcopy(signal_desc)
                new_signal_desc_sec_seg = deepcopy(signal_desc)
                # Check bounds for partial signals
                new_signal_desc_first_seg.lower_frequency = -0.5 if new_signal_desc_first_seg.lower_frequency < -0.5 else new_signal_desc_first_seg.lower_frequency
                new_signal_desc_first_seg.upper_frequency = 0.5 if new_signal_desc_first_seg.upper_frequency > 0.5 else new_signal_desc_first_seg.upper_frequency
                new_signal_desc_first_seg.bandwidth = new_signal_desc_first_seg.upper_frequency - new_signal_desc_first_seg.lower_frequency
                new_signal_desc_first_seg.center_frequency = new_signal_desc_first_seg.lower_frequency + new_signal_desc_first_seg.bandwidth * 0.5
                
                # Update time for original segment if present in segment and add to list
                if new_signal_desc_first_seg.start < start_shift:
                    new_signal_desc_first_seg.stop = start_shift if new_signal_desc_first_seg.stop > start_shift else new_signal_desc_first_seg.stop
                    new_signal_desc_first_seg.duration = new_signal_desc_first_seg.stop - new_signal_desc_first_seg.start
                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc_first_seg)                

                # Begin second segment processing
                new_signal_desc_sec_seg.lower_frequency = -0.5 if new_signal_desc_sec_seg.lower_frequency < -0.5 else new_signal_desc_sec_seg.lower_frequency
                new_signal_desc_sec_seg.upper_frequency = 0.5 if new_signal_desc_sec_seg.upper_frequency > 0.5 else new_signal_desc_sec_seg.upper_frequency
                new_signal_desc_sec_seg.bandwidth = new_signal_desc_sec_seg.upper_frequency - new_signal_desc_sec_seg.lower_frequency
                new_signal_desc_sec_seg.center_frequency = new_signal_desc_sec_seg.lower_frequency + new_signal_desc_sec_seg.bandwidth * 0.5
                    
                # Update freqs for next segment
                new_signal_desc_sec_seg.lower_frequency += freq_shift
                new_signal_desc_sec_seg.upper_frequency += freq_shift
                new_signal_desc_sec_seg.center_frequency += freq_shift

                # Check bounds for aliasing
                if new_signal_desc_sec_seg.lower_frequency >= 0.5 or new_signal_desc_sec_seg.upper_frequency <= -0.5:
                    avoid_aliasing = True
                    continue
                if new_signal_desc_sec_seg.lower_frequency < -0.45 or new_signal_desc_sec_seg.upper_frequency > 0.45:
                    avoid_aliasing = True
                new_signal_desc_sec_seg.lower_frequency = -0.5 if new_signal_desc_sec_seg.lower_frequency < -0.5 else new_signal_desc_sec_seg.lower_frequency
                new_signal_desc_sec_seg.upper_frequency = 0.5 if new_signal_desc_sec_seg.upper_frequency > 0.5 else new_signal_desc_sec_seg.upper_frequency

                # Update bw & fc
                new_signal_desc_sec_seg.bandwidth = new_signal_desc_sec_seg.upper_frequency - new_signal_desc_sec_seg.lower_frequency
                new_signal_desc_sec_seg.center_frequency = new_signal_desc_sec_seg.lower_frequency + new_signal_desc_sec_seg.bandwidth * 0.5
                
                # Update time for shifted segment if present in segment and add to list
                if new_signal_desc_sec_seg.stop > start_shift:
                    new_signal_desc_sec_seg.start = start_shift if new_signal_desc_sec_seg.start < start_shift else new_signal_desc_sec_seg.start
                    new_signal_desc_sec_seg.stop = new_signal_desc_sec_seg.stop
                    new_signal_desc_sec_seg.duration = new_signal_desc_sec_seg.stop - new_signal_desc_sec_seg.start
                    # Append SignalDescription to list
                    new_signal_description.append(new_signal_desc_sec_seg)
                    
            # Update with the new SignalDescription
            new_data.signal_description = new_signal_description

            # Perform augmentation
            if avoid_aliasing:
                # If any potential aliasing detected, perform shifting at higher sample rate
                new_data.iq_data[int(start_shift*num_iq_samples):] = functional.freq_shift_avoid_aliasing(
                    data.iq_data[int(start_shift*num_iq_samples):], 
                    freq_shift
                )
            else:
                # Otherwise, use faster freq shifter
                new_data.iq_data[int(start_shift*num_iq_samples):] = functional.freq_shift(
                    data.iq_data[int(start_shift*num_iq_samples):], 
                    freq_shift
                )
            
        return new_data
    
    
class LocalOscillatorDrift(SignalTransform):
    """LocalOscillatorDrift is a transform modelling a local oscillator's drift in frequency by
    a random walk in frequency.

        Args:
            max_drift (FloatParameter, optional):
                [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
            max_drift_rate (FloatParameter, optional):
                [description]. Defaults to uniform_continuous_distribution(0.001,0.01).

    """
    def __init__(
        self,
        max_drift: FloatParameter = uniform_continuous_distribution(0.005,0.015),
        max_drift_rate: FloatParameter = uniform_continuous_distribution(0.001,0.01),
        **kwargs
    ):
        super(LocalOscillatorDrift, self).__init__(**kwargs)
        self.max_drift = to_distribution(max_drift, self.random_generator)
        self.max_drift_rate = to_distribution(max_drift_rate, self.random_generator)

    def __call__(self, data: Any) -> Any:
        max_drift = self.max_drift()
        max_drift_rate = self.max_drift_rate()
        
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        
        # Apply drift as a random walk.
        random_walk = self.random_generator.choice([-1, 1], size=iq_data.shape[0])

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        frequency = np.cumsum(random_walk) * max_drift_rate / np.sqrt(iq_data.shape[0])

        # Every time frequency hits max_drift, reset to zero.
        while np.argmax(np.abs(frequency) > max_drift):
            idx = np.argmax(np.abs(frequency) > max_drift)
            offset = max_drift if frequency[idx] < 0 else -max_drift
            frequency[idx:] += offset
        min_offset = min(frequency)
        max_offset = max(frequency)

        complex_phase = np.exp(2j*np.pi*np.cumsum(frequency))
        iq_data = iq_data*complex_phase
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                
                # Expand frequency labels
                new_signal_desc.lower_frequency += min_offset
                new_signal_desc.upper_frequency += max_offset
                new_signal_desc.bandwidth = new_signal_desc.upper_frequency - new_signal_desc.lower_frequency
                
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
            new_data.iq_data = iq_data
        else:
            new_data = iq_data
            
        return new_data
    
    
class GainDrift(SignalTransform):
    """GainDrift is a transform modelling a front end gain controller's drift in gain by
    a random walk in gain values.

        Args:
            max_drift (FloatParameter, optional):
                [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
            min_drift (FloatParameter, optional):
                [description]. Defaults to uniform_continuous_distribution(0.005,0.015).
            drift_rate (FloatParameter, optional):
                [description]. Defaults to uniform_continuous_distribution(0.001,0.01).

    """
    def __init__(
        self,
        max_drift: FloatParameter = uniform_continuous_distribution(0.005,0.015),
        min_drift: FloatParameter = uniform_continuous_distribution(0.005,0.015),
        drift_rate: FloatParameter = uniform_continuous_distribution(0.001,0.01),
        **kwargs
    ):
        super(GainDrift, self).__init__(**kwargs)
        self.max_drift = to_distribution(max_drift, self.random_generator)
        self.min_drift = to_distribution(min_drift, self.random_generator)
        self.drift_rate = to_distribution(drift_rate, self.random_generator)

    def __call__(self, data: Any) -> Any:
        max_drift = self.max_drift()
        min_drift = self.min_drift()
        drift_rate = self.drift_rate()
        
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        
        # Apply drift as a random walk.
        random_walk = self.random_generator.choice([-1, 1], size=iq_data.shape[0])

        # limit rate of change to at most 1/max_drift_rate times the length of the data sample
        gain = np.cumsum(random_walk) * drift_rate / np.sqrt(iq_data.shape[0])

        # Every time gain hits max_drift, reset to zero
        while np.argmax(gain > max_drift):
            idx = np.argmax(gain > max_drift)
            offset = gain[idx] - max_drift
            gain[idx:] -= offset
        # Every time gain hits min_drift, reset to zero
        while np.argmax(gain < min_drift):
            idx = np.argmax(gain < min_drift)
            offset = min_drift - gain[idx]
            gain[idx:] += offset
        iq_data = iq_data * (1 + gain)
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            new_data.iq_data = iq_data
        else:
            new_data = iq_data
            
        return new_data
    
    
class AutomaticGainControl(SignalTransform):
    """Automatic gain control (AGC) implementation

    Args:
        rand_scale (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Random scaling of alpha values
            * If Callable, produces a sample by calling rand_scale()
            * If int or float, rand_scale is fixed at the value provided
            * If list, rand_scale is any element in the list
            * If tuple, rand_scale is in range of (tuple[0], tuple[1])

        initial_gain_db (:obj:`float`):
            Initial gain value in linear units

        alpha_smooth (:obj:`float`):
            Alpha for averaging the measured signal level level_n = level_n*alpha + level_n-1*(1 - alpha)

        alpha_track (:obj:`float`):
            Amount by which to adjust gain when in tracking state

        alpha_overflow (:obj:`float`):
            Amount by which to adjust gain when in overflow state [level_db + gain_db] >= max_level

        alpha_acquire (:obj:`float`):
            Amount by which to adjust gain when in acquire state abs([ref_level_db - level_db - gain_db]) >= track_range_db

        ref_level_db (:obj:`float`):
            Level to which we intend to adjust gain to achieve

        track_range_db (:obj:`float`):
            Range from ref_level_linear for which we can deviate before going into acquire state

        low_level_db (:obj:`float`):
            Level below which we disable AGC

        high_level_db (:obj:`float`):
            Level above which we go into overflow state
            
    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.AutomaticGainControl(rand_scale=(1.0,10.0))

    """
    def __init__(
        self, 
        rand_scale: FloatParameter = uniform_continuous_distribution(1.0,10.0),
        initial_gain_db: float = 0.0,
        alpha_smooth: float = 0.00004,
        alpha_overflow: float = 0.3,
        alpha_track: float = 0.0004,
        alpha_acquire: float = 0.04,
        ref_level_db: float = 0.0, 
        track_range_db: float = 1.0, 
        low_level_db: float = -80.0, 
        high_level_db: float = 6.0,
    ):
        super(AutomaticGainControl, self).__init__()
        self.rand_scale = to_distribution(rand_scale, self.random_generator)
        self.initial_gain_db = initial_gain_db
        self.alpha_smooth = alpha_smooth
        self.alpha_overflow = alpha_overflow
        self.alpha_track = alpha_track
        self.alpha_acquire = alpha_acquire
        self.ref_level_db = ref_level_db
        self.track_range_db = track_range_db
        self.low_level_db = low_level_db
        self.high_level_db = high_level_db

    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        rand_scale = self.rand_scale()
        alpha_acquire = np.random.uniform(self.alpha_acquire / rand_scale, self.alpha_acquire * rand_scale, 1)
        alpha_overflow = np.random.uniform(self.alpha_overflow /  rand_scale, self.alpha_overflow * rand_scale, 1)
        alpha_track = np.random.uniform(self.alpha_track /  rand_scale, self.alpha_track * rand_scale, 1)
        alpha_smooth = np.random.uniform(self.alpha_smooth /  rand_scale, self.alpha_smooth * rand_scale, 1)

        ref_level_db = np.random.uniform(-.5 + self.ref_level_db, .5 + self.ref_level_db, 1)
        
        iq_data = functional.agc(
            np.ascontiguousarray(iq_data, dtype=np.complex64),
            np.float64(self.initial_gain_db),
            np.float64(alpha_smooth),
            np.float64(alpha_track),
            np.float64(alpha_overflow),
            np.float64(alpha_acquire),
            np.float64(ref_level_db),
            np.float64(self.track_range_db),
            np.float64(self.low_level_db),
            np.float64(self.high_level_db)
        )
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            new_data.iq_data = iq_data
        else:
            new_data = iq_data
            
        return new_data
    
    
class IQImbalance(SignalTransform):
    """Applies various types of IQ imbalance to a tensor

    Args:
        iq_amplitude_imbalance_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_amplitude_imbalance()
            * If int or float, iq_amplitude_imbalance is fixed at the value provided
            * If list, iq_amplitude_imbalance is any element in the list
            * If tuple, iq_amplitude_imbalance is in range of (tuple[0], tuple[1])

        iq_phase_imbalance (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_phase_imbalance()
            * If int or float, iq_phase_imbalance is fixed at the value provided
            * If list, iq_phase_imbalance is any element in the list
            * If tuple, iq_phase_imbalance is in range of (tuple[0], tuple[1])

        iq_dc_offset_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling iq_dc_offset()
            * If int or float, iq_dc_offset_db is fixed at the value provided
            * If list, iq_dc_offset is any element in the list
            * If tuple, iq_dc_offset is in range of (tuple[0], tuple[1])

    Note:
        For more information about IQ imbalance in RF systems, check out
        https://www.mathworks.com/help/comm/ref/iqimbalance.html

    Example:
        >>> import torchsig.transforms as ST
        >>> # IQ imbalance with default params
        >>> transform = ST.IQImbalance()

    """
    def __init__(
            self,
            iq_amplitude_imbalance_db: NumericParameter = (0, 3),
            iq_phase_imbalance: NumericParameter = (-np.pi*1.0/180.0, np.pi*1.0/180.0),
            iq_dc_offset_db: NumericParameter = (-.1, .1)
    ):
        super(IQImbalance, self).__init__()
        self.amp_imbalance = to_distribution(iq_amplitude_imbalance_db, self.random_generator)
        self.phase_imbalance = to_distribution(iq_phase_imbalance, self.random_generator)
        self.dc_offset = to_distribution(iq_dc_offset_db, self.random_generator)

    def __call__(self, data: Any) -> Any:
        amp_imbalance = self.amp_imbalance()
        phase_imbalance = self.phase_imbalance()
        dc_offset = self.dc_offset()

        if isinstance(data, SignalData):
            data.iq_data = functional.iq_imbalance(
                data.iq_data,
                amp_imbalance,
                phase_imbalance,
                dc_offset
            )
        else:
            data = functional.iq_imbalance(
                data,
                amp_imbalance,
                phase_imbalance,
                dc_offset
            )
        return data

    
class RollOff(SignalTransform):
    """Applies a band-edge RF roll-off effect simulating front end filtering
    
    Args:
        low_freq (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling low_freq()
            * If int or float, low_freq is fixed at the value provided
            * If list, low_freq is any element in the list
            * If tuple, low_freq is in range of (tuple[0], tuple[1])
            
        upper_freq (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling upper_freq()
            * If int or float, upper_freq is fixed at the value provided
            * If list, upper_freq is any element in the list
            * If tuple, upper_freq is in range of (tuple[0], tuple[1])
            
        low_cut_apply (:obj:`float`):
            Probability that the low frequency provided above is applied
            
        upper_cut_apply (:obj:`float`):
            Probability that the upper frequency provided above is applied
            
        order (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling order()
            * If int or float, order is fixed at the value provided
            * If list, order is any element in the list
            * If tuple, order is in range of (tuple[0], tuple[1])
    
    """
    def __init__(
        self,
        low_freq: NumericParameter = (0.00, 0.05),
        upper_freq: NumericParameter = (0.95, 1.00),
        low_cut_apply: float = 0.5,
        upper_cut_apply: float = 0.5,
        order: NumericParameter = (6, 20),
    ):
        super(RollOff, self).__init__()
        self.low_freq = to_distribution(low_freq, self.random_generator)
        self.upper_freq = to_distribution(upper_freq, self.random_generator)
        self.low_cut_apply = low_cut_apply
        self.upper_cut_apply = upper_cut_apply
        self.order = to_distribution(order, self.random_generator)
        
    def __call__(self, data: Any) -> Any:
        low_freq = self.low_freq() if np.random.rand() < self.low_cut_apply else 0.0
        upper_freq = self.upper_freq() if np.random.rand() < self.upper_cut_apply else 1.0
        order = self.order()
        if isinstance(data, SignalData):
            data.iq_data = functional.roll_off(data.iq_data, low_freq, upper_freq, int(order))
        else:
            data = functional.roll_off(data, low_freq, upper_freq, int(order))
        return data

    
class AddSlope(SignalTransform):
    """Add the slope of each sample with its preceeding sample to itself.
    Creates a weak 0 Hz IF notch filtering effect
    
    """
    def __init__(self, **kwargs):
        super(AddSlope, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Apply data augmentation
            new_data.iq_data = functional.add_slope(data.iq_data)
            
        else:
            new_data = functional.add_slope(data)
        return new_data
    
    
class SpectralInversion(SignalTransform):
    """Applies a spectral inversion
    
    """
    def __init__(self):
        super(SpectralInversion, self).__init__()
        
    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Perform data augmentation
            new_data.iq_data = functional.spectral_inversion(data.iq_data)
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                
                # Invert frequency labels
                original_lower = new_signal_desc.lower_frequency
                original_upper = new_signal_desc.upper_frequency
                new_signal_desc.lower_frequency = original_upper * -1
                new_signal_desc.upper_frequency = original_lower * -1
                new_signal_desc.center_frequency *= -1
                
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
                
        else:
            new_data = functional.spectral_inversion(data)
        return new_data
    
    
class ChannelSwap(SignalTransform):
    """Transform that swaps the I and Q channels of complex input data
    
    """
    def __init__(self):
        super(ChannelSwap, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                
                # Invert frequency labels
                original_lower = new_signal_desc.lower_frequency
                original_upper = new_signal_desc.upper_frequency
                new_signal_desc.lower_frequency = original_upper * -1
                new_signal_desc.upper_frequency = original_lower * -1
                new_signal_desc.center_frequency *= -1
                
                new_signal_description.append(new_signal_desc)
                
            new_data.signal_description = new_signal_description
            
            # Perform data augmentation
            new_data.iq_data = functional.channel_swap(data.iq_data)
                
        else:
            new_data = functional.channel_swap(data)
        return new_data
    
    
class RandomMagRescale(SignalTransform):
    """Randomly apply a magnitude rescaling, emulating a change in a receiver's
    gain control
    
    Args:
         start (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            start sets the time when the rescaling kicks in
            * If Callable, produces a sample by calling start()
            * If int or float, start is fixed at the value provided
            * If list, start is any element in the list
            * If tuple, start is in range of (tuple[0], tuple[1])
            
        scale (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            scale sets the magnitude of the rescale
            * If Callable, produces a sample by calling scale()
            * If int or float, scale is fixed at the value provided
            * If list, scale is any element in the list
            * If tuple, scale is in range of (tuple[0], tuple[1])
    
    """
    def __init__(
        self,
        start: NumericParameter = uniform_continuous_distribution(0.0,0.9),
        scale: NumericParameter = uniform_continuous_distribution(-4.0,4.0),
    ):
        super(RandomMagRescale, self).__init__()
        self.start = to_distribution(start, self.random_generator)
        self.scale = to_distribution(scale, self.random_generator)

    def __call__(self, data: Any) -> Any:
        start = self.start()
        scale = self.scale()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Perform data augmentation
            new_data.iq_data = functional.mag_rescale(data.iq_data, start, scale)
                
        else:
            new_data = functional.mag_rescale(data, start, scale)
        return new_data
    
    
class RandomDropSamples(SignalTransform):
    """Randomly drop IQ samples from the input data of specified durations and
    with specified fill techniques:
    * `ffill` (front fill): replace drop samples with the last previous value
    * `bfill` (back fill): replace drop samples with the next value
    * `mean`: replace drop samples with the mean value of the full data
    * `zero`: replace drop samples with zeros
       
    Transform is based off of the
    `TSAug Dropout Transform <https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/dropout.py>`_.

    Args:
         drop_rate (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            drop_rate sets the rate at which to drop samples
            * If Callable, produces a sample by calling drop_rate()
            * If int or float, drop_rate is fixed at the value provided
            * If list, drop_rate is any element in the list
            * If tuple, drop_rate is in range of (tuple[0], tuple[1])
            
        size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            size sets the size of each instance of dropped samples
            * If Callable, produces a sample by calling size()
            * If int or float, size is fixed at the value provided
            * If list, size is any element in the list
            * If tuple, size is in range of (tuple[0], tuple[1])
            
        fill (:py:class:`~Callable`, :obj:`list`, :obj:`str`):
            fill sets the method of how the dropped samples should be filled
            * If Callable, produces a sample by calling fill()
            * If list, fill is any element in the list
            * If str, fill is fixed at the method provided
    
    """
    def __init__(
        self,
        drop_rate: NumericParameter = uniform_continuous_distribution(0.01,0.05),
        size: NumericParameter = uniform_discrete_distribution(np.arange(1,10)),
        fill: Union[List, str] = uniform_discrete_distribution(["ffill", "bfill", "mean", "zero"]),
    ):
        super(RandomDropSamples, self).__init__()
        self.drop_rate = to_distribution(drop_rate, self.random_generator)
        self.size = to_distribution(size, self.random_generator)
        self.fill = to_distribution(fill, self.random_generator)

    def __call__(self, data: Any) -> Any:
        drop_rate = self.drop_rate()
        fill = self.fill()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Perform data augmentation
            drop_instances = int(data.iq_data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(1, data.iq_data.shape[0]-max(drop_sizes)-1, drop_instances).astype(int)
            
            new_data.iq_data = functional.drop_samples(data.iq_data, drop_starts, drop_sizes, fill)
                
        else:
            drop_instances = int(data.shape[0] * drop_rate)
            drop_sizes = self.size(drop_instances).astype(int)
            drop_starts = np.random.uniform(0, data.shape[0]-max(drop_sizes), drop_instances).astype(int)
            
            new_data = functional.drop_samples(data, drop_starts, drop_sizes, fill)
        return new_data
    
    
class Quantize(SignalTransform):
    """Quantize the input to the number of levels specified
    
    Args:
         num_levels (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            num_levels sets the number of quantization levels
            * If Callable, produces a sample by calling num_levels()
            * If int or float, num_levels is fixed at the value provided
            * If list, num_levels is any element in the list
            * If tuple, num_levels is in range of (tuple[0], tuple[1])
        
        round_type (:py:class:`~Callable`, :obj:`str`, :obj:`list`):
            round_type sets the rounding direction of the quantization. Options
            include: 'floor', 'middle', & 'ceiling'
            * If Callable, produces a sample by calling round_type()
            * If str, round_type is fixed at the value provided
            * If list, round_type is any element in the list
    """
    def __init__(
        self,
        num_levels: NumericParameter = uniform_discrete_distribution([16,24,32,40,48,56,64]),
        round_type: Union[List, str] = ["floor", "middle", "ceiling"],
    ):
        super(Quantize, self).__init__()
        self.num_levels = to_distribution(num_levels, self.random_generator)
        self.round_type = to_distribution(round_type, self.random_generator)

    def __call__(self, data: Any) -> Any:
        num_levels = self.num_levels()
        round_type = self.round_type()

        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Perform data augmentation
            new_data.iq_data = functional.quantize(data.iq_data, num_levels, round_type)
                
        else:
            new_data = functional.quantize(data, num_levels, round_type)
        return new_data
    

class Clip(SignalTransform):
    """Clips the input values to a percentage of the max/min values
    
    Args:
        clip_percentage (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Specifies the percentage of the max/min values to clip
            * If Callable, produces a sample by calling clip_percentage()
            * If int or float, clip_percentage is fixed at the value provided
            * If list, clip_percentage is any element in the list
            * If tuple, clip_percentage is in range of (tuple[0], tuple[1])
    
    """
    
    def __init__(
        self,
        clip_percentage: NumericParameter = uniform_continuous_distribution(0.75, 0.95),
        **kwargs,
    ):
        super(Clip, self).__init__(**kwargs)
        self.clip_percentage = to_distribution(clip_percentage)

    def __call__(self, data: Any) -> Any:
        clip_percentage = self.clip_percentage()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Apply data augmentation
            new_data.iq_data = functional.clip(data.iq_data, clip_percentage)
            
        else:
            new_data = functional.clip(data, clip_percentage)
        return new_data
    

class RandomConvolve(SignalTransform):
    """Convolve a random complex filter with the input data
    
    Args:
        num_taps (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Number of taps for the random filter
            * If Callable, produces a sample by calling num_taps()
            * If int or float, num_taps is fixed at the value provided
            * If list, num_taps is any element in the list
            * If tuple, num_taps is in range of (tuple[0], tuple[1])
            
        alpha (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            The effect of the filtered data is dampened using an alpha factor
            that determines the weightings for the summing of the filtered data
            and the original data. `alpha` should be in range `[0,1]` where a
            value of 0 applies all of the weight to the original data, and a 
            value of 1 applies all of the weight to the filtered data
            * If Callable, produces a sample by calling alpha()
            * If int or float, alpha is fixed at the value provided
            * If list, alpha is any element in the list
            * If tuple, alpha is in range of (tuple[0], tuple[1])
    
    """
    def __init__(
        self, 
        num_taps: IntParameter = uniform_continuous_distribution(2, 5), 
        alpha: FloatParameter = uniform_continuous_distribution(0.1, 0.5), 
        **kwargs,
    ):
        super(RandomConvolve, self).__init__(**kwargs)
        self.num_taps = to_distribution(num_taps, self.random_generator)
        self.alpha = to_distribution(alpha, self.random_generator)

    def __call__(self, data: Any) -> Any:
        num_taps = int(self.num_taps())
        alpha = self.alpha()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=data.signal_description,
            )
            
            # Apply data augmentation
            new_data.iq_data = functional.random_convolve(data.iq_data, num_taps, alpha)
            
        else:
            new_data = functional.random_convolve(data, num_taps, alpha)
        return new_data
