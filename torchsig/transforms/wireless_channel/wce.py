from copy import deepcopy
import numpy as np
from typing import Optional, Tuple, List, Union, Any

from torchsig.utils.types import SignalData, SignalDescription
from torchsig.transforms.transforms import SignalTransform
from torchsig.transforms.wireless_channel import wce_functional as F
from torchsig.transforms.functional import NumericParameter, FloatParameter, IntParameter
from torchsig.transforms.functional import to_distribution, uniform_continuous_distribution, uniform_discrete_distribution

    
class TargetSNR(SignalTransform):
    """Adds zero-mean complex additive white Gaussian noise to a provided 
    tensor to achieve a target SNR. The provided signal is assumed to be 
    entirely the signal of interest. Note that this transform relies on 
    information contained within the SignalData object's SignalDescription. The 
    transform also assumes that only one signal is present in the IQ data. If
    multiple signals' SignalDescriptions are detected, the transform will raise a 
    warning.

    Args:
        target_snr (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

        eb_no (:obj:`bool`):
            Defines SNR as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2))*samples_per_symbol/bits_per_symbol.
            Defining SNR this way effectively normalized the noise level with respect to spectral efficiency and
            bandwidth. Normalizing this way is common in comparing systems in terms of power efficiency.
            If True, bits_per_symbol in the the SignalData will be used in the calculation of SNR. To achieve SNR in
            terms of E_b/N_0, samples_per_symbol must also be provided.

        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB.

    """
    def __init__(
        self,
        target_snr: NumericParameter = uniform_continuous_distribution(-10, 10),
        eb_no: Optional[bool] = False,
        linear: Optional[bool] = False,
        **kwargs
    ):
        super(TargetSNR, self).__init__(**kwargs)
        self.target_snr = to_distribution(target_snr, self.random_generator)
        self.eb_no = eb_no
        self.linear = linear

    def __call__(self, data: Any) -> Any:
        target_snr = self.target_snr()
        target_snr_linear = 10**(target_snr/10) if not self.linear else target_snr
        if isinstance(data, SignalData):
            if len(data.signal_description) > 1:
                raise ValueError(
                    "Expected single `SignalDescription` for input `SignalData` but {} detected."
                    .format(len(data.signal_description))
                )
            signal_power = np.mean(np.abs(data.iq_data)**2, axis=self.time_dim)
            class_name = data.signal_description[0].class_name
            if "ofdm" not in class_name:
                # EbNo not available for OFDM
                target_snr_linear *= data.signal_description[0].bits_per_symbol if self.eb_no else 1
            occupied_bw = 1 / data.signal_description[0].samples_per_symbol
            noise_power_linear = signal_power / (target_snr_linear * occupied_bw)
            noise_power_db = 10*np.log10(noise_power_linear)
            data.iq_data = F.awgn(data.iq_data, noise_power_db)
            data.signal_description[0].snr = target_snr
            return data
        else:
            raise ValueError(
                "Expected input type `SignalData`. Received {}. \n\t\
                The `TargetSNR` transform depends on metadata from a `SignalData` object. \n\t\
                Please reference the `AddNoise` transform as an alternative."
                .format(type(data))
            )
            
    
class AddNoise(SignalTransform):
    """ Add random AWGN at specified power levels
    
    Note:
        Differs from the TargetSNR() transform in that this transform adds
        noise at a specified power level, whereas AddNoise() 
        assumes a basebanded signal and adds noise to achieve a specified SNR
        level for the signal of interest. This transform, 
        AddNoise() is useful for simply adding a randomized
        level of noise to either a narrowband or wideband input.
    
    Args:
        noise_power_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])
        
        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB.
        
    Example:
        >>> import torchsig.transforms as ST
        >>> # Added AWGN power range is (-40, -20) dB
        >>> transform = ST.AddNoiseTransform((-40, -20))
    
    """
    
    def __init__(
        self,
        noise_power_db : NumericParameter = uniform_continuous_distribution(-80, -60),
        linear: Optional[bool] = False,
        **kwargs,
    ):
        super(AddNoise, self).__init__(**kwargs)
        self.noise_power_db = to_distribution(noise_power_db)
        self.linear = linear

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Apply data augmentation
            noise_power_db = self.noise_power_db()
            noise_power_db = 10*np.log10(noise_power_db) if self.linear else noise_power_db
            new_data.iq_data = F.awgn(data.iq_data, noise_power_db)
            
            # Update SignalDescription
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                new_signal_desc.snr -= noise_power_db
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            
        else:
            noise_power_db = self.noise_power_db(size=data.shape[0])
            noise_power_db = 10*np.log10(noise_power_db) if self.linear else noise_power_db
            new_data = F.awgn(data, noise_power_db)
        return new_data

    
class TimeVaryingNoise(SignalTransform):
    """Add time-varying random AWGN at specified input parameters
    
    Args:
        noise_power_db_low (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.
            * If Callable, produces a sample by calling noise_power_db_low()
            * If int or float, noise_power_db_low is fixed at the value provided
            * If list, noise_power_db_low is any element in the list
            * If tuple, noise_power_db_low is in range of (tuple[0], tuple[1])
            
        noise_power_db_high (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.
            * If Callable, produces a sample by calling noise_power_db_low()
            * If int or float, noise_power_db_low is fixed at the value provided
            * If list, noise_power_db_low is any element in the list
            * If tuple, noise_power_db_low is in range of (tuple[0], tuple[1])
            
        inflections (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Number of inflection points in time-varying noise
            * If Callable, produces a sample by calling inflections()
            * If int or float, inflections is fixed at the value provided
            * If list, inflections is any element in the list
            * If tuple, inflections is in range of (tuple[0], tuple[1])
        
        random_regions (:py:class:`~Callable`, :obj:`bool`, :obj:`list`, :obj:`tuple`):
            If inflections > 0, random_regions specifies whether each 
            inflection point should be randomly selected or evenly divided 
            among input data
            * If Callable, produces a sample by calling random_regions()
            * If bool, random_regions is fixed at the value provided
            * If list, random_regions is any element in the list
            * If tuple, random_regions is in range of (tuple[0], tuple[1])
        
        linear (:obj:`bool`):
            If True, powers input are on linear scale not dB.
    
    """
    def __init__(
        self,
        noise_power_db_low: NumericParameter = uniform_continuous_distribution(-80, -60),
        noise_power_db_high: NumericParameter = uniform_continuous_distribution(-40, -20),
        inflections: IntParameter = uniform_continuous_distribution(0, 10),
        random_regions: Optional[Union[Tuple, bool]] = (False, True),
        linear: Optional[bool] = False,
        **kwargs,
    ):
        super(TimeVaryingNoise, self).__init__(**kwargs)
        self.noise_power_db_low = to_distribution(noise_power_db_low)
        self.noise_power_db_high = to_distribution(noise_power_db_high)
        self.inflections = to_distribution(inflections)
        self.random_regions = to_distribution(random_regions)
        self.linear = linear

    def __call__(self, data: Any) -> Any:
        noise_power_db_low = self.noise_power_db_low()
        noise_power_db_high = self.noise_power_db_high()
        noise_power_db_low = 10*np.log10(noise_power_db_low) if self.linear else noise_power_db_low
        noise_power_db_high = 10*np.log10(noise_power_db_high) if self.linear else noise_power_db_high
        inflections = int(self.inflections())
        random_regions = self.random_regions()
        
        if isinstance(data, SignalData):
            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            
            # Apply data augmentation
            new_data.iq_data = F.time_varying_awgn(data.iq_data, noise_power_db_low, noise_power_db_high, inflections, random_regions)
            
            # Update SignalDescription with average of added noise (Note: this is merely an approximation)
            new_signal_description = []
            signal_description = [data.signal_description] if isinstance(data.signal_description, SignalDescription) else data.signal_description
            noise_power_db_change = np.abs(noise_power_db_high - noise_power_db_low)
            avg_noise_power_db = min(noise_power_db_low, noise_power_db_high) + noise_power_db_change / 2
            for signal_desc in signal_description:
                new_signal_desc = deepcopy(signal_desc)
                new_signal_desc.snr -= avg_noise_power_db
                new_signal_description.append(new_signal_desc)
            new_data.signal_description = new_signal_description
            
        else:
            new_data = F.time_varying_awgn(data, noise_power_db_low, noise_power_db_high, inflections, random_regions)
        return new_data

    
class RayleighFadingChannel(SignalTransform):
    """Applies Rayleigh fading channel to tensor.

    Note:
        A Rayleigh fading channel can be modeled as an FIR filter with Gaussian distributed taps which vary over time.
        The length of the filter determines the coherence bandwidth of the channel and is inversely proportional to
        the delay spread. The rate at which the channel taps vary over time is related to the coherence time and this is
        inversely proportional to the maximum Doppler spread. This time variance is not included in this model.

    Args:
        coherence_bandwidth (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling coherence_bandwidth()
            * If int or float, coherence_bandwidth is fixed at the value provided
            * If list, coherence_bandwidth is any element in the list
            * If tuple, coherence_bandwidth is in range of (tuple[0], tuple[1])

        power_delay_profile (:obj:`list`, :obj:`tuple`):
            A list of positive values assigning power to taps of the channel model. When the number of taps
            exceeds the number of items in the provided power_delay_profile, the list is linearly interpolated
            to provide values for each tap of the channel

    Example:
        >>> import torchsig.transforms as ST
        >>> # Rayleigh Fading with coherence bandwidth uniformly distributed between fs/100 and fs/10
        >>> transform = ST.RayleighFadingChannel(lambda size: np.random.uniform(.01, .1, size))
        >>> # Rayleigh Fading with coherence bandwidth normally distributed clipped between .01 and .1
        >>> transform = ST.RayleighFadingChannel(lambda size: np.clip(np.random.normal(0, .1, size), .01, .1))
        >>> # Rayleigh Fading with coherence bandwidth uniformly distributed between fs/100 and fs/10
        >>> transform = ST.RayleighFadingChannel((.01, .1))
        >>> # Rayleigh Fading with coherence bandwidth either .02 or .01
        >>> transform = ST.RayleighFadingChannel([.02, .01])
        >>> # Rayleigh Fading with fixed coherence bandwidth at .1
        >>> transform = ST.RayleighFadingChannel(.1)
        >>> # Rayleigh Fading with fixed coherence bandwidth at .1 and pdp (1.0, .7, .1)
        >>> transform = ST.RayleighFadingChannel((.01, .1), power_delay_profile=(1.0, .7, .1))
    """

    def __init__(
        self,
        coherence_bandwidth: FloatParameter = uniform_continuous_distribution(.01, .1),
        power_delay_profile: Union[Tuple, List, np.ndarray] = (1, 1),
        **kwargs
    ):
        super(RayleighFadingChannel, self).__init__(**kwargs)
        self.coherence_bandwidth = to_distribution(coherence_bandwidth, self.random_generator)
        self.power_delay_profile = np.asarray(power_delay_profile)

    def __call__(self, data: Any) -> Any:
        coherence_bandwidth = self.coherence_bandwidth()
        if isinstance(data, SignalData):
            data.iq_data = F.rayleigh_fading(data.iq_data, coherence_bandwidth, self.power_delay_profile)
        else:
            data = F.rayleigh_fading(data, coherence_bandwidth, self.power_delay_profile)
        return data

    
class ImpulseInterferer(SignalTransform):
    """Applies an impulse interferer

    Args:
        amp (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling amp()
            * If int or float, amp is fixed at the value provided
            * If list, amp is any element in the list
            * If tuple, amp is in range of (tuple[0], tuple[1])

        pulse_offset (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling phase_offset()
            * If int or float, pulse_offset is fixed at the value provided
            * If list, phase_offset is any element in the list
            * If tuple, phase_offset is in range of (tuple[0], tuple[1])

    """
    def __init__(
        self,
        amp: FloatParameter = uniform_continuous_distribution(.1, 100.),
        pulse_offset: FloatParameter = uniform_continuous_distribution(0., 1),
        **kwargs
    ):
        super(ImpulseInterferer, self).__init__(**kwargs)
        self.amp = to_distribution(amp, self.random_generator)
        self.pulse_offset = to_distribution(pulse_offset, self.random_generator)
        self.BETA = .3
        self.SPS = .1

    def __call__(self, data: Any) -> Any:
        amp = self.amp()
        pulse_offset = self.pulse_offset() 
        pulse_offset = 1. if pulse_offset > 1. else np.max((0., pulse_offset))
        if isinstance(data, SignalData):
            data.iq_data = F.impulsive_interference(data.iq_data, amp, self.pulse_offset)
        else:
            data = F.impulsive_interference(data, amp, self.pulse_offset)
        return data
    

class RandomPhaseShift(SignalTransform):
    """Applies a random phase offset to tensor

    Args:
        phase_offset (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling phase_offset()
            * If int or float, phase_offset is fixed at the value provided
            * If list, phase_offset is any element in the list
            * If tuple, phase_offset is in range of (tuple[0], tuple[1])

    Example:
        >>> import torchsig.transforms as ST
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift(uniform_continuous_distribution(-1, 1))
        >>> # Phase Offset from [-pi/2, 0, and pi/2]
        >>> transform = ST.RandomPhaseShift(uniform_discrete_distribution([-.5, 0, .5]))
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = ST.RandomPhaseShift((-1, 1))
        >>> # Phase Offset either -pi/4 or pi/4
        >>> transform = ST.RandomPhaseShift([-.25, .25])
        >>> # Phase Offset is fixed at -pi/2
        >>> transform = ST.RandomPhaseShift(-.5)
    """
    def __init__(
        self,
        phase_offset: FloatParameter = uniform_continuous_distribution(-1, 1),
        **kwargs
    ):
        super(RandomPhaseShift, self).__init__(**kwargs)
        self.phase_offset = to_distribution(phase_offset, self.random_generator)

    def __call__(self, data: Any) -> Any:
        phases = self.phase_offset()
        if isinstance(data, SignalData):
            data.iq_data = F.phase_offset(data.iq_data, phases*np.pi)
        else:
            data = F.phase_offset(data, phases*np.pi)
        return data
