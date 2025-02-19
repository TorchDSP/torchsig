"""SignalTransforms on Signal objects.
"""

__all__ = [
    "SignalTransform",
    # "AdjacentChannelInterferenceSignalTransform",
    # "AtmospherericDuctSignalTransform",
    # "CochannelInterferenceSignalTransform",    
    "CarrierPhaseOffsetSignalTransform",
    # "ClockSignalTransform",   
    # "DopplerSignalTransform",
    "Fading",
    # "IntermodulationProductsSignalTransform",
    "IQImbalanceSignalTransform",
    # "LocalOscillatorPhaseNoiseSignalTransform",
    # "LocalOscillatorFrequencyDriftSignalTransform",
    # "NonlinearAmplifierSignalTransform",
    # "PassbandRippleSignalTransform",    
    # "ShadowingSignalTransform",
    "SpectralInversionSignalTransform",
    # "TimeShiftSignalTransform"
]

# TorchSig
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_types import Signal
import torchsig.transforms.functional as F
from torchsig.utils.dsp import torchsig_complex_data_type
from torchsig.transforms.transform_utils import (
    get_distribution,
    FloatParameter,
    NumericParameter
)

# Third Party
import numpy as np

# Built-In
from typing import Tuple, List


class SignalTransform(Transform):
    """SignalTransform parent class.
    """

    def update(self, signal: Signal) -> None:
        """Updates bookkeeping to transforms in Signal's SignalMetadata and checks signal valididty.
        Inherited classes should always call self.update() after performing transform operation (inside __call__).

        Args:
            signal (Signal): Transformed signal.

        """
        signal.metadata.applied_transforms.append(self)
        # signal.verify()

    def __call__(self, signal: Signal) -> Signal:
        """Performs transforms.

        Args:
            signal (Signal): Signal to be transformed.

        Raises:
            NotImplementedError: Inherited classes must override this method.

        Returns:
            Signal: Transformed Signal.

        """
        raise NotImplementedError


class CarrierPhaseOffsetSignalTransform(SignalTransform):
    """SignalTransform that applies a randomized carrier phase offset to Signal IQ data.

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

        self.phase_offset_distribution = get_distribution(self.phase_offset_range, self.random_generator)
        
    
    def __call__(self, signal: Signal) -> Signal:
        phase_offset = self.phase_offset_distribution()

        signal.data = F.phase_offset(signal.data, phase_offset)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class Fading(SignalTransform): # slow, fast, block fading
    """SignalTransform that applies a channel fading model. 

    Note, currently only performs Rayleigh fading:
        A Rayleigh fading channel can be modeled as an FIR filter with Gaussian distributed
        taps which vary over time. The length of the filter determines the coherence bandwidth
        of the channel and is inversely proportional to the delay spread. The rate at which 
        the channel taps vary over time is related to the coherence time and this is inversely
        proportional to the maximum Doppler spread. This time variance is not included in this model.
    
    Attributes:
        coherence_bandwidth (FloatParameter, optional): Coherence bandwidth sampling parameters. 
                Defaults to (0.01, 0.1).
        coherence_bandwidth_distribution (Callable[[], float]): Random draw from coherence bandwidth distribution.
        power_delay_profile (Tuple | List | np.ndarray, optional): A list of positive values 
            assigning power to taps of the channel model. When the number of taps exceeds the number 
            of items in the provided power_delay_profile, the list is linearly interpolated to 
            provide values for each tap of the channel. Defaults to (1, 1).

    """    
    def __init__(
        self, 
        coherence_bandwidth: FloatParameter = (0.01, 0.1),
        power_delay_profile: Tuple | List | np.ndarray = (1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.coherence_bandwidth = coherence_bandwidth
        self.power_delay_profile = np.asarray(power_delay_profile)

        self.coherence_bandwidth_distribution = get_distribution(self.coherence_bandwidth, self.random_generator)
        
        

    def __call__(self, signal: Signal) -> Signal:
        coherence_bandwidth = self.coherence_bandwidth_distribution()

        signal.data = F.fading(
            signal.data, 
            coherence_bandwidth, 
            self.power_delay_profile,
            self.random_generator
        )

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class IQImbalanceSignalTransform(SignalTransform):
    """Applies a set of IQImbalance effects to a Signal: amplitude, phase, and DC offset.

    Attributes:
        amplitude_imbalance (NumericParameter, optional): Range bounds of IQ amplitude imbalance (dB).    
        amplitude_imbalance_distribution (Callable[[], float]): Random draw from amplitude imbalance distribution.
        phase_imbalance (NumericParameter, optional): Range bounds of IQ phase imbalance (radians).        
        phase_imbalance (Callable[[], float]): Random draw from phase imbalance distribution.
        dc_offset (Tuple, optional): Range bounds for I and Q component DC offsets (NumericParameters).
        dc_offset (Callable[[], (float, float)]): Random draw from dc_offset distribution.
        
    """
    def __init__(
        self,
        amplitude_imbalance: NumericParameter = (-1., 1.),
        phase_imbalance: NumericParameter = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0),
        dc_offset: NumericParameter = ((-0.1, 0.1),(-0.1, 0.1)),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.amplitude_imbalance = amplitude_imbalance
        self.phase_imbalance = phase_imbalance
        self.dc_offset = dc_offset 
        
        self.amplitude_imbalance_distribution = get_distribution(self.amplitude_imbalance, self.random_generator)
        self.phase_imbalance_distribution = get_distribution(self.phase_imbalance, self.random_generator)
        self.dc_offset_distribution = get_distribution(self.dc_offset, self.random_generator)
        
    def __call__(self, signal: Signal) -> Signal:
        amplitude_imbalance = self.amplitude_imbalance_distribution()
        phase_imbalance = self.phase_imbalance_distribution()
        dc_offset = self.dc_offset_distribution()

        signal.data = F.iq_imbalance(signal.data, amplitude_imbalance, phase_imbalance, dc_offset)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class SpectralInversionSignalTransform(SignalTransform):
    """Inverts spectrum of complex IQ data.
    """
    def __call__(self, signal: Signal) -> Signal:
        signal.data = F.spectral_inversion(signal.data)

        signal.data = signal.data.astype(torchsig_complex_data_type)

        signal.metadata.center_freq *= -1
        
        self.update(signal)
        return signal




# TBD -------------------------

class AdjacentChannelInterferenceSignalTransform(SignalTransform):
    """Unimplemented SignalTransform that applies an adjacent channel interference model to Signal IQ data.
    """


class AtmosphericDuctSignalTransform(SignalTransform):
    """Unimplemented SignalTransform that models atmospheric ducting effects.
    """    


class ClockSignalTransform(SignalTransform):
    """Unimplemented SignalTransform that models Clock nonidealities.
    """        


class CoChannelInterferenceSignalTransform(SignalTransform):
    """Unimplemented SignalTransform that adds co-channel interfence.
    """


class DopplerSignalTransform(SignalTransform):
    """Unimplemented SignalTransform for consistent Doppler effect.
    """    


class IntermodulationProductsSignalTransform(SignalTransform):
    """Unimplemented SignalTransform for modeling IntermodulationProducts in and near baseband.
    """       


# TODO: rename to CarrierFrequencyDrift()
# a slow, random walk over the carrier frequency. the center frequency would 
# be modeled as CF + gaussian RV with mean = 0 and some non-zero variance. but 
# since we are implementing this at complex baseband the carrier can be modeled
# with the following pseudo code:
#
# freq_noise_var = 1e-6 # is this a good number? need some insights here
# random_phase = rng.random.normal(0,phase_noise_var,len(iq_sample_array))
# random_freq = np.cumsum(random_phase)
# carrier_with_freq_drift = np.exp(1j*np.pi*random_freq)
# signal *= carrier_with_freq_drift
#
# should be randomizing over the input variance as well
class LocalOscillatorFrequencyDriftSignalTransform(SignalTransform):
    """Unimplemented SignalTransform for modeling Local Oscillator drift in frequency.
    """      


class LocalOscillatorPhaseNoiseSignalTransform(SignalTransform):
    """Unimplemented SignalTransform for modeling Local Oscillator phase noise.
    """          


class NonlinearAmplifierSignalTransform(SignalTransform):
    """Unimplemented SignalTransform for memoryless nonlinear amplifier response.
    """        


class PassbandRippleSignalTransform(SignalTransform):
    """Unimplemented SignalTransform to create passband ripple filter effects within the sampling bandwidth.
    """            
    # analog and digital


class ShadowingSignalTransform(SignalTransform):
    """Unimplemented SignalTransform to effect RF channel shadowing.
    """            


class TimeShiftSignalTransform(SignalTransform):
    """Unimplemented SignalTransform that shifts signal samples in time.
    """
