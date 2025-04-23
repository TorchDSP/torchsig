"""SignalTransforms on Signal objects.
"""

__all__ = [
    "SignalTransform",
    "AdditiveNoiseSignalTransform",
    "AdjacentChannelInterference",   
    "CarrierPhaseOffsetSignalTransform",
    "CochannelInterference", 
    "DopplerSignalTransform",
    "Fading",
    "IntermodulationProductsSignalTransform",
    "IQImbalanceSignalTransform",
    "LocalOscillatorFrequencyDriftSignalTransform",
    "LocalOscillatorPhaseNoiseSignalTransform",
    "NonlinearAmplifierSignalTransform",
    "PassbandRippleSignalTransform",
    "QuantizeSignalTransform",
    "ShadowingSignalTransform",
    "SpectralInversionSignalTransform",
]

# TorchSig
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_types import Signal
import torchsig.transforms.functional as F
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    low_pass
)

# Third Party
import numpy as np
import scipy as sp

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


class AdditiveNoiseSignalTransform(SignalTransform):
    """Adds noise with specified properties to Signal data.

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
    
    def __call__(self, signal: Signal) -> Signal:
        add_noise_power = self.power_distribution()
        
        if self.measure:
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
    """Applies adjacent channel interference to Signal.

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
    
    def __call__(self, signal: Signal) -> Signal:
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
        self.phase_offset_distribution = self.get_distribution(self.phase_offset_range)
    
    def __call__(self, signal: Signal) -> Signal:
        phase_offset = self.phase_offset_distribution()

        signal.data = F.phase_offset(signal.data, phase_offset)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class CochannelInterference(SignalTransform):
    """Applies cochannel interference to Signal.

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
    
    def __call__(self, signal: Signal) -> Signal:
        cochan_noise_power = self.power_distribution()
        
        if self.measure:
            # Treat SNR as SINR/AWGN for full sampled band, assuming independent noise
            snr_linear = 10 ** (signal.metadata.snr_db / 10)
            total_power = np.sum(np.abs(signal.data)**2)/len(signal.data)
            sig_power = total_power / (1 + 1/snr_linear)
            noise_power = sig_power / snr_linear
            new_snr = sig_power / (noise_power + cochan_noise_power)
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


class DopplerSignalTransform(SignalTransform):
    """SignalTransform that applies wideband Doppler to Signal IQ data.

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
    
    def __call__(self, signal: Signal) -> Signal:
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
        signal.metadata.center_freq *= alpha
        signal.metadata.bandwidth *= alpha
        
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
        
    def __call__(self, signal: Signal) -> Signal:
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


class IntermodulationProductsSignalTransform(SignalTransform):
    """Applies simulated intermodulation products to a Signal.

    Attributes:
        model_order (List[int]): The choices model order, 3rd or 5th order. Defaults to [3,5].
        coeffs_range (Tuple[float, float]): Range bounds for each intermodulation coefficient. 
            Defaults to (0., 1.).
        
    """
    def __init__(
        self,
        model_order: List[int] = [3, 5],
        coeffs_range: Tuple[float, float] = (1e-7, 1e-5),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.model_order = model_order
        self.model_order_distribution = self.get_distribution(self.model_order)
        self.coeffs_range = coeffs_range
        self.coeffs_distribution = self.get_distribution(self.coeffs_range,'log10')
    
    def __call__(self, signal: Signal) -> Signal:
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


class IQImbalanceSignalTransform(SignalTransform):
    """Applies a set of IQImbalance effects to a Signal: amplitude, phase, and DC offset.

    Attributes:
        amplitude_imbalance (optional): Range bounds of IQ amplitude imbalance (dB).    
        amplitude_imbalance_distribution (Callable[[], float]): Random draw from amplitude imbalance distribution.
        phase_imbalance (optional): Range bounds of IQ phase imbalance (radians).        
        phase_imbalance (Callable[[], float]): Random draw from phase imbalance distribution.
        dc_offset (Tuple, optional): Range bounds for I and Q component DC offsets.
        dc_offset (Callable[[], (float, float)]): Random draw from dc_offset distribution.
        
    """
    def __init__(
        self,
        amplitude_imbalance = (-1., 1.),
        phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0),
        dc_offset = ((-0.1, 0.1),(-0.1, 0.1)),
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.amplitude_imbalance = amplitude_imbalance
        self.phase_imbalance = phase_imbalance
        self.dc_offset = dc_offset 
        
        self.amplitude_imbalance_distribution = self.get_distribution(self.amplitude_imbalance)
        self.phase_imbalance_distribution = self.get_distribution(self.phase_imbalance)
        self.dc_offset_distribution = self.get_distribution(self.dc_offset)
        
    def __call__(self, signal: Signal) -> Signal:
        amplitude_imbalance = self.amplitude_imbalance_distribution()
        phase_imbalance = self.phase_imbalance_distribution()
        dc_offset = self.dc_offset_distribution()

        signal.data = F.iq_imbalance(signal.data, amplitude_imbalance, phase_imbalance, dc_offset)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class LocalOscillatorFrequencyDriftSignalTransform(SignalTransform):
    """SignalTransform that applies LO frequency drift to Signal IQ data.

    Attributes:
        drift_std_range (Tuple[float, float]): Drift standard deviation. Default (10, 100).
        drift_std_distribution (Callable[[], float]): Random draw from drift_std_range distribution.
        
    """
    def __init__(
        self, 
        drift_std_range: Tuple[float, float] = (10, 100),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.drift_std_range = drift_std_range
        self.drift_std_distribution = self.get_distribution(self.drift_std_range)
    
    def __call__(self, signal: Signal) -> Signal:
        drift_std = self.drift_std_distribution()

        signal.data = F.local_oscillator_frequency_drift(
            data = signal.data, 
            drift_std = drift_std, 
            sample_rate = signal.metadata.sample_rate,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)       
        self.update(signal)
        return signal


class LocalOscillatorPhaseNoiseSignalTransform(SignalTransform):
    """SignalTransform that applies LO phase noise to Signal IQ data.

    Attributes:
       phase_noise_std (Tuple[float, float]): Range for phase noise standard deviation. Defaults to (10, 100).
       phase_noise_std_distribution (Callable[[], float]): Random draw from phase_noise_std distribution.
        
    """
    def __init__(
        self, 
        phase_noise_std: Tuple[float, float] = (10, 100),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.phase_noise_std = phase_noise_std
        self.phase_noise_std_distribution = self.get_distribution(self.phase_noise_std)
    
    def __call__(self, signal: Signal) -> Signal:
        phase_noise_std = self.phase_noise_std_distribution()

        signal.data = F.local_oscillator_phase_noise(
            data = signal.data,
            phase_noise_std = phase_noise_std,
            sample_rate = signal.metadata.sample_rate,
            rng = self.random_generator
        )

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class NonlinearAmplifierSignalTransform(SignalTransform):
    """Applies a memoryless nonlinear amplifier model to Signal.

    Attributes:
        gain_range (Tuple[float, float]): Small-signal gain range (linear). Defaults to (1.0, 4.0).
        gain_distribution (Callable[[], float]): Random draw from gain distribution.
        psat_backoff_range (Tuple[float, float]): Psat backoff factor (linear) reflecting saturated
            power level (Psat) relative to input signal mean power. Defaults to (5.0, 20.0).
        past_backoff_distribution (Callable[[], float]): Random draw from psat_backoff distribution.   
        phi_range (Tuple[float, float]): Maximum signal relative phase shift at 
            saturation power level (radians). Defaults to (0.0, 0.0).
        phi_distribution (Callable[[], float]): Random draw from phi distribution.        
        auto_scale (bool): Automatically rescale output power to match full-scale peak 
            input power prior to transform, based on peak estimates. Default True.
        
    """
    def __init__(
        self,
        gain_range: Tuple[float, float] = (1.0, 4.0),
        psat_backoff_range: Tuple[float, float] = (5.0, 20.0),
        phi_range: Tuple[float, float] = (0.0, 0.0),
        auto_scale: bool = True,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.gain_range = gain_range
        self.gain_distribution = self.get_distribution(self.gain_range)
        self.psat_backoff_range = psat_backoff_range
        self.psat_backoff_distribution = self.get_distribution(self.psat_backoff_range)        
        self.phi_range = phi_range
        self.phi_distribution = self.get_distribution(self.phi_range)
        self.auto_scale = auto_scale
    
    def __call__(self, signal: Signal) -> Signal:
        gain = self.gain_distribution()
        psat_backoff = self.psat_backoff_distribution()
        phi = self.phi_distribution()

        signal.data = F.nonlinear_amplifier(
            data = signal.data,
            gain = gain,
            psat_backoff = psat_backoff,
            phi_rad = phi,
            auto_scale = self.auto_scale
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class PassbandRippleSignalTransform(SignalTransform):
    """SignalTransform that models analog filter passband ripple for Signal IQ data.

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
    
    def __call__(self, signal: Signal) -> Signal:
        signal.data = F.passband_ripple(
            data = signal.data,
            filter_coeffs = self.fir_coeffs,
            normalize = True
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class Shadowing(SignalTransform):
    """SignalTransform that applies RF channel shadowing to Signal IQ data. This
    slow channel obstruction effect is applied as a block to the whole data.

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

    def __call__(self, signal: Signal) -> Signal:
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


class QuantizeSignalTransform(SignalTransform):
    """SignalTransform that models Quantization in DAC.

    Attributes:
        num_bits (float): Range of number of bits in DAC to simulate. Defaults 4 bits to 18 bits.
        ref_level_adjustment_db (float): Reference level (in dB) to increase or decrease relative to full scale. Defaults to -10 dB to +3 dB.
        
    """
    def __init__(
        self, 
        num_bits:  Tuple[int, int] = (6, 18),
        ref_level_adjustment_db:  Tuple[float, float] = (-10, 3),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_bits = num_bits
        self.num_bits_distribution = self.get_distribution(self.num_bits)
        self.ref_level_adjustment_db = ref_level_adjustment_db
        self.ref_level_adjustment_db_distribution = self.get_distribution(self.ref_level_adjustment_db)

    def __call__(self, signal: Signal) -> Signal:

        num_bits = self.num_bits_distribution()
        ref_level_adjustment_db = self.ref_level_adjustment_db_distribution()

        # apply quantization
        signal.data = F.quantize(
            data = signal.data,
            num_bits = num_bits,
            ref_level_adjustment_db = ref_level_adjustment_db,
        )

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

