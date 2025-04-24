"""DatasetTransforms on DatasetSignal objects.
"""

__all__ = [
    "DatasetTransform",

    ### RF Transforms
    "AdditiveNoiseDatasetTransform"
    "AGC",
    "AWGN",
    "BlockAGC",
    "CarrierPhaseOffsetDatasetTransform",
    "ComplexTo2D",
    "IQImbalanceDatasetTransform",
    "LocalOscillatorFrequencyDriftDatasetTransform"
    "LocalOscillatorPhaseNoiseDatasetTransform",
    "NonlinearAmplifierDatasetTransform",
    "PassbandRippleDatasetTransform",
    "QuantizeDatasetTransform",
    "Spectrogram",
    "SpectralInversionDatasetTransform",
    "TimeVaryingNoise",

    ### ML Tranforms
    "AddSlope",
    "ChannelSwap",
    "CutOut",
    # "DropSpectrogram",
    "PatchShuffle",
    "RandomDropSamples",    
    "RandomMagRescale",
    "SpectrogramDropSamples",
    "TimeReversal"
]

# TorchSig
from torchsig.transforms.base_transforms import Transform
from torchsig.signals.signal_types import DatasetSignal, SignalMetadata
import torchsig.transforms.functional as F
from torchsig.utils.dsp import torchsig_complex_data_type, torchsig_float_data_type

# Third Party
import numpy as np
import scipy as sp

# Built-In
from typing import List, Tuple
from copy import copy


class DatasetTransform(Transform):
    """Dataset Transform base class

    Dataset Transforms are transforms applied to DatasetSignals.

    """

    def update(self, signal: DatasetSignal) -> None:
        """Updates bookkeeping to transforms in DatasetSignal's SignalMetadata and checks signal valididty.
        Inherited classes should always call self.update() after performing transform operation (inside __call__).

        Args:
            signal (DatasetSignal): transformed DatasetSignal.

        """
        for m in signal.metadata:
            m.applied_transforms.append(self)
        # signal.verify()

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """Performs transforms.

        Args:
            signal (DatasetSignal): DatasetSignal to be transformed.

        Raises:
            NotImplementedError: Inherited classes must override this method.

        Returns:
            DatasetSignal: Transformed DatasetSignal.

        """
        raise NotImplementedError

### RF Transforms

class AdditiveNoiseDatasetTransform(DatasetTransform):
    """Apply wideband additive noise with specified parameters to DatasetSignal.

    Attributes:  
        power_range (Tuple[float, float]): Range bounds for interference power level (W). 
            Defaults to (0.01, 10.0).
        power_distribution (float): Random draw of interference power.
        color (str): Noise color, supports 'white', 'pink', or 'red' noise frequency spectrum types. Defaults to 'white'.
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

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        add_noise_power = self.power_distribution()
        
        if self.measure:
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


class AGC(DatasetTransform):
    """Automatic Gain Control performing sample-by-sample AGC algorithm.

    Attributes:
        rand_scale (Tuple): setting the random scaling bounds for each sample update. 
        rand_scale_distribution (Callable[[], float]): Random draw from rand_scale distribution.
        initial_gain_db (float): Inital gain value in dB.
        alpha_smooth (float): Alpha for avergaing the measure signal level `level_n = level_n * alpha + level_n-1(1-alpha)`
        alpha_track (float): Amount to adjust gain when in tracking state.
        alpha_overflow (float): Amount to adjust gain when in overflow state `[level_db + gain_db] >= max_level`.
        alpha_acquire (float): Amount to adjust gain when in acquire state.
        ref_level_db (float): Reference level goal for algorithm to achieve, in dB units. 
        track_range_db (float): dB range for operating in tracking state.
        low_level_db (float): minimum magnitude value (dB) to perform any gain control adjustment.
        high_level_db (float): magnitude value (dB) to enter overflow state.

    """
    def __init__(
        self,
        rand_scale = (1.0, 10.0),
        initial_gain_db: float = 0.0,
        alpha_smooth: float = 0.00004,
        alpha_track: float = 0.0004,
        alpha_overflow: float = 0.3,
        alpha_acquire: float = 0.04,
        ref_level_db: float = 0.0,
        track_range_db: float = 1.0,
        low_level_db: float = -80.0,
        high_level_db: float = 6.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rand_scale = rand_scale        
        self.initial_gain_db = initial_gain_db
        self.alpha_smooth = alpha_smooth
        self.alpha_track = alpha_track
        self.alpha_overflow = alpha_overflow
        self.alpha_acquire = alpha_acquire
        self.ref_level_db = ref_level_db
        self.track_range_db = track_range_db
        self.low_level_db = low_level_db
        self.high_level_db = high_level_db

        self.rand_scale_distribution = self.get_distribution(self.rand_scale )
        

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        rand_scale = self.rand_scale_distribution()

        alpha_acquire = self.random_generator.uniform(
            low = self.alpha_acquire / rand_scale,
            high = self.alpha_acquire * rand_scale,
            size = 1
        )[0]
        alpha_overflow = self.random_generator.uniform(
            low = self.alpha_overflow / rand_scale,
            high = self.alpha_overflow * rand_scale,
            size = 1
        )[0]
        alpha_track = self.random_generator.uniform(
            low = self.alpha_track / rand_scale,
            high = self.alpha_track * rand_scale,
            size = 1
        )[0]
        alpha_smooth = self.random_generator.uniform(
            low = self.alpha_smooth / rand_scale,
            high = self.alpha_smooth * rand_scale,
            size = 1
        )[0]
        ref_level_db = self.random_generator.uniform(
            low = -0.5 + self.ref_level_db,
            high = 0.5 + self.ref_level_db,
            size = 1
        )[0]

        signal.data = F.agc(
            np.ascontiguousarray(signal.data, dtype=np.complex64),
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

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal
    

class AWGN(DatasetTransform):
    """Apply Additive White Gaussian Noise to DatasetSignal.

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

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:

        if self.measure:            
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

        signal.data = F.awgn(
            signal.data,
            noise_power_db = self.noise_power_db,
            rng = self.random_generator
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class BlockAGC(DatasetTransform):
    """Implements a large instantaneous jump in receiver gain.

    Attributes:
        gain_change_db_range (Tuple): Sets the (min, max) gain change in dB.
        gain_change_db_distribution (Callable[[], float]): Random draw from gain_change_db distribution.
        
    """
    def __init__(
        self, 
        max_gain_change_db: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # define the range (min,max) for possible change in gain (assume range is symmetric)
        self.gain_change_db_range = (-max_gain_change_db,max_gain_change_db)
        self.gain_change_db_distribution = self.get_distribution(self.gain_change_db_range )
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        # select a gain value change from distribution
        gain_change_db = self.gain_change_db_distribution()
        # determine which samples gain change will be applied to. minimum index is 1, and maximum
        # index is second to last sample, such that at minimum the gain will be applied to one
        # sample or at maximum it will be applied to all less 1 samples. applying to zero samples
        # or to all samples does not have a practical effect for this specific transform.
        start_index = self.random_generator.integers(1,len(signal.data)-1)
        
        signal.data = F.block_agc(signal.data, gain_change_db, start_index)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class CarrierPhaseOffsetDatasetTransform(DatasetTransform):
    """Apply randomized phase offset to signal I/Q data.

    Attributes: 
        phase_offset_range (Tuple[float, float]): Phase range bounds in radians.
        phase_offset_distribution (Callable[[], float]): Random draw from phase_offset distribution.
    
    """
    def __init__(
        self,
        phase_offset_range: Tuple[float, float] = (0, 2*np.pi),
        **kwargs
    ):
        super().__init__(**kwargs)
        # by default, randomizes the phase across the 0 to 2pi radians range
        self.phase_offset_range = phase_offset_range
        self.phase_offset_distribution = self.get_distribution(self.phase_offset_range )
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        phase_offset = self.phase_offset_distribution()
        signal.data = F.phase_offset(signal.data, phase_offset)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class ComplexTo2D(DatasetTransform):
    """Converts IQ data to two channels (real and imaginary parts).
    """
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.complex_to_2d(signal.data)
        signal.data = signal.data.astype(torchsig_float_data_type)
        self.update(signal)
        return signal


class IQImbalanceDatasetTransform(DatasetTransform):
    """Applies a set of IQImbalance effects to a DatasetSignal: amplitude, phase, and DC offset.

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
        self.dc_offset = dc_offset #dc_offset, both I/Q components
        self.amplitude_imbalance_distribution = self.get_distribution(self.amplitude_imbalance )
        self.phase_imbalance_distribution = self.get_distribution(self.phase_imbalance )
        self.dc_offset_distribution = self.get_distribution(self.dc_offset )
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        amplitude_imbalance = self.amplitude_imbalance_distribution()
        phase_imbalance = self.phase_imbalance_distribution()
        dc_offset = self.dc_offset_distribution()

        signal.data = F.iq_imbalance(signal.data, amplitude_imbalance, phase_imbalance, dc_offset)

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class LocalOscillatorFrequencyDriftDatasetTransform(DatasetTransform):
    """Apply LO frequency drift to DatasetSignal.

    Attributes:
        drift_ppm (Tuple[float, float]): Drift in parts per million (ppm). Default (0.1, 1).
        
    """
    def __init__(
        self, 
        drift_ppm: Tuple[float, float] = (0.1, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.drift_ppm = drift_ppm
        self.drift_ppm_distribution = self.get_distribution(self.drift_ppm,'log10')
    
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        drift_ppm = self.drift_ppm_distribution()

        signal.data = F.local_oscillator_frequency_drift(
            data = signal.data, 
            drift_ppm = drift_ppm,
            rng = self.random_generator
        )

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class LocalOscillatorPhaseNoiseDatasetTransform(DatasetTransform):
    """Applies LO phase noise to DatasetSignal.

    Attributes:
        phase_noise_degrees (Tuple[float, float]): Range of phase noise (in degrees). Defaults to (0.25,1).
        
    """
    def __init__(
        self, 
        phase_noise_degrees: Tuple[float, float] = (0.25, 1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.phase_noise_degrees = phase_noise_degrees
        self.phase_noise_degrees_distribution = self.get_distribution(self.phase_noise_degrees)
    
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        phase_noise_degrees = self.phase_noise_degrees_distribution()

        signal.data = F.local_oscillator_phase_noise(
            data = signal.data,
            phase_noise_degrees = phase_noise_degrees,
            rng = self.random_generator
        )

        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class NonlinearAmplifierDatasetTransform(DatasetTransform):
    """Applies a memoryless nonlinear amplifier model to DatasetSignal.

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

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        
        # apply a wideband nonlinear response to entire band
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


class PassbandRippleDatasetTransform(DatasetTransform):
    """Applies a model of wideband analog filter passband ripple for DatasetSignals.

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
    
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.passband_ripple(
            data = signal.data,
            filter_coeffs = self.fir_coeffs,
            normalize = True
        )
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class QuantizeDatasetTransform(DatasetTransform):
    """Quantize signal I/Q samples into specified levels with a rounding method.

    Attributes:
        num_levels: Number of quantization levels.
        num_levels_distribution (Callable[[], int]): Random draw from num_levels distribution.
        round_type (str, List[str]): Quantization rounding method. Must be 'floor', 
                'nearest' or 'ceiling'. Defaults to 'ceiling'.
        round_type_distribution (Callable[[], str]): Random draw from round_type distribution.
    
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
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
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


class SpectralInversionDatasetTransform(DatasetTransform):
    """Invert spectrum of a DatasetSignal.

    """
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        
        signal.data = F.spectral_inversion(signal.data)
        
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # inverted data is mirrored in frequency
        for m in signal.metadata:
            m.center_freq *= -1
        
        self.update(signal)
        return signal  

          
class TimeVaryingNoise(DatasetTransform):
    """Add time-varying noise to DatasetSignal regions.

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
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
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
        

#### ML Transforms -------------------------
class AddSlope(DatasetTransform):
    """Add the slope of each sample with its preceeding sample to itself.
    Creates a weak 0 Hz IF notch filtering effect.

    """
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.add_slope(signal.data) 
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class ChannelSwap(DatasetTransform):
    """Swaps the I and Q channels of complex input data.
    
    """
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.channel_swap(signal.data)

        signal.data = signal.data.astype(torchsig_complex_data_type)

        #metadata: swapping I/Q channels creates a frequency mirroring
        for m in signal.metadata:
            m.center_freq *= -1
        
        self.update(signal)
        return signal
        

class CutOut(DatasetTransform):
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

        self.duration_distribution = self.get_distribution(self.duration )
        self.cut_type_distribution = self.get_distribution(self.cut_type )
        
        

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

        self.update(signal)
        return signal


class PatchShuffle(DatasetTransform):
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
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
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


class RandomDropSamples(DatasetTransform):
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
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
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


class RandomMagRescale(DatasetTransform):
    """Randomly apply a magnitude rescaling, emulating a change in a receiver's
    gain control.

    Attributes:
         start (int, float, list, tuple):
            start sets the time when the rescaling kicks in
            * If int or float, start is fixed at the value provided.
            * If list, start is any element in the list.
            * If tuple, start is in range of (tuple[0], tuple[1]).
        start_distribution (Callable[[], float]): Random draw from start distribution.
        scale (int, float, list, tuple):
            scale sets the magnitude of the rescale
            * If int or float, scale is fixed at the value provided.
            * If list, scale is any element in the list.
            * If tuple, scale is in range of (tuple[0], tuple[1]).
        scale_distribution (Callable[[], float]): Random draw from scale distribution.

    """

    def __init__(
        self,
        start = (0.0, 0.9),
        scale = (-4.0, 4.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.start = start
        self.scale = scale

        self.start_distribution = self.get_distribution(self.start )
        self.scale_distribution = self.get_distribution(self.scale )

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        start = self.start_distribution()
        scale = self.scale_distribution()
        
        signal.data = F.mag_rescale(signal.data, start, scale)
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class Spectrogram(DatasetTransform):
    """Computes the spectogram of IQ data.

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

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.spectrogram(
            signal.data, 
            self.fft_size, 
            self.fft_stride, 
        )
        self.update(signal)
        return signal


class SpectrogramDropSamples(DatasetTransform):
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
        
    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
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

            # SpectrogramDropSamples can have complicated signal feature effects in practice.
            # Any desired metadata updates should be made manually.
            
            self.update(signal)
        
        return signal


class TimeReversal(DatasetTransform):
    """Applies a time reversal to the input. 
    
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

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = F.time_reversal(signal.data)

        do_si = self.random_generator.random() > self.allow_spectral_inversion

        if do_si:
            signal.data = F.spectral_inversion(signal.data)

        signal.data = signal.data.astype(torchsig_complex_data_type)

        num_data_samples = len(signal.data)
        for m in signal.metadata:
            original_stop = m.stop_in_samples
            m.start_in_samples = num_data_samples - original_stop
            if not do_si:
                m.center_freq *= -1
        
        self.update(signal)
        return signal

