from scipy.signal import ShortTimeFFT
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splrep, splev
from PIL import Image
import numpy as np
import pywt
import cv2

def real_imag_vstacked_cwt_image(signal, wavelet='morl', scales=np.arange(1, 64), sampling_rate=1.0, output_size=(224, 112)):
    """
    Converts a complex signal to a CWT image using the specified wavelet and scales.

    Parameters:
        signal (np.ndarray): Complex input signal.
        wavelet (str): Type of wavelet to use for the CWT. Defaults to 'morl' (Morlet wavelet).
        scales (np.ndarray): Scales to use for the CWT. Defaults to np.arange(1, 128).
        sampling_rate (float): Sampling rate of the signal. Defaults to 1.0.
        output_size (tuple): Size of the output image. Defaults to (224, 224).

    Returns:
        np.array cv2 image of real and imaginary cwt of signal vertically stacked.
    """
    # Perform the Continuous Wavelet Transform
    real_coeffs, _ = pywt.cwt(signal.real, scales=scales, wavelet=wavelet, sampling_period=1/sampling_rate)
    imag_coeffs, _ = pywt.cwt(signal.imag, scales=scales, wavelet=wavelet, sampling_period=1/sampling_rate)

    
    # Convert the CWT coefficients to power (magnitude squared)
    real_power = np.abs(real_coeffs)**2
    imag_power = np.abs(imag_coeffs)**2

    # Normalize the power spectrum to the range [0, 255]
    real_normalized = np.uint8(255 * (real_power / np.max(real_power)))
    imag_normalized = np.uint8(255 * (imag_power / np.max(imag_power)))

    # Resize the CWT image to the desired output size using cv2
    real_resized = cv2.resize(real_normalized, output_size, interpolation=cv2.INTER_LINEAR)
    imag_resized = cv2.resize(imag_normalized, output_size, interpolation=cv2.INTER_LINEAR)

    # Apply the Twilight colormap using cv2
    real_colormap = cv2.applyColorMap(real_resized, cv2.COLORMAP_HSV)
    imag_colormap = cv2.applyColorMap(imag_resized, cv2.COLORMAP_HSV)

    result = cv2.vconcat([real_colormap, imag_colormap])

    # Convert to PIL Image and return
    return result



def spectrogram_image(signal, nfft=64):
    # Calculate the spectrogram
    window = np.blackman(nfft)
    STF = ShortTimeFFT(win=window, hop=window.shape[0]+1, fs=len(signal), fft_mode='centered', scale_to='psd')
    spec = STF.spectrogram(signal)
    spec = 10*np.log10(spec)
    img = np.zeros((spec.shape[0], spec.shape[1], 3), dtype=np.float32)
    img = cv2.normalize(spec, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    return img

class PLL:
    def __init__(self, sample_rate, loop_bandwidth, damping_factor):
        self.sample_rate = sample_rate
        self.loop_bandwidth = loop_bandwidth
        self.damping_factor = damping_factor
        self.phase_error_integral = 0
        self.omega = 2 * np.pi * self.loop_bandwidth
        self.filter_alpha = 1 / (1 + (self.omega / (2 * np.pi * self.damping_factor))**2)
        self.filter_beta = (self.filter_alpha ** 2) / (4 * self.damping_factor)

    def phase_detector(self, signal, reference):
        """Compute the phase difference between the signal and the reference."""
        phase_diff = np.angle(signal * np.conj(reference))
        return phase_diff

    def loop_filter(self, phase_error):
        """Apply a PI controller to the phase error."""
        proportional = self.filter_alpha * np.mean(phase_error)
        integral = self.filter_beta * self.phase_error_integral
        self.phase_error_integral += np.mean(phase_error)
        control_signal = proportional + integral
        return control_signal

    def vco(self, control_signal, length):
        """Generate a signal with frequency adjusted by the control signal."""
        time = np.arange(length) / self.sample_rate
        return np.exp(1j * (2 * np.pi * np.cumsum(np.full(length, control_signal)) / self.sample_rate))

    def correct_signal(self, signal):
        """Correct the input signal using the PLL."""
        reference = np.exp(1j * np.angle(signal))  # Initial reference
        phase_error = self.phase_detector(signal, reference)
        control_signal = self.loop_filter(phase_error)
        corrected_signal = signal * np.conj(self.vco(control_signal, len(signal)))
        return corrected_signal

def upsample_iq(iq, target_length):
    iqx, iqy = iq.real, iq.imag
    original_indices = np.arange(len(iq))
    target_indices = np.linspace(0, len(iq) - 1, target_length)
    
    # Compute the spline representation for both real and imaginary parts
    tck_iqx = splrep(original_indices, iqx, s=0)
    tck_iqy = splrep(original_indices, iqy, s=0)
    
    # Evaluate the spline over the target indices
    iqx_interp = splev(target_indices, tck_iqx, der=0)
    iqy_interp = splev(target_indices, tck_iqy, der=0)
    
    # Recombine into a complex signal
    iq_upsampled = iqx_interp + 1j * iqy_interp
    return iq_upsampled

def complex_iq_to_heatmap(iq, output_size=(224, 224), loop_bandwidth=1000, damping_factor=1, amp_factor=.5):
    sample_rate = len(iq)
    pll = PLL(sample_rate, loop_bandwidth, damping_factor)
    iq_corrected = pll.correct_signal(iq)
    
    target_length = output_size[0] * output_size[1]
    iq_upsampled = upsample_iq(iq_corrected, target_length)
    
    iqx, iqy = iq_upsampled.real, iq_upsampled.imag
    iqx = (iqx - iqx.min()) / (iqx.max() - iqx.min() + 1e-8)
    iqy = (iqy - iqy.min()) / (iqy.max() - iqy.min() + 1e-8)
    
    heatmap = np.zeros(output_size)
    ix = (iqx * (output_size[1] - 1)).astype(int)
    iy = (iqy * (output_size[0] - 1)).astype(int)
    
    np.add.at(heatmap, (iy, ix), 1)
    
    heatmap = np.power(heatmap, amp_factor)
    
    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=1.0)
    
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    
    heatmap_8bit = np.uint8(heatmap * 255)
    
    # Contrast adjustment
    heatmap_8bit = cv2.equalizeHist(heatmap_8bit)
    
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_TWILIGHT)
    
    return heatmap_colored
