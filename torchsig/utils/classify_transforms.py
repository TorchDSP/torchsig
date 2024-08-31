from scipy.signal import ShortTimeFFT
from scipy.interpolate import interp1d
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

def upsample_iq(iq, target_length):
    # Separate real and imaginary components
    iqx, iqy = iq.real, iq.imag
    
    # Create a linear space for interpolation
    original_indices = np.arange(len(iq))
    target_indices = np.linspace(0, len(iq) - 1, target_length)
    
    # Interpolate real and imaginary parts
    iqx_interp = interp1d(original_indices, iqx, kind='cubic')(target_indices)
    iqy_interp = interp1d(original_indices, iqy, kind='cubic')(target_indices)
    
    # Recombine into a complex signal
    iq_upsampled = iqx_interp + 1j * iqy_interp
    return iq_upsampled

def complex_iq_to_heatmap(iq, output_size=[128, 128]):
    # Calculate resolution based on the square root of the data length
    resolution = int(np.sqrt(len(iq)))
    
    # Calculate the target upsample length
    target_length = output_size[0] * output_size[1]
    
    # Upsample I/Q signal to match the output size dimensions
    iq_upsampled = upsample_iq(iq, target_length)
    
    # Separate real and imaginary components
    iqx, iqy = iq_upsampled.real, iq_upsampled.imag
    
    # Normalize the real and imaginary components to [0, 1]
    iqx = (iqx - iqx.min()) / (iqx.max() - iqx.min() + 1e-8)
    iqy = (iqy - iqy.min()) / (iqy.max() - iqy.min() + 1e-8)
    
    # Initialize an empty heatmap
    heatmap = np.zeros(output_size)
    
    # Map the I/Q values to the output size grid
    ix = (iqx * (output_size[1] - 1)).astype(int)
    iy = (iqy * (output_size[0] - 1)).astype(int)
    
    # Accumulate counts in the heatmap
    np.add.at(heatmap, (iy, ix), 1)
    
    # Normalize the heatmap to [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    
    # Convert the heatmap to 8-bit for visualization
    heatmap_8bit = np.uint8(heatmap * 255)
    
    # Apply a colormap to the heatmap using OpenCV
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_TWILIGHT)
    
    return heatmap_colored
