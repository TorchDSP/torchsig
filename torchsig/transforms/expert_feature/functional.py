import pywt
import numpy as np
from scipy import signal
from typing import Callable


def interleave_complex(tensor: np.ndarray) -> np.ndarray:
    """Converts complex vectors to real interleaved IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Interleaved vectors.
    """
    new_tensor = np.empty((tensor.shape[0]*2,))
    new_tensor[::2] = np.real(tensor)
    new_tensor[1::2] = np.imag(tensor)
    return new_tensor


def complex_to_2d(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to two channels representing real and imaginary

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Expanded vectors
    """

    new_tensor = np.zeros((2, tensor.shape[0]), dtype=np.float64)
    new_tensor[0] = np.real(tensor).astype(np.float64)
    new_tensor[1] = np.imag(tensor).astype(np.float64)
    return new_tensor


def real(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a real-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            real(tensor)
    """
    return np.real(tensor)


def imag(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a imaginary-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            imag(tensor)
    """
    return np.imag(tensor)


def complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a complex magnitude vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            abs(tensor)
    """
    return np.abs(tensor)


def wrapped_phase(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a wrapped-phase vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            angle(tensor)
    """
    return np.angle(tensor)


def discrete_fourier_transform(tensor: np.ndarray) -> np.ndarray:
    """Computes DFT of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            fft(tensor). normalization is 1/sqrt(n)
    """
    return np.fft.fft(tensor, norm="ortho")


def spectrogram(
        tensor: np.ndarray,
        nperseg: int,
        noverlap: int,
        nfft: int,
        window_fcn: Callable[[int], np.ndarray],
        mode: str,
) -> np.ndarray:
    """Computes spectrogram of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        nperseg (:obj:`int`):
            Length of each segment. If window is str or tuple, is set to 256,
            and if window is array_like, is set to the length of the window.

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 8.

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            If None, the FFT length is nperseg.

        window_fcn (:obj:`Callable`):
            Function generating the window for each FFT
        
        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            
    Returns:
        transformed (:class:`numpy.ndarray`):
            Spectrogram of tensor along time dimension
    """
    _, _, spectrograms = signal.spectrogram(
        tensor,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window_fcn(nperseg),
        return_onesided=False,
        mode=mode,
        axis=0
    )
    return np.fft.fftshift(spectrograms, axes=0)


def continuous_wavelet_transform(
        tensor: np.ndarray,
        wavelet: str,
        nscales: int,
        sample_rate: float
) -> np.ndarray:
    """Computes the continuous wavelet transform resulting in a Scalogram of the complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        wavelet (:obj:`str`):
            Name of the mother wavelet.
            If None, wavename = 'mexh'.

        nscales (:obj:`int`):
            Number of scales to use in the Scalogram.
            If None, nscales = 33.

        sample_rate (:obj:`float`):
            Sample rate of the signal.
            If None, fs = 1.0.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Scalogram of tensor along time dimension
    """
    scales = np.arange(1, nscales)
    cwtmatr, _ = pywt.cwt(
        tensor, 
        scales=scales, 
        wavelet=wavelet, 
        sampling_period=1.0/sample_rate
    )

    # if the dtype is complex then return the magnitude
    if np.iscomplexobj(cwtmatr):
        cwtmatr = abs(cwtmatr)

    return cwtmatr
