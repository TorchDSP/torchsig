import numpy as np
from typing import Callable, Tuple, Any

from torchsig.utils.types import SignalData
from torchsig.transforms.expert_feature import functional as F
from torchsig.transforms.transforms import SignalTransform


class InterleaveComplex(SignalTransform):
    """ Converts complex IQ samples to interleaved real and imaginary floating
    point values.

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.InterleaveComplex()

    """
    def __init__(self):
        super(InterleaveComplex, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.interleave_complex(data.iq_data)
        else:
            data = F.interleave_complex(data)
        return data


class ComplexTo2D(SignalTransform):
    """ Takes a vector of complex IQ samples and converts two channels of real 
    and imaginary parts

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ComplexTo2D()

    """
    def __init__(self):
        super(ComplexTo2D, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.complex_to_2d(data.iq_data)
        else:
            data = F.complex_to_2d(data)
        return data


class Real(SignalTransform):
    """ Takes a vector of complex IQ samples and returns Real portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Real()

    """
    def __init__(self):
        super(Real, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.real(data.iq_data)
        else:
            data = F.real(data)
        return data


class Imag(SignalTransform):
    """ Takes a vector of complex IQ samples and returns Imaginary portions

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Imag()

    """
    def __init__(self):
        super(Imag, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.imag(data.iq_data)
        else:
            data = F.imag(data)
        return data
    

class ComplexMagnitude(SignalTransform):
    """ Takes a vector of complex IQ samples and returns the complex magnitude

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ComplexMagnitude()

    """
    def __init__(self):
        super(ComplexMagnitude, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.complex_magnitude(data.iq_data)
        else:
            data = F.complex_magnitude(data)
        return data


class WrappedPhase(SignalTransform):
    """ Takes a vector of complex IQ samples and returns wrapped phase (-pi, pi)

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.WrappedPhase()

    """
    def __init__(self):
        super(WrappedPhase, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.wrapped_phase(data.iq_data)
        else:
            data = F.wrapped_phase(data)
        return data
        

class DiscreteFourierTransform(SignalTransform):
    """ Calculates DFT using FFT

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.DiscreteFourierTransform()

    """
    def __init__(self):
        super(DiscreteFourierTransform, self).__init__()

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.discrete_fourier_transform(data.iq_data)
        else:
            data = F.discrete_fourier_transform(data)
        return data


class ChannelConcatIQDFT(SignalTransform):
    """ Converts the input IQ into 2D tensor of the real & imaginary components
    concatenated in the channel dimension. Next, calculate the DFT using the 
    FFT, convert the complex DFT into a 2D tensor of real & imaginary frequency
    components. Finally, stack the 2D IQ and the 2D DFT components in the 
    channel dimension.
    
    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.ChannelConcatIQDFT()
    
    """
    def __init__(self):
        super(ChannelConcatIQDFT, self).__init__()
    
    def __call__(self, data: Any) -> Any:
        iq_data = data.iq_data if isinstance(data, SignalData) else data
        dft_data = F.discrete_fourier_transform(iq_data)
        iq_data = F.complex_to_2d(iq_data)
        dft_data = F.complex_to_2d(dft_data)
        output_data = np.concatenate([iq_data, dft_data], axis=0)
        if isinstance(data, SignalData):
            data.iq_data = output_data
        else:
            data = output_data
        return data

    
class Spectrogram(SignalTransform):
    """ Calculates power spectral density over time

    Args:
        nperseg (:obj:`int`):
            Length of each segment. If window is str or tuple, is set to 256,
            and if window is array_like, is set to the length of the window.

        noverlap (:obj:`int`):
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 8.

        nfft (:obj:`int`):
            Length of the FFT used, if a zero padded FFT is desired.
            If None, the FFT length is nperseg.
            
        window_fcn (:obj:`str`):
            Window to be used in spectrogram operation.
            Default value is 'np.blackman'.

        mode (:obj:`str`):
            Mode of the spectrogram to be computed.
            Default value is 'psd'.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Spectrogram with seg_size=256, overlap=64, nfft=256, window=blackman_harris
        >>> transform = ST.Spectrogram()
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in time)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64)
        >>> # Spectrogram with seg_size=128, overlap=0, nfft=128, window=blackman_harris (critically sampled)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=0)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=blackman_harris (2x oversampled in frequency)
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256)
        >>> # Spectrogram with seg_size=128, overlap=64, nfft=128, window=rectangular
        >>> transform = ST.Spectrogram(nperseg=128, noverlap=64, nfft=256, window_fcn=np.ones)

    """
    def __init__(
        self,
        nperseg: int = 256,
        noverlap: int = None,
        nfft: int = None,
        window_fcn: Callable[[int], np.ndarray] = np.blackman,
        mode: str = 'psd'
    ):
        super(Spectrogram, self).__init__()
        self.nperseg = nperseg
        self.noverlap = nperseg/4 if noverlap is None else noverlap
        self.nfft = nperseg if nfft is None else nfft
        self.window_fcn = window_fcn
        self.mode = mode

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.spectrogram(data.iq_data)
            if self.mode == "complex":
                new_tensor = np.zeros((2, data.iq_data.shape[0], data.iq_data.shape[1]), dtype=np.float32)
                new_tensor[0, :, :] = np.real(data.iq_data).astype(np.float32)
                new_tensor[1, :, :] = np.imag(data.iq_data).astype(np.float32)
                data.iq_data = new_tensor
        else:
            data = F.spectrogram(data)
            if self.mode == "complex":
                new_tensor = np.zeros((2, data.shape[0], data.shape[1]), dtype=np.float32)
                new_tensor[0, :, :] = np.real(data).astype(np.float32)
                new_tensor[1, :, :] = np.imag(data).astype(np.float32)
                data = new_tensor
        return data


class ContinuousWavelet(SignalTransform):
    """Computes the continuous wavelet transform resulting in a Scalogram of 
    the complex IQ vector

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

    Example:
        >>> import torchsig.transforms as ST
        >>> # ContinuousWavelet SignalTransform using the 'mexh' mother wavelet with 33 scales
        >>> transform = ST.ContinuousWavelet()
        
    """
    def __init__(
            self,
            wavelet: str = 'mexh',
            nscales: int = 33,
            sample_rate: float = 1.0
    ):
        super(ContinuousWavelet, self).__init__()
        self.wavelet = wavelet
        self.nscales = nscales
        self.sample_rate = sample_rate

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = F.continuous_wavelet_transform(
                data.iq_data, 
                self.wavelet, 
                self.nscales, 
                self.sample_rate,
            )
        else:
            data = F.continuous_wavelet_transform(
                data, 
                self.wavelet, 
                self.nscales, 
                self.sample_rate,
            )
        return data
    
    
class ReshapeTransform(SignalTransform):
    """Reshapes the input data to the specified shape
    
    Args:
        new_shape (obj:`tuple`):
            The new shape for the input data

    """
    def __init__(self, new_shape: Tuple, **kwargs):
        super(ReshapeTransform, self).__init__(**kwargs)
        self.new_shape = new_shape

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            data.iq_data = data.iq_data.reshape(*self.new_shape)
        else:
            data = data.reshape(*self.new_shape)
        return data
