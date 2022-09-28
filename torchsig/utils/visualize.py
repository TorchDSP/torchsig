import pywt
import numpy as np
from scipy import ndimage
from scipy import signal as sp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from typing import Optional, Callable, Iterable, Union, Tuple, List


class Visualizer:
    """A non-entirely abstract class which represents a visualization of a dataset

    Args:
        data_loader:
            A Dataloader to sample from for plotting
            
        visualize_transform:
            Defines how to transform the data prior to plotting
            
        visualize_target_transform:
            Defines how to transform the target prior to plotting
            
    """
    def __init__(
        self,
        data_loader: DataLoader,
        visualize_transform: Optional[Callable] = None,
        visualize_target_transform: Optional[Callable] = None
    ):
        self.data_loader = iter(data_loader)
        self.visualize_transform = visualize_transform
        self.visualize_target_transform = visualize_target_transform

    def __iter__(self) -> Iterable:
        self.data_iter = iter(self.data_loader)
        return self

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)
        if self.visualize_transform:
            iq_data = self.visualize_transform(iq_data)

        if self.visualize_target_transform:
            targets = self.visualize_target_transform(targets)

        return self._visualize(iq_data, targets)

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        raise NotImplementedError


class SpectrogramVisualizer(Visualizer):
    """Visualize spectrograms using SciPy spectrogram function

    Args:
        sample_rate:
            The sample rate of the input data
            
        window:
            The window for use in the spectrogram
            
        nperseg:
            Specify the segments for the spectrogram
            
        noverlap:
            Specify the overlap for the spectrogram
            
        nfft:
            Specify the number of FFT bins for the spectrogram
            
        **\*\*kwargs:**
            Keyword arguments
    
    """
    def __init__(
        self,
        sample_rate: float = 1.0,
        window: Optional[Union[str, Tuple, np.ndarray]] = sp.windows.tukey(256, .25),
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        **kwargs
    ):
        super(SpectrogramVisualizer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            _, _, spectrogram = sp.spectrogram(
                x=iq_data[sample_idx],
                fs=self.sample_rate,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                return_onesided=False
            )
            spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram),axes=0))
            plt.imshow(
                spectrogram,
                vmin=np.min(spectrogram[spectrogram != -np.inf]),
                vmax=np.max(spectrogram[spectrogram != np.inf]),
                aspect="auto",
                cmap="jet"
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class WaveletVisualizer(Visualizer):
    """Visualize wavelet using PyWavelet spectrogram function

    Args:
        wavelet:
            The wavelet to apply to the data prior to plotting
            
        nscales:
            Specify the number of wavelet scales
            
        sample_rate:
            The sample rate of the input data
            
        **\*\*kwargs:**
            Keyword arguments
            
    """
    def __init__(
        self,
        wavelet: str = 'mexh',
        nscales: int = 33,
        sample_rate: float = 1.0,
        **kwargs
    ):
        super(WaveletVisualizer, self).__init__(**kwargs)
        self.wavelet = wavelet
        self.nscales = nscales
        self.sample_rate = sample_rate

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            scales = np.arange(1, self.nscales)
            cwt_matrix, freqs = pywt.cwt(
                iq_data[sample_idx],
                scales=scales,
                wavelet=self.wavelet,
                sampling_period=1.0 / self.sample_rate
            )
            ts = np.arange(len(cwt_matrix[0])) / self.sample_rate
            plt.imshow(
                np.abs(cwt_matrix),
                extent=[ts[0], ts[-1], freqs[-1], freqs[0]],
                vmin=0,
                vmax=np.abs(cwt_matrix).max(),
                aspect="auto",
                cmap="jet"  # 'PRGn'
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class ConstellationVisualizer(Visualizer):
    """Visualize a constellation

    Args:
        **\*\*kwargs:**
            Keyword arguments
            
    """
    def __init__(self, **kwargs):
        super(ConstellationVisualizer, self).__init__(**kwargs)

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            plt.scatter(np.real(iq_data[sample_idx]),
                        np.imag(iq_data[sample_idx]))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class IQVisualizer(Visualizer):
    """Visualize time-series IQ data

    Args:
        **\*\*kwargs:**
            Keyword arguments
            
    """
    def __init__(self, **kwargs):
        super(IQVisualizer, self).__init__(**kwargs)

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            plt.plot(np.real(iq_data[sample_idx]))
            plt.plot(np.imag(iq_data[sample_idx]))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class TimeSeriesVisualizer(Visualizer):
    """Visualize time-series data directly

    Args:
        **\*\*kwargs:**
            Keyword arguments
    """
    def __init__(self, **kwargs):
        super(TimeSeriesVisualizer, self).__init__(**kwargs)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            plt.plot(data[sample_idx])
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class ImageVisualizer(Visualizer):
    """Visualize image data directly

    Args:
        **\*\*kwargs:**
            Keyword arguments
            
    """
    def __init__(self, **kwargs):
        super(ImageVisualizer, self).__init__(**kwargs)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(int(np.ceil(np.sqrt(batch_size))),
                        int(np.sqrt(batch_size)), sample_idx + 1)
            plt.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx][data[sample_idx] != -np.inf]),
                vmax=np.max(data[sample_idx][data[sample_idx] != np.inf]),
                aspect="auto",
                cmap="jet"
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))

        return figure

    
###############################################################################
# Visualizer Transform Functions
###############################################################################

def two_channel_to_complex(tensor: np.ndarray) -> np.ndarray:
    """Visualizer data transform: Transform two channel IQ data to complex IQ 
    data for visualization

    """
    batch_size = tensor.shape[0]
    new_tensor = np.zeros((batch_size, tensor.shape[2]), dtype=np.complex128)
    for idx in range(tensor.shape[0]):
        new_tensor[idx].real = tensor[idx, 0]
        new_tensor[idx].imag = tensor[idx, 1]
    return new_tensor


def complex_spectrogram_to_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Visualizer data transform: Transform two channel spectrogram data for
    spectrogram magnitude visualization (mode = 'complex')

    """
    batch_size = tensor.shape[0]
    new_tensor = np.zeros(
        (batch_size, tensor.shape[2], tensor.shape[3]), dtype=np.float64)
    for idx in range(tensor.shape[0]):
        new_tensor[idx] = 20 * np.log10(tensor[idx, 0]**2 + tensor[idx, 1]**2)
    return new_tensor


def magnitude_spectrogram(tensor: np.ndarray) -> np.ndarray:
    """Visualizer data transform: Transform magnitude spectrogram data for 
    plotting (mode = 'psd')

    """
    batch_size = tensor.shape[0]
    new_tensor = np.zeros(
        (batch_size, tensor.shape[1], tensor.shape[2]), dtype=np.float64)
    for idx in range(tensor.shape[0]):
        new_tensor[idx] = 20 * np.log10(tensor[idx])
    return new_tensor


def iq_to_complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Visualizer data transform: Complex IQ to time series magnitude for 
    TimeSeriesVisualizer

    """
    batch_size = tensor.shape[0]
    new_tensor = np.zeros((batch_size, tensor.shape[1]))
    for idx in range(batch_size):
        new_tensor[idx] = np.abs(tensor[idx])
    return new_tensor


def binary_label_format(tensor: np.ndarray) -> List[str]:
    """Visualizer target transform: Format binary labels for titles in 
    visualizer

    """
    batch_size = tensor.shape[0]
    label = []
    for idx in range(batch_size):
        label.append("Label: {}".format(tensor[idx].numpy()))
    return label


def onehot_label_format(tensor: np.ndarray) -> List[str]:
    """Visualizer target transform: Format onehot labels for titles in 
    visualizer

    """
    batch_size = tensor.shape[0]
    label = []
    for idx in range(batch_size):
        label.append("Class: {}".format(np.where(tensor[idx] > 0)[0][0]))
    return label


def multihot_label_format(tensor: np.ndarray, class_list: List[str]) -> List[List[str]]:
    """Target Transform: Format multihot labels for titles in visualizer
    
    """
    batch_size = tensor.shape[0]
    label = []
    for idx in range(batch_size):
        curr_label = []
        for class_idx in range(len(class_list)):
            if tensor[idx][class_idx] > (1/len(class_list)):
                curr_label.append(class_list[class_idx])
        label.append(curr_label)
    return label
