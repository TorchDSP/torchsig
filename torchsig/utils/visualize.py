from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pywt
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage
from scipy import signal as sp


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
        data_loader,
        visualize_transform: Optional[Callable] = None,
        visualize_target_transform: Optional[Callable] = None,
    ) -> None:
        self.data_loader = iter(data_loader)
        self.visualize_transform = visualize_transform
        self.visualize_target_transform = visualize_target_transform

    def __iter__(self) -> Iterable:
        self.data_iter = iter(self.data_loader)
        return self  # type: ignore

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

        ****kwargs:**
            Keyword arguments

    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        window: Union[str, Tuple, np.ndarray] = sp.windows.tukey(256, 0.25),
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        **kwargs,
    ) -> None:
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
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            _, _, spectrogram = sp.spectrogram(
                x=iq_data[sample_idx],
                fs=self.sample_rate,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                return_onesided=False,
            )
            spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram), axes=0))
            plt.imshow(
                spectrogram,
                vmin=np.min(spectrogram[spectrogram != -np.inf]),
                vmax=np.max(spectrogram[spectrogram != np.inf]),
                aspect="auto",
                cmap="jet",
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

        ****kwargs:**
            Keyword arguments

    """

    def __init__(
        self,
        wavelet: str = "mexh",
        nscales: int = 33,
        sample_rate: float = 1.0,
        **kwargs,
    ) -> None:
        super(WaveletVisualizer, self).__init__(**kwargs)
        self.wavelet = wavelet
        self.nscales = nscales
        self.sample_rate = sample_rate

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            scales = np.arange(1, self.nscales)
            cwt_matrix, freqs = pywt.cwt(
                iq_data[sample_idx],
                scales=scales,
                wavelet=self.wavelet,
                sampling_period=1.0 / self.sample_rate,
            )
            ts = np.arange(len(cwt_matrix[0])) / self.sample_rate
            plt.imshow(
                np.abs(cwt_matrix),
                extent=[ts[0], ts[-1], freqs[-1], freqs[0]],
                vmin=0,
                vmax=np.abs(cwt_matrix).max(),
                aspect="auto",
                cmap="jet",  # 'PRGn'
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class ConstellationVisualizer(Visualizer):
    """Visualize a constellation

    Args:
        ****kwargs:**
            Keyword arguments

    """

    def __init__(self, **kwargs) -> None:
        super(ConstellationVisualizer, self).__init__(**kwargs)

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            plt.scatter(np.real(iq_data[sample_idx]), np.imag(iq_data[sample_idx]))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class IQVisualizer(Visualizer):
    """Visualize time-series IQ data

    Args:
        ****kwargs:**
            Keyword arguments

    """

    def __init__(self, **kwargs) -> None:
        super(IQVisualizer, self).__init__(**kwargs)

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            plt.plot(np.real(iq_data[sample_idx]))
            plt.plot(np.imag(iq_data[sample_idx]))
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class TimeSeriesVisualizer(Visualizer):
    """Visualize time-series data directly

    Args:
        **kwargs:**
            Keyword arguments
    """

    def __init__(self, **kwargs) -> None:
        super(TimeSeriesVisualizer, self).__init__(**kwargs)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            plt.plot(data[sample_idx])
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))
        return figure


class ImageVisualizer(Visualizer):
    """Visualize image data directly

    Args:
        ****kwargs:**
            Keyword arguments

    """

    def __init__(self, **kwargs) -> None:
        super(ImageVisualizer, self).__init__(**kwargs)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            plt.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx][data[sample_idx] != -np.inf]),
                vmax=np.max(data[sample_idx][data[sample_idx] != np.inf]),
                aspect="auto",
                cmap="jet",
            )
            plt.xticks([])
            plt.yticks([])
            plt.title(str(targets[sample_idx]))

        return figure


class PSDVisualizer(Visualizer):
    """Visualize a PSD

    Args:
        fft_size:
        **kwargs:
    """

    def __init__(self, fft_size: int = 1024, **kwargs) -> None:
        super(PSDVisualizer, self).__init__(**kwargs)
        self.fft_size = fft_size

    def _visualize(self, iq_data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = iq_data.shape[0]
        figure = plt.figure()
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            Pxx, freqs = plt.psd(iq_data[sample_idx], NFFT=self.fft_size, Fs=1)
            plt.xticks()
            plt.yticks()
            plt.title(str(targets[sample_idx]))
        return figure


class MaskVisualizer(Visualizer):
    """Visualize data with mask label information overlaid

    Args:
        **kwargs:
    """

    def __init__(self, **kwargs) -> None:
        super(MaskVisualizer, self).__init__(**kwargs)

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)
        if self.visualize_transform:
            iq_data = self.visualize_transform(deepcopy(iq_data))

        if self.visualize_target_transform:
            targets = self.visualize_target_transform(deepcopy(targets))
        else:
            targets = None

        return self._visualize(iq_data, targets)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure(frameon=False)
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            extent = 0, data.shape[1], 0, data.shape[2]
            data_img = plt.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
                cmap="jet",
                extent=extent,
            )
            if targets is not None:
                label = targets[sample_idx]
                label_img = plt.imshow(
                    label,
                    vmin=np.min(label),
                    vmax=np.max(label),
                    cmap="gray",
                    alpha=0.5,
                    interpolation="none",
                    extent=extent,
                )
            plt.xticks([])
            plt.yticks([])
            plt.title("Data")

        return figure


class MaskClassVisualizer(Visualizer):
    """
    Visualize data with mask label information overlaid and the class of the
    mask included in the title

    Args:
        **kwargs:
    """

    def __init__(self, class_list, **kwargs) -> None:
        super(MaskClassVisualizer, self).__init__(**kwargs)
        self.class_list = class_list

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)
        if self.visualize_transform:
            iq_data = self.visualize_transform(deepcopy(iq_data))

        if self.visualize_target_transform:
            classes, targets = self.visualize_target_transform(deepcopy(targets))
        else:
            targets = None

        return self._visualize(iq_data, targets, classes)

    def _visualize(  # type: ignore
        self, data: np.ndarray, targets: np.ndarray, classes: List[str]
    ) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure(frameon=False)
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            extent = 0, data.shape[1], 0, data.shape[2]
            data_img = plt.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
                cmap="jet",
                extent=extent,
            )
            title = []
            if targets is not None:
                class_idx = classes[sample_idx]
                mask = targets[sample_idx]
                mask_img = plt.imshow(
                    mask,
                    vmin=np.min(mask),
                    vmax=np.max(mask),
                    cmap="gray",
                    alpha=0.5,
                    interpolation="none",
                    extent=extent,
                )
                title = [self.class_list[idx] for idx in class_idx]
            else:
                title = "Data"
            plt.xticks([])
            plt.yticks([])
            plt.title(title)

        return figure


class SemanticMaskClassVisualizer(Visualizer):
    """
    Visualize data with mask label information overlaid and the class of the
    mask included in the title

    Args:
        **kwargs:
    """

    def __init__(self, class_list, **kwargs) -> None:
        super(SemanticMaskClassVisualizer, self).__init__(**kwargs)
        self.class_list = class_list

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)
        if self.visualize_transform:
            iq_data = self.visualize_transform(deepcopy(iq_data))

        if self.visualize_target_transform:
            targets = self.visualize_target_transform(deepcopy(targets))

        return self._visualize(iq_data, targets)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure(frameon=False)
        for sample_idx in range(batch_size):
            plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )
            extent = 0, data.shape[1], 0, data.shape[2]
            data_img = plt.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
                cmap="jet",
                extent=extent,
            )
            title = []
            if targets is not None:
                mask = np.ma.masked_where(targets[sample_idx] < 1, targets[sample_idx])
                mask_img = plt.imshow(
                    mask,
                    alpha=0.5,
                    interpolation="none",
                    extent=extent,
                )
                classes_present = list(set(targets[sample_idx].flatten().tolist()))
                classes_present.remove(0.0)  # Remove 'background' class
                title = [
                    self.class_list[int(class_idx - 1)] for class_idx in classes_present
                ]
            else:
                title = "Data"
            plt.xticks([])
            plt.yticks([])
            plt.title(title)

        return figure


class BoundingBoxVisualizer(Visualizer):
    """Visualize data with bounding box label information overlaid

    Args:
        **kwargs:
    """

    def __init__(self, **kwargs) -> None:
        super(BoundingBoxVisualizer, self).__init__(**kwargs)

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)

        if self.visualize_transform:
            iq_data = self.visualize_transform(deepcopy(iq_data))

        if self.visualize_target_transform:
            targets = self.visualize_target_transform(deepcopy(targets))
        else:
            targets = targets

        return self._visualize(iq_data, targets)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure(frameon=False)
        for sample_idx in range(batch_size):
            ax = plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )

            # Retrieve individual label
            ax.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
                cmap="jet",
            )
            label = targets[sample_idx]
            pixels_per_cell_x = data[sample_idx].shape[0] / label.shape[0]
            pixels_per_cell_y = data[sample_idx].shape[1] / label.shape[1]

            for grid_cell_x_idx in range(label.shape[0]):
                for grid_cell_y_idx in range(label.shape[1]):
                    if label[grid_cell_x_idx, grid_cell_y_idx, 0] == 1:
                        duration = (
                            label[grid_cell_x_idx, grid_cell_y_idx, 2]
                            * data[sample_idx].shape[0]
                        )
                        bandwidth = (
                            label[grid_cell_x_idx, grid_cell_y_idx, 4]
                            * data[sample_idx].shape[1]
                        )
                        start_pixel = (
                            (grid_cell_x_idx * pixels_per_cell_x)
                            + (
                                label[grid_cell_x_idx, grid_cell_y_idx, 1]
                                * pixels_per_cell_x
                            )
                            - duration / 2
                        )
                        low_freq = (
                            (grid_cell_y_idx * pixels_per_cell_y)
                            + (
                                label[grid_cell_x_idx, grid_cell_y_idx, 3]
                                * pixels_per_cell_y
                            )
                            - (
                                label[grid_cell_x_idx, grid_cell_y_idx, 4]
                                / 2
                                * data[sample_idx].shape[1]
                            )
                        )

                        rect = patches.Rectangle(
                            (start_pixel, low_freq),
                            duration,
                            bandwidth,  # Bandwidth (pixels)
                            linewidth=3,
                            edgecolor="b",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
            plt.imshow(
                data[sample_idx],
                aspect="auto",
                cmap="jet",
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
            )
            plt.xticks([])
            plt.yticks([])
            plt.title("Data")

        return figure


class AnchorBoxVisualizer(Visualizer):
    """Visualize data with anchor box label information overlaid

    Args:
        **kwargs:
    """

    def __init__(
        self,
        data_loader,
        anchor_boxes: List,
        visualize_transform: Optional[Callable] = None,
        visualize_target_transform: Optional[Callable] = None,
    ) -> None:
        self.data_loader = iter(data_loader)
        self.anchor_boxes = anchor_boxes
        self.visualize_transform = visualize_transform
        self.visualize_target_transform = visualize_target_transform
        self.num_anchor_boxes = len(anchor_boxes)

    def __next__(self) -> Figure:
        iq_data, targets = next(self.data_iter)

        if self.visualize_transform:
            iq_data = self.visualize_transform(deepcopy(iq_data))

        if self.visualize_target_transform:
            targets = self.visualize_target_transform(deepcopy(targets))
        else:
            targets = targets

        return self._visualize(iq_data, targets)

    def _visualize(self, data: np.ndarray, targets: np.ndarray) -> Figure:
        batch_size = data.shape[0]
        figure = plt.figure(frameon=False)
        for sample_idx in range(batch_size):
            ax = plt.subplot(
                int(np.ceil(np.sqrt(batch_size))),
                int(np.sqrt(batch_size)),
                sample_idx + 1,
            )

            # Retrieve individual label
            ax.imshow(
                data[sample_idx],
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
                cmap="jet",
            )
            label = targets[sample_idx]
            pixels_per_cell_x = data[sample_idx].shape[0] / label.shape[0]
            pixels_per_cell_y = data[sample_idx].shape[1] / label.shape[1]

            for grid_cell_x_idx in range(label.shape[0]):
                for grid_cell_y_idx in range(label.shape[1]):
                    for anchor_idx in range(self.num_anchor_boxes):
                        if (
                            label[grid_cell_x_idx, grid_cell_y_idx, 0 + 5 * anchor_idx]
                            == 1
                        ):
                            duration = (
                                label[
                                    grid_cell_x_idx, grid_cell_y_idx, 2 + 5 * anchor_idx
                                ]
                                * self.anchor_boxes[anchor_idx][0]
                                * data[sample_idx].shape[0]
                            )
                            bandwidth = (
                                label[
                                    grid_cell_x_idx, grid_cell_y_idx, 4 + 5 * anchor_idx
                                ]
                                * self.anchor_boxes[anchor_idx][1]
                                * data[sample_idx].shape[1]
                            )
                            start_pixel = (
                                (grid_cell_x_idx * pixels_per_cell_x)
                                + (
                                    label[
                                        grid_cell_x_idx,
                                        grid_cell_y_idx,
                                        1 + 5 * anchor_idx,
                                    ]
                                    * pixels_per_cell_x
                                )
                                - duration / 2
                            )
                            low_freq = (
                                (grid_cell_y_idx * pixels_per_cell_y)
                                + (
                                    label[
                                        grid_cell_x_idx,
                                        grid_cell_y_idx,
                                        3 + 5 * anchor_idx,
                                    ]
                                    * pixels_per_cell_y
                                )
                                - (
                                    label[
                                        grid_cell_x_idx,
                                        grid_cell_y_idx,
                                        4 + 5 * anchor_idx,
                                    ]
                                    * self.anchor_boxes[anchor_idx][1]
                                    / 2
                                    * data[sample_idx].shape[1]
                                )
                            )

                            rect = patches.Rectangle(
                                (start_pixel, low_freq),
                                duration,
                                bandwidth,  # Bandwidth (pixels)
                                linewidth=3,
                                edgecolor="b",
                                facecolor="none",
                            )
                            ax.add_patch(rect)

            plt.imshow(
                data[sample_idx],
                aspect="auto",
                cmap="jet",
                vmin=np.min(data[sample_idx]),
                vmax=np.max(data[sample_idx]),
            )
            plt.xticks([])
            plt.yticks([])
            plt.title("Data")

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
        (batch_size, tensor.shape[2], tensor.shape[3]), dtype=np.float64
    )
    for idx in range(tensor.shape[0]):
        new_tensor[idx] = 20 * np.log10(tensor[idx, 0] ** 2 + tensor[idx, 1] ** 2)
    return new_tensor


def magnitude_spectrogram(tensor: np.ndarray) -> np.ndarray:
    """Visualizer data transform: Transform magnitude spectrogram data for
    plotting (mode = 'psd')

    """
    batch_size = tensor.shape[0]
    new_tensor = np.zeros(
        (batch_size, tensor.shape[1], tensor.shape[2]), dtype=np.float64
    )
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
    """Target Transform: Format multihot labels for titles in visualizer"""
    batch_size = tensor.shape[0]
    label: List[List[str]] = []
    for idx in range(batch_size):
        curr_label: List[str] = []
        for class_idx in range(len(class_list)):
            if tensor[idx][class_idx] > (1 / len(class_list)):
                curr_label.append(class_list[class_idx])
        label.append(curr_label)
    return label


def mask_to_outline(tensor: np.ndarray) -> List[str]:
    """Target Transform: Transforms masks for all bursts to outlines for the
    MaskVisualizer. Overlapping mask outlines are represented as a single
    polygon.

    """
    batch_size = tensor.shape[0]
    labels = []
    struct = ndimage.generate_binary_structure(2, 2)
    for idx in range(batch_size):
        label = tensor[idx].numpy()
        label = np.sum(label, axis=0)
        label[label > 0] = 1
        label = label - ndimage.binary_erosion(label)
        label = ndimage.binary_dilation(label, structure=struct, iterations=3).astype(
            label.dtype
        )
        label = np.ma.masked_where(label == 0, label)
        labels.append(label)
    return labels


def mask_to_outline_overlap(tensor: np.ndarray) -> List[str]:
    """Target Transform: Transforms masks for each burst to individual outlines
    for the MaskVisualizer. Overlapping mask outlines are still shown as
    overlapping.

    """
    batch_size = tensor.shape[0]
    labels = []
    struct = ndimage.generate_binary_structure(2, 2)
    for idx in range(batch_size):
        label = tensor[idx].numpy()
        for individual_burst_idx in range(label.shape[0]):
            label[individual_burst_idx] = label[
                individual_burst_idx
            ] - ndimage.binary_erosion(label[individual_burst_idx])
        label = np.sum(label, axis=0)
        label[label > 0] = 1
        label = ndimage.binary_dilation(label, structure=struct, iterations=2).astype(
            label.dtype
        )
        label = np.ma.masked_where(label == 0, label)
        labels.append(label)
    return labels


def overlay_mask(tensor: np.ndarray) -> List[str]:
    """Target Transform: Transforms multi-dimensional mask to binary overlay of
    full mask.

    """
    batch_size = tensor.shape[0]
    labels = []
    for idx in range(batch_size):
        label = torch.sum(tensor[idx], axis=0).numpy()  # type: ignore
        label[label > 0] = 1
        label = np.ma.masked_where(label == 0, label)
        labels.append(label)
    return labels


def mask_class_to_outline(tensor: np.ndarray) -> Tuple[List[List[int]], List[Any]]:
    """Target Transform: Transforms masks for each burst to individual outlines
    for the MaskClassVisualizer. Overlapping mask outlines are still shown as
    overlapping. Each bursts' class index is also returned.

    """
    batch_size = tensor.shape[0]
    labels = []
    class_idx = []
    struct = ndimage.generate_binary_structure(2, 2)
    for idx in range(batch_size):
        label = tensor[idx].numpy()
        class_idx_curr = []
        for individual_burst_idx in range(label.shape[0]):
            if np.count_nonzero(label[individual_burst_idx]) > 0:
                class_idx_curr.append(individual_burst_idx)
            label[individual_burst_idx] = label[
                individual_burst_idx
            ] - ndimage.binary_erosion(label[individual_burst_idx])
        label = np.sum(label, axis=0)
        label[label > 0] = 1
        label = ndimage.binary_dilation(label, structure=struct, iterations=2).astype(
            label.dtype
        )
        label = np.ma.masked_where(label == 0, label)
        class_idx.append(class_idx_curr)
        labels.append(label)
    return class_idx, labels
