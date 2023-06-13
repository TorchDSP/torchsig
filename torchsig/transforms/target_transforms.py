from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from torchsig.transforms.transforms import Transform
from torchsig.utils.types import SignalDescription

__all__ = [
    "DescToClassName",
    "DescToClassNameSNR",
    "DescToClassIndex",
    "DescToClassIndexSNR",
    "DescToMask",
    "DescToMaskSignal",
    "DescToMaskFamily",
    "DescToMaskClass",
    "DescToSemanticClass",
    "DescToBBox",
    "DescToAnchorBoxes",
    "DescPassThrough",
    "DescToBinary",
    "DescToCustom",
    "DescToClassEncoding",
    "DescToWeightedMixUp",
    "DescToWeightedCutMix",
    "DescToBBoxDict",
    "DescToBBoxSignalDict",
    "DescToBBoxFamilyDict",
    "DescToInstMaskDict",
    "DescToSignalInstMaskDict",
    "DescToSignalFamilyInstMaskDict",
    "DescToListTuple",
    "ListTupleToDesc",
    "LabelSmoothing",
]


class DescToClassName(Transform):
    """Transform to transform SignalDescription into either the single class name
    or a list of the classes present if there are multiple classes

    """

    def __init__(self) -> None:
        super(DescToClassName, self).__init__()

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[List[str], str]:
        classes: List[str] = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            curr_class_name: Optional[str] = (
                signal_desc.class_name[0]  # type: ignore
                if isinstance(signal_desc.class_name, list)
                else signal_desc.class_name
            )
            if curr_class_name is not None:
                classes.append(curr_class_name)  # type: ignore
        if len(classes) > 1:
            return classes
        elif len(classes) == 1:
            return classes[0]
        else:
            return []


class DescToClassNameSNR(Transform):
    """Transform to transform SignalDescription into either the single class name
    or a list of the classes present if there are multiple classes along with
    the SNRs for each

    """

    def __init__(self) -> None:
        super(DescToClassNameSNR, self).__init__()

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[Tuple[List[str], List[float]], Tuple[str, float]]:
        classes: List[str] = []
        snrs: List[float] = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            if signal_desc.class_name is not None:
                classes.append(signal_desc.class_name)  # type: ignore
            if signal_desc.snr is not None:
                snrs.append(signal_desc.snr)  # type: ignore
        if len(classes) > 1:
            return classes, snrs
        else:
            return classes[0], snrs[0]


class DescToClassIndex(Transform):
    """Transform to transform SignalDescription into either the single class index
    or a list of the class indices present if there are multiple classes. Note:
    if the SignalDescription contains classes not present in the provided
    `class_list`, the SignalDescription is interpretted as having no classes
    present

    Args:
        class_list (:obj:`List[str]`):
            A full list of classes to map the class names to indices

    """

    def __init__(self, class_list: List[str]) -> None:
        super(DescToClassIndex, self).__init__()
        self.class_list = class_list

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[List[int], int]:
        classes: List[int] = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            if signal_desc.class_name in self.class_list:
                curr_class: str = signal_desc.class_name
                classes.append(self.class_list.index(curr_class))
        if len(classes) > 1:
            return classes
        else:
            return classes[0]


class DescToClassIndexSNR(Transform):
    """Transform to transform SignalDescription into either the single class index
    or a list of the class indices present if there are multiple classes along
    with the SNRs of each. Note: if the SignalDescription contains classes not
    present in the provided `class_list`, the SignalDescription is interpretted as
    having no classes present

    Args:
        class_list (:obj:`List[str]`):
            A full list of classes to map the class names to indices

    """

    def __init__(self, class_list: List[str]) -> None:
        super(DescToClassIndexSNR, self).__init__()
        self.class_list = class_list

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[Tuple[List[int], List[Optional[float]]], Tuple[int, Optional[float]]]:
        classes = []
        snrs = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            if signal_desc.class_name in self.class_list:
                classes.append(self.class_list.index(signal_desc.class_name))
                snrs.append(signal_desc.snr)
        if len(classes) > 1:
            return classes, snrs
        else:
            return classes[0], snrs[0]


class DescToMask(Transform):
    """Transform to transform SignalDescriptions into spectrogram masks

    Args:
        max_bursts (:obj:`int`):
            Maximum number of bursts to label in their own target channel
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    def __init__(self, max_bursts: int, width: int, height: int) -> None:
        super(DescToMask, self).__init__()
        self.max_bursts = max_bursts
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        masks: np.ndarray = np.zeros((self.max_bursts, self.height, self.width))
        idx = 0
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            idx += 1
        return masks


class DescToMaskSignal(Transform):
    """Transform to transform SignalDescriptions into spectrogram masks for binary
    signal detection

    Args:
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    def __init__(self, width: int, height: int) -> None:
        super(DescToMaskSignal, self).__init__()
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        masks: np.ndarray = np.zeros((self.height, self.width))
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
        return masks


class DescToMaskFamily(Transform):
    """Transform to transform SignalDescriptions into spectrogram masks with
    different channels for each class's family. If no `class_family_dict`
    provided, the default mapping for the WBSig53 modulation families is used.

    Args:
        class_family_dict (:obj:`dict`):
            Dictionary mapping all class names to their families
        family_list (:obj:`list`):
            List of all of the families
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    class_family_dict: Dict[str, str] = {
        "4ask": "ask",
        "8ask": "ask",
        "16ask": "ask",
        "32ask": "ask",
        "64ask": "ask",
        "ook": "pam",
        "4pam": "pam",
        "8pam": "pam",
        "16pam": "pam",
        "32pam": "pam",
        "64pam": "pam",
        "2fsk": "fsk",
        "2gfsk": "fsk",
        "2msk": "fsk",
        "2gmsk": "fsk",
        "4fsk": "fsk",
        "4gfsk": "fsk",
        "4msk": "fsk",
        "4gmsk": "fsk",
        "8fsk": "fsk",
        "8gfsk": "fsk",
        "8msk": "fsk",
        "8gmsk": "fsk",
        "16fsk": "fsk",
        "16gfsk": "fsk",
        "16msk": "fsk",
        "16gmsk": "fsk",
        "bpsk": "psk",
        "qpsk": "psk",
        "8psk": "psk",
        "16psk": "psk",
        "32psk": "psk",
        "64psk": "psk",
        "16qam": "qam",
        "32qam": "qam",
        "32qam_cross": "qam",
        "64qam": "qam",
        "128qam_cross": "qam",
        "256qam": "qam",
        "512qam_cross": "qam",
        "1024qam": "qam",
        "ofdm-64": "ofdm",
        "ofdm-72": "ofdm",
        "ofdm-128": "ofdm",
        "ofdm-180": "ofdm",
        "ofdm-256": "ofdm",
        "ofdm-300": "ofdm",
        "ofdm-512": "ofdm",
        "ofdm-600": "ofdm",
        "ofdm-900": "ofdm",
        "ofdm-1024": "ofdm",
        "ofdm-1200": "ofdm",
        "ofdm-2048": "ofdm",
    }

    def __init__(
        self,
        width: int,
        height: int,
        class_family_dict: Optional[Dict[str, str]] = None,
        family_list: Optional[List[str]] = None,
        label_encode: bool = False,
    ) -> None:
        super(DescToMaskFamily, self).__init__()
        self.class_family_dict: Dict[str, str] = (
            class_family_dict if class_family_dict else self.class_family_dict
        )
        self.family_list: List[str] = (
            family_list if family_list else sorted(list(set(self.class_family_dict.values())))
        )
        self.width = width
        self.height = height
        self.label_encode = label_encode

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        masks: np.ndarray = np.zeros((len(self.family_list), self.height, self.width))
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.class_name is not None
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if isinstance(signal_desc.class_name, list):
                signal_desc.class_name = signal_desc.class_name[0]
            family_name = self.class_family_dict[signal_desc.class_name]
            family_idx = self.family_list.index(family_name)
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    family_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    family_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
        if self.label_encode:
            background_mask: np.ndarray = np.zeros((1, self.height, self.height))
            masks = np.concatenate([background_mask, masks], axis=0)
            masks = np.argmax(masks, axis=0)
        return masks


class DescToMaskClass(Transform):
    """Transform to transform list of SignalDescriptions into spectrogram masks
    with classes

    Args:
        num_classes (:obj:`int`):
            Integer number of classes, setting the channel dimension of the resultant mask
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    def __init__(self, num_classes: int, width: int, height: int) -> None:
        super(DescToMaskClass, self).__init__()
        self.num_classes = num_classes
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        masks: np.ndarray = np.zeros((self.num_classes, self.height, self.width))
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    signal_desc.class_index,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    signal_desc.class_index,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
        return masks


class DescToSemanticClass(Transform):
    """Transform to transform SignalDescriptions into spectrogram semantic
    segmentation mask with class information denoted as a value, rather than by
    a one/multi-hot vector in an additional channel like the
    DescToMaskClass does. Note that the class indicies are all
    incremented by 1 in order to reserve the 0 class for "background". Note
    that cases of overlapping bursts are currently resolved by comparing SNRs,
    labeling the pixel by the stronger signal. Ties in SNR are awarded to the
    burst that appears later in the burst collection.

    Args:
        num_classes (:obj:`int`):
            Integer number of classes, setting the channel dimension of the resultant mask
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    def __init__(self, num_classes: int, width: int, height: int) -> None:
        super(DescToSemanticClass, self).__init__()
        self.num_classes = num_classes
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        masks: np.ndarray = np.zeros((self.height, self.width))
        curr_snrs: np.ndarray = np.ones((self.height, self.width)) * -np.inf
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.snr is not None
            assert signal_desc.class_index is not None
            # Normalize freq values to [0,1]
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5

            # Convert to pixels
            height_start: int = max(0, int((signal_desc.lower_frequency + 0.5) * self.height))
            height_stop: int = min(
                int((signal_desc.upper_frequency + 0.5) * self.height), self.height
            )
            width_start: int = max(0, int(signal_desc.start * self.width))
            width_stop: int = min(int(signal_desc.stop * self.width), self.width)

            # Account for signals with bandwidths < a pixel
            if height_start == height_stop:
                height_stop = min(height_stop + 1, self.height)

            # Loop through pixels
            for height_idx in range(height_start, height_stop):
                for width_idx in range(width_start, width_stop):
                    # Check SNR against currently stored SNR at pixel
                    if signal_desc.snr >= curr_snrs[height_idx, width_idx]:
                        # If SNR >= currently stored class's SNR, update class & snr
                        masks[height_start:height_stop, width_start:width_stop,] = (
                            signal_desc.class_index + 1
                        )
                        curr_snrs[
                            height_start:height_stop,
                            width_start:width_stop,
                        ] = signal_desc.snr
        return masks


class DescToBBox(Transform):
    """Transform to transform SignalDescriptions into spectrogram bounding boxes
    with dimensions: <grid_width, grid_height, 5>, where the last 5 represents:
        - 0: presence ~ 1 if center of burst in current cell, else 0
        - 1: center_time ~ normalized to cell
        - 2: dur_time ~ normalized to full spec time
        - 3: center_freq ~ normalized to cell
        - 4: bw_freq ~ normalized to full spec bw

    Args:
        grid_width (:obj:`int`):
            Width of grid celling
        grid_height (:obj:`int`):
            Height of grid celling

    """

    def __init__(self, grid_width: int, grid_height: int) -> None:
        super(DescToBBox, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        boxes: np.ndarray = np.zeros((self.grid_width, self.grid_height, 5))
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.duration is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            # Time conversions
            if signal_desc.start >= 1.0:
                # Burst starts outside of window of capture
                continue
            elif signal_desc.start + signal_desc.duration * 0.5 >= 1.0:
                # Center is outside grid cell; re-center to truncated burst
                signal_desc.duration = 1 - signal_desc.start
            x: float = (signal_desc.start + signal_desc.duration * 0.5) * self.grid_width
            time_cell: int = int(np.floor(x))
            center_time: float = x - time_cell

            # Freq conversions
            if signal_desc.lower_frequency > 0.5 or signal_desc.upper_frequency < -0.5:
                # Burst is fully outside of capture bandwidth
                continue
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            signal_desc.bandwidth = signal_desc.upper_frequency - signal_desc.lower_frequency
            signal_desc.center_frequency = signal_desc.lower_frequency + signal_desc.bandwidth / 2
            y: float = (signal_desc.center_frequency + 0.5) * self.grid_height
            freq_cell: int = int(np.floor(y))
            center_freq: float = y - freq_cell

            if time_cell >= self.grid_width:
                print("Error: time_cell idx is greater than grid_width")
                print("time_cell: {}".format(time_cell))
                print("burst.start: {}".format(signal_desc.start))
                print("burst.duration: {}".format(signal_desc.duration))
                print("x: {}".format(x))
            if freq_cell >= self.grid_height:
                print("Error: freq_cell idx is greater than grid_height")
                print("freq_cell: {}".format(freq_cell))
                print("burst.lower_frequency: {}".format(signal_desc.lower_frequency))
                print("burst.upper_frequency: {}".format(signal_desc.upper_frequency))
                print("burst.center_frequency: {}".format(signal_desc.center_frequency))
                print("y: {}".format(y))

            # Assign to label
            boxes[time_cell, freq_cell, 0] = 1
            boxes[time_cell, freq_cell, 1] = center_time
            boxes[time_cell, freq_cell, 2] = signal_desc.duration
            boxes[time_cell, freq_cell, 3] = center_freq
            boxes[time_cell, freq_cell, 4] = signal_desc.bandwidth
        return boxes


class DescToAnchorBoxes(Transform):
    """Transform to transform BurstCollections into spectrogram bounding boxes
    using anchor boxes, such that the output target shape will have the
    dimensions: <grid_width, grid_height, 5*num_anchor_boxes>, where the last 5 represents:
        - 0: objectness ~ 1 if burst associated with current cell & anchor, else 0
        - 1: center_time ~ normalized to cell
        - 2: dur_offset ~ offset in duration with anchor box duration
        - 3: center_freq ~ normalized to cell
        - 4: bw_offset ~ offset in bandwidth with anchor box duration

    Args:
        grid_width (:obj:`int`):
            Width of grid celling
        grid_height (:obj:`int`):
            Height of grid celling
        anchor_boxes:
            List of tuples describing the anchor boxes (normalized values)
                Example format: [(dur1, bw1), (dur2, bw2)]

    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        anchor_boxes: List[Tuple[float, float]],
    ) -> None:
        super(DescToAnchorBoxes, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.anchor_boxes = anchor_boxes
        self.num_anchor_boxes: int = len(anchor_boxes)

    def iou(
        self,
        start_a: float,
        dur_a: float,
        center_freq_a: float,
        bw_a: float,
        start_b: float,
        dur_b: float,
        center_freq_b: float,
        bw_b: float,
    ) -> float:
        """
        Method to compute the intersection over union (IoU)

        """
        # Convert to start/stops
        x_start_a: float = start_a
        x_stop_a: float = start_a + dur_a
        y_start_a: float = center_freq_a - bw_a / 2
        y_stop_a: float = center_freq_a + bw_a / 2

        x_start_b: float = start_b
        x_stop_b: float = start_b + dur_b
        y_start_b: float = center_freq_b - bw_b / 2
        y_stop_b: float = center_freq_b + bw_b / 2

        # Determine the (x, y)-coordinates of the intersection
        x_start_int: float = max(x_start_a, x_start_b)
        y_start_int: float = max(y_start_a, y_start_b)
        x_stop_int: float = min(x_stop_a, x_stop_b)
        y_stop_int: float = min(y_stop_a, y_stop_b)

        # Compute the area of intersection
        inter_area: float = abs(
            max((x_stop_int - x_start_int, 0)) * max((y_stop_int - y_start_int), 0)
        )
        if inter_area == 0:
            return 0
        # Compute the area of both the prediction and ground-truth
        area_a: float = abs((x_stop_a - x_start_a) * (y_stop_a - y_start_a))
        area_b: float = abs((x_stop_b - x_start_b) * (y_stop_b - y_start_b))

        # Compute the intersection over union
        iou: float = inter_area / float(area_a + area_b - inter_area)
        return iou

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        boxes: np.ndarray = np.zeros((self.grid_width, self.grid_height, 5 * self.num_anchor_boxes))
        for signal_desc in signal_description_list:
            assert signal_desc.start is not None
            assert signal_desc.duration is not None
            assert signal_desc.center_frequency is not None
            assert signal_desc.bandwidth is not None
            assert signal_desc.duration is not None
            # Time conversions
            if signal_desc.start > 1.0:
                # Error handling (TODO: should fix within dataset)
                continue
            elif signal_desc.start + signal_desc.duration * 0.5 > 1.0:
                # Center is outside grid cell; re-center to truncated burst
                signal_desc.duration = 1 - signal_desc.start
            x: float = (signal_desc.start + signal_desc.duration * 0.5) * self.grid_width
            time_cell: int = int(np.floor(x))
            center_time: float = x - time_cell

            # Freq conversions
            y: float = (signal_desc.center_frequency + 0.5) * self.grid_height
            freq_cell: int = int(np.floor(y))
            center_freq: float = y - freq_cell

            # Debugging messages for potential errors
            if time_cell > self.grid_width:
                print("Error: time_cell idx is greater than grid_width")
                print("time_cell: {}".format(time_cell))
                print("burst.start: {}".format(signal_desc.start))
                print("burst.duration: {}".format(signal_desc.duration))
                print("x: {}".format(x))
            if freq_cell > self.grid_height:
                print("Error: freq_cell idx is greater than grid_height")
                print("freq_cell: {}".format(freq_cell))
                print("burst.center_frequency: {}".format(signal_desc.center_frequency))
                print("y: {}".format(y))

            # Determine which anchor box to associate burst with
            best_iou_score: float = -1
            best_iou_idx: int = 0
            best_anchor_duration: float = 0
            best_anchor_bw: float = 0
            for anchor_idx, anchor_box in enumerate(self.anchor_boxes):
                # anchor_start = ((time_cell+0.5) / self.grid_width) - (anchor_box[0]*0.5) # Anchor centered on cell
                anchor_start: float = (
                    signal_desc.start + 0.5 * signal_desc.duration - anchor_box[0] * 0.5
                )  # Anchor overlaid on burst
                anchor_duration: float = anchor_box[0]
                # anchor_center_freq = (freq_cell+0.5) / self.grid_height # Anchor centered on cell
                anchor_center_freq: float = signal_desc.center_frequency  # Anchor overlaid on burst
                anchor_bw: float = anchor_box[1]
                iou_score: float = self.iou(
                    signal_desc.start,
                    signal_desc.duration,
                    signal_desc.center_frequency,
                    signal_desc.bandwidth,
                    anchor_start,
                    anchor_duration,
                    anchor_center_freq,
                    anchor_bw,
                )
                if (
                    iou_score > best_iou_score
                    and boxes[time_cell, freq_cell, 0 + 5 * anchor_idx] != 1
                ):
                    # If IoU score is the best out of all anchors and anchor hasn't already been used for another burst, save results
                    best_iou_score = iou_score
                    best_iou_idx = anchor_idx
                    best_anchor_duration = anchor_duration
                    best_anchor_bw = anchor_bw

            # Convert absolute coordinates to anchor-box offsets
            # centers are normalized values like previous code segment below
            # width/height are relative values to anchor boxes
            #   -- if anchor width is 0.6; true width is 0.5; label width should be 0.5/0.6
            #   -- if anchor height is 0.6; true height is 0.7; label height should be 0.7/0.6
            #   -- loss & inference will require predicted_box_wh = (sigmoid(model_output_wh)*2)**2 * anchor_wh
            if best_iou_score > 0:
                # Detection:
                boxes[time_cell, freq_cell, 0 + 5 * best_iou_idx] = 1
                # Center time & freq
                boxes[time_cell, freq_cell, 1 + 5 * best_iou_idx] = center_time
                boxes[time_cell, freq_cell, 3 + 5 * best_iou_idx] = center_freq
                # Duration/Bandwidth (Width/Height)
                boxes[time_cell, freq_cell, 2 + 5 * best_iou_idx] = (
                    signal_desc.duration / best_anchor_duration
                )
                boxes[time_cell, freq_cell, 4 + 5 * best_iou_idx] = (
                    signal_desc.bandwidth / best_anchor_bw
                )
        return boxes


class DescPassThrough(Transform):
    """Transform to simply pass the SignalDescription through. Same as applying no
    transform in most cases.

    """

    def __init__(self) -> None:
        super(DescPassThrough, self).__init__()

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Union[List[SignalDescription], SignalDescription]:
        return signal_description


class DescToBinary(Transform):
    """Transform to transform SignalDescription into binary 0/1 label

    Args:
        label (:obj:`int`):
            Binary label to assign

    """

    def __init__(self, label: int) -> None:
        super(DescToBinary, self).__init__()
        self.label = label

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> int:
        return self.label


class DescToCustom(Transform):
    """Transform to transform SignalDescription into any static value

    Args:
        label (:obj:`Any`):
            Custom static label to assign

    """

    def __init__(self, label: Any) -> None:
        super(DescToCustom, self).__init__()
        self.label = label

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Any:
        return self.label


class DescToClassEncoding(Transform):
    """Transform to transform SignalDescription into one- or multi-hot class
    encodings. Note that either the number of classes or the full class list
    must be provided as input. If neither are provided, the transform will
    raise an error, and if both are provided, the transform will default to
    using the full class list. If only the number of classes are provided,
    the SignalDescription objects must contain the class index field

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list

        num_classes (:obj:`Optional[int]`):
            Number of classes in the encoding

    """

    def __init__(
        self,
        class_list: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        super(DescToClassEncoding, self).__init__()
        self.class_list = class_list
        self.num_classes: int = 0
        if num_classes:
            self.num_classes = num_classes
        elif class_list:
            self.num_classes = len(class_list)
        else:
            raise ValueError("class_list or num_classes must be provided")

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        encoding: np.ndarray = np.zeros((self.num_classes,))
        for signal_desc in signal_description_list:
            if self.class_list:
                assert signal_desc.class_name is not None
                encoding[self.class_list.index(signal_desc.class_name)] = 1.0
            else:
                assert signal_desc.class_index is not None
                encoding[signal_desc.class_index] = 1.0
        return encoding


class DescToWeightedMixUp(Transform):
    """Transform to transform SignalDescription into weighted multi-hot class
    encodings.

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list

    """

    def __init__(
        self,
        class_list: List[str],
    ) -> None:
        super(DescToWeightedMixUp, self).__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        encoding: np.ndarray = np.zeros((self.num_classes,))
        # Instead of a binary value for the encoding, set it to the SNR
        for signal_desc in signal_description_list:
            assert signal_desc.class_name is not None
            assert signal_desc.snr is not None
            encoding[self.class_list.index(signal_desc.class_name)] += signal_desc.snr
        # Next, normalize to the total of all SNR values
        encoding = encoding / np.sum(encoding)
        return encoding


class DescToWeightedCutMix(Transform):
    """Transform to transform SignalDescription into weighted multi-hot class
    encodings.

    Args:
        class_list (:obj:`Optional[List[str]]`):
            Class list

    """

    def __init__(
        self,
        class_list: List[str],
    ) -> None:
        super(DescToWeightedCutMix, self).__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> np.ndarray:
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        encoding: np.ndarray = np.zeros((self.num_classes,))
        # Instead of a binary value for the encoding, set it to the cumulative duration
        for signal_desc in signal_description_list:
            assert signal_desc.class_name is not None
            assert signal_desc.duration is not None
            encoding[self.class_list.index(signal_desc.class_name)] += signal_desc.duration
        # Normalize on total signals durations
        encoding = encoding / np.sum(encoding)
        return encoding


class DescToBBoxDict(Transform):
    """Transform to transform SignalDescriptions into the class bounding box format
    using dictionaries of labels and boxes, similar to the COCO image dataset

    Args:
        class_list (:obj:`list`):
            List of class names. Used when converting SignalDescription class names
            to indices

    """

    def __init__(self, class_list: List[str]) -> None:
        super(DescToBBoxDict, self).__init__()
        self.class_list = class_list

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        labels: List[int] = []
        boxes: np.ndarray = np.empty((len(signal_description_list), 4))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.class_name is not None
            # xcycwh
            duration: float = signal_desc.stop - signal_desc.start
            bandwidth: float = signal_desc.upper_frequency - signal_desc.lower_frequency
            boxes[signal_desc_idx] = np.array(
                [
                    signal_desc.start + 0.5 * duration,
                    signal_desc.lower_frequency + 0.5 + 0.5 * bandwidth,
                    duration,
                    bandwidth,
                ]
            )[0]
            labels.append(self.class_list.index(signal_desc.class_name))

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "boxes": torch.Tensor(boxes),
        }
        return targets


class DescToBBoxSignalDict(Transform):
    """Transform to transform SignalDescriptions into the class bounding box format
    using dictionaries of labels and boxes, similar to the COCO image dataset.
    Differs from the `SignalDescriptionToBoundingBoxDictTransform` in the ommission
    of signal-specific class labels, grouping all objects into the 'signal'
    class.

    """

    def __init__(self) -> None:
        super(DescToBBoxSignalDict, self).__init__()
        self.class_list: List[str] = ["signal"]

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        labels: List[int] = []
        boxes: np.ndarray = np.empty((len(signal_description_list), 4))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            # xcycwh
            duration: float = signal_desc.stop - signal_desc.start
            bandwidth: float = signal_desc.upper_frequency - signal_desc.lower_frequency
            boxes[signal_desc_idx] = np.array(
                [
                    signal_desc.start + 0.5 * duration,
                    signal_desc.lower_frequency + 0.5 + 0.5 * bandwidth,
                    duration,
                    bandwidth,
                ]
            )[0]
            labels.append(self.class_list.index(self.class_list[0]))

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "boxes": torch.Tensor(boxes),
        }
        return targets


class DescToBBoxFamilyDict(Transform):
    """Transform to transform SignalDescriptions into the class bounding box format
    using dictionaries of labels and boxes, similar to the COCO image dataset.
    Differs from the `DescToBBoxDict` transform in the grouping
    of fine-grain classes into their signal family as defined by an input
    `class_family_dict` dictionary.

    Args:
        class_family_dict (:obj:`dict`):
            Dictionary mapping all class names to their families

    """

    class_family_dict: Dict[str, str] = {
        "4ask": "ask",
        "8ask": "ask",
        "16ask": "ask",
        "32ask": "ask",
        "64ask": "ask",
        "ook": "pam",
        "4pam": "pam",
        "8pam": "pam",
        "16pam": "pam",
        "32pam": "pam",
        "64pam": "pam",
        "2fsk": "fsk",
        "2gfsk": "fsk",
        "2msk": "fsk",
        "2gmsk": "fsk",
        "4fsk": "fsk",
        "4gfsk": "fsk",
        "4msk": "fsk",
        "4gmsk": "fsk",
        "8fsk": "fsk",
        "8gfsk": "fsk",
        "8msk": "fsk",
        "8gmsk": "fsk",
        "16fsk": "fsk",
        "16gfsk": "fsk",
        "16msk": "fsk",
        "16gmsk": "fsk",
        "bpsk": "psk",
        "qpsk": "psk",
        "8psk": "psk",
        "16psk": "psk",
        "32psk": "psk",
        "64psk": "psk",
        "16qam": "qam",
        "32qam": "qam",
        "32qam_cross": "qam",
        "64qam": "qam",
        "128qam_cross": "qam",
        "256qam": "qam",
        "512qam_cross": "qam",
        "1024qam": "qam",
        "ofdm-64": "ofdm",
        "ofdm-72": "ofdm",
        "ofdm-128": "ofdm",
        "ofdm-180": "ofdm",
        "ofdm-256": "ofdm",
        "ofdm-300": "ofdm",
        "ofdm-512": "ofdm",
        "ofdm-600": "ofdm",
        "ofdm-900": "ofdm",
        "ofdm-1024": "ofdm",
        "ofdm-1200": "ofdm",
        "ofdm-2048": "ofdm",
    }

    def __init__(
        self,
        class_family_dict: Optional[Dict[str, str]] = None,
        family_list: Optional[List[str]] = None,
    ) -> None:
        super(DescToBBoxFamilyDict, self).__init__()
        self.class_family_dict: Dict[str, str] = (
            class_family_dict if class_family_dict else self.class_family_dict
        )
        self.family_list: List[str] = (
            family_list if family_list else sorted(list(set(self.class_family_dict.values())))
        )

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        labels: List[int] = []
        boxes: np.ndarray = np.empty((len(signal_description_list), 4))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.class_name is not None
            # xcycwh
            duration: float = signal_desc.stop - signal_desc.start
            bandwidth: float = signal_desc.upper_frequency - signal_desc.lower_frequency
            boxes[signal_desc_idx] = np.array(
                [
                    signal_desc.start + 0.5 * duration,
                    signal_desc.lower_frequency + 0.5 + 0.5 * bandwidth,
                    duration,
                    bandwidth,
                ]
            )
            if isinstance(signal_desc.class_name, list):
                signal_desc.class_name = signal_desc.class_name[0]
            family_name: str = self.class_family_dict[signal_desc.class_name]
            labels.append(self.family_list.index(family_name))

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "boxes": torch.Tensor(boxes),
        }
        return targets


class DescToInstMaskDict(Transform):
    """Transform to transform SignalDescriptions into the class mask format
    using dictionaries of labels and masks, similar to the COCO image dataset

    Args:
        class_list (:obj:`list`):
            List of class names. Used when converting SignalDescription class names
            to indices
        width (:obj:`int`):
            Width of masks
        heigh (:obj:`int`):
            Height of masks

    """

    def __init__(
        self,
        class_list: List[str],
        width: int = 512,
        height: int = 512,
    ) -> None:
        super(DescToInstMaskDict, self).__init__()
        self.class_list = class_list
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        num_objects: int = len(signal_description_list)
        labels: List[int] = []
        masks: np.ndarray = np.zeros((num_objects, self.height, self.width))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.class_name is not None
            labels.append(self.class_list.index(signal_desc.class_name))
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "masks": torch.Tensor(masks.astype(bool)),
        }
        return targets


class DescToSignalInstMaskDict(Transform):
    """Transform to transform SignalDescriptions into the class mask format
    using dictionaries of labels and masks, similar to the COCO image dataset

    Args:
        width (:obj:`int`):
            Width of masks
        heigh (:obj:`int`):
            Height of masks

    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
    ) -> None:
        super(DescToSignalInstMaskDict, self).__init__()
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        num_objects: int = len(signal_description_list)
        labels: List[int] = []
        masks: np.ndarray = np.zeros((num_objects, self.height, self.width))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            labels.append(0)
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "masks": torch.Tensor(masks.astype(bool)),
        }
        return targets


class DescToSignalFamilyInstMaskDict(Transform):
    """Transform to transform SignalDescriptions into the class mask format
    using dictionaries of labels and masks, similar to the COCO image dataset.
    The labels with this target transform are set to be the class's family. If
    no `class_family_dict` is provided, the default mapping for the WBSig53
    modulation families is used.

    Args:
        class_family_dict (:obj:`dict`):
            Dictionary mapping all class names to their families
        family_list (:obj:`list`):
            List of all of the families
        width (:obj:`int`):
            Width of resultant spectrogram mask
        height (:obj:`int`):
            Height of resultant spectrogram mask

    """

    class_family_dict: Dict[str, str] = {
        "4ask": "ask",
        "8ask": "ask",
        "16ask": "ask",
        "32ask": "ask",
        "64ask": "ask",
        "ook": "pam",
        "4pam": "pam",
        "8pam": "pam",
        "16pam": "pam",
        "32pam": "pam",
        "64pam": "pam",
        "2fsk": "fsk",
        "2gfsk": "fsk",
        "2msk": "fsk",
        "2gmsk": "fsk",
        "4fsk": "fsk",
        "4gfsk": "fsk",
        "4msk": "fsk",
        "4gmsk": "fsk",
        "8fsk": "fsk",
        "8gfsk": "fsk",
        "8msk": "fsk",
        "8gmsk": "fsk",
        "16fsk": "fsk",
        "16gfsk": "fsk",
        "16msk": "fsk",
        "16gmsk": "fsk",
        "bpsk": "psk",
        "qpsk": "psk",
        "8psk": "psk",
        "16psk": "psk",
        "32psk": "psk",
        "64psk": "psk",
        "16qam": "qam",
        "32qam": "qam",
        "32qam_cross": "qam",
        "64qam": "qam",
        "128qam_cross": "qam",
        "256qam": "qam",
        "512qam_cross": "qam",
        "1024qam": "qam",
        "ofdm-64": "ofdm",
        "ofdm-72": "ofdm",
        "ofdm-128": "ofdm",
        "ofdm-180": "ofdm",
        "ofdm-256": "ofdm",
        "ofdm-300": "ofdm",
        "ofdm-512": "ofdm",
        "ofdm-600": "ofdm",
        "ofdm-900": "ofdm",
        "ofdm-1024": "ofdm",
        "ofdm-1200": "ofdm",
        "ofdm-2048": "ofdm",
    }

    def __init__(
        self,
        width: int,
        height: int,
        class_family_dict: Optional[Dict[str, str]] = None,
        family_list: Optional[List[str]] = None,
    ) -> None:
        super(DescToSignalFamilyInstMaskDict, self).__init__()
        self.class_family_dict: Dict[str, str] = (
            class_family_dict if class_family_dict else self.class_family_dict
        )
        self.family_list: List[str] = (
            family_list if family_list else sorted(list(set(self.class_family_dict.values())))
        )
        self.width = width
        self.height = height

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> Dict[str, torch.Tensor]:
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        num_objects: int = len(signal_description_list)
        labels: List[int] = []
        masks: np.ndarray = np.zeros((num_objects, self.height, self.width))
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.lower_frequency is not None
            assert signal_desc.upper_frequency is not None
            assert signal_desc.class_name is not None

            family_name: str = self.class_family_dict[signal_desc.class_name]
            family_idx: int = self.family_list.index(family_name)
            labels.append(family_idx)
            if signal_desc.lower_frequency < -0.5:
                signal_desc.lower_frequency = -0.5
            if signal_desc.upper_frequency > 0.5:
                signal_desc.upper_frequency = 0.5
            if int((signal_desc.lower_frequency + 0.5) * self.height) == int(
                (signal_desc.upper_frequency + 0.5) * self.height
            ):
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    )
                    + 1,
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0
            else:
                masks[
                    signal_desc_idx,
                    int((signal_desc.lower_frequency + 0.5) * self.height) : int(
                        (signal_desc.upper_frequency + 0.5) * self.height
                    ),
                    int(signal_desc.start * self.width) : int(signal_desc.stop * self.width),
                ] = 1.0

        targets: Dict[str, torch.Tensor] = {
            "labels": torch.Tensor(labels).long(),
            "masks": torch.Tensor(masks.astype(bool)),
        }
        return targets


class DescToListTuple(Transform):
    """Transform to transform SignalDescription into a list of tuples containing
    the modulation, start time, stop time, center frequency, bandwidth, and SNR
    for each signal present

    Args:
        precision (:obj: `np.dtype`):
            Specify the data type precision for the tuple's information

    """

    def __init__(self, precision: np.dtype = np.dtype(np.float16)) -> None:
        super(DescToListTuple, self).__init__()
        self.precision = precision

    def __call__(
        self, signal_description: Union[List[SignalDescription], SignalDescription]
    ) -> List[Tuple[str, float, float, float, float, float]]:
        output: List[Tuple[str, float, float, float, float, float]] = []
        # Handle cases of both SignalDescriptions and lists of SignalDescriptions
        signal_description_list: List[SignalDescription] = (
            [signal_description]
            if isinstance(signal_description, SignalDescription)
            else signal_description
        )
        # Loop through SignalDescription's, converting values of interest to tuples
        for signal_desc_idx, signal_desc in enumerate(signal_description_list):
            assert signal_desc.start is not None
            assert signal_desc.stop is not None
            assert signal_desc.center_frequency is not None
            assert signal_desc.bandwidth is not None
            assert signal_desc.snr is not None
            assert signal_desc.class_name is not None
            curr_tuple: Tuple[str, float, float, float, float, float] = (
                signal_desc.class_name[0],
                self.precision.type(signal_desc.start),
                self.precision.type(signal_desc.stop),
                self.precision.type(signal_desc.center_frequency),
                self.precision.type(signal_desc.bandwidth),
                self.precision.type(signal_desc.snr),
            )
            output.append(curr_tuple)
        return output


class ListTupleToDesc(Transform):
    """Transform to transform a list of tuples to a list of SignalDescriptions
    Sample rate and number of IQ samples optional arguments are provided in
    order to fill in additional information if desired. If a class list is
    provided, the class names are used with the list to fill in class indices

    Args:
        sample_rate (:obj: `Optional[float]`):
            Optionally provide the sample rate for the SignalDescriptions

        num_iq_samples (:obj: `Optional[int]`):
            Optionally provide the number of IQ samples for the SignalDescriptions

        class_list (:obj: `List`):
            Optionally provide the class list to fill in class indices

    """

    def __init__(
        self,
        sample_rate: Optional[float] = None,
        num_iq_samples: Optional[int] = None,
        class_list: Optional[List[str]] = None,
    ) -> None:
        super(ListTupleToDesc, self).__init__()
        self.sample_rate = sample_rate
        self.num_iq_samples = num_iq_samples
        self.class_list = class_list

    def __call__(
        self,
        list_tuple: List[Tuple[str, float, float, float, float, float]],
    ) -> List[SignalDescription]:
        output: List[SignalDescription] = []
        # Loop through SignalDescription's, converting values of interest to tuples
        for tuple_idx, curr_tuple in enumerate(list_tuple):
            processed_curr_tuple: Tuple[Any, ...] = tuple(
                [l.numpy() if isinstance(l, torch.Tensor) else l for l in curr_tuple]
            )
            curr_signal_desc: SignalDescription = SignalDescription(
                sample_rate=self.sample_rate,
                num_iq_samples=self.num_iq_samples,
                class_name=processed_curr_tuple[0],
                class_index=self.class_list.index(processed_curr_tuple[0])
                if self.class_list
                else None,
                start=processed_curr_tuple[1],
                stop=processed_curr_tuple[2],
                center_frequency=processed_curr_tuple[3],
                bandwidth=processed_curr_tuple[4],
                lower_frequency=processed_curr_tuple[3] - processed_curr_tuple[4] / 2,
                upper_frequency=processed_curr_tuple[3] + processed_curr_tuple[4] / 2,
                snr=processed_curr_tuple[5],
            )
            output.append(curr_signal_desc)
        return output


class LabelSmoothing(Transform):
    """Transform to transform a numpy array encoding to a smoothed version to
    assist with overconfidence. The input hyperparameter `alpha` determines the
    degree of smoothing with the following equation:

        output = (1 - alpha) / num_hot * input + alpha / num_classes,

    Where,
        output ~ Smoothed output encoding
        alpha ~ Degree of smoothing to apply
        num_hot ~ Number of positively-labeled classes
        input ~ Input one/multi-hot encoding
        num_classes ~ Number of classes

    Note that the `LabelSmoothing` transform accepts a numpy encoding input,
    and as such, should be used in conjunction with a preceeding
    DescTo... transform that maps the SignalDescription to the expected
    numpy encoding format.

    Args:
        alpha (:obj:`float`):
            Degree of smoothing to apply

    """

    def __init__(self, alpha: float = 0.1) -> None:
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha

    def __call__(self, encoding: np.ndarray) -> np.ndarray:
        output: np.ndarray = (1 - self.alpha) / np.sum(encoding) * encoding + (
            self.alpha / encoding.shape[0]
        )
        return output
