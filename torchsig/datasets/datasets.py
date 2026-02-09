"""Dataset Base Classes for creation and static loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from torchsig.datasets.dataset_utils import frequency_shift_signal
from torchsig.signals.builder import BaseSignalGenerator, ConcatSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.coordinate_system import Coordinate, Rectangle, is_rectangle_overlap
from torchsig.utils.dsp import compute_spectrogram
from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.random import Seedable
from torchsig.utils.signal_building import lookup_signal_generator_by_string

# Type checking imports
if TYPE_CHECKING:
    from torchsig.transforms.base_transforms import Transform


def apply_label_to_signal(sample: Signal, target_label: str) -> list:
    """Recursively applies the specified label to a signal sample and its components.

    Args:
        sample: The signal sample to apply the label to.
        target_label: The label that should be used to identify relevant values in the signal sample.

    Returns:
        A list of values corresponding to the label specified in the sample and its component signals.
    """
    values = []
    if hasattr(sample, target_label):
        values += [getattr(sample, target_label)]
    for component_signal in sample.component_signals:
        values += apply_label_to_signal(component_signal, target_label)
    return values


def apply_transforms_and_labels_to_signal(
    sample: Signal, transforms: list[Transform | callable], target_labels: list
) -> Signal | np.ndarray | tuple:
    """Applies a series of transformations to a signal sample and retrieves specified label values.

    Args:
        sample: The signal sample to process.
        transforms: A list of function objects, each taking a Signal object and returning a transformed Signal object.
        target_labels: Labels to be retrieved from the signal sample after transformations. If None, the transformed signal is returned. If an empty list, the signal data is returned.

    Returns:
        - If target_labels is None, a Signal object with all applied transforms.
        - If target_labels is an empty list, the numpy.ndarray data of the sample.
        - If target_labels contains one label, a tuple of (sample_data, target_value).
        - If target_labels contains multiple labels, a tuple of (sample_data, [target_values]).
    """
    # apply user transforms
    for transform in transforms:
        sample = transform(sample)

    # apply metadata transforms
    # just return data if target_labels is None or empty list
    if target_labels is None:
        # return Signal object
        return sample
    if len(target_labels) < 1:
        # just return np.ndarray data
        return sample.data

    targets = {}
    for key in target_labels:
        values = apply_label_to_signal(sample, key)
        if sample["num_signals_max"] == 1 and len(values) == 1:
            values = values[0]
        targets[key] = values
    if len(target_labels) == 1:
        return sample.data, targets[target_labels[0]]

    return sample.data, [targets[key] for key in targets]


class TorchSigIterableDataset(HierarchicalMetadataObject, IterableDataset):
    """Base class for generating signals.

    The dataset will continue to generate samples infinitely.

    Attributes:
        signal_generators: The signal generators to use. Can be a string, ConcatSignalGenerator, or list.
        transforms: List of transforms to apply to the entire signal.
        component_transforms: List of transforms to apply to individual signal components.
        target_labels: Labels to extract from the signal.
        validate_init: Whether to validate metadata during initialization.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        signal_generators: str | ConcatSignalGenerator | list = "all",
        transforms: list[Transform | callable] = [],
        component_transforms: list[Transform | callable] = [],
        target_labels: list | None = None,
        # will try to validate required metadata in this dataset; can be turned off if a dataset needs to be initialized before it's metadata is known
        validate_init: bool = True,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            signal_generators: The signal generators to use. Can be a string, ConcatSignalGenerator, or list.
            transforms: List of transforms to apply to the entire signal.
            component_transforms: List of transforms to apply to individual signal components.
            target_labels: Labels to extract from the signal.
            validate_init: Whether to validate metadata during initialization.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        HierarchicalMetadataObject.__init__(self, **kwargs)
        self.validate_init = validate_init
        self.signal_generators = []
        self.signal_likelihoods = []
        self.signal_probabilities = []
        self.total_likelihood = 0
        self.target_labels = target_labels
        self.transforms = transforms
        self.component_transforms = component_transforms
        if not hasattr(self, "class_names"):
            self["class_names"] = []
        if "num_signals_min" not in self.keys():
            self["num_signals_min"] = 1
        if "num_signals_max" not in self.keys():
            self["num_signals_max"] = 1
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        for transform in self.component_transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        if isinstance(signal_generators, str):
            signal_generators = lookup_signal_generator_by_string(signal_generators)
        if isinstance(signal_generators, ConcatSignalGenerator):
            signal_generators = signal_generators.signal_generators
        for generator in signal_generators:
            self.init_signal_generator(generator)

    def init_signal_generator(self, signal_generator: str | callable) -> None:
        """Initializes the signal generator.

        Args:
            signal_generator: The signal generator to be initialized. If a string, it is first looked up to retrieve the corresponding signal generator function.

        Raises:
            TypeError: If the signal_generator is neither a string nor a callable.
        """
        if isinstance(signal_generator, str):
            self.add_signal_generator(
                lookup_signal_generator_by_string(signal_generator)
            )
        else:
            self.add_signal_generator(signal_generator)

    def add_signal_generator(
        self,
        signal_generator: callable,
        class_name: str | None = None,
        class_index: int | None = None,
        likelihood: int = 1,
    ) -> None:
        """Adds a signal generator to this dataset.

        Args:
            signal_generator: A callable object which takes no arguments and returns a Signal.
            class_name: (optional) A name for this signal class in the dataset. If None, the signal will be generated and added to the data, but no labels will be made for the signal.
            likelihood: (optional) The relative likelihood of this signal type in the dataset. Doubling the likelihood will make this signal twice as likely to be placed in the data.
        """
        if isinstance(signal_generator, Seedable):
            signal_generator.add_parent(self)
        try:
            if self.validate_init:
                signal_generator.validate_metadata_fields()
        except AttributeError:
            pass  # there is no validate function; ignore and assume the best; a user who doesn't write a validate function does so at their own risk
        signal_generator["class_index"] = len(self.signal_generators)
        if class_index is None:
            signal_generator["class_index"] = len(self.signal_generators)
        else:
            signal_generator["class_index"] = class_index
        self.signal_generators += [signal_generator]
        if class_name is not None:
            signal_generator["class_name"] = class_name
        if (
            hasattr(signal_generator, "class_name")
            and signal_generator["class_name"] is not None
        ):
            self["class_names"] += [signal_generator["class_name"]]
        self.signal_likelihoods += [likelihood]
        self.total_likelihood += likelihood
        self.signal_probabilities = np.array(
            [
                likelihood / self.total_likelihood
                for likelihood in self.signal_likelihoods
            ]
        )

    def validate_metadata_fields(self) -> bool:
        """Validates signal metadata for each signal generators.

        Returns:
            Whether Signal metadata is valid.
        """
        for generator in self.signal_generators:
            generator.validate_metadata_fields()
        return True

    def __iter__(self):
        """Returns an iterator object for the dataset.

        Returns:
            An iterator object that yields samples from the dataset.
        """
        return self

    def __next__(self) -> Signal | np.ndarray | tuple:
        """Returns a dataset sample and (optionally) corresponding targets for a given index.

        Returns:
            The sample data and the target values.

        Raises:
            IndexError: If the index is out of bounds of the generated samples.
        """
        # user requesting another sample at index +1 larger than current list of generates samples
        # generate new sample
        sample = self.__generate_new_signal__()
        return apply_transforms_and_labels_to_signal(
            sample, self.transforms, self.target_labels
        )

    def __call__(self) -> Signal | np.ndarray | tuple:
        """Same as next(); returns the next item in the dataset.

        Allows datasets to be treated as signal generators for other datasets.
        """
        return next(self)

    def __repr__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            String representation of the dataset.
        """
        repr_str = f"{self.__class__.__name__}("
        if self.metadata is not None:
            repr_str += "metadata="
            repr_str += str(self.metadata)
            repr_str += ", "
        if self.transforms is not None:
            repr_str += "transforms="
            repr_str += str(self.transforms)
            repr_str += ", "
        if self.signal_generators is not None:
            repr_str += "signal_generators="
            repr_str += str(self.signal_generators)
            repr_str += ", "
        repr_str += ")"
        return repr_str

    def _build_noise_floor(self) -> np.ndarray:
        """Generates the noise floor for the dataset by creating an IQ sample and applying a frequency-domain noise estimation.

        Returns:
            The generated IQ samples representing the noise floor.
        """
        real_samples = self.random_generator.normal(
            0, 1, self["num_iq_samples_dataset"]
        )
        imag_samples = self.random_generator.normal(
            0, 1, self["num_iq_samples_dataset"]
        )
        # combine real and imaginary portions of noise
        iq_samples = real_samples + 1j * imag_samples
        # compute an estimate of the noise floor in the frequency domain. use a large stride to process a subset
        # of the data since not many FFTs are needed to be averaged for the noise
        noise_spectrogram_db = compute_spectrogram(
            iq_samples, self["fft_size"], self["fft_stride"] * 16
        )
        # average over time
        noise_fft_db = np.mean(noise_spectrogram_db, axis=1)
        # estimate the average noise value in dB in the frequency domain
        noise_avg_db = np.mean(noise_fft_db)
        # compute the correction factor as the distance from the desired level
        correction_db = self["noise_power_db"] - noise_avg_db
        # apply the correction
        correction = 10 ** (correction_db / 10)
        iq_samples = np.sqrt(correction) * iq_samples
        return iq_samples.astype(np.complex64)

    def __generate_new_signal__(self) -> Signal:
        """Generates a new dataset signal/sample.

        This method creates a new signal by:
        1. Building a noise floor
        2. Generating multiple signal components
        3. Placing them in the frequency domain
        4. Combining them into a final signal

        Returns:
            A new generated dataset signal containing the data and metadata.

        Raises:
            RuntimeError: If unable to generate a valid signal after maximum attempts.
            ValueError: If signal parameters are invalid.
        """
        # build noise floor
        iq_samples = self._build_noise_floor()

        # empty signal list initialization
        signals = []

        # determine number of signals in sample
        num_signals_to_generate = self.random_generator.integers(
            low=self["num_signals_min"], high=self["num_signals_max"] + 1
        )

        # list of rectangles representing the individual signals within the dataset IQ
        signal_rectangle_list = []

        # counter to avoid stuck in infinite loop
        infinite_loop_counter = 0
        infinite_loop_counter_max = 10 * num_signals_to_generate

        # generate individual bursts
        num_signals_created = 0
        while (
            num_signals_created < num_signals_to_generate
            and infinite_loop_counter < infinite_loop_counter_max
        ):

            # increment fail-safe counter
            infinite_loop_counter += 1

            # choose random signal
            generator = self._random_signal_generator()

            # generate signal at complex baseband
            new_signal = generator()

            # apply component transforms
            for ctransform in self.component_transforms:
                new_signal = ctransform(new_signal)

            # frequency shift signal
            # after signal transforms applied at complex baseband
            new_signal = frequency_shift_signal(
                new_signal,
                center_freq_min=self["signal_center_freq_min"],
                center_freq_max=self["signal_center_freq_max"],
                sample_rate=self["sample_rate"],
                frequency_max=self["frequency_max"],
                frequency_min=self["frequency_min"],
                random_generator=self.random_generator,
            )

            # map the signal bounding box into a rectangle in cartesian coordinate system
            if len(iq_samples) - len(new_signal.data) < 1:
                warnings.warn(
                    "generated signal is too large to fit in spectrogram; it will be cut off",
                    UserWarning,
                    stacklevel=2
                )
            start_sample = self.random_generator.integers(
                low=0, high=max(len(iq_samples) - len(new_signal.data), 1)
            )
            new_rectangle = self._map_to_coordinates(new_signal, start_sample)

            # check if the new_rectangle overlaps with any others in spectrogram
            has_overlap = self._check_if_overlap(new_rectangle, signal_rectangle_list)

            # signal is used if there is no overlap OR with some random chance
            if (
                has_overlap is False
                or self.random_generator.uniform(0, 1)
                < self["cochannel_overlap_probability"]
            ):
                num_signals_created += 1
                # store the rectangle for future overlap checking
                signal_rectangle_list.append(new_rectangle)
                # place signal on iq sample cut
                iq_samples[
                    start_sample : start_sample + len(new_signal.data)
                ] += new_signal.data
                # append the signal on the list
                new_signal["start_in_samples"] = start_sample
                signals.append(new_signal)
        # form the sample (dataset object)
        sample = Signal(
            data=iq_samples,
            component_signals=signals,
            center_freq=0,
            bandwidth=max([0] + [signal.bandwidth for signal in signals]),
        )
         # Set class name if available
        if hasattr(self, "class_name"):
            sample.class_name = self.class_name

        if sample.parent is None:
            sample.add_parent(self)
        return sample

    def _map_to_coordinates(self, new_signal: Signal, start_sample: int) -> Rectangle:
        """Maps a new signal to coordinates based on the start sample and signal characteristics.

        Args:
            new_signal: The new signal to map.
            start_sample: The starting sample index of the new signal.

        Returns:
            A rectangle object representing the mapped coordinates of the new signal in the frequency domain.

        Notes:
            This function computes the start and stop times in terms of Fast Fourier Transform (FFT) length using the provided
            start sample and the length of the new signal's data. It also calculates the bin positions in the FFT based on
            the signal's center frequency, bandwidth, and the sample rate. Finally, it maps these positions into rectangle
            coordinates, which it returns as a `Rectangle` object.
        """
        # calculate start and stop time in terms of FFT number
        fft_start_time = np.round(start_sample / self["fft_size"])
        fft_stop_time = np.round(
            (start_sample + len(new_signal.data)) / self["fft_size"]
        )
        # calculate bin position in FFT
        fs = self["sample_rate"]
        fft_start_bin_norm = (
            (new_signal.center_freq - new_signal.bandwidth) + (fs / 2)
        ) / (fs / 2)
        fft_stop_bin_norm = (
            (new_signal.center_freq + new_signal.bandwidth) + (fs / 2)
        ) / (fs / 2)
        fft_start_bin_index = np.round(fft_start_bin_norm * self["fft_size"])
        fft_stop_bin_index = np.round(fft_stop_bin_norm * self["fft_size"])
        # map the position into retangle coordinates
        lower_left_coord = Coordinate(fft_start_time, fft_start_bin_index)
        upper_right_coord = Coordinate(fft_stop_time, fft_stop_bin_index)
        # turn into a rectangle
        return Rectangle(lower_left_coord, upper_right_coord)

    def _check_if_overlap(
        self, new_rectangle: Rectangle, signal_rectangle_list: list
    ) -> bool:
        """Determines if a new rectangle overlaps with any of the rectangles in a list.

        Args:
            new_rectangle: The new rectangle to check for overlap.
            signal_rectangle_list: A list of rectangles to check against for overlap.

        Returns:
            True if the new rectangle overlaps with any rectangle in the list, otherwise False.
        """
        # initialize the boolean value which determines if there is overlap or not
        has_overlap = False
        # determine if overlap
        if len(signal_rectangle_list) > 0:
            # check to see if the current rectangle overlaps with any signals currently
            # in the spectrogram
            for reference_box in signal_rectangle_list:
                # check for invidivual overlap
                individual_overlap = is_rectangle_overlap(new_rectangle, reference_box)
                # combine with previous potential overlap checks
                has_overlap = has_overlap or individual_overlap
        return has_overlap

    def _random_signal_generator(self) -> BaseSignalGenerator:
        """Randomly selects which signal generator to use next"""
        return self.random_generator.choice(
            self.signal_generators, p=self.signal_probabilities
        )


class StaticTorchSigDataset(Dataset, Seedable):
    """Static Dataset class, which loads pre-generated data from a directory.

    Args:
        root: The root directory where the dataset is stored.
        transforms: Transforms to apply to the data (default: []).
        file_handler_class: Class used for reading the dataset (default: HDF5FileHandler).
    """

    def __init__(
        self,
        root: str,
        file_handler_class=HDF5Reader,
        transforms: list = [],
        target_labels: list | None = None,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            root: The root directory where the dataset is stored.
            file_handler_class: Class used for reading the dataset.
            transforms: Transforms to apply to the data.
            target_labels: Labels to extract from the signal.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.root = Path(root)
        self.reader = file_handler_class(root=self.root)

        Seedable.__init__(self, **kwargs)
        self.transforms = transforms
        for transform in self.transforms:
            transform.add_parent(self)
        self.target_labels = target_labels

        # dataset size
        self.dataset_length = len(self.reader)

        self._verify()

    def _verify(self) -> None:
        """Checks if root exists

        Raises:
            ValueError: Root does not exist.
        """
        # check root

        if not self.root.exists():
            raise ValueError(f"root does not exist: {self.root}")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= idx < len(self):
            sample = self.reader.read(idx=idx)
            return apply_transforms_and_labels_to_signal(
                sample, self.transforms, self.target_labels
            )

        raise IndexError(
            f"Index {idx} is out of bounds. Must be [0, {self.__len__() - 1}]"
        )

    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        """Returns a detailed string representation of the dataset.

        Returns:
            A detailed string representation of the dataset.
        """
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"file_handler_class={self.reader}"
        )
