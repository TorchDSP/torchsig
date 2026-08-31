"""Dataset Base Classes for creation and static loading."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from torchsig.datasets.dataset_utils import frequency_shift_signal
from torchsig.signals.builder import BaseSignalGenerator, ConcatSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import GroupingLabel
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.coordinate_system import Coordinate, Rectangle, is_rectangle_overlap
from torchsig.utils.dsp import compute_spectrogram, update_signal_snr_bandwidth
from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.metadata_logging import (
    get_metadata_logging_context,
    metadata_logging_context,
)
from torchsig.utils.random import Seedable
from torchsig.utils.signal_building import lookup_signal_generator_by_string

from .pipeline_failover import PipelineFailOverEnabled

log = logging.getLogger(__name__)


# Type checking imports
if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from torchsig.transforms.base_transforms import Transform

__all__ = ["StaticTorchSigDataset", "TorchSigDatasetConfig", "TorchSigIterableDataset", "apply_label_to_signal", "apply_transforms_and_labels_to_signal"]
"""Dataset Base Classes for creation and static loading."""


@dataclass(frozen=True)
class TorchSigDatasetConfig:
    """Configuration dataclass for TorchSig datasets.

    Attributes:
        dataset_id: A unique identifier for the dataset.
        dataset_length: The total number of samples in the dataset.
        seed: A random seed for reproducibility.
        impairment_level: The level of impairment to apply to the signals.
        output_representation: The representation of the output data (e.g., "iq" or "spectrogram").
        output_spectrogram_fft: The FFT size to use when generating spectrograms (if output_representation is "spectrogram").
        signal_sampling_mode: The mode for sampling signals, either "per_signal" or "per_family".
        dataset_metadata: A dictionary containing additional metadata about the dataset.
        target_labels: A list of target_labels. Defaults to ["class_index"].
    """

    dataset_id: str
    dataset_length: int
    seed: int
    impairment_level: int
    output_representation: Literal["iq", "spectrogram"]
    output_spectrogram_fft: int | None
    signal_sampling_mode: Literal["per_signal", "per_family"]
    dataset_metadata: dict[str, Any]
    target_labels: list[str] = field(default_factory=lambda: ["class_index"])  # default classification use case


def apply_label_to_signal(sample: Signal, target_label: str) -> list:
    """Extract a target label from a signal and its component signals.

    Target labels are resolved through the public ``Signal`` interface rather
    than directly from the underlying metadata dictionary. This allows both
    stored metadata fields (e.g. ``class_name``) and computed properties
    (e.g. ``start``, ``stop``, ``lower_freq``, and ``upper_freq``) to be
    requested uniformly.

    A label stored directly on the sample, but not on any component, is
    returned as one sample-level value. Otherwise, if the signal contains
    component signals, labels are extracted only from the components. If it
    has no components, the label is extracted from the signal itself. This
    avoids returning duplicate labels for both a composite signal and its
    children while supporting aggregate labels such as multi-hot vectors.

    Args:
        sample: Signal from which to extract target labels.
        target_label: Name of the target label or ``Signal`` property to
            extract.

    Returns:
        A list containing a sample-level value, one value for each component
        signal, or a single value for a signal without components.
    """
    is_sample_level_label = target_label in sample.metadata and not any(target_label in component.metadata for component in sample.component_signals)
    if is_sample_level_label:
        return [sample[target_label]]

    values = []

    signals = sample.component_signals or [sample]

    for signal in signals:
        if target_label == "class_index":
            if hasattr(signal, "class_index"):
                values.append(int(signal.class_index))
            elif hasattr(signal, "class_name"):
                class_names = signal.get_full_metadata()["class_names"]
                values.append(int(list(class_names).index(signal.class_name)))
        elif hasattr(signal, target_label):
            values.append(getattr(signal, target_label))

    return values


def apply_transforms_and_labels_to_signal(sample: Signal, transforms: list[Transform | callable], target_labels: list) -> Signal | np.ndarray | tuple:
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
        is_sample_level_label = key in sample.metadata and not any(key in component.metadata for component in sample.component_signals)
        if len(values) == 1 and (is_sample_level_label or sample["num_signals_max"] == 1):
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
        transforms: list[Transform | callable] | None = None,
        component_transforms: list[Transform | callable] | None = None,
        target_labels: list | None = None,
        sampling_grouping: (GroupingLabel | str | Path | Mapping[str, Any] | None) = None,
        per_signal_metadata: dict[str, dict[str, int | float]] | None = None,
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
            sampling_grouping: A GroupingLabel, grouping YAML path, or
                grouping configuration mapping. When provided, signal
                generators use each group's configured ``probability`` or
                normalized ``likelihood``. Without weights, represented
                groups are equiprobable. Classes within a group are uniform.
            per_signal_metadata: Optional metadata overrides keyed by signal
                class name. Supported overrides are the minimum and maximum
                values for SNR, signal duration in samples, and bandwidth.
            validate_init: Whether to validate metadata during initialization.
            **kwargs: Additional keyword arguments passed to the parent class.

        Example:
            Configure different SNR ranges for two signal classes::

                dataset = TorchSigIterableDataset(
                    signal_generators=["qpsk", "am-dsb-sc"],
                    per_signal_metadata={
                        "qpsk": {"snr_db_min": 0, "snr_db_max": 10},
                        "am-dsb-sc": {"snr_db_min": 15, "snr_db_max": 30},
                    },
                    metadata=dataset_metadata,
                )
        """
        HierarchicalMetadataObject.__init__(self, **kwargs)
        self.validate_init = validate_init
        self.signal_generators = []
        self.signal_likelihoods = []
        self.signal_probabilities = np.array([], dtype=float)
        self._signal_probability_mode = "likelihood"
        self.target_labels = target_labels
        self.per_signal_metadata = self._validate_per_signal_metadata(per_signal_metadata)
        self.transforms = [] if transforms is None else transforms
        self.component_transforms = [] if component_transforms is None else component_transforms
        self._metadata_logging_sample_index = 0
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
        if sampling_grouping is not None:
            self.balance_signal_generators_by_group(sampling_grouping)
        unmatched_classes = set(self.per_signal_metadata).difference(generator.class_name for generator in self.signal_generators if hasattr(generator, "class_name"))
        if unmatched_classes:
            raise ValueError(f"per_signal_metadata contains classes not configured in signal_generators: {sorted(unmatched_classes)}")
        if self.validate_init:
            self.validate_metadata_fields()

    @staticmethod
    def _validate_per_signal_metadata(
        per_signal_metadata: dict[str, dict[str, int | float]] | None,
    ) -> dict[str, dict[str, int | float]]:
        """Validate and copy per-signal metadata overrides."""
        if per_signal_metadata is None:
            return {}
        if not isinstance(per_signal_metadata, dict):
            raise TypeError("per_signal_metadata must be a dictionary")

        supported_fields = {
            "snr_db_min",
            "snr_db_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
            "bandwidth_min",
            "bandwidth_max",
        }
        validated: dict[str, dict[str, int | float]] = {}
        for class_name, overrides in per_signal_metadata.items():
            if not isinstance(class_name, str):
                raise TypeError("per_signal_metadata class names must be strings")
            if not isinstance(overrides, dict):
                raise TypeError(f"per_signal_metadata[{class_name!r}] must be a dictionary")

            unknown_fields = set(overrides).difference(supported_fields)
            if unknown_fields:
                raise ValueError(f"unsupported per-signal metadata fields for {class_name!r}: {sorted(unknown_fields)}")

            validated_overrides: dict[str, int | float] = {}
            for metadata_field, value in overrides.items():
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    raise TypeError(f"per_signal_metadata[{class_name!r}][{metadata_field!r}] must be a real number")
                if not np.isfinite(value):
                    raise ValueError(f"per_signal_metadata[{class_name!r}][{metadata_field!r}] must be finite")
                if metadata_field.startswith(("bandwidth_", "signal_duration_")) and value <= 0:
                    raise ValueError(f"per_signal_metadata[{class_name!r}][{metadata_field!r}] must be positive")
                validated_overrides[metadata_field] = value
            validated[class_name] = validated_overrides

        return validated

    def _apply_per_signal_metadata(self, signal_generator: callable) -> None:
        """Apply local metadata overrides for a configured signal class."""
        if not hasattr(signal_generator, "class_name"):
            return
        overrides = self.per_signal_metadata.get(signal_generator.class_name, {})
        for metadata_field, value in overrides.items():
            signal_generator[metadata_field] = value

    @staticmethod
    def _validate_positive_weight(value: float, parameter_name: str) -> float:
        """Validate a likelihood/probability value used for class selection."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{parameter_name} must be a real number, got {type(value).__name__}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{parameter_name} must be finite")
        if value <= 0.0:
            raise ValueError(f"{parameter_name} must be > 0")

        return value

    def _validate_signal_sampling_configuration(self, require_complete: bool = True) -> None:
        """Validate the dataset's configured class-selection distribution."""
        if len(self.signal_generators) == 0:
            return

        if self._signal_probability_mode == "probability":
            probabilities = np.asarray(self.signal_probabilities, dtype=float)
            if probabilities.shape[0] != len(self.signal_generators):
                raise ValueError("signal probability count does not match number of generators")
            if np.any(probabilities < 0.0):
                raise ValueError("all signal probabilities must be >= 0")
            if not np.any(probabilities > 0.0):
                raise ValueError("at least one signal probability must be > 0")

            probability_sum = float(np.sum(probabilities))
            if probability_sum > 1.0 + 1e-8:
                raise ValueError(f"signal probabilities must sum to 1.0, found {probability_sum}")
            if require_complete and not np.isclose(probability_sum, 1.0, atol=1e-8):
                raise ValueError(f"signal probabilities must sum to 1.0 before sampling, found {probability_sum}")
            return

        likelihoods = np.asarray(self.signal_likelihoods, dtype=float)
        if likelihoods.shape[0] != len(self.signal_generators):
            raise ValueError("signal likelihood count does not match number of generators")
        if np.any(likelihoods <= 0.0):
            raise ValueError("all signal likelihoods must be > 0")

    def _refresh_signal_probabilities(self) -> None:
        """Recompute normalized sampling probabilities from configured weights."""
        if len(self.signal_generators) == 0:
            self.signal_probabilities = np.array([], dtype=float)
            return

        self._validate_signal_sampling_configuration(require_complete=False)

        if self._signal_probability_mode == "probability":
            self.signal_probabilities = np.asarray(
                self.signal_probabilities,
                dtype=float,
            )
            return

        likelihoods = np.asarray(self.signal_likelihoods, dtype=float)
        self.signal_probabilities = likelihoods / np.sum(likelihoods)

    def balance_signal_generators_by_group(
        self,
        grouping: GroupingLabel | str | Path | Mapping[str, Any],
    ) -> None:
        """Weight generators using the configured group sampling weights.

        Groups are resolved from metadata already available on each signal
        generator. A grouping based on generated metadata, such as a realized
        bandwidth, cannot be balanced before generation and raises an error.
        Group probabilities are used directly and must sum to one. Group
        likelihoods are normalized over represented groups. When neither is
        configured, represented groups receive equal probability.

        Args:
            grouping: A GroupingLabel instance, YAML path, or configuration
                mapping.

        Raises:
            ValueError: If there are no generators, a generator lacks the
                grouping source field, or a generator value matches no group.
        """
        if not isinstance(grouping, GroupingLabel):
            grouping = GroupingLabel(grouping)
        if not self.signal_generators:
            raise ValueError("cannot balance groups without signal generators")

        generator_groups = []
        for generator in self.signal_generators:
            if not hasattr(generator, grouping.source):
                class_name = getattr(generator, "class_name", "<unlabeled>")
                raise ValueError(f"cannot balance class {class_name!r}: grouping source {grouping.source!r} is not available before generation")
            source_value = getattr(generator, grouping.source)
            try:
                group_name, _ = grouping.match(source_value)
            except ValueError as error:
                raise ValueError(f"cannot balance generator value {source_value!r} from source {grouping.source!r}: no group matched") from error
            generator_groups.append(group_name)

        group_counts = {group_name: generator_groups.count(group_name) for group_name in set(generator_groups)}
        configured_groups = {group["name"]: group for group in grouping.groups}
        uses_probabilities = any("probability" in group for group in grouping.groups)
        uses_likelihoods = any("likelihood" in group for group in grouping.groups)

        if uses_probabilities:
            unrepresented_probability_groups = [group_name for group_name, group in configured_groups.items() if group_name not in group_counts and group.get("probability", 0.0) > 0.0]
            if unrepresented_probability_groups:
                raise ValueError(f"cannot assign positive probability to groups without signal generators: {unrepresented_probability_groups}")
            group_probabilities = {
                group_name: configured_groups[group_name].get(
                    "probability",
                    0.0,
                )
                for group_name in group_counts
            }
        elif uses_likelihoods:
            group_likelihoods = {
                group_name: configured_groups[group_name].get(
                    "likelihood",
                    1.0,
                )
                for group_name in group_counts
            }
            likelihood_sum = float(sum(group_likelihoods.values()))
            group_probabilities = {group_name: likelihood / likelihood_sum for group_name, likelihood in group_likelihoods.items()}
        else:
            equal_group_probability = 1.0 / len(group_counts)
            group_probabilities = dict.fromkeys(
                group_counts,
                equal_group_probability,
            )

        self._signal_probability_mode = "probability"
        self.signal_probabilities = np.asarray(
            [group_probabilities[group_name] / group_counts[group_name] for group_name in generator_groups],
            dtype=float,
        )
        self.signal_likelihoods = []
        self.sampling_grouping = grouping
        self._sampling_generator_groups = generator_groups
        self._validate_signal_sampling_configuration(require_complete=True)

    def init_signal_generator(self, signal_generator: str | callable) -> None:
        """Initializes the signal generator.

        Args:
            signal_generator: The signal generator to be initialized. If a string, it is first looked up to retrieve the corresponding signal generator function.

        Raises:
            TypeError: If the signal_generator is neither a string nor a callable.
        """
        if isinstance(signal_generator, str):
            signal_generator = lookup_signal_generator_by_string(signal_generator)

        if isinstance(signal_generator, ConcatSignalGenerator):
            for child_generator in signal_generator.signal_generators:
                self.init_signal_generator(child_generator)
            return

        self.add_signal_generator(signal_generator)

    def add_signal_generator(
        self,
        signal_generator: callable,
        class_name: str | None = None,
        class_index: int | None = None,
        likelihood: float | None = None,
        probability: float | None = None,
    ) -> None:
        """Add a signal generator to this dataset.

        Args:
            signal_generator: Callable that takes no arguments and returns a Signal.
            class_name: Optional name for this signal class. If omitted, the signal
                is generated and added to the data without a class-name label.
            class_index: Optional class index. If omitted, the generator's position
                in the dataset is used.
            likelihood: Relative sampling weight for this signal class. When
                explicit probabilities are not used, omitted likelihoods default
                to 1.0.
            probability: Explicit sampling probability for this signal class. Once
                explicit probability mode is selected, every generator must specify
                a probability. The final probabilities must sum to 1.0 before
                sampling.

        Raises:
            TypeError: If a likelihood or probability is not a real number.
            ValueError: If the sampling configuration is invalid.
        """
        if likelihood is not None and probability is not None:
            raise ValueError("Specify only one of likelihood or probability for a signal generator")

        has_generators = bool(self.signal_generators)
        using_probability_mode = self._signal_probability_mode == "probability"
        using_likelihood_mode = self._signal_probability_mode == "likelihood" and has_generators

        if using_probability_mode and probability is None:
            raise ValueError("All signal generators must specify probability once probability mode is used")

        if using_likelihood_mode and probability is not None:
            raise ValueError("Cannot mix explicit probability with likelihood/default likelihood generators")

        if probability is not None:
            probability = self._validate_positive_weight(
                probability,
                "probability",
            )

            candidate_probabilities = np.append(
                np.asarray(self.signal_probabilities, dtype=float),
                probability,
            )
            probability_sum = float(np.sum(candidate_probabilities))

            if probability_sum > 1.0 + 1e-8:
                raise ValueError(f"signal probabilities must sum to 1.0 or less while configuring the dataset, found {probability_sum}")
        else:
            likelihood = 1.0 if likelihood is None else likelihood
            likelihood = self._validate_positive_weight(
                likelihood,
                "likelihood",
            )

        if class_name is not None:
            signal_generator["class_name"] = class_name

        self._apply_per_signal_metadata(signal_generator)

        if isinstance(signal_generator, Seedable):
            signal_generator.add_parent(self)

        for minimum_field, maximum_field in (
            ("snr_db_min", "snr_db_max"),
            (
                "signal_duration_in_samples_min",
                "signal_duration_in_samples_max",
            ),
            ("bandwidth_min", "bandwidth_max"),
        ):
            if hasattr(signal_generator, minimum_field) and hasattr(signal_generator, maximum_field) and signal_generator[minimum_field] > signal_generator[maximum_field]:
                raise ValueError(f"{minimum_field} must be less than or equal to {maximum_field} for signal class {signal_generator.class_name!r}")

        try:
            if self.validate_init:
                signal_generator.validate_metadata_fields()
        except AttributeError:
            pass  # Proceed without validation at the caller's risk.

        resolved_class_index = len(self.signal_generators) if class_index is None else class_index
        signal_generator["class_index"] = resolved_class_index

        self.signal_generators.append(signal_generator)

        if hasattr(signal_generator, "class_name") and signal_generator["class_name"] is not None:
            self["class_names"].append(signal_generator["class_name"])

        if probability is not None:
            self._signal_probability_mode = "probability"
            self.signal_probabilities = candidate_probabilities
        else:
            self._signal_probability_mode = "likelihood"
            self.signal_likelihoods.append(likelihood)

        self._refresh_signal_probabilities()

    def validate_metadata_fields(self) -> bool:
        """Validate dataset and signal generator metadata."""
        self._validate_signal_duration_limits()

        for generator in self.signal_generators:
            generator.validate_metadata_fields()

        return True

    def _validate_signal_duration_limits(self) -> None:
        """Warn when configured signal durations exceed the dataset sample length.

        Signals longer than the dataset IQ buffer are supported, but they will be
        truncated during sample generation. This warning helps identify metadata
        configurations that are likely to truncate every generated component.
        """
        if self["signal_duration_in_samples_max"] > self["num_iq_samples_dataset"]:
            warnings.warn(
                "signal_duration_in_samples_max exceeds num_iq_samples_dataset; generated component signals may be truncated to fit within the dataset sample",
                UserWarning,
                stacklevel=2,
            )

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
        sample_index = self._metadata_logging_sample_index
        self._metadata_logging_sample_index += 1
        if not self._metadata_logging_is_active():
            sample = self.__generate_new_signal__()
            return apply_transforms_and_labels_to_signal(
                sample,
                self.transforms,
                self.target_labels,
            )

        with self._metadata_logging_sample_context(sample_index):
            sample = self._run_with_metadata_logging_stage(
                "generate",
                self.__generate_new_signal__,
            )
            return self._run_with_metadata_logging_stage(
                "transform",
                lambda: self._transform_and_log_metadata_snapshot(
                    sample,
                    lambda: apply_transforms_and_labels_to_signal(
                        sample,
                        self.transforms,
                        self.target_labels,
                    ),
                ),
            )

    @contextmanager
    def _metadata_logging_sample_context(
        self,
        sample_index: int,
    ) -> Iterator[None]:
        """Establish correlation fields for one generated dataset sample."""
        outer_context = get_metadata_logging_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        local_metadata = object.__getattribute__(self, "_metadata")
        dataset_id = local_metadata.get(
            "dataset_id",
            outer_context.dataset_id or type(self).__name__,
        )
        with metadata_logging_context(
            dataset_id=str(dataset_id),
            sample_index=sample_index,
            worker_id=worker_id,
        ):
            yield

    def _metadata_logging_is_active(self) -> bool:
        """Return whether sample correlation should be established."""
        instance_attributes = object.__getattribute__(self, "__dict__")
        if instance_attributes.get("_metadata_debug_enabled", False):
            return True
        return get_metadata_logging_context().session_id is not None

    @staticmethod
    def _run_with_metadata_logging_stage(
        stage: str,
        operation: Callable[[], Any],
    ) -> Any:
        """Run one generation stage with an optional correlation field."""
        if get_metadata_logging_context().session_id is None:
            return operation()
        with metadata_logging_context(fields={"stage": stage}):
            return operation()

    def _transform_and_log_metadata_snapshot(
        self,
        sample: Signal,
        operation: Callable[[], Any],
    ) -> Any:
        """Run final transforms and optionally log the completed metadata."""
        result = operation()
        session = object.__getattribute__(self, "__dict__").get("_metadata_debug_session")
        if self.metadata_debug_enabled and session is not None and "snapshot" in session.config.events:
            self.log_metadata_snapshot(sample, include_components=True)
        return result

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
        real_samples = self.random_generator.normal(0, 1, self["num_iq_samples_dataset"])
        imag_samples = self.random_generator.normal(0, 1, self["num_iq_samples_dataset"])
        # combine real and imaginary portions of noise
        iq_samples = real_samples + 1j * imag_samples
        # compute an estimate of the noise floor in the frequency domain. use a large stride to process a subset
        # of the data since not many FFTs are needed to be averaged for the noise
        noise_spectrogram_db = compute_spectrogram(iq_samples, self["fft_size"], self["fft_stride"] * 16)
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
        """Generate a new synthetic dataset sample.

        The generated sample consists of a noise floor plus zero or more component
        signals placed into the IQ buffer. Each component is generated at complex
        baseband, transformed, updated with SNR and bandwidth metadata, frequency
        shifted, checked for overlap, and inserted into the dataset sample.

        Returns:
            A generated Signal containing IQ data and component signal metadata.
        """
        iq_samples = self._build_noise_floor()
        component_signals = []
        signal_rectangles = []

        num_signals_to_generate = int(
            self.random_generator.integers(
                low=self["num_signals_min"],
                high=self["num_signals_max"] + 1,
            )
        )
        max_attempts = 10 * num_signals_to_generate
        retry_group = None

        for _ in range(max_attempts):
            if len(component_signals) >= num_signals_to_generate:
                break

            generator = self._random_signal_generator(group_name=retry_group) if hasattr(self, "sampling_grouping") else None
            if retry_group is None and generator is not None:
                source_value = getattr(
                    generator,
                    self.sampling_grouping.source,
                )
                retry_group, _ = self.sampling_grouping.match(source_value)
            new_signal = self._generate_component_signal() if generator is None else self._generate_component_signal(generator)
            start_sample = self._choose_start_sample(
                iq_samples,
                new_signal,
            )

            # Truncate before calculating the overlap rectangle so the rectangle
            # reflects the portion of the component that will actually be inserted.
            self._truncate_component_signal(
                iq_samples,
                new_signal,
                start_sample,
            )
            new_rectangle = self._map_to_coordinates(
                new_signal,
                start_sample,
            )
            has_overlap = self._check_if_overlap(
                new_rectangle,
                signal_rectangles,
            )

            if has_overlap and not self._allow_cochannel_overlap():
                continue

            self._insert_component_signal(
                iq_samples,
                new_signal,
                start_sample,
            )
            signal_rectangles.append(new_rectangle)
            component_signals.append(new_signal)
            retry_group = None

        sample = Signal(
            data=iq_samples,
            component_signals=component_signals,
            center_freq=0,
            bandwidth=max([0] + [signal.bandwidth for signal in component_signals]),
        )

        if hasattr(self, "class_name"):
            sample.class_name = self.class_name

        if sample.parent is None:
            # The generated sample needs access to dataset metadata, but registering
            # it as a child would retain every generated sample in memory.
            sample.add_parent(self, register=False)

        return sample

    def _generate_component_signal(
        self,
        generator: BaseSignalGenerator | None = None,
    ) -> Signal:
        """Generate and prepare one component signal."""
        if generator is None:
            generator = self._random_signal_generator()
        signal = generator()

        for component_transform in self.component_transforms:
            signal = component_transform(signal)

        update_signal_snr_bandwidth(self, signal)

        return frequency_shift_signal(
            signal,
            center_freq_min=self["signal_center_freq_min"],
            center_freq_max=self["signal_center_freq_max"],
            sample_rate=self["sample_rate"],
            frequency_max=self["frequency_max"],
            frequency_min=self["frequency_min"],
            random_generator=self.random_generator,
        )

    def _choose_start_sample(
        self,
        iq_samples: np.ndarray,
        signal: Signal,
    ) -> int:
        """Choose a valid start index for a component signal."""
        num_available_samples = len(iq_samples)
        num_signal_samples = len(signal.data)

        if num_signal_samples > num_available_samples:
            warnings.warn(
                "generated signal is too large to fit in the dataset sample; it will be cut off",
                UserWarning,
                stacklevel=2,
            )

        # NumPy's upper bound is exclusive, so add one to include the final valid
        # start position.
        num_valid_start_positions = max(
            num_available_samples - num_signal_samples + 1,
            1,
        )

        return int(
            self.random_generator.integers(
                low=0,
                high=num_valid_start_positions,
            )
        )

    def _truncate_component_signal(
        self,
        iq_samples: np.ndarray,
        signal: Signal,
        start_sample: int,
    ) -> None:
        """Truncate a component to the portion that fits in the IQ buffer."""
        num_samples_to_add = max(
            min(
                len(signal.data),
                len(iq_samples) - start_sample,
            ),
            0,
        )

        if num_samples_to_add < len(signal.data):
            signal.data = signal.data[:num_samples_to_add]
            signal["duration_in_samples"] = num_samples_to_add

    def _allow_cochannel_overlap(self) -> bool:
        """Return whether an overlapping component should be accepted."""
        return bool(self.random_generator.uniform(0, 1) < self["cochannel_overlap_probability"])

    def _insert_component_signal(
        self,
        iq_samples: np.ndarray,
        signal: Signal,
        start_sample: int,
    ) -> None:
        """Insert the portion of a component signal that fits in the IQ buffer."""
        # Keep this method safe when called directly, even though normal generation
        # truncates the signal before overlap detection.
        self._truncate_component_signal(
            iq_samples,
            signal,
            start_sample,
        )

        stop_sample = start_sample + len(signal.data)

        iq_samples[start_sample:stop_sample] += signal.data
        signal["start_in_samples"] = start_sample
        signal["duration_in_samples"] = len(signal.data)

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
        fft_stop_time = np.round((start_sample + len(new_signal.data)) / self["fft_size"])
        # calculate bin position in FFT
        fs = self["sample_rate"]
        fft_start_bin_norm = ((new_signal.center_freq - new_signal.bandwidth) + (fs / 2)) / (fs / 2)
        fft_stop_bin_norm = ((new_signal.center_freq + new_signal.bandwidth) + (fs / 2)) / (fs / 2)
        fft_start_bin_index = np.round(fft_start_bin_norm * self["fft_size"])
        fft_stop_bin_index = np.round(fft_stop_bin_norm * self["fft_size"])
        # map the position into retangle coordinates
        lower_left_coord = Coordinate(fft_start_time, fft_start_bin_index)
        upper_right_coord = Coordinate(fft_stop_time, fft_stop_bin_index)
        # turn into a rectangle
        return Rectangle(lower_left_coord, upper_right_coord)

    def _check_if_overlap(self, new_rectangle: Rectangle, signal_rectangle_list: list) -> bool:
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

    def _random_signal_generator(
        self,
        group_name: str | None = None,
    ) -> BaseSignalGenerator:
        """Randomly select a generator, optionally within one sampling group."""
        if len(self.signal_generators) == 0:
            raise ValueError("cannot sample from a dataset with no signal generators")

        if group_name is not None:
            matching_generators = [
                generator
                for generator, generator_group in zip(
                    self.signal_generators,
                    self._sampling_generator_groups,
                    strict=True,
                )
                if generator_group == group_name
            ]
            if not matching_generators:
                raise ValueError(f"sampling group {group_name!r} has no signal generators")
            return self.random_generator.choice(matching_generators)

        self._validate_signal_sampling_configuration(require_complete=True)
        self._refresh_signal_probabilities()
        return self.random_generator.choice(self.signal_generators, p=self.signal_probabilities)


class SafeTorchSigIterableDataset(PipelineFailOverEnabled, TorchSigIterableDataset):
    """A fault-tolerant version of TorchSigIterableDataset with automatic error recovery.

    This class behaves exactly like :class:`TorchSigIterableDataset` but adds
    built-in recovery mechanisms for when transforms fail during data generation.
    It's designed to prevent dataset generation from stopping due to transform errors
    by either retrying failed operations or falling back to safe outputs.

    The class maintains full compatibility with the parent class API while adding
    configurable error handling through the ``pipeline_fallback`` and
    ``pipeline_max_retries`` attributes.

    Example:
        >>> ds = SafeTorchSigIterableDataset(signal_generators="all", transforms=[MyTransform()], target_labels=["class_index"])
        >>> # Configure fallback behavior
        >>> ds.pipeline_fallback = "retry"
        >>> ds.pipeline_max_retries = 3
        >>> # Dataset will now retry failed transforms up to 3 times
        >>> sample = next(ds)
    """

    def __next__(self) -> Any:
        """Generate the next dataset sample with pipeline fault tolerance.

        Sample creation is performed in two stages:

        1. Generate a raw signal, including any component-level transforms.
        2. Apply whole-signal transforms and target label generation.

        Each stage is executed through the configured failover mechanism. If a
        stage raises an exception, the behavior is controlled by
        ``pipeline_fallback``:

        - ``"original"``: Return the original raw signal when available.
        - ``"zero"``: Return a zero-filled signal with matching shape.
        - ``"retry"``: Retry the failed stage up to
        ``pipeline_max_retries`` times before falling back.

        If failure occurs during raw signal generation, no original signal exists
        and only retry or zero fallbacks are possible. If failure occurs during
        whole-signal transforms or label generation, the generated raw signal can
        be used as the fallback result.

        Returns:
            A successfully generated sample, or a fallback sample if recovery
            logic is triggered.

        Note:
            All pipeline failures are logged to aid debugging and monitoring.
        """
        sample_index = self._metadata_logging_sample_index
        self._metadata_logging_sample_index += 1
        if not self._metadata_logging_is_active():
            raw_signal = self._run_with_fallback(
                self.__generate_new_signal__,
                fallback_raw_signal=None,
            )
            return self._run_with_fallback(
                lambda: apply_transforms_and_labels_to_signal(
                    raw_signal,
                    self.transforms,
                    self.target_labels,
                ),
                fallback_raw_signal=raw_signal,
            )

        with self._metadata_logging_sample_context(sample_index):
            # Stage 1: generation + component_transforms. If this fails, no
            # raw/original sample exists yet.
            raw_signal = self._run_with_metadata_logging_stage(
                "generate",
                lambda: self._run_with_fallback(
                    self.__generate_new_signal__,
                    fallback_raw_signal=None,
                ),
            )

            # Stage 2: whole-signal transforms + labels. If this fails,
            # raw_signal exists, so original fallback is possible.
            return self._run_with_metadata_logging_stage(
                "transform",
                lambda: self._transform_and_log_metadata_snapshot(
                    raw_signal,
                    lambda: self._run_with_fallback(
                        lambda: apply_transforms_and_labels_to_signal(
                            raw_signal,
                            self.transforms,
                            self.target_labels,
                        ),
                        fallback_raw_signal=raw_signal,
                    ),
                ),
            )

    def set_fallback_policy(
        self,
        fallback: Literal["original", "zero", "retry"] = "original",
        max_retries: int | None = None,
    ) -> None:
        """Configure the dataset's error recovery behavior.

        Args:
            fallback: The recovery strategy to use when transforms fail:
                - "original": Return the untransformed signal
                - "zero": Return a zero-filled array of matching shape
                - "retry": Attempt the transform again (requires max_retries)

            max_retries: Maximum number of retry attempts when fallback="retry".
                Must be a positive integer. Ignored for other fallback modes.

        Raises:
            ValueError: If max_retries is provided with a fallback mode other than "retry"

        Example:
            >>> ds = SafeTorchSigIterableDataset(...)
            >>> # Configure to retry failed transforms up to 5 times
            >>> ds.set_fallback_policy(fallback="retry", max_retries=5)
        """
        self.pipeline_fallback = fallback
        if max_retries is not None:
            if fallback != "retry":
                raise ValueError("max_retries is only allowed with fallback='retry'")
            self.pipeline_max_retries = max_retries


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
            return apply_transforms_and_labels_to_signal(sample, self.transforms, self.target_labels)

        raise IndexError(f"Index {idx} is out of bounds. Must be [0, {self.__len__() - 1}]")

    def __getitems__(
        self,
        indices: list[int],
    ) -> list[Signal | np.ndarray | tuple]:
        """Retrieve a DataLoader batch, using native contiguous reads when available.

        Readers without ``read_signals_batch`` and non-contiguous index lists use
        the existing single-item path.
        """
        if not indices:
            return []
        if any(idx < 0 or idx >= len(self) for idx in indices):
            invalid_idx = next(idx for idx in indices if idx < 0 or idx >= len(self))
            raise IndexError(f"Index {invalid_idx} is out of bounds. Must be [0, {self.__len__() - 1}]")

        read_signals_batch = getattr(self.reader, "read_signals_batch", None)
        contiguous = all(idx == indices[0] + offset for offset, idx in enumerate(indices))
        if read_signals_batch is None or not contiguous:
            return [self[idx] for idx in indices]

        samples = read_signals_batch(indices[0], indices[-1] + 1)
        return [
            apply_transforms_and_labels_to_signal(
                sample,
                self.transforms,
                self.target_labels,
            )
            for sample in samples
        ]

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
        return f"{self.__class__.__name__}(root={self.root}, file_handler_class={self.reader})"
