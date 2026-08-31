"""Signal and Signal Metadata classes.
This module defines the `Signal` class and its associated functionality,
which is used to represent and manipulate signal data and metadata.

Examples:
    Signal:
        >>> from torchsig.signals import Signal
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0])
        >>> new_sig = Signal(data=data)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.dsp import (
    bandwidth_from_lower_upper_freq,
    center_freq_from_lower_upper_freq,
    lower_freq_from_center_freq_bandwidth,
    upper_freq_from_center_freq_bandwidth,
)

__all__ = ["Signal", "SignalMetadataObject"]


class SignalMetadataObject(HierarchicalMetadataObject):
    """Represents metadata associated with a signal.

    This class extends HierarchicalMetadataObject to provide signal-specific
    metadata properties and calculations. ``center_freq`` is expressed in Hz
    relative to the containing dataset's frequency origin; an unplaced
    baseband component normally has a center frequency of zero. ``bandwidth``
    is the full two-sided bandwidth in Hz. These are the canonical frequency
    fields, and ``lower_freq`` and ``upper_freq`` are derived as
    ``center_freq - bandwidth / 2`` and ``center_freq + bandwidth / 2``.

    This metadata object does not independently enforce dataset frequency
    limits. Wideband placement is responsible for choosing or filtering a
    center frequency that satisfies the dataset's configured bounds. The field
    is a signed offset, not an absolute RF tuning frequency. Applications with
    an external RF center can calculate an absolute component frequency by
    adding that external center to ``center_freq``.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the SignalMetadata object.

        Args:
            **kwargs: Metadata key-value pairs to initialize the object.
        """
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set metadata and invalidate frequency edges when canonical fields change.

        ``center_freq`` and ``bandwidth`` are the canonical frequency metadata.
        Lower and upper frequency edges are derived from them, so any local
        legacy or pending edge values must not survive an update to either
        canonical field.

        Args:
            key: Metadata key to set.
            value: Metadata value to store.
        """
        if key in {"center_freq", "bandwidth"}:
            self._metadata.pop("_lower_frequency", None)
            self._metadata.pop("_upper_frequency", None)
        super().__setitem__(key, value)

    @property
    def start(self) -> float:
        """Signal start normalized to duration of signal.

        Returns:
            float: Signal start as a percentage of total time (0-1).
        """
        return self.start_in_samples / self.num_iq_samples_dataset

    @start.setter
    def start(self, new_start: float) -> None:
        """Sets signal start.

        Args:
            new_start: Signal start as a percentage of total time (0-1).
        """
        self["start_in_samples"] = int(new_start * self.num_iq_samples_dataset)

    @property
    def stop(self) -> float:
        """Signal stop normalized to duration of signal.

        Returns:
            float: Signal stop as a percentage of total time (0-1).
        """
        return self.stop_in_samples / self.num_iq_samples_dataset

    @stop.setter
    def stop(self, new_stop: float) -> None:
        """Sets signal stop.

        Args:
            new_stop: Signal stop as a percentage of total time (0-1).
        """
        self["duration_in_samples"] = (new_stop * self.num_iq_samples_dataset) - self.start_in_samples

    @property
    def duration(self) -> float:
        """Signal duration normalized to 0-1.0.

        Returns:
            float: Signal duration as a percentage of total time (0-1).
        """
        return self.duration_in_samples / self.num_iq_samples_dataset

    @duration.setter
    def duration(self, new_duration: float) -> None:
        """Sets the duration of the signal.

        Args:
            new_duration: Duration as a percentage of total time (0-1).
        """
        self["duration_in_samples"] = new_duration * self.num_iq_samples_dataset

    @property
    def stop_in_samples(self) -> int:
        """Signal stop in samples.

        Returns:
            int: Signal stop time in samples.
        """
        return self.start_in_samples + self.duration_in_samples

    @stop_in_samples.setter
    def stop_in_samples(self, new_stop_in_samples: int) -> None:
        """Sets the stop time of the signal in samples.

        Args:
            new_stop_in_samples: Stop time in samples.
        """
        self["duration_in_samples"] = new_stop_in_samples - self.start_in_samples

    @property
    def upper_freq(self) -> float:
        """Calculate the upper edge of the full two-sided signal bandwidth.

        ``center_freq`` and ``bandwidth`` are canonical when both are present.
        A stored ``_upper_frequency`` is used only as a pending or legacy value
        when the canonical pair is unavailable. The returned value uses the
        same signed-Hz coordinate system and reference origin as
        ``center_freq``; it is not an absolute RF frequency.

        Returns:
            float: Upper frequency in Hz.

        Raises:
            ValueError: If center_freq or bandwidth are not available.
        """
        try:
            return upper_freq_from_center_freq_bandwidth(
                self.center_freq,
                self.bandwidth,
            )
        except (AttributeError, KeyError) as e:
            if "_upper_frequency" in self._metadata:
                return self._metadata["_upper_frequency"]
            raise ValueError("Cannot calculate upper frequency: missing center_freq or bandwidth") from e

    @upper_freq.setter
    def upper_freq(self, new_upper_freq: float) -> None:
        """Set the upper edge in the ``center_freq`` reference coordinate.

        When the lower edge is available, setting this edge recalculates the
        canonical full bandwidth and center frequency.

        Args:
            new_upper_freq: Upper frequency in Hz.
        """
        try:
            lower_freq = self.lower_freq
        except ValueError:
            self["_upper_frequency"] = new_upper_freq
            return

        self["bandwidth"] = bandwidth_from_lower_upper_freq(
            lower_freq,
            new_upper_freq,
        )
        self["center_freq"] = center_freq_from_lower_upper_freq(
            lower_freq,
            new_upper_freq,
        )

    @property
    def lower_freq(self) -> float:
        """Calculate the lower edge of the full two-sided signal bandwidth.

        ``center_freq`` and ``bandwidth`` are canonical when both are present.
        A stored ``_lower_frequency`` is used only as a pending or legacy value
        when the canonical pair is unavailable. The returned value uses the
        same signed-Hz coordinate system and reference origin as
        ``center_freq``; it is not an absolute RF frequency.

        Returns:
            float: Lower frequency in Hz.

        Raises:
            ValueError: If center_freq or bandwidth are not available.
        """
        try:
            return lower_freq_from_center_freq_bandwidth(
                self.center_freq,
                self.bandwidth,
            )
        except (AttributeError, KeyError) as e:
            if "_lower_frequency" in self._metadata:
                return self._metadata["_lower_frequency"]
            raise ValueError("Cannot calculate lower frequency: missing center_freq or bandwidth") from e

    @lower_freq.setter
    def lower_freq(self, new_lower_freq: float) -> None:
        """Set the lower edge in the ``center_freq`` reference coordinate.

        When the upper edge is available, setting this edge recalculates the
        canonical full bandwidth and center frequency.

        Args:
            new_lower_freq: Lower frequency in Hz.
        """
        try:
            upper_freq = self.upper_freq
        except ValueError:
            self["_lower_frequency"] = new_lower_freq
            return

        self["bandwidth"] = bandwidth_from_lower_upper_freq(
            new_lower_freq,
            upper_freq,
        )
        self["center_freq"] = center_freq_from_lower_upper_freq(
            new_lower_freq,
            upper_freq,
        )

    @property
    def oversampling_rate(self) -> float:
        """Calculates the oversampling rate for a signal.

        Returns:
            float: Oversampling rate (sample_rate / bandwidth).
        """
        return self.sample_rate / self.bandwidth

    def to_dict(self) -> dict[str, Any]:
        """Return signal metadata as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing metadata fields, excluding transient/internal
            fields that should not be serialized as signal metadata.
        """
        excluded_fields = {
            "applied_transforms",
            "dataset_metadata",
            "_dataset_metadata",
            "_center_freq_set",
        }

        metadata = dict(self.metadata)

        for field in excluded_fields:
            metadata.pop(field, None)

        return metadata


class Signal(SignalMetadataObject):
    """Represents a signal with data and metadata.

    This class extends SignalMetadataObject to include actual signal data
    and component signals. For a generated wideband container,
    ``center_freq=0`` identifies the dataset frequency origin; it does not
    assert that the composite signal's spectral energy is centered at DC.
    Component signals store their own signed offsets from that origin.

    Args:
        data: Signal IQ data. Defaults to empty numpy array.
        component_signals: List of component signals. Defaults to empty list.
        **kwargs: Additional metadata key-value pairs.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        component_signals: list[Signal] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Signal with data and metadata.

        Args:
            data: Signal IQ data. Defaults to np.array([]).
            component_signals: List of component signals. Defaults to [].
            **kwargs: Additional metadata key-value pairs.
        """
        super().__init__(**kwargs)
        self.data = np.array([]) if data is None else np.asarray(data)
        if "duration_in_samples" not in self.keys():
            self["duration_in_samples"] = len(self.data)
        self.component_signals = list(component_signals) if component_signals is not None else []

    def __repr__(self) -> str:
        """Returns a string representation of the Signal.

        Returns:
            str: String representation showing class name, metadata, and component signals.
        """
        return f"{self.__class__.__name__}(data={type(self.data)}. metadata={self.metadata}, component_signals={self.component_signals})"

    def copy(
        self,
        *,
        preserve_parent: bool = True,
    ) -> Signal:
        """Create a deep copy of the signal.

        Creates a new ``Signal`` with copied IQ data, component signals,
        and metadata. By default, the copy retains the same parent metadata
        relationship as the original, but this can be disabled to create a
        detached copy.

        Args:
            preserve_parent: If ``True`` (default), preserve the parent
                relationship in the copied signal. If ``False``, the copy
                is created without a parent.

        Returns:
            Signal: A new ``Signal`` instance with copied data, metadata,
            and component signals.
        """
        return Signal(
            data=self.data.copy(),
            component_signals=[sig.copy(preserve_parent=preserve_parent) for sig in self.component_signals],
            parent=self.parent if preserve_parent else None,
            **self.get_full_metadata(),
        )
