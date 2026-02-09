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


class SignalMetadataObject(HierarchicalMetadataObject):
    """Represents metadata associated with a signal.

    This class extends HierarchicalMetadataObject to provide signal-specific
    metadata properties and calculations.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the SignalMetadata object.

        Args:
            **kwargs: Metadata key-value pairs to initialize the object.
        """
        super().__init__(**kwargs)

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
        self["duration_in_samples"] = (
            new_stop * self.num_iq_samples_dataset
        ) - self.start_in_samples

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
        """Calculates the upper frequency of a signal.

        Returns:
            float: Upper frequency in Hz.

        Raises:
            ValueError: If center_freq or bandwidth are not available.
        """
        try:
            self["_upper_frequency"] = upper_freq_from_center_freq_bandwidth(
                self.center_freq, self.bandwidth
            )
        except (AttributeError, KeyError) as e:
            raise ValueError(
                "Cannot calculate upper frequency: missing center_freq or bandwidth"
            ) from e
        else:
            return self._upper_frequency

    @upper_freq.setter
    def upper_freq(self, new_upper_freq: float) -> None:
        """Sets the upper frequency of the signal.

        Args:
            new_upper_freq: Upper frequency in Hz.
        """
        self["_upper_frequency"] = new_upper_freq
        if hasattr(self, "_lower_frequency") and self._lower_frequency is not None:
            self["bandwidth"] = bandwidth_from_lower_upper_freq(
                new_upper_freq, self.lower_freq
            )
            self["center_freq"] = center_freq_from_lower_upper_freq(
                new_upper_freq, self.lower_freq
            )

    @property
    def lower_freq(self) -> float:
        """Calculates the lower frequency of a signal.

        Returns:
            float: Lower frequency in Hz.

        Raises:
            ValueError: If center_freq or bandwidth are not available.
        """
        try:
            self["_lower_frequency"] = lower_freq_from_center_freq_bandwidth(
                self.center_freq, self.bandwidth
            )
        except (AttributeError, KeyError) as e:
            raise ValueError(
                "Cannot calculate lower frequency: missing center_freq or bandwidth"
            ) from e
        else:
            return self._lower_frequency

    @lower_freq.setter
    def lower_freq(self, new_lower_freq: float) -> None:
        """Sets the lower frequency of the signal.

        Args:
            new_lower_freq: Lower frequency in Hz.
        """
        self["_lower_frequency"] = new_lower_freq
        if hasattr(self, "_upper_frequency") and self._upper_frequency is not None:
            self["bandwidth"] = bandwidth_from_lower_upper_freq(
                self.upper_freq, new_lower_freq
            )
            self["center_freq"] = center_freq_from_lower_upper_freq(
                self.upper_freq, new_lower_freq
            )

    @property
    def oversampling_rate(self) -> float:
        """Calculates the oversampling rate for a signal.

        Returns:
            float: Oversampling rate (sample_rate / bandwidth).
        """
        return self.sample_rate / self.bandwidth

    def to_dict(self) -> dict[str, Any]:
        """Returns SignalMetadataExternal as a full dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all metadata attributes.
        """
        attributes_original = self.__dict__.copy()  # Start with the instance variables
        attributes = attributes_original.copy()
        # exclude certain variables
        for var in attributes_original:
            if var in [
                "applied_transforms",
                "dataset_metadata",
                "_dataset_metadata",
                "_center_freq_set",
            ]:
                del attributes[var]
        return attributes


class Signal(SignalMetadataObject):
    """Represents a signal with data and metadata.

    This class extends SignalMetadataObject to include actual signal data
    and component signals.

    Args:
        data: Signal IQ data. Defaults to empty numpy array.
        component_signals: List of component signals. Defaults to empty list.
        **kwargs: Additional metadata key-value pairs.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        component_signals: list[Signal] = [],
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
        self["duration_in_samples"] = len(self.data)
        self.component_signals = component_signals

    def __repr__(self) -> str:
        """Returns a string representation of the Signal.

        Returns:
            str: String representation showing class name, metadata, and component signals.
        """
        return f"{self.__class__.__name__}(data={type(self.data)}. metadata={self.metadata}, component_signals={self.component_signals})"

    def copy(self) -> Signal:
        """Returns a copy of the Signal.

        Note:
            Parent relationships are not guaranteed to be preserved across copies.

        Returns:
            Signal: A new Signal instance with copied data and metadata.
        """
        return Signal(
            metadata=self.get_full_metadata(),
            data=self.data.copy(),
            component_signals=[sig.copy() for sig in self.component_signals],
        )
