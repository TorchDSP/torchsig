"""Tone Signal Builder and Modulator Module"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import TorchSigComplexDataType


def tone_modulator(num_samples: int) -> np.ndarray:
    """Implements a tone modulator.

    Generates a constant tone signal at baseband (all ones).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        np.ndarray: Tone signal (array of ones) with shape (num_samples,).

    Raises:
        ValueError: If num_samples is not positive.
    """
    # Input validation
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # Generate tone signal (all ones at baseband)
    return np.ones(num_samples, dtype=TorchSigComplexDataType)


class ToneSignalGenerator(BaseSignalGenerator):
    """Tone Signal Generator.

    Implements tone waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes Tone Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("tone")

    def generate(self) -> Signal:
        """Generates a tone signal based on the configured parameters.

        Returns:
            Signal: Generated tone signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        # Get parameters from metadata
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )

        # Generate signal
        signal_data = tone_modulator(num_iq_samples_signal)

        return Signal(
            data=signal_data, center_freq=0, bandwidth=1  # Tone has 1Hz bandwidth
        )
