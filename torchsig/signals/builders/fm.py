"""FM Signal Builder and Modulator Module"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    convolve,
    low_pass_iterative_design,
)


def fm_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Frequency Modulator (FM) signal generator.

    Generates FM signals using Carson's Rule for bandwidth calculation.

    Args:
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: FM modulated signal at the appropriate bandwidth.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth exceeds sample_rate/2.
        ValueError: If num_samples is not positive.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> fm_signal = fm_modulator(1000, 10000, 1000, rng)
        >>> fm_signal.shape
        (1000,)
    """
    # Input validation
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Randomly determine modulation index
    mod_index = rng.uniform(1, 10)

    # Calculate frequency deviation using Carson's Rule
    fdev = (bandwidth / 2) / (1 + (1 / mod_index))

    # Calculate maximum deviation
    fmax = fdev / mod_index

    # Generate and scale message signal
    message = rng.normal(0, 1, num_samples)
    message = message / np.sqrt(np.mean(np.abs(message) ** 2))  # Scale to unit power

    # Design LPF to limit frequencies
    lpf = low_pass_iterative_design(
        cutoff=fmax, transition_bandwidth=fmax, sample_rate=sample_rate
    )

    # Apply LPF to limit bandwidth
    source = convolve(message, lpf)

    # Apply FM modulation
    modulated = np.exp(2j * np.pi * np.cumsum(source) * fdev / sample_rate)

    return modulated.astype(TorchSigComplexDataType)


class FMSignalGenerator(BaseSignalGenerator):
    """FM Signal Generator.

    Implements frequency modulation (FM) waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes FM Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("fm")

    def generate(self) -> Signal:
        """Generates an FM signal based on the configured parameters.

        Returns:
            Signal: Generated FM signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        # Get parameters from metadata
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(
            low=self["bandwidth_min"], high=self["bandwidth_max"] + 1
        )

        # Generate signal
        signal_data = fm_modulator(
            bandwidth, sample_rate, num_iq_samples_signal, self.random_generator
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
