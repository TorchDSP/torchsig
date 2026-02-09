"""AM Signal Generator Module"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    convolve,
    frequency_shift,
    low_pass_iterative_design,
    polyphase_decimator,
)


def am_modulator(
    am_mode: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Amplitude Modulator (AM) signal generator.

    Generates AM signals in various modes (DSB, DSB-SC, LSB, USB).

    Args:
        am_mode: Mode of the AM signal ('dsb', 'dsb-sc', 'lsb', 'usb').
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: AM modulated signal at the appropriate bandwidth.

    Raises:
        ValueError: If bandwidth is not positive or exceeds sample_rate/2.
        ValueError: If am_mode is not one of the supported modes.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> dsb_signal = am_modulator('dsb', 1000, 10000, 1000, rng)
        >>> dsb_sc_signal = am_modulator('dsb-sc', 1000, 10000, 1000, rng)
    """
    # Input validation
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if am_mode not in ["dsb", "dsb-sc", "lsb", "usb"]:
        raise ValueError("am_mode must be one of: 'dsb', 'dsb-sc', 'lsb', 'usb'")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Determine number of samples for modulation
    num_samples_mod = (
        2 * num_samples if "lsb" in am_mode or "usb" in am_mode else num_samples
    )

    # Generate random message signal
    message = rng.normal(0, 1, num_samples_mod).astype(TorchSigComplexDataType)
    message = message / np.sqrt(np.mean(np.abs(message) ** 2))  # Scale to unit power

    # Design bandwidth-limiting filter
    cutoff = bandwidth / 2
    max_transition_bandwidth = (sample_rate / 2) - cutoff
    transition_bandwidth = rng.uniform(0.05, 0.25) * max_transition_bandwidth / 2
    lpf = low_pass_iterative_design(
        cutoff=cutoff,
        transition_bandwidth=transition_bandwidth,
        sample_rate=sample_rate,
    )

    # Apply bandwidth-limiting filter
    shaped_message = convolve(message, lpf)

    # Generate baseband signal based on modulation mode
    if am_mode == "dsb-sc":
        baseband_signal = shaped_message
    elif am_mode == "dsb":
        modulation_index = rng.uniform(0.8, 4)
        shaped_message_max = np.max(np.abs(shaped_message))
        carrier = (shaped_message_max / modulation_index) * np.ones(len(shaped_message))
        baseband_signal = (modulation_index * shaped_message) + carrier
    elif am_mode == "lsb":
        dsb_upconverted = frequency_shift(shaped_message, bandwidth / 2, sample_rate)
        lsb_signal_at_if = convolve(dsb_upconverted, lpf)
        baseband_signal_oversampled = frequency_shift(
            lsb_signal_at_if, -bandwidth / 4, sample_rate
        )
        baseband_signal = polyphase_decimator(baseband_signal_oversampled, 2) * 2
    elif am_mode == "usb":
        dsb_downconverted = frequency_shift(shaped_message, -bandwidth / 2, sample_rate)
        usb_signal_atif = convolve(dsb_downconverted, lpf)
        baseband_signal_oversampled = frequency_shift(
            usb_signal_atif, bandwidth / 4, sample_rate
        )
        baseband_signal = polyphase_decimator(baseband_signal_oversampled, 2) * 2

    return baseband_signal.astype(TorchSigComplexDataType)


class AMSignalGenerator(BaseSignalGenerator):
    """AM Signal Generator.

    Implements various AM modulation schemes (DSB, DSB-SC, LSB, USB).
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes AM Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - am_mode: AM modulation mode ('dsb', 'dsb-sc', 'lsb', 'usb')
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
            "am_mode",
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("am-" + str(self["am_mode"]))

    def generate(self) -> Signal:
        """Generates an AM signal based on the configured parameters.

        Returns:
            Signal: Generated AM signal with metadata.

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
        am_mode = self["am_mode"]

        # Generate signal
        signal_data = am_modulator(
            am_mode,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
