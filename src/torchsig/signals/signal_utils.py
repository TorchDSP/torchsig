"""Utility functions for dealing with the Signal type"""

from __future__ import annotations

import numpy as np

from torchsig.utils.dsp import low_pass_iterative_design


def check_signal_class(name: str, possible_names: list[str]) -> bool:
    """Check if the provided signal name matches any of the possible signal names.

    This function performs a substring match against each possible name.

    Args:
        name: The signal name to check.
        possible_names: A list of possible signal names to compare against.

    Returns:
        bool: True if the signal name matches any of the possible names, otherwise False.

    Examples:
        >>> check_signal_class("4fsk", ["fsk", "msk"])
        True
        >>> check_signal_class("am-dsb", ["am-"])
        True
        >>> check_signal_class("ofdm-64", ["ofdm"])
        True
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(possible_names, list) or not all(
        isinstance(n, str) for n in possible_names
    ):
        raise TypeError("possible_names must be a list of strings")

    return any(n in name for n in possible_names)


def random_limiting_filter_design(
    bandwidth: float, sample_rate: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Design a coarse bandwidth limiting filter with randomized parameters.

    Uses TorchSig's iterative designer function to create a low-pass filter
    with randomized cutoff and transition bandwidth parameters.

    Args:
        bandwidth: Occupied bandwidth for signal (Hz).
        sample_rate: Signal sampling rate (Hz).
        rng: Random number generator. If None, creates a new default generator.

    Returns:
        np.ndarray: Filter taps of designed low-pass filter.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth is greater than sample_rate/2.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> lpf = random_limiting_filter_design(1000, 10000, rng)
        >>> lpf.shape
        (101,)
    """
    # Input validation
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Randomize the cutoff (80-95% of Nyquist frequency)
    cutoff = rng.uniform(0.8 * bandwidth / 2, 0.95 * sample_rate / 2)

    # Calculate maximum transition bandwidth
    max_transition_bandwidth = sample_rate / 2 - cutoff

    # Transition bandwidth is randomized value (50-150% of max)
    transition_bandwidth = rng.uniform(0.5, 1.5) * max_transition_bandwidth

    # Design bandwidth-limiting filter
    return low_pass_iterative_design(
        cutoff=cutoff,
        transition_bandwidth=transition_bandwidth,
        sample_rate=sample_rate,
    )
