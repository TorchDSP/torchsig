"""Chirp Signal Generator Module"""

from __future__ import annotations

import numpy as np

__all__ = ["chirp"]


def draw_samples_per_symbol(
    max_num_samples: int,
    rng: np.random.Generator,
    samples_per_symbol_min: int | None,
    samples_per_symbol_max: int | None,
) -> int:
    """Choose an inclusive sweep length, preserving the legacy draw when unbounded.

    Args:
        max_num_samples: Maximum permitted baseband sweep length when bounds are set.
        rng: Random number generator used to draw the sweep length.
        samples_per_symbol_min: Inclusive lower bound, or None for the legacy draw.
        samples_per_symbol_max: Inclusive upper bound, or None for the legacy draw.

    Returns:
        The sweep length in baseband samples. With both bounds omitted, draws
        from [128, 4095] as before. Equal bounds consume no random draw.

    Raises:
        ValueError: If only one bound is set, either bound is not an integer,
            or the bounds do not satisfy 1 <= min <= max <= max_num_samples.
    """
    if samples_per_symbol_min is None and samples_per_symbol_max is None:
        return rng.integers(low=128, high=4096)
    if (
        not isinstance(samples_per_symbol_min, (int, np.integer))
        or isinstance(samples_per_symbol_min, (bool, np.bool_))
        or not isinstance(samples_per_symbol_max, (int, np.integer))
        or isinstance(samples_per_symbol_max, (bool, np.bool_))
        or not 1 <= samples_per_symbol_min <= samples_per_symbol_max <= max_num_samples
    ):
        raise ValueError("samples_per_symbol_min and samples_per_symbol_max must be integers in [1, max_num_samples], with min <= max")
    if samples_per_symbol_min == samples_per_symbol_max:
        return int(samples_per_symbol_min)
    return rng.integers(low=samples_per_symbol_min, high=samples_per_symbol_max + 1)


def chirp(f0: float, f1: float, samples_per_symbol: int, phi: float = 0.0) -> np.ndarray:
    """Generates a linear frequency modulated (LFM) chirp signal.

    Creates a chirp waveform that linearly sweeps from frequency f0 to f1.

    Args:
        f0: Starting frequency (Hz).
        f1: Ending frequency (Hz).
        samples_per_symbol: Number of samples for the chirp symbol.
        phi: Starting phase in degrees (default: 0).

    Returns:
        np.ndarray: Complex-valued chirp waveform of shape (samples_per_symbol,).

    Raises:
        ValueError: If samples_per_symbol is not positive.
        ValueError: If f0 or f1 are not finite numbers.

    Examples:
        >>> chirp(100, 200, 100)
        array([1.+0.j, 1.+0.j, ..., 1.+0.j], dtype=complex64)
        >>> chirp(100, 200, 100, phi=45)
        array([1.+0.j, 1.+0.j, ..., 1.+0.j], dtype=complex64)
    """
    # Input validation
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if not np.isfinite(f0) or not np.isfinite(f1):
        raise ValueError("f0 and f1 must be finite numbers")

    # Create time array
    t = np.arange(samples_per_symbol, dtype=np.float64)

    # Calculate chirp rate
    b = (f1 - f0) / (t[-1] - t[0]) if t[-1] != t[0] else 0.0

    # Calculate phase
    phase = 2 * np.pi * (f0 * t + 0.5 * b * t * t)  # Linear FM

    # Convert phase from degrees to radians
    phi_rad = np.deg2rad(phi)

    # Generate chirp waveform
    return np.exp(1j * (phase + phi_rad), dtype=np.complex64)
