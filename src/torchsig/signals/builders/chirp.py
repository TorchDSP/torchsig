"""Chirp Signal Generator Module"""

from __future__ import annotations

import numpy as np


def chirp(
    f0: float, f1: float, samples_per_symbol: int, phi: float = 0.0
) -> np.ndarray:
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
