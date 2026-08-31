"""RF propagation utilities for path loss calculations.

This module provides functions for calculating RF propagation effects such as
free-space path loss for geographic signal modeling.
"""

from __future__ import annotations

import numpy as np

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

__all__ = ["SPEED_OF_LIGHT_M_PER_S", "free_space_path_loss_db"]


def free_space_path_loss_db(
    distance: float,
    frequency: float,
    propagation_constant: float = 1.0,
) -> float:
    """Calculate free-space path loss in dB.

    The free space path loss formula is::

        L = 20 * log10(4 * pi * d / lambda)

    where d is distance, lambda is wavelength (c/f), c is speed of light.

    Args:
        distance: Propagation distance in meters
        frequency: Signal frequency in Hz
        propagation_constant: Scaling factor for speed of light. Defaults to 1.0
            (vacuum). Scaling factor applied to the propagation speed. Defaults to 1.0
            for vacuum. Values below 1.0 may approximate propagation through a
            homogeneous medium but do not model medium-specific absorption,
            scattering, or guided-wave losses.

    Returns:
        float: Free-space path loss in dB

    Raises:
        ValueError: If distance is not positive, frequency is not positive, or propagation_constant is not positive.

    Example:
        >>> # Path loss for 1 km at 2.4 GHz
        >>> free_space_path_loss_db(1000, 2.4e9)
        100.052...
        >>> free_space_path_loss_db(1000, 2.4e9, propagation_constant=2 / 3)
    """
    if not isinstance(distance, (int, float)) or not np.isfinite(distance):
        raise ValueError(f"free_space_path_loss_db.distance must be finite, got {distance}")
    if distance <= 0:
        raise ValueError(f"free_space_path_loss_db.distance must be positive, got {distance}")
    if not isinstance(frequency, (int, float)) or not np.isfinite(frequency):
        raise ValueError(f"free_space_path_loss_db.frequency must be finite, got {frequency}")
    if frequency <= 0:
        raise ValueError(f"free_space_path_loss_db.frequency must be positive, got {frequency}")
    if not isinstance(propagation_constant, (int, float)) or not np.isfinite(propagation_constant):
        raise ValueError(f"free_space_path_loss_db.propagation_constant must be finite, got {propagation_constant}")
    if propagation_constant <= 0:
        raise ValueError(f"free_space_path_loss_db.propagation_constant must be positive, got {propagation_constant}")
    effective_speed = SPEED_OF_LIGHT_M_PER_S * propagation_constant
    wavelength = effective_speed / frequency
    return float(20 * np.log10(4 * np.pi * distance / wavelength))
