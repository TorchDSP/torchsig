"""Constellation Maps"""

import numpy as np

__all__ = ["all_symbol_maps", "remove_corners"]


def remove_corners(const):
    """Removes corners from 'cross' modulations.

    The function is applied during the formation of the 32-QAM, 128-QAM
    and 512-QAM constellations in order to remove the corners to produce
    a cross or plus shape constellation.

    Returns:
        list[float]: A symbol map (list of symbols) without corners
    """
    spacing = 2.0 / (np.sqrt(len(const)) - 1)
    cutoff = spacing * (np.sqrt(len(const)) / 6 - 0.5)
    return [p for p in const if np.abs(np.real(p)) < 1.0 - cutoff or np.abs(np.imag(p)) < 1.0 - cutoff]


def apsk_rings(
    ring_counts: list[int],
    ring_radii: list[float],
    ring_offsets: list[float],
) -> np.ndarray:
    """Builds an Amplitude-Phase-Shift-Keying (APSK) constellation from rings.

    Used to form the DVB-S2 16-APSK (4+12) and 32-APSK (4+12+16) constellations.
    Each ring contributes equally spaced points at a fixed radius and angular
    offset. The result is normalized to unit average power.

    Args:
        ring_counts: Number of points on each concentric ring.
        ring_radii: Radius of each ring.
        ring_offsets: Angular offset (radians) of the first point on each ring.

    Returns:
        np.ndarray: Complex symbol map normalized to unit average power.
    """
    points = []
    for count, radius, offset in zip(ring_counts, ring_radii, ring_offsets):
        angles = offset + 2.0 * np.pi * np.arange(count) / count
        points.append(radius * np.exp(1j * angles))
    const = np.concatenate(points)
    return const / np.sqrt(np.mean(np.abs(const) ** 2))


# retains all of the symbol maps for QAM/PSK/ASK/OOK modulations
all_symbol_maps = {
    "ook": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 2), 0j))),
    "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
    "qpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2)))),
    "8psk": np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0),
    "16psk": np.exp(2j * np.pi * np.linspace(0, 15, 16) / 16.0),
    "32psk": np.exp(2j * np.pi * np.linspace(0, 31, 32) / 32.0),
    "64psk": np.exp(2j * np.pi * np.linspace(0, 63, 64) / 64.0),
    "4ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 0j))),
    "8ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 0j))),
    "16ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 0j))),
    "32ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 0j))),
    "64ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 64), 0j))),
    "16qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4)))),
    "32qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 8)))),
    "64qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 1j * np.linspace(-1, 1, 8)))),
    "256qam": np.add(
        *map(
            np.ravel,
            np.meshgrid(np.linspace(-1, 1, 16), 1j * np.linspace(-1, 1, 16)),
        )
    ),
    "1024qam": np.add(
        *map(
            np.ravel,
            np.meshgrid(np.linspace(-1, 1, 32), 1j * np.linspace(-1, 1, 32)),
        )
    ),
    "32qam_cross": remove_corners(
        np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 6), 1j * np.linspace(-1, 1, 6)),
            )
        )
    ),
    "128qam_cross": remove_corners(
        np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 12), 1j * np.linspace(-1, 1, 12)),
            )
        )
    ),
    "512qam_cross": remove_corners(
        np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 24), 1j * np.linspace(-1, 1, 24)),
            )
        )
    ),
    # DVB-S2 APSK constellations (representative ~rate-3/4 ring radius ratios).
    "16apsk": apsk_rings([4, 12], [1.0, 2.85], [np.pi / 4, np.pi / 12]),
    "32apsk": apsk_rings([4, 12, 16], [1.0, 2.84, 5.27], [np.pi / 4, np.pi / 12, np.pi / 16]),
}
