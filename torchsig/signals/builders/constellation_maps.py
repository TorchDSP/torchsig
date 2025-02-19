"""Constellation Maps
"""
from collections import OrderedDict
import numpy as np

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
    return [
        p
        for p in const
        if np.abs(np.real(p)) < 1.0 - cutoff or np.abs(np.imag(p)) < 1.0 - cutoff
    ]

# retains all of the symbol maps for QAM/PSK/ASK/OOK modulations
all_symbol_maps = OrderedDict(
    {
        "ook": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 2), 0j))),
        "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
        "4ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 0j))),
        "qpsk": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2))
            )
        ),
        "8ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 0j))),
        "8psk": np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0),
        "16qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4))
            )
        ),
        "16ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 0j))),
        "16psk": np.exp(2j * np.pi * np.linspace(0, 15, 16) / 16.0),
        "32qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 8))
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
        "32ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 0j))),
        "32psk": np.exp(2j * np.pi * np.linspace(0, 31, 32) / 32.0),
        "64qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 1j * np.linspace(-1, 1, 8))
            )
        ),
        "64ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 64), 0j))),
        "64psk": np.exp(2j * np.pi * np.linspace(0, 63, 64) / 64.0),
        "128qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 12), 1j * np.linspace(-1, 1, 12)),
                )
            )
        ),
        "256qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 16), 1j * np.linspace(-1, 1, 16)),
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
        "1024qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 32), 1j * np.linspace(-1, 1, 32)),
            )
        ),
    }
)
