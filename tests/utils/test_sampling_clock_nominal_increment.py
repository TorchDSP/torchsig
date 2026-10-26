"""Tests for the drift-adjusted nominal sampling-position increment."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


class ZeroRng:
    """Return zero-valued draws while preserving each implementation's RNG calls."""

    def normal(self, _loc, _scale, size=None):
        """Return zeros matching the requested draw shape."""
        if size is None:
            return 0.0
        return np.zeros(size)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_signed_drift_changes_nominal_output_length(implementation) -> None:
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 0.0,
        "rng": ZeroRng(),
    }

    nominal = implementation(drift_ppm=0.0, **kwargs)
    faster = implementation(drift_ppm=500_000.0, **kwargs)
    slower = implementation(drift_ppm=-500_000.0, **kwargs)

    assert len(faster) < len(nominal) < len(slower)
