"""Tests for out-of-range impaired sampling positions."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


class OffsetRng:
    """Provide absolute offsets that push two positions out of range."""

    def __init__(self) -> None:
        self.values = iter((100.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def normal(self, _loc, scale, size=None):
        """Return deterministic values in scalar or paired-array form."""
        if size is None:
            return next(self.values) / scale if scale != 0.0 else next(self.values)
        return np.array(
            [[100.0, 0.0], [-100.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_out_of_range_position_is_skipped_instead_of_clipped(implementation) -> None:
    output = implementation(
        h=np.array([1.0], dtype=np.float32),
        x=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex64),
        uprate=1,
        drate=1.0,
        jitter_ppm=1_000_000.0,
        drift_ppm=0.0,
        rng=OffsetRng(),
    )

    np.testing.assert_array_equal(output, np.array([3.0, 0.0], dtype=np.complex64))
