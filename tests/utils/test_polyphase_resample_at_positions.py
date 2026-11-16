"""Tests for explicit-position polyphase interpolation."""

import numpy as np
import pytest

from torchsig.utils.dsp import (
    polyphase_resample_at_positions,
    sampling_clock_impairments,
)


def test_polyphase_resample_at_positions_known_values():
    """A single-tap filter selects the expected input samples."""
    output = polyphase_resample_at_positions(
        h=np.array([1.0], dtype=np.float32),
        x=np.array([1, 2, 3, 4], dtype=np.complex64),
        uprate=1,
        positions=np.array([0.0, 1.0, 3.0, 4.0]),
    )

    np.testing.assert_array_equal(
        output,
        np.array([1, 2, 4, 0], dtype=np.complex64),
    )


def test_polyphase_resample_at_positions_empty():
    output = polyphase_resample_at_positions(
        h=np.array([1.0], dtype=np.float32),
        x=np.array([1, 2, 3], dtype=np.complex64),
        uprate=1,
        positions=np.array([], dtype=np.float64),
    )

    assert output.shape == (0,)
    assert output.dtype == np.dtype(np.complex64)


def test_polyphase_resample_at_positions_interpolates_subphase_remainder():
    output = polyphase_resample_at_positions(
        h=np.array([1.0, 2.0], dtype=np.float32),
        x=np.array([1.0, 2.0], dtype=np.complex64),
        uprate=2,
        positions=np.array([0.5, 1.5]),
    )

    np.testing.assert_array_equal(output, np.array([3.0, 4.0], dtype=np.complex64))


def test_polyphase_resample_at_positions_interpolates_across_last_phase():
    output = polyphase_resample_at_positions(
        h=np.array([1.0, 2.0], dtype=np.float32),
        x=np.array([1.0, 2.0], dtype=np.complex64),
        uprate=2,
        positions=np.array([1.75]),
    )

    np.testing.assert_array_equal(output, np.array([4.0], dtype=np.complex64))


@pytest.mark.parametrize(
    "positions",
    [
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_polyphase_resample_at_positions_invalid(positions):
    with pytest.raises(ValueError):
        polyphase_resample_at_positions(
            h=np.array([1.0], dtype=np.float32),
            x=np.ones(4, dtype=np.complex64),
            uprate=1,
            positions=positions,
        )


@pytest.mark.parametrize(
    ("jitter_ppm", "drift_ppm"),
    [
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 100.0),
        (100.0, 100.0),
    ],
)
def test_sampling_clock_impairments_deterministic(
    jitter_ppm,
    drift_ppm,
):
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.arange(1, 17, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": jitter_ppm,
        "drift_ppm": drift_ppm,
    }

    first = sampling_clock_impairments(
        **kwargs,
        rng=np.random.default_rng(42),
    )
    second = sampling_clock_impairments(
        **kwargs,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first, second)
