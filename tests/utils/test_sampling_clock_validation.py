"""Input-validation tests for sampling-clock impairment implementations."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


def _kwargs() -> dict:
    return {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(8, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 0.0,
        "drift_ppm": 0.0,
        "rng": np.random.default_rng(7),
    }


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"uprate": 0}, "uprate must be a positive integer"),
        ({"uprate": 1.5}, "uprate must be a positive integer"),
        ({"drate": 0.0}, "drate must be finite and positive"),
        ({"drate": np.inf}, "drate must be finite and positive"),
        ({"jitter_ppm": -1.0}, "jitter_ppm must be finite and nonnegative"),
        ({"jitter_ppm": np.nan}, "jitter_ppm must be finite and nonnegative"),
        ({"drift_ppm": np.nan}, "drift_ppm must be finite"),
        ({"initial_phase": -0.1}, "initial_phase must be finite and in the interval"),
        ({"initial_phase": 1.0}, "initial_phase must be finite and in the interval"),
        ({"initial_phase": np.nan}, "initial_phase must be finite and in the interval"),
        (
            {"drift_ppm": -1_000_000.0},
            "drift_ppm produces a nonfinite or nonpositive sampling-position increment",
        ),
    ],
)
def test_sampling_clock_rejects_invalid_parameters(
    implementation,
    overrides: dict,
    message: str,
) -> None:
    kwargs = _kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        implementation(**kwargs)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_sampling_clock_creates_default_rng(implementation) -> None:
    kwargs = _kwargs()
    kwargs["rng"] = None

    output = implementation(**kwargs)

    assert output.dtype == np.complex64
