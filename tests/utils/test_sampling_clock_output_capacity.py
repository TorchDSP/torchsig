"""Tests for sampling-clock output-capacity calculation."""

import numpy as np
import pytest

from torchsig.utils import dsp, dsp_numba
from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_small_drate_with_initial_position_past_bound_returns_empty(implementation) -> None:
    output = implementation(
        h=np.array([1.0], dtype=np.float32),
        x=np.ones(4, dtype=np.complex64),
        uprate=1,
        drate=1e-12,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(7),
    )

    assert output.shape == (0,)
    assert output.dtype == np.complex64


@pytest.mark.parametrize(
    ("implementation", "module", "constant_name"),
    [
        (sampling_clock_impairments, dsp, "_MAX_SAMPLING_CLOCK_OUTPUT_SAMPLES"),
        (sampling_clock_impairments_numba_wrapper, dsp_numba, "_MAX_OUTPUT_SAMPLES"),
    ],
)
def test_capacity_limit_is_enforced_before_allocation(
    implementation,
    module,
    constant_name: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, constant_name, 100)

    with pytest.raises(RuntimeError, match="sampling clock output capacity exhausted"):
        implementation(
            h=np.array([1.0], dtype=np.float32),
            x=np.ones(100, dtype=np.complex64),
            uprate=1,
            drate=0.1,
            jitter_ppm=0.0,
            drift_ppm=0.0,
            rng=np.random.default_rng(7),
        )
