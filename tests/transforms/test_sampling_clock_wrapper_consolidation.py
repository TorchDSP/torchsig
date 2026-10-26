"""Regression tests for the shared sampling-clock transform wrapper."""

# ruff: noqa: INP001

import numpy as np
import pytest

from torchsig.transforms import functional as F  # noqa: N812
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    TorchSigRealDataType,
    prototype_polyphase_filter,
    sampling_clock_impairments,
)
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (sampling_clock_impairments, sampling_clock_impairments_numba_wrapper)


def _legacy_wrapper_output(data, implementation, *, seed: int, **impairment_kwargs):
    """Reproduce the wrapper setup that preceded helper consolidation."""
    complex_data = data.astype(TorchSigComplexDataType)
    uprate = 5000
    prototype_filter = prototype_polyphase_filter(num_branches=uprate).astype(TorchSigRealDataType)
    impaired = implementation(
        h=prototype_filter,
        x=complex_data,
        uprate=uprate,
        drate=uprate,
        rng=np.random.default_rng(seed),
        **impairment_kwargs,
    )
    filter_delay = (len(prototype_filter) - 1) / (2 * uprate)
    return F._preserve_sampling_clock_output(  # noqa: SLF001
        impaired,
        len(complex_data),
        start_position=filter_delay,
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("drift_ppm", [0.0, 2_500.0])
def test_clock_drift_matches_preconsolidation_setup(monkeypatch, implementation, drift_ppm: float) -> None:
    time = np.arange(192)
    data = np.exp(2j * np.pi * 0.08 * time).astype(np.complex64)
    seed = 31
    impairment_kwargs = {
        "jitter_ppm": 0.0,
        "drift_ppm": drift_ppm,
        "initial_phase": 0.375,
        "drift_model": "constant",
        "drift_rate_ppm_per_sec": 0.0,
        "drift_noise_ppm_per_sqrt_sec": 0.0,
        "sample_rate_hz": 1.0,
    }
    expected = _legacy_wrapper_output(data, implementation, seed=seed, **impairment_kwargs)
    monkeypatch.setattr(F, "_sampling_clock_impairments", implementation)

    actual = F.clock_drift(
        data,
        drift_ppm=drift_ppm,
        initial_phase=0.375,
        rng=np.random.default_rng(seed),
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == data.shape
    assert actual.dtype == np.complex64


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("jitter_ppm", [0.0, 20_000.0])
def test_clock_jitter_matches_preconsolidation_setup(monkeypatch, implementation, jitter_ppm: float) -> None:
    time = np.arange(192)
    data = np.exp(2j * np.pi * 0.08 * time).astype(np.complex64)
    seed = 47
    impairment_kwargs = {
        "jitter_ppm": jitter_ppm,
        "drift_ppm": 0.0,
        "initial_phase": 0.625,
        "jitter_model": "ar1",
        "jitter_alpha": -0.3,
    }
    expected = _legacy_wrapper_output(data, implementation, seed=seed, **impairment_kwargs)
    monkeypatch.setattr(F, "_sampling_clock_impairments", implementation)

    actual = F.clock_jitter(
        data,
        jitter_ppm=jitter_ppm,
        initial_phase=0.625,
        jitter_model="ar1",
        jitter_alpha=-0.3,
        rng=np.random.default_rng(seed),
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == data.shape
    assert actual.dtype == np.complex64
