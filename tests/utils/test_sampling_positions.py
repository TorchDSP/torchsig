"""Tests for sampling-clock position generation."""

import numpy as np
import pytest

from torchsig.utils.dsp import generate_sampling_positions, sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper


def _positions(model: str, *, seed: int = 17, alpha: float = 0.75):
    return generate_sampling_positions(
        num_input_samples=200_000,
        taps_per_phase=1,
        uprate=8,
        drate=8.0,
        jitter_ppm=20_000.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(seed),
        jitter_model=model,
        jitter_alpha=alpha,
    )


def test_independent_jitter_statistics() -> None:
    error = _positions("independent").timing_error
    assert abs(np.mean(error)) < 3e-4
    assert np.var(error) == pytest.approx(0.02**2, rel=0.03)
    assert np.corrcoef(error[:-1], error[1:])[0, 1] == pytest.approx(0.0, abs=0.02)


def test_period_jitter_increment_statistics() -> None:
    increments = np.diff(_positions("period").timing_error)
    assert np.var(increments) == pytest.approx(0.02**2, rel=0.03)
    assert np.corrcoef(increments[:-1], increments[1:])[0, 1] == pytest.approx(0.0, abs=0.02)


def test_ar1_jitter_statistics() -> None:
    error = _positions("ar1").timing_error
    assert np.var(error) == pytest.approx(0.02**2, rel=0.04)
    assert np.corrcoef(error[:-1], error[1:])[0, 1] == pytest.approx(0.75, abs=0.02)


def test_zero_impairments_and_input_sample_accessors() -> None:
    positions = generate_sampling_positions(8, 2, 4, 4.0, 0.0, 0.0, initial_phase=0.25)
    np.testing.assert_array_equal(positions.actual, positions.nominal)
    np.testing.assert_array_equal(positions.nominal_indices, np.arange(len(positions.nominal)))
    np.testing.assert_array_equal(positions.nominal_input_samples, positions.nominal / 4)
    np.testing.assert_array_equal(positions.actual_input_samples, positions.actual / 4)


@pytest.mark.parametrize("drift_ppm", [-250_000.0, 250_000.0])
def test_constant_drift_has_linear_timing_error(drift_ppm: float) -> None:
    positions = generate_sampling_positions(100, 1, 4, 2.0, 0.0, drift_ppm)
    expected = positions.nominal_indices * 2.0 / 4.0 * drift_ppm * 1e-6
    np.testing.assert_allclose(positions.timing_error, expected, atol=1e-14)


def test_linear_drift_uses_elapsed_physical_time() -> None:
    positions = generate_sampling_positions(
        100,
        1,
        4,
        2.0,
        0.0,
        100.0,
        drift_model="linear",
        drift_rate_ppm_per_sec=20.0,
        sample_rate_hz=10.0,
    )
    n = positions.nominal_indices.astype(np.float64)
    expected = 2.0 / 4.0 * (100e-6 * n + 20e-6 / 10.0 * n * (n - 1.0) / 2.0)
    np.testing.assert_allclose(positions.timing_error, expected, atol=1e-14)


def test_random_walk_frequency_innovation_variance() -> None:
    positions = generate_sampling_positions(
        50_000,
        1,
        8,
        8.0,
        0.0,
        0.0,
        rng=np.random.default_rng(5),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=100.0,
        sample_rate_hz=100.0,
    )
    period_error = np.diff(positions.timing_error)
    innovations = np.diff(period_error)
    assert np.var(innovations) == pytest.approx((100e-6 / np.sqrt(100.0)) ** 2, rel=0.03)


def test_random_walk_timing_variance_has_cubic_growth() -> None:
    sample_indices = (80, 160)
    errors = np.empty((300, len(sample_indices)))
    for seed in range(len(errors)):
        positions = generate_sampling_positions(
            400,
            1,
            1,
            1.0,
            0.0,
            0.0,
            rng=np.random.default_rng(seed),
            drift_model="random_walk",
            drift_noise_ppm_per_sqrt_sec=1_000.0,
            sample_rate_hz=1.0,
        )
        errors[seed] = positions.timing_error[list(sample_indices)]
    variance_ratio = np.var(errors[:, 1]) / np.var(errors[:, 0])
    assert variance_ratio == pytest.approx((sample_indices[1] / sample_indices[0]) ** 3, rel=0.2)


def test_random_walk_time_scaling_is_sample_rate_independent() -> None:
    low_rate = generate_sampling_positions(
        2_000,
        1,
        1,
        1.0,
        0.0,
        0.0,
        rng=np.random.default_rng(9),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=5_000.0,
        sample_rate_hz=100.0,
    )
    high_rate = generate_sampling_positions(
        4_000,
        1,
        1,
        1.0,
        0.0,
        0.0,
        rng=np.random.default_rng(9),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=5_000.0,
        sample_rate_hz=400.0,
    )
    low_innovations = np.diff(np.diff(low_rate.timing_error))
    high_innovations = np.diff(np.diff(high_rate.timing_error))
    assert np.var(low_innovations) / np.var(high_innovations) == pytest.approx(4.0, rel=0.12)


def test_positions_are_reproducible() -> None:
    first = _positions("ar1", seed=123)
    second = _positions("ar1", seed=123)
    np.testing.assert_array_equal(first.actual, second.actual)


def test_invalid_positions_are_skipped_without_losing_nominal_alignment() -> None:
    class OffsetRng:
        def __init__(self):
            self.values = iter((0.0, -3.0, 0.0, 0.0))

        def normal(self, _mean, scale):
            return next(self.values) * scale

    positions = generate_sampling_positions(4, 1, 1, 1.0, 1_000_000.0, 0.0, rng=OffsetRng())
    np.testing.assert_array_equal(positions.nominal_indices, [0, 2, 3])
    np.testing.assert_array_equal(positions.nominal, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(positions.actual, [1.0, 3.0, 4.0])


@pytest.mark.parametrize("model", ["independent", "period", "ar1"])
def test_python_numba_parity_for_all_jitter_models(model: str) -> None:
    kwargs = {
        "h": np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32),
        "x": np.arange(32, dtype=np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 50_000.0,
        "drift_ppm": 10.0,
        "jitter_model": model,
        "jitter_alpha": -0.4,
    }
    expected = sampling_clock_impairments(rng=np.random.default_rng(91), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(91), **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"jitter_model": "unknown"}, "jitter_model"),
        ({"jitter_alpha": 1.0}, "jitter_alpha"),
        ({"jitter_alpha": -1.0}, "jitter_alpha"),
        ({"drift_model": "unknown"}, "drift_model"),
        ({"drift_noise_ppm_per_sqrt_sec": -1.0}, "drift_noise"),
        ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
    ],
)
def test_jitter_model_validation(overrides, message: str) -> None:
    kwargs = {
        "num_input_samples": 8,
        "taps_per_phase": 1,
        "uprate": 2,
        "drate": 2.0,
        "jitter_ppm": 1.0,
        "drift_ppm": 0.0,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        generate_sampling_positions(**kwargs)
