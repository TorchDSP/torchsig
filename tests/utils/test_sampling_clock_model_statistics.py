"""Statistical validation for configurable sampling-clock models."""

import numpy as np
import pytest

from torchsig.utils.dsp import generate_sampling_positions

ENSEMBLE_SIZE = 800
SIGMA_MULTIPLIER = 5.0


def _jitter_positions(model: str, *, seed: int = 17, alpha: float = 0.75):
    """Generate a long jitter-only trajectory for stable moment estimates."""
    return generate_sampling_positions(
        num_input_samples=150_000,
        taps_per_phase=1,
        uprate=8,
        drate=8.0,
        jitter_ppm=20_000.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(seed),
        jitter_model=model,
        jitter_alpha=alpha,
    )


def _random_walk_positions(seed: int, *, samples: int = 256):
    """Generate a unit-increment random-walk drift trajectory."""
    return generate_sampling_positions(
        num_input_samples=samples,
        taps_per_phase=1,
        uprate=1,
        drate=1.0,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(seed),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=1_000.0,
        sample_rate_hz=100.0,
    )


def test_independent_jitter_matches_white_gaussian_statistics() -> None:
    """Independent jitter has configured variance and negligible lag-one correlation."""
    timing_error = _jitter_positions("independent").timing_error
    expected_variance = (20_000e-6) ** 2
    standard_error_of_mean = np.sqrt(expected_variance / len(timing_error))

    assert np.mean(timing_error) == pytest.approx(0.0, abs=SIGMA_MULTIPLIER * standard_error_of_mean)
    assert np.var(timing_error, ddof=1) == pytest.approx(expected_variance, rel=0.025)
    assert np.corrcoef(timing_error[:-1], timing_error[1:])[0, 1] == pytest.approx(0.0, abs=0.015)


def test_period_jitter_increments_match_white_gaussian_statistics() -> None:
    """Period jitter has independent increments with the configured variance."""
    timing_error = _jitter_positions("period").timing_error
    increments = np.diff(timing_error)
    expected_variance = (20_000e-6) ** 2

    assert np.mean(increments) == pytest.approx(
        0.0,
        abs=SIGMA_MULTIPLIER * np.sqrt(expected_variance / len(increments)),
    )
    assert np.var(increments, ddof=1) == pytest.approx(expected_variance, rel=0.025)
    assert np.corrcoef(increments[:-1], increments[1:])[0, 1] == pytest.approx(0.0, abs=0.015)


@pytest.mark.parametrize("alpha", [-0.65, 0.65])
def test_ar1_jitter_matches_stationary_variance_and_correlation(alpha: float) -> None:
    """AR(1) jitter preserves variance and has the configured lag-one correlation."""
    timing_error = _jitter_positions("ar1", alpha=alpha).timing_error
    expected_variance = (20_000e-6) ** 2

    assert np.mean(timing_error) == pytest.approx(0.0, abs=4e-4)
    assert np.var(timing_error, ddof=1) == pytest.approx(expected_variance, rel=0.035)
    assert np.corrcoef(timing_error[:-1], timing_error[1:])[0, 1] == pytest.approx(alpha, abs=0.015)


def test_random_walk_drift_ensemble_matches_theoretical_variances() -> None:
    """Random-walk innovations, period error, and timing error follow theory."""
    period_index = 80
    timing_index = 100
    innovations = np.empty(ENSEMBLE_SIZE)
    period_errors = np.empty(ENSEMBLE_SIZE)
    timing_errors = np.empty(ENSEMBLE_SIZE)

    for seed in range(ENSEMBLE_SIZE):
        timing_error = _random_walk_positions(seed).timing_error
        period_error = np.diff(timing_error)
        innovations[seed] = np.diff(period_error)[20]
        period_errors[seed] = period_error[period_index]
        timing_errors[seed] = timing_error[timing_index]

    innovation_variance = (1_000e-6) ** 2 / 100.0
    variance_relative_tolerance = SIGMA_MULTIPLIER * np.sqrt(2.0 / (ENSEMBLE_SIZE - 1))
    assert np.mean(innovations) == pytest.approx(
        0.0,
        abs=SIGMA_MULTIPLIER * np.sqrt(innovation_variance / ENSEMBLE_SIZE),
    )
    assert np.var(innovations, ddof=1) == pytest.approx(
        innovation_variance,
        rel=variance_relative_tolerance,
    )
    assert np.var(period_errors, ddof=1) == pytest.approx(
        period_index * innovation_variance,
        rel=variance_relative_tolerance,
    )
    expected_timing_variance = (
        innovation_variance * (timing_index - 1) * timing_index * (2 * timing_index - 1) / 6.0
    )
    assert np.var(timing_errors, ddof=1) == pytest.approx(
        expected_timing_variance,
        rel=variance_relative_tolerance,
    )


def test_random_walk_drift_scales_with_physical_sample_rate() -> None:
    """One-second period-error variance is independent of the sample rate."""
    rates = (100, 400)
    endpoint_errors = []
    for sample_rate_hz in rates:
        errors = np.empty(ENSEMBLE_SIZE)
        sample_index = sample_rate_hz
        for seed in range(ENSEMBLE_SIZE):
            positions = generate_sampling_positions(
                num_input_samples=sample_index + 8,
                taps_per_phase=1,
                uprate=1,
                drate=1.0,
                jitter_ppm=0.0,
                drift_ppm=0.0,
                rng=np.random.default_rng(seed),
                drift_model="random_walk",
                drift_noise_ppm_per_sqrt_sec=1_000.0,
                sample_rate_hz=float(sample_rate_hz),
            )
            errors[seed] = np.diff(positions.timing_error)[sample_index - 1]
        endpoint_errors.append(errors)

    variance_ratio = np.var(endpoint_errors[1], ddof=1) / np.var(endpoint_errors[0], ddof=1)
    assert variance_ratio == pytest.approx(1.0, rel=0.16)


def test_constant_and_linear_drift_are_deterministic() -> None:
    """Deterministic drift models do not vary with the random-generator seed."""
    common = {
        "num_input_samples": 512,
        "taps_per_phase": 1,
        "uprate": 8,
        "drate": 8.0,
        "jitter_ppm": 0.0,
        "drift_ppm": 100.0,
        "drift_rate_ppm_per_sec": 20.0,
        "sample_rate_hz": 100.0,
    }
    for model in ("constant", "linear"):
        first = generate_sampling_positions(rng=np.random.default_rng(1), drift_model=model, **common)
        second = generate_sampling_positions(rng=np.random.default_rng(2), drift_model=model, **common)
        np.testing.assert_array_equal(first.actual, second.actual)
