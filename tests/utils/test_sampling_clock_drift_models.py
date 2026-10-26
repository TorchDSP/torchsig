"""Mathematical and ensemble validation for sampling-clock drift models."""

import numpy as np
import pytest

from torchsig.utils.dsp import generate_sampling_positions

ENSEMBLE_SIZE = 1_200
SIGMA_MULTIPLIER = 4.5


def _positions(
    *,
    samples: int = 512,
    seed: int = 0,
    sample_rate_hz: float = 100.0,
    noise_ppm_per_sqrt_sec: float = 1_000.0,
):
    """Generate a drift-only, unit-increment random-walk realization."""
    return generate_sampling_positions(
        num_input_samples=samples,
        taps_per_phase=1,
        uprate=1,
        drate=1.0,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(seed),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=noise_ppm_per_sqrt_sec,
        sample_rate_hz=sample_rate_hz,
    )


@pytest.mark.parametrize("drift_ppm", [-2_000.0, 2_000.0])
def test_constant_drift_matches_linear_position_error(drift_ppm: float) -> None:
    """A signed constant period error integrates to linear position error."""
    positions = generate_sampling_positions(1_000, 1, 8, 4.0, 0.0, drift_ppm)
    n = positions.nominal_indices
    expected_polyphase_error = 4.0 * n * drift_ppm * 1e-6
    np.testing.assert_allclose(positions.actual - positions.nominal, expected_polyphase_error, atol=2e-12)


def test_linear_drift_matches_quadratic_position_error() -> None:
    """The discrete linear period trend integrates to n(n-1)/2 timing error."""
    sample_rate_hz = 200.0
    drift_rate = 10_000.0
    positions = generate_sampling_positions(
        1_000,
        1,
        8,
        4.0,
        0.0,
        0.0,
        drift_model="linear",
        drift_rate_ppm_per_sec=drift_rate,
        sample_rate_hz=sample_rate_hz,
    )
    n = positions.nominal_indices.astype(np.float64)
    expected_polyphase_error = 4.0 * drift_rate * 1e-6 / sample_rate_hz * n * (n - 1.0) / 2.0
    np.testing.assert_allclose(positions.actual - positions.nominal, expected_polyphase_error, atol=2e-11)


def test_random_walk_ensemble_statistics_match_theory() -> None:
    """Innovation, period, and integrated-position ensembles match theory."""
    sample_rate_hz = 100.0
    noise = 1_000.0
    period_index = 80
    position_index = 100
    innovations = np.empty(ENSEMBLE_SIZE)
    period_errors = np.empty(ENSEMBLE_SIZE)
    position_errors = np.empty(ENSEMBLE_SIZE)
    for seed in range(ENSEMBLE_SIZE):
        positions = _positions(seed=seed, sample_rate_hz=sample_rate_hz, noise_ppm_per_sqrt_sec=noise)
        period_error = np.diff(positions.timing_error)
        innovations[seed] = np.diff(period_error)[20]
        period_errors[seed] = period_error[period_index]
        position_errors[seed] = positions.timing_error[position_index]

    innovation_variance = (noise * 1e-6) ** 2 / sample_rate_hz
    mean_tolerance = SIGMA_MULTIPLIER * np.sqrt(innovation_variance / ENSEMBLE_SIZE)
    variance_relative_tolerance = SIGMA_MULTIPLIER * np.sqrt(2.0 / (ENSEMBLE_SIZE - 1))
    assert np.mean(innovations) == pytest.approx(0.0, abs=mean_tolerance)
    assert np.var(innovations, ddof=1) == pytest.approx(
        innovation_variance,
        rel=variance_relative_tolerance,
    )
    assert np.var(period_errors, ddof=1) == pytest.approx(
        period_index * innovation_variance,
        rel=variance_relative_tolerance,
    )
    expected_position_variance = (
        innovation_variance * (position_index - 1) * position_index * (2 * position_index - 1) / 6.0
    )
    assert np.var(position_errors, ddof=1) == pytest.approx(
        expected_position_variance,
        rel=variance_relative_tolerance,
    )


def test_random_walk_physical_time_scaling_across_sample_rates() -> None:
    """One-second period and physical timing statistics are rate invariant."""
    noise = 1_000.0
    duration_seconds = 1.0
    rates = (100, 400)
    period_ensembles = []
    time_error_ensembles = []
    for sample_rate_hz in rates:
        sample_index = int(duration_seconds * sample_rate_hz)
        period_errors = np.empty(ENSEMBLE_SIZE)
        time_errors = np.empty(ENSEMBLE_SIZE)
        for seed in range(ENSEMBLE_SIZE):
            positions = _positions(
                samples=sample_index + 32,
                seed=seed,
                sample_rate_hz=float(sample_rate_hz),
                noise_ppm_per_sqrt_sec=noise,
            )
            period_errors[seed] = np.diff(positions.timing_error)[sample_index - 1]
            time_errors[seed] = positions.timing_error[sample_index] / sample_rate_hz
        period_ensembles.append(period_errors)
        time_error_ensembles.append(time_errors)

    statistical_tolerance = 5.0 * np.sqrt(2.0 / (ENSEMBLE_SIZE - 1))
    assert np.var(period_ensembles[1], ddof=1) / np.var(period_ensembles[0], ddof=1) == pytest.approx(
        1.0,
        rel=statistical_tolerance,
    )
    assert np.var(time_error_ensembles[1], ddof=1) / np.var(time_error_ensembles[0], ddof=1) == pytest.approx(
        1.0,
        rel=statistical_tolerance,
    )


def test_random_walk_is_reproducible_with_fixed_seed() -> None:
    """A fixed generator seed reproduces the complete accepted trajectory."""
    first = _positions(seed=91)
    second = _positions(seed=91)
    np.testing.assert_array_equal(first.nominal_indices, second.nominal_indices)
    np.testing.assert_array_equal(first.actual, second.actual)
