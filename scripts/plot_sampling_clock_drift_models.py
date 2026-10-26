# ruff: noqa: INP001
"""Plot diagnostics for the three sampling-clock drift models."""

import matplotlib.pyplot as plt
import numpy as np

from torchsig.utils.dsp import generate_sampling_positions


def main() -> None:
    """Generate timing, period-error, autocorrelation, and cumulative-MSE plots."""
    models = {
        "constant": {"drift_ppm": 1_000.0},
        "linear": {"drift_ppm": 0.0, "drift_rate_ppm_per_sec": 10_000.0},
        "random_walk": {"drift_ppm": 0.0, "drift_noise_ppm_per_sqrt_sec": 1_000.0},
    }
    _, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for model, extra in models.items():
        positions = generate_sampling_positions(
            num_input_samples=20_000,
            taps_per_phase=1,
            uprate=8,
            drate=8.0,
            jitter_ppm=0.0,
            drift_ppm=extra.pop("drift_ppm"),
            rng=np.random.default_rng(0),
            drift_model=model,
            sample_rate_hz=1_000.0,
            **extra,
        )
        error = positions.timing_error
        period_error = np.diff(error)
        cumulative_mse = np.cumsum(error**2) / np.arange(1, len(error) + 1)
        axes[0, 0].plot(error, label=model)
        axes[0, 1].plot(period_error * 1e6, label=model)
        if model == "constant":
            axes[1, 0].plot([], label="constant: ACF undefined")
        elif model == "linear":
            n = positions.nominal_indices[:-1]
            theoretical_ppm = 10_000.0 * n / 1_000.0
            axes[1, 0].plot(period_error * 1e6 - theoretical_ppm, label="linear residual")
        else:
            innovations = np.diff(period_error)
            centered = innovations - np.mean(innovations)
            autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1 : len(centered) + 99]
            autocorrelation /= autocorrelation[0]
            axes[1, 0].plot(autocorrelation, label="random-walk innovation ACF")
        axes[1, 1].loglog(np.arange(1, len(error) + 1), cumulative_mse, label=model)

    for axis, title, ylabel in zip(
        axes.flat,
        ("Sampling-position error", "Period-error trajectory", "Model-appropriate residual / ACF", "Cumulative timing-error MSE"),
        ("input samples", "ppm", "correlation", "input samples squared"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(visible=True)
        axis.legend()
    axes[1, 0].set_xlabel("lag")
    axes[1, 1].set_xlabel("output sample")
    plt.show()


if __name__ == "__main__":
    main()
