"""Compare TorchSig sampling-clock drift models on a tone and pulse-shaped QPSK.

Run from the TorchSig checkout with its editable environment. Jitter is always
zero. Timing plots use the same sampling positions passed to the resampler.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, spectrogram, welch

from torchsig.signals.builders.constellation import constellation_modulator
from torchsig.utils.dsp import SamplingPositions, generate_sampling_positions, polyphase_resample_at_positions

MODELS = ("constant", "linear", "random_walk")
MIN_SAMPLES = 256
MIN_UPRATE = 2
NYQUIST_FREQUENCY = 0.5


@dataclass(frozen=True)
class SignalRun:
    """Pair resampled IQ data with the positions used to produce it."""

    samples: np.ndarray
    positions: SamplingPositions


@dataclass(frozen=True)
class DriftConfig:
    """Parameters for one drift-model diagnostic run."""

    drift_ppm: float
    drift_rate_ppm_per_sec: float = 0.0
    drift_noise_ppm_per_sqrt_sec: float = 0.0


def comparison_configs(args: argparse.Namespace) -> dict[str, dict[str, DriftConfig]]:
    """Return shared-offset and isolated-model diagnostic configurations."""
    shared = {
        "constant": DriftConfig(args.drift_ppm),
        "linear": DriftConfig(args.drift_ppm, drift_rate_ppm_per_sec=args.drift_rate_ppm_per_sec),
        "random_walk": DriftConfig(
            args.drift_ppm,
            drift_noise_ppm_per_sqrt_sec=args.drift_noise_ppm_per_sqrt_sec,
        ),
    }
    isolated = {
        "constant": DriftConfig(args.isolated_constant_ppm),
        "linear": DriftConfig(0.0, drift_rate_ppm_per_sec=args.isolated_drift_rate_ppm_per_sec),
        "random_walk": DriftConfig(
            0.0,
            drift_noise_ppm_per_sqrt_sec=args.isolated_drift_noise_ppm_per_sqrt_sec,
        ),
    }
    return {"shared_offset": shared, "isolated_models": isolated}


def rms(x: np.ndarray) -> float:
    """Return the root-mean-square magnitude of a signal."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def make_signals(n: int, seed: int, tone_frequency: float, qpsk_bandwidth: float) -> dict[str, np.ndarray]:
    """Create deterministic tone and pulse-shaped QPSK inputs."""
    qpsk = constellation_modulator(
        constellation_name="qpsk",
        pulse_shape_name="srrc",
        bandwidth=qpsk_bandwidth,
        sample_rate=1.0,
        num_samples=n,
        alpha_rolloff=0.35,
        rng=np.random.default_rng(seed),
    )
    tone = np.exp(2j * np.pi * tone_frequency * np.arange(n))
    return {"tone": tone.astype(np.complex64), "qpsk": np.asarray(qpsk, dtype=np.complex64)}


def design_filter(uprate: int, taps_per_phase: int = 16) -> np.ndarray:
    """Design the prototype filter used by the diagnostic resampler."""
    return firwin(uprate * taps_per_phase + 1, cutoff=0.9 / uprate, window="hann")


def sample_with_positions(
    x: np.ndarray,
    h: np.ndarray,
    args: argparse.Namespace,
    model: str,
    config: DriftConfig,
) -> SignalRun:
    """Generate one clock trajectory and interpolate at exactly those positions."""
    taps_per_phase = int(np.ceil(len(h) / args.uprate))
    positions = generate_sampling_positions(
        num_input_samples=len(x),
        taps_per_phase=taps_per_phase,
        uprate=args.uprate,
        drate=float(args.uprate),
        jitter_ppm=0.0,
        drift_ppm=config.drift_ppm,
        rng=np.random.default_rng(args.seed + 101),
        drift_model=model,
        drift_rate_ppm_per_sec=config.drift_rate_ppm_per_sec,
        drift_noise_ppm_per_sqrt_sec=config.drift_noise_ppm_per_sqrt_sec,
        sample_rate_hz=args.sample_rate_hz,
    )
    samples = polyphase_resample_at_positions(h=h, x=x, uprate=args.uprate, positions=positions.actual)
    return SignalRun(samples=samples, positions=positions)


def baseline_with_positions(x: np.ndarray, h: np.ndarray, args: argparse.Namespace) -> SignalRun:
    """Resample with an ideal clock for aligned comparisons."""
    taps_per_phase = int(np.ceil(len(h) / args.uprate))
    positions = generate_sampling_positions(
        len(x), taps_per_phase, args.uprate, float(args.uprate), 0.0, 0.0
    )
    samples = polyphase_resample_at_positions(h=h, x=x, uprate=args.uprate, positions=positions.actual)
    return SignalRun(samples=samples, positions=positions)


def acf(x: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """Compute a normalized autocorrelation sequence."""
    centered = np.asarray(x, dtype=float) - np.mean(x)
    energy = np.dot(centered, centered)
    if not energy:
        return np.zeros(max_lag + 1)
    return np.array(
        [np.dot(centered[: len(centered) - lag], centered[lag:]) / energy for lag in range(max_lag + 1)]
    )


def aligned(baseline: SignalRun, impaired: SignalRun) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match samples by nominal iteration index when positions were skipped."""
    indices, baseline_idx, impaired_idx = np.intersect1d(
        baseline.positions.nominal_indices,
        impaired.positions.nominal_indices,
        assume_unique=True,
        return_indices=True,
    )
    return indices, baseline.samples[baseline_idx], impaired.samples[impaired_idx]


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and center a signal's power spectral density."""
    frequency, power = welch(x, fs=1, nperseg=min(1024, len(x)), return_onesided=False)
    return np.fft.fftshift(frequency), 10 * np.log10(np.maximum(np.fft.fftshift(power), 1e-14))


def plot_signal(
    name: str,
    comparison_name: str,
    baseline: SignalRun,
    outputs: dict[str, SignalRun],
    args: argparse.Namespace,
    folder: Path,
) -> None:
    """Plot waveform, spectrogram, PSD, error, and constellation diagnostics."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(
        f"{name.upper()} | {comparison_name.replace('_', ' ')} | jitter=0",
        fontsize=15,
    )
    base_frequency, base_psd = spectrum(baseline.samples)
    spectrograms = {}
    for model, run in outputs.items():
        # Keep multiple time bins even for the minimum supported signal length.
        segment_length = max(32, min(256, len(run.samples) // 4))
        frequency, time, power = spectrogram(
            run.samples,
            fs=1,
            nperseg=segment_length,
            noverlap=3 * segment_length // 4,
            return_onesided=False,
            mode="psd",
        )
        order = np.argsort(frequency)
        spectrograms[model] = (frequency[order], time, 10 * np.log10(np.maximum(power[order], 1e-14)))
    vmin = min(np.percentile(values[2], 5) for values in spectrograms.values())
    vmax = max(np.percentile(values[2], 99) for values in spectrograms.values())

    for column, model in enumerate(MODELS):
        run = outputs[model]
        indices, reference, impaired = aligned(baseline, run)
        displayed = min(args.display_samples, len(reference))
        axes[0, column].plot(indices[:displayed], reference.real[:displayed], label="Ideal baseline", lw=1)
        axes[0, column].plot(indices[:displayed], impaired.real[:displayed], label=model, lw=1, alpha=0.75)
        axes[0, column].set_title(f"Real IQ | output length={len(run.samples)}")
        axes[0, column].set_xlabel("Nominal output index (aligned)")
        if column == 0:
            axes[0, column].legend(fontsize=8)
        frequency, time, power_db = spectrograms[model]
        axes[1, column].pcolormesh(time, frequency, power_db, shading="auto", vmin=vmin, vmax=vmax)
        axes[1, column].set_title("Drifted spectrogram (shared dB scale)")
        axes[1, column].set_ylabel("Cycles/sample")
        axes[1, column].set_xlabel("Output index")
        frequency, power_db = spectrum(run.samples)
        axes[2, column].plot(base_frequency, base_psd, label="Baseline", lw=1)
        axes[2, column].plot(frequency, power_db, label=model, lw=1, alpha=0.8)
        axes[2, column].set_title("Welch PSD")
        axes[2, column].set_xlabel("Cycles/sample")
        axes[2, column].set_ylabel("dB")
        if column == 0:
            axes[2, column].legend(fontsize=8)
        difference = impaired - reference
        count = min(800, len(difference))
        axes[3, column].plot(indices[:count], np.abs(difference[:count]), lw=0.8)
        axes[3, column].set_title("|Drifted - baseline| (nominal-index aligned)")
        axes[3, column].set_xlabel("Nominal output index")
        axes[3, column].set_ylabel("Magnitude")
    fig.savefig(folder / f"{name}_signal.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for column, model in enumerate(MODELS):
        samples = outputs[model].samples
        count = min(2000, len(samples))
        axes[column].scatter(samples.real[:count], samples.imag[:count], s=3, alpha=0.25)
        axes[column].set_title(f"{name.upper()} | {model}")
        axes[column].set_xlabel("I")
        axes[column].set_ylabel("Q")
        axes[column].set_aspect("equal", adjustable="box")
    fig.savefig(folder / f"{name}_iq.png", dpi=150)
    plt.close(fig)

    if name == "tone":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for column, model in enumerate(MODELS):
            indices, _, _ = aligned(baseline, outputs[model])
            position_lookup = dict(
                zip(outputs[model].positions.nominal_indices, outputs[model].positions.timing_error, strict=True)
            )
            timing_error = np.array([position_lookup[index] for index in indices])
            phase_error = 2 * np.pi * args.tone_frequency * timing_error
            axes[column].plot(indices / args.sample_rate_hz, phase_error)
            axes[column].set_title(f"{model} | phase error predicted from timing")
            axes[column].set_xlabel("Elapsed time (seconds)")
            axes[column].set_ylabel("Radians")
            axes[column].grid(visible=True)
        fig.savefig(folder / "tone_phase_error.png", dpi=150)
        plt.close(fig)


def plot_timing(
    positions_by_model: dict[str, SamplingPositions],
    configs: dict[str, DriftConfig],
    args: argparse.Namespace,
    folder: Path,
) -> None:
    """Plot timing and period errors from the actual accepted positions."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), constrained_layout=True)
    fig.suptitle("Actual resampler sampling positions | jitter=0", fontsize=15)
    for column, model in enumerate(MODELS):
        positions = positions_by_model[model]
        timing_error = positions.timing_error
        indices = positions.nominal_indices
        consecutive = np.diff(indices) == 1
        # Derive period error directly from actual positions to avoid
        # cancellation from differencing accumulated timing errors.
        period_error = (np.diff(positions.actual)[consecutive] / args.uprate - 1.0)
        period_indices = indices[:-1][consecutive]
        axes[0, column].plot(indices, timing_error, lw=0.75)
        axes[0, column].set_title(f"{model} | timing error")
        axes[0, column].set_xlabel("Nominal output index")
        axes[0, column].set_ylabel("Input-sample periods")
        if model == "constant":
            axes[1, column].axhline(configs[model].drift_ppm, lw=0.75)
            maximum_residual = np.max(np.abs(period_error * 1e6 - configs[model].drift_ppm))
            axes[1, column].text(
                0.02,
                0.95,
                f"max numerical residual: {maximum_residual:.2e} ppm",
                va="top",
                transform=axes[1, column].transAxes,
            )
        else:
            axes[1, column].plot(period_indices / args.sample_rate_hz, period_error * 1e6, lw=0.75)
        axes[1, column].set_title("Instantaneous period error")
        axes[1, column].set_xlabel("Elapsed time (seconds)")
        axes[1, column].set_ylabel("ppm")
        max_lag = min(60, len(period_error) - 1)
        if model == "constant":
            axes[2, column].axis("off")
            axes[2, column].text(
                0.5,
                0.5,
                "Constant period error\nNormalized ACF undefined\n(zero variance)",
                ha="center",
                va="center",
                transform=axes[2, column].transAxes,
            )
        elif model == "linear":
            theoretical = (
                configs[model].drift_ppm
                + configs[model].drift_rate_ppm_per_sec * period_indices / args.sample_rate_hz
            )
            residual_ppm = period_error * 1e6 - theoretical
            axes[2, column].plot(period_indices / args.sample_rate_hz, residual_ppm)
            axes[2, column].set_title("Residual after theoretical linear trend")
            axes[2, column].set_xlabel("Elapsed time (seconds)")
            axes[2, column].set_ylabel("ppm")
        else:
            innovations_ppm = np.diff(period_error) * 1e6
            axes[2, column].plot(acf(innovations_ppm, max_lag=min(max_lag, len(innovations_ppm) - 1)))
            axes[2, column].set_title("Random-walk innovation ACF")
            axes[2, column].set_xlabel("Lag")
        cumulative_mse = np.cumsum(timing_error**2) / np.arange(1, len(timing_error) + 1)
        axes[3, column].plot(indices, cumulative_mse)
        axes[3, column].set_title("Cumulative mean squared timing error")
        axes[3, column].set_xlabel("Nominal output index")
        axes[3, column].set_ylabel("Sample periods squared")
    fig.savefig(folder / "timing_models.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate sampling-clock drift diagnostic plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--drift-ppm", type=float, default=10_000.0)
    parser.add_argument("--drift-rate-ppm-per-sec", type=float, default=1_000.0)
    parser.add_argument("--drift-noise-ppm-per-sqrt-sec", type=float, default=500.0)
    parser.add_argument("--isolated-constant-ppm", type=float, default=1_000.0)
    parser.add_argument("--isolated-drift-rate-ppm-per-sec", type=float, default=10_000.0)
    parser.add_argument("--isolated-drift-noise-ppm-per-sqrt-sec", type=float, default=1_000.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1_000.0)
    parser.add_argument("--uprate", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tone-frequency", type=float, default=0.12, help="Cycles per sample")
    parser.add_argument("--qpsk-bandwidth", type=float, default=0.15, help="Normalized bandwidth")
    parser.add_argument("--display-samples", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("drift_diagnostics"))
    args = parser.parse_args()
    if args.samples < MIN_SAMPLES or args.uprate < MIN_UPRATE:
        parser.error("Require samples>=256 and uprate>=2")
    drift_parameters = np.array(
        [
            args.drift_ppm,
            args.drift_rate_ppm_per_sec,
            args.drift_noise_ppm_per_sqrt_sec,
            args.isolated_constant_ppm,
            args.isolated_drift_rate_ppm_per_sec,
            args.isolated_drift_noise_ppm_per_sqrt_sec,
            args.sample_rate_hz,
        ]
    )
    if (
        np.any(~np.isfinite(drift_parameters))
        or args.drift_ppm <= -1_000_000
        or args.drift_noise_ppm_per_sqrt_sec < 0
        or args.isolated_drift_noise_ppm_per_sqrt_sec < 0
        or args.sample_rate_hz <= 0
    ):
        parser.error("Require drift-ppm>-1000000, nonnegative drift noise, and positive sample rate")
    if not 0 < args.tone_frequency < NYQUIST_FREQUENCY or not 0 < args.qpsk_bandwidth < NYQUIST_FREQUENCY:
        parser.error("Require tone-frequency and qpsk-bandwidth in (0, 0.5)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prototype_filter = design_filter(args.uprate)
    print(f"Filter sum: {np.sum(prototype_filter):.6f}; uprate=drate={args.uprate}; jitter_ppm=0")
    signals = make_signals(args.samples, args.seed, args.tone_frequency, args.qpsk_bandwidth)
    for comparison_name, configs in comparison_configs(args).items():
        comparison_folder = args.output_dir / comparison_name
        comparison_folder.mkdir(parents=True, exist_ok=True)
        timing_positions = None
        for name, signal in signals.items():
            baseline = baseline_with_positions(signal, prototype_filter, args)
            outputs = {
                model: sample_with_positions(signal, prototype_filter, args, model, configs[model])
                for model in MODELS
            }
            if timing_positions is None:
                timing_positions = {model: run.positions for model, run in outputs.items()}
            lengths = {model: len(run.samples) for model, run in outputs.items()}
            print(
                f"{comparison_name}/{name}: input RMS={rms(signal):.5f}, "
                f"baseline RMS={rms(baseline.samples):.5f}, lengths={lengths}"
            )
            plot_signal(name, comparison_name, baseline, outputs, args, comparison_folder)
        if timing_positions is not None:
            plot_timing(timing_positions, configs, args, comparison_folder)
    print(f"Saved 12 PNGs under {args.output_dir}")
    print("Waveform differences are aligned by nominal sampling iteration.")
    print("Timing plots show actual accepted resampler positions (not a separate simulation).")


if __name__ == "__main__":
    main()
