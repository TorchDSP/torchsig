"""Compare TorchSig sampling-clock jitter on a tone and pulse-shaped QPSK.

Run from the TorchSig checkout with its editable environment. Drift is always zero.
Timing plots use the same sampling positions passed to the polyphase resampler.
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
from torchsig.utils.dsp import (
    SamplingPositions,
    generate_sampling_positions,
    polyphase_resample_at_positions,
)

MODELS = ("independent", "period", "ar1")
MIN_SAMPLES = 256
MIN_UPRATE = 2
NYQUIST_FREQUENCY = 0.5


def rms(x: np.ndarray) -> float:
    """Return the root-mean-square magnitude of a signal."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def make_signals(n: int, seed: int, tone_frequency: float, qpsk_bandwidth: float) -> dict[str, np.ndarray]:
    """Create deterministic tone and pulse-shaped QPSK inputs."""
    # TorchSig's modulation routine generates the pulse-shaped QPSK waveform.
    qpsk = constellation_modulator(
        constellation_name="qpsk",
        pulse_shape_name="srrc",
        bandwidth=qpsk_bandwidth,
        sample_rate=1.0,
        num_samples=n,
        alpha_rolloff=0.35,
        rng=np.random.default_rng(seed),
    )
    # Deterministic analytic reference tone, with normalized frequency in cycles/sample.
    tone = np.exp(2j * np.pi * tone_frequency * np.arange(n))
    return {"tone": tone.astype(np.complex64), "qpsk": np.asarray(qpsk, dtype=np.complex64)}


def design_filter(uprate: int, taps_per_phase: int = 16) -> np.ndarray:
    """Design the prototype filter used by the diagnostic resampler."""
    # TorchSig's partition_polyphase normalizes branch sums to the total FIR sum.
    return firwin(uprate * taps_per_phase + 1, cutoff=0.9 / uprate, window="hann")


def sample_with_positions(x: np.ndarray, h: np.ndarray, args: argparse.Namespace, model: str, ppm: float):
    """Generate one clock trajectory and interpolate at exactly those positions."""
    taps_per_phase = int(np.ceil(len(h) / args.uprate))
    positions = generate_sampling_positions(
        num_input_samples=len(x),
        taps_per_phase=taps_per_phase,
        uprate=args.uprate,
        drate=float(args.uprate),
        jitter_ppm=ppm,
        drift_ppm=0.0,
        rng=np.random.default_rng(args.seed + 101),
        jitter_model=model,
        jitter_alpha=args.alpha,
    )
    samples = polyphase_resample_at_positions(
        h=h,
        x=x,
        uprate=args.uprate,
        positions=positions.actual,
    )
    return SignalRun(samples=samples, positions=positions)


@dataclass(frozen=True)
class SignalRun:
    """Pair resampled IQ data with the positions used to produce it."""

    samples: np.ndarray
    positions: SamplingPositions


def acf(x: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """Compute a normalized autocorrelation sequence."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    variance = np.dot(x, x)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / variance for k in range(max_lag + 1)]) if variance else np.zeros(max_lag + 1)


def aligned(baseline: SignalRun, impaired: SignalRun):
    """Match samples by nominal iteration index, even if positions were skipped."""
    indices, baseline_idx, impaired_idx = np.intersect1d(
        baseline.positions.nominal_indices,
        impaired.positions.nominal_indices,
        assume_unique=True,
        return_indices=True,
    )
    return indices, baseline.samples[baseline_idx], impaired.samples[impaired_idx]


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and center a signal's power spectral density."""
    freq, power = welch(x, fs=1, nperseg=min(1024, len(x)), return_onesided=False)
    return np.fft.fftshift(freq), 10 * np.log10(np.maximum(np.fft.fftshift(power), 1e-14))


def plot_signal(name: str, baseline: SignalRun, outputs: dict[str, SignalRun], args: argparse.Namespace, folder: Path) -> None:
    """Plot waveform, spectrogram, PSD, error, and constellation diagnostics."""
    fig, ax = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"{name.upper()} | TorchSig jitter | {args.jitter_ppm:g} ppm | drift=0", fontsize=15)
    base_freq, base_psd = spectrum(baseline.samples)
    # Shared spectrogram color limits across models.
    specs = {}
    for model, run in outputs.items():
        y = run.samples
        f, t, p = spectrogram(y, fs=1, nperseg=min(256, len(y)), noverlap=192, return_onesided=False, mode="psd")
        order = np.argsort(f)
        specs[model] = (f[order], t, 10 * np.log10(np.maximum(p[order], 1e-14)))
    vmin = min(np.percentile(v[2], 5) for v in specs.values())
    vmax = max(np.percentile(v[2], 99) for v in specs.values())
    for col, model in enumerate(MODELS):
        y = outputs[model].samples
        indices, b, z = aligned(baseline, outputs[model])
        m = min(args.display_samples, len(b))
        ax[0, col].plot(indices[:m], b.real[:m], label="Zero-jitter baseline", lw=1)
        ax[0, col].plot(indices[:m], z.real[:m], label=model, lw=1, alpha=0.75)
        ax[0, col].set_title(f"Real IQ | output length={len(y)}")
        ax[0, col].set_xlabel("Nominal output index (aligned)")
        if col == 0:
            ax[0, col].legend(fontsize=8)
        f, t, db = specs[model]
        ax[1, col].pcolormesh(t, f, db, shading="auto", vmin=vmin, vmax=vmax)
        ax[1, col].set_title("Jittered spectrogram (shared dB scale)")
        ax[1, col].set_ylabel("Cycles/sample")
        ax[1, col].set_xlabel("Output index")
        freq, psd = spectrum(y)
        ax[2, col].plot(base_freq, base_psd, label="Baseline", lw=1)
        ax[2, col].plot(freq, psd, label=model, lw=1, alpha=0.8)
        ax[2, col].set_title("Welch PSD")
        ax[2, col].set_xlabel("Cycles/sample")
        ax[2, col].set_ylabel("dB")
        if col == 0:
            ax[2, col].legend(fontsize=8)
        difference = z - b
        ax[3, col].plot(indices[: min(800, len(difference))], np.abs(difference[: min(800, len(difference))]), lw=0.8)
        ax[3, col].set_title("|Jittered - baseline| (nominal-index aligned)")
        ax[3, col].set_xlabel("Nominal output index")
        ax[3, col].set_ylabel("Magnitude")
    fig.savefig(folder / f"{name}_signal.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for col, model in enumerate(MODELS):
        y = outputs[model].samples
        axes[col].scatter(y.real[: min(2000, len(y))], y.imag[: min(2000, len(y))], s=3, alpha=0.25)
        axes[col].set_title(f"{name.upper()} | {model}")
        axes[col].set_xlabel("I")
        axes[col].set_ylabel("Q")
        axes[col].set_aspect("equal", adjustable="box")
    fig.savefig(folder / f"{name}_iq.png", dpi=150)
    plt.close(fig)


def plot_timing(positions_by_model: dict, folder: Path) -> None:
    """Plot actual positions used to interpolate, not independently simulated jitter."""
    fig, ax = plt.subplots(4, 3, figsize=(16, 13), constrained_layout=True)
    fig.suptitle("Actual resampler sampling positions | drift=0", fontsize=15)
    for col, model in enumerate(MODELS):
        positions = positions_by_model[model]
        tau = positions.timing_error
        indices = positions.nominal_indices
        # For period jitter, use increments only for consecutive nominal indices.
        consecutive = np.diff(indices) == 1
        increments = np.diff(tau)[consecutive]
        ax[0, col].plot(indices, tau, lw=0.75)
        ax[0, col].set_title(f"{model} | actual timing error")
        ax[0, col].set_xlabel("Nominal output index")
        ax[0, col].set_ylabel("Input-sample periods")
        ax[1, col].hist(tau, bins=65, density=True)
        ax[1, col].set_title("Accepted-sample timing-error distribution")
        series = increments if model == "period" else tau
        ax[2, col].plot(acf(series, max_lag=min(60, len(series) - 1)))
        ax[2, col].set_title("Increment ACF" if model == "period" else "Timing-error ACF")
        ax[2, col].set_xlabel("Lag")
        ax[3, col].plot(indices, np.cumsum(tau**2) / np.arange(1, len(tau) + 1))
        ax[3, col].set_title("Cumulative mean squared timing error")
        ax[3, col].set_xlabel("Nominal output index")
        ax[3, col].set_ylabel("Sample periods squared")
    fig.savefig(folder / "timing_models.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate sampling-clock jitter diagnostic plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--jitter-ppm", type=float, default=25000)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--uprate", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tone-frequency", type=float, default=0.12, help="Cycles per sample")
    parser.add_argument("--qpsk-bandwidth", type=float, default=0.15, help="Normalized bandwidth (sample_rate=1)")
    parser.add_argument("--display-samples", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("jitter_diagnostics"))
    args = parser.parse_args()
    if args.samples < MIN_SAMPLES or args.uprate < MIN_UPRATE or args.jitter_ppm < 0 or not 0 <= args.alpha < 1:
        parser.error("Require samples>=256, uprate>=2, jitter-ppm>=0, 0<=alpha<1")
    if not 0 < args.tone_frequency < NYQUIST_FREQUENCY or not 0 < args.qpsk_bandwidth < NYQUIST_FREQUENCY:
        parser.error("Require tone-frequency and qpsk-bandwidth in (0, 0.5)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h = design_filter(args.uprate)
    print(f"Filter sum: {np.sum(h):.6f}; uprate=drate={args.uprate}; drift_ppm=0")
    timing_positions = None
    for name, x in make_signals(args.samples, args.seed, args.tone_frequency, args.qpsk_bandwidth).items():
        baseline = sample_with_positions(x, h, args, "independent", 0.0)
        outputs = {model: sample_with_positions(x, h, args, model, args.jitter_ppm) for model in MODELS}
        if timing_positions is None:
            timing_positions = {model: run.positions for model, run in outputs.items()}
        lengths = {model: len(run.samples) for model, run in outputs.items()}
        print(f"{name}: input RMS={rms(x):.5f}, baseline RMS={rms(baseline.samples):.5f}, gain={rms(baseline.samples) / rms(x):.5f}; lengths={lengths}")
        plot_signal(name, baseline, outputs, args, args.output_dir)
    plot_timing(timing_positions, args.output_dir)
    print(f"Saved 5 PNGs in {args.output_dir}")
    print("Waveform differences are aligned by nominal sampling iteration.")
    print("Timing plots show actual accepted resampler positions (not a separate simulation).")


if __name__ == "__main__":
    main()
