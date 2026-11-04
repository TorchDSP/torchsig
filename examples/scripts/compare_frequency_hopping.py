"""Visualize an existing BPSK waveform with and without frequency hopping.

The same BPSK symbols are used in both panels. ``FrequencyHopping`` applies a
phase-continuous frequency translation without regenerating those symbols.

Example::

    python examples/scripts/compare_frequency_hopping.py \
        --num-samples 4096 --sample-rate 1000 --bandwidth 400 \
        --seed 7 --output frequency_hopping_comparison.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import FrequencyHopping
from torchsig.utils.dsp import TorchSigComplexDataType, time_varying_frequency_shift

MIN_SAMPLES = 16
SPECTROGRAM_FLOOR_DB = -80.0
IQ_DISPLAY_DURATION_SECONDS = 0.5


@dataclass(frozen=True)
class Comparison:
    """Waveforms and internal diagnostics for one hopping realization."""

    original: np.ndarray
    hopped: np.ndarray
    frequency_offsets: np.ndarray
    phasor: np.ndarray
    max_reconstruction_error: float


def make_comparison(
    num_samples: int,
    sample_rate: float,
    bandwidth: float,
    seed: int,
) -> Comparison:
    """Return matched BPSK waveforms and the actual applied hopping trace."""
    if num_samples < MIN_SAMPLES:
        raise ValueError(f"num_samples must be at least {MIN_SAMPLES}")

    samples_per_symbol = 16
    symbol_rng = np.random.default_rng(0)
    symbols = symbol_rng.choice((-1.0, 1.0), size=(num_samples + samples_per_symbol - 1) // samples_per_symbol)
    original = np.repeat(symbols, samples_per_symbol)[:num_samples].astype(TorchSigComplexDataType)
    signal = Signal(
        data=original.copy(),
        sample_rate=sample_rate,
        center_freq=0.0,
        bandwidth=sample_rate / samples_per_symbol,
        class_name="bpsk",
    )
    transform = FrequencyHopping(frequency_span=bandwidth, seed=seed)
    hopped = transform(signal)
    # This experimental example intentionally consumes the transform's internal
    # diagnostic trace; it is not part of the supported transform API.
    if transform._last_frequency_offsets is None:  # noqa: SLF001  # pragma: no cover
        raise RuntimeError("frequency-hopping transform did not retain its diagnostic trace")
    frequency_offsets = transform._last_frequency_offsets  # noqa: SLF001
    phasor = time_varying_frequency_shift(
        np.ones(num_samples, dtype=TorchSigComplexDataType),
        frequency_offsets,
        sample_rate,
    )
    recovered = hopped.data * np.conj(phasor)
    max_reconstruction_error = float(np.max(np.abs(original - recovered)))
    return Comparison(original, hopped.data, frequency_offsets, phasor, max_reconstruction_error)


def calculate_spectrogram(
    samples: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a centered power spectrogram in decibels."""
    segment_length = min(256, len(samples) // 2)
    frequencies, times, power = spectrogram(
        samples,
        fs=sample_rate,
        nperseg=segment_length,
        noverlap=3 * segment_length // 4,
        detrend=False,
        return_onesided=False,
        mode="psd",
    )
    order = np.argsort(frequencies)
    ordered_power = power[order]
    peak_power = max(float(np.max(ordered_power)), np.finfo(float).tiny)
    power_db = 10 * np.log10(np.maximum(ordered_power / peak_power, np.finfo(float).tiny))
    power_db = np.maximum(power_db, SPECTROGRAM_FLOOR_DB)
    return frequencies[order], times, power_db


def plot_comparison(
    comparison: Comparison,
    sample_rate: float,
    bandwidth: float,
    seed: int,
) -> plt.Figure:
    """Build the original versus frequency-hopped BPSK comparison figure."""
    original_spectrogram = calculate_spectrogram(comparison.original, sample_rate)
    hopped_spectrogram = calculate_spectrogram(comparison.hopped, sample_rate)
    color_min = SPECTROGRAM_FLOOR_DB
    color_max = 0.0

    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    figure.suptitle(f"Experimental frequency hopping comparison | span={bandwidth:g} Hz | seed={seed} | inverse max error={comparison.max_reconstruction_error:.2e}")

    sample_times = np.arange(len(comparison.original)) / sample_rate
    display_samples = min(len(comparison.original), max(1, round(IQ_DISPLAY_DURATION_SECONDS * sample_rate)))
    display_duration = display_samples / sample_rate
    axes[0, 0].plot(sample_times[:display_samples], comparison.original.real[:display_samples], linewidth=0.8)
    axes[0, 0].set_title(f"Without hopping: BPSK IQ (real), first {display_duration:g} s")
    axes[0, 1].plot(sample_times[:display_samples], comparison.hopped.real[:display_samples], linewidth=0.8)
    axes[0, 1].set_title(f"With hopping: BPSK IQ (real), first {display_duration:g} s")
    for axis in axes[0]:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        axis.set_ylim(-1.1, 1.1)
        axis.grid(visible=True, alpha=0.25)

    for axis, (frequencies, times, power_db), title in (
        (axes[1, 0], original_spectrogram, "Without hopping: BPSK spectrogram"),
        (axes[1, 1], hopped_spectrogram, "With hopping: BPSK spectrogram"),
    ):
        image = axis.pcolormesh(times, frequencies, power_db, shading="auto", vmin=color_min, vmax=color_max)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_ylim(-sample_rate / 2, sample_rate / 2)

    figure.colorbar(image, ax=axes[1], label="Power spectral density (dB)")

    hop_boundaries = np.flatnonzero(np.diff(comparison.frequency_offsets) != 0) + 1
    boundary = int(hop_boundaries[0])
    window_radius = min(16, boundary, len(comparison.phasor) - boundary)
    window = slice(boundary - window_radius, boundary + window_radius + 1)
    axes[2, 0].plot(sample_times[window], comparison.phasor.real[window], marker=".", label="Real")
    axes[2, 0].plot(sample_times[window], comparison.phasor.imag[window], marker=".", label="Imaginary")
    axes[2, 0].axvline(boundary / sample_rate, color="black", linestyle="--", label="Hop boundary")
    axes[2, 0].set_title("Phase continuity at first hop (frequency-shifting phasor)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].legend()
    axes[2, 0].grid(visible=True, alpha=0.25)

    # phase[n] - phase[n-1] uses f[n-1], so align each estimate with the
    # originating interval rather than the later phasor sample.
    frequency_times = np.arange(len(comparison.phasor) - 1) / sample_rate
    estimated_frequency = np.angle(comparison.phasor[1:] * np.conj(comparison.phasor[:-1])) * sample_rate / (2 * np.pi)
    axes[2, 1].step(
        frequency_times,
        comparison.frequency_offsets[:-1],
        where="post",
        label="Requested offset f[n-1]",
    )
    axes[2, 1].plot(frequency_times, estimated_frequency, linestyle="--", label="Estimated from h[n]·conj(h[n-1])")
    for hop_boundary in hop_boundaries:
        axes[2, 1].axvline(hop_boundary / sample_rate, color="black", alpha=0.2, linewidth=0.8)
    axes[2, 1].set_title("Requested versus applied frequency offset")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    axes[2, 1].legend()
    axes[2, 1].grid(visible=True, alpha=0.25)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--bandwidth", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("frequency_hopping_comparison.png"))
    return parser.parse_args()


def main() -> None:
    """Generate and save the comparison figure."""
    args = parse_args()
    comparison = make_comparison(
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        bandwidth=args.bandwidth,
        seed=args.seed,
    )
    figure = plot_comparison(comparison, args.sample_rate, args.bandwidth, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved comparison to {args.output}")
    print(f"Maximum inverse-hopping reconstruction error: {comparison.max_reconstruction_error:.3e}")


if __name__ == "__main__":
    main()
