"""Write, read, and display fixed-shape IQ Signals with homogeneous HDF5.

Example:
    python examples/scripts/visualize_homogeneous_hdf5.py --samples 4
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import compute_spectrogram
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)

MIN_IQ_SAMPLES = 8
MIN_FFT_SIZE = 2
SIGNAL_CLASSES = ("tone", "bpsk", "qpsk", "lfm")


def parse_args() -> argparse.Namespace:
    """Parse dataset-generation and plotting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Dataset directory containing data.h5. A temporary directory is "
            "used when omitted."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of fixed-shape top-level Signals to generate.",
    )
    parser.add_argument(
        "--num-iq-samples",
        type=int,
        default=2_048,
        help="Number of complex IQ samples in every top-level Signal.",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=64,
        help="FFT size used to display each spectrogram.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing --root directory to be replaced.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path at which to save the generated figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create and optionally save the figure without opening a window.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.num_iq_samples < MIN_IQ_SAMPLES:
        parser.error(
            f"--num-iq-samples must be at least {MIN_IQ_SAMPLES}"
        )
    if (
        args.fft_size < MIN_FFT_SIZE
        or args.fft_size > args.num_iq_samples
    ):
        parser.error(
            f"--fft-size must be at least {MIN_FFT_SIZE} and no larger "
            "than --num-iq-samples"
        )
    if args.root is not None and args.root.exists() and not args.overwrite:
        parser.error("the --root directory exists; pass --overwrite to replace it")
    return args


def make_component(
    class_name: str,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate one recognizable baseband modulation."""
    time = np.arange(length, dtype=np.float32)
    samples_per_symbol = 8

    if class_name == "tone":
        center_frequency = -0.22
        data = np.exp(2j * np.pi * center_frequency * time)
    elif class_name == "bpsk":
        center_frequency = -0.08
        symbol_count = int(np.ceil(length / samples_per_symbol))
        symbols = rng.choice((-1.0, 1.0), size=symbol_count)
        baseband = np.repeat(symbols, samples_per_symbol)[:length]
        data = baseband * np.exp(
            2j * np.pi * center_frequency * time
        )
    elif class_name == "qpsk":
        center_frequency = 0.1
        symbol_count = int(np.ceil(length / samples_per_symbol))
        symbol_indices = rng.integers(0, 4, size=symbol_count)
        symbols = np.exp(0.5j * np.pi * symbol_indices)
        baseband = np.repeat(symbols, samples_per_symbol)[:length]
        data = baseband * np.exp(
            2j * np.pi * center_frequency * time
        )
    elif class_name == "lfm":
        start_frequency = -0.3
        stop_frequency = 0.3
        chirp_rate = (stop_frequency - start_frequency) / length
        center_frequency = 0.0
        phase = 2 * np.pi * (
            start_frequency * time + 0.5 * chirp_rate * time**2
        )
        data = np.exp(1j * phase)
    else:
        raise ValueError(f"Unknown signal class: {class_name}")

    return (0.7 * data).astype(np.complex64), center_frequency


def make_signals(count: int, num_iq_samples: int) -> list[Signal]:
    """Create homogeneous wideband IQ arrays with varied signal metadata."""
    rng = np.random.default_rng(0)
    signals = []

    for signal_index in range(count):
        components = []
        iq = (
            0.03
            * (
                rng.standard_normal(num_iq_samples)
                + 1j * rng.standard_normal(num_iq_samples)
            )
        ).astype(np.complex64)

        for component_index in range(2 + signal_index % 3):
            class_name = SIGNAL_CLASSES[
                (2 * signal_index + component_index) % len(SIGNAL_CLASSES)
            ]
            component_length = num_iq_samples // (2 + component_index)
            component, center_frequency = make_component(
                class_name,
                component_length,
                rng,
            )
            available_start = num_iq_samples - component_length
            start = (
                signal_index * 173 + component_index * 251
            ) % (available_start + 1)
            stop = start + component_length
            iq[start:stop] += component
            components.append(
                Signal(
                    data=component,
                    component_index=component_index,
                    class_name=class_name,
                    start_in_samples=start,
                    duration_in_samples=component_length,
                    normalized_center_frequency=center_frequency,
                )
            )

        signals.append(
            Signal(
                data=iq,
                component_signals=components,
                sample_index=signal_index,
                signal_classes=tuple(
                    component["class_name"] for component in components
                ),
                description="homogeneous HDF5 example",
            )
        )

    return signals


def write_signals(root: Path, signals: list[Signal]) -> None:
    """Write Signals in two sequential batches."""
    split = max(1, len(signals) // 2)
    batches = [signals[:split], signals[split:]]
    with HomogeneousHDF5Writer(
        root,
        compression="lzf",
        chunk_samples=2,
    ) as writer:
        for batch_index, batch in enumerate(batches):
            if batch:
                writer.write(batch_index, batch)


def read_signals(root: Path) -> list[Signal]:
    """Read complete Signals, including metadata and components."""
    reader = HomogeneousHDF5Reader(root)
    try:
        return reader.read_signals_batch(0, len(reader))
    finally:
        reader.teardown()


def verify_round_trip(
    written: list[Signal],
    read_back: list[Signal],
) -> None:
    """Verify top-level arrays, metadata, and component arrays."""
    if len(read_back) != len(written):
        raise AssertionError(
            f"Expected {len(written)} Signals, read {len(read_back)}"
        )
    for expected, actual in zip(written, read_back, strict=True):
        np.testing.assert_array_equal(actual.data, expected.data)
        if actual.data.dtype != expected.data.dtype:
            raise AssertionError(
                f"Expected dtype {expected.data.dtype}, got {actual.data.dtype}"
            )
        if actual["sample_index"] != expected["sample_index"]:
            raise AssertionError("Signal metadata did not round-trip")
        if actual["signal_classes"] != expected["signal_classes"]:
            raise AssertionError("Signal class metadata did not round-trip")
        if len(actual.component_signals) != len(expected.component_signals):
            raise AssertionError("Component count did not round-trip")
        for expected_component, actual_component in zip(
            expected.component_signals,
            actual.component_signals,
            strict=True,
        ):
            np.testing.assert_array_equal(
                actual_component.data,
                expected_component.data,
            )
            for key in (
                "class_name",
                "start_in_samples",
                "duration_in_samples",
                "normalized_center_frequency",
            ):
                if actual_component[key] != expected_component[key]:
                    raise AssertionError(
                        f"Component metadata {key!r} did not round-trip"
                    )


def plot_spectrogram_comparison(
    before_write: list[Signal],
    after_read: list[Signal],
    fft_size: int,
) -> plt.Figure:
    """Display before-write and after-read spectrograms side by side."""
    figure, axes = plt.subplots(
        len(before_write),
        3,
        figsize=(16, 4 * len(before_write)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (original, reconstructed) in enumerate(
        zip(before_write, after_read, strict=True)
    ):
        original_spectrogram = compute_spectrogram(
            original.data,
            fft_size=fft_size,
            fft_stride=fft_size // 2,
        )
        reconstructed_spectrogram = compute_spectrogram(
            reconstructed.data,
            fft_size=fft_size,
            fft_stride=fft_size // 2,
        )
        difference = np.abs(
            reconstructed_spectrogram - original_spectrogram
        )
        value_min = min(
            float(np.min(original_spectrogram)),
            float(np.min(reconstructed_spectrogram)),
        )
        value_max = max(
            float(np.max(original_spectrogram)),
            float(np.max(reconstructed_spectrogram)),
        )
        max_iq_error = float(
            np.max(np.abs(reconstructed.data - original.data))
        )
        original_classes = ", ".join(original["signal_classes"])
        reconstructed_classes = ", ".join(
            reconstructed["signal_classes"]
        )

        before_image = axes[row, 0].imshow(
            original_spectrogram,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axes[row, 0].set_title(
            f"Before write — sample {original['sample_index']}\n"
            f"{original_classes}"
        )

        axes[row, 1].imshow(
            reconstructed_spectrogram,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axes[row, 1].set_title(
            f"After read — {len(reconstructed.component_signals)} components\n"
            f"{reconstructed_classes}"
        )

        difference_image = axes[row, 2].imshow(
            difference,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=max(float(np.max(difference)), np.finfo(float).eps),
        )
        axes[row, 2].set_title(
            f"Absolute difference\nmax IQ error={max_iq_error:.3g}"
        )

        for axis in axes[row]:
            axis.set_xlabel("Time bin")
            axis.set_ylabel("Frequency bin")
        figure.colorbar(
            before_image,
            ax=axes[row, :2],
            label="Power spectral density (dB)",
        )
        figure.colorbar(
            difference_image,
            ax=axes[row, 2],
            label="Absolute difference (dB)",
        )

    return figure


def main() -> None:
    """Round-trip homogeneous Signals and display the reconstructed data."""
    args = parse_args()
    root = (
        args.root.expanduser().resolve()
        if args.root is not None
        else Path(tempfile.mkdtemp(prefix="torchsig-homogeneous-"))
    )
    written = make_signals(args.samples, args.num_iq_samples)
    write_signals(root, written)
    read_back = read_signals(root)
    verify_round_trip(written, read_back)

    figure = plot_spectrogram_comparison(
        written,
        read_back,
        args.fft_size,
    )
    if args.save is not None:
        save_path = args.save.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    print(f"Homogeneous HDF5 file: {root / 'data.h5'}")
    print(f"Verified and displayed {len(read_back)} Signals read from disk")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
