"""Build a small dataset of three structured signals (DMR, 802.11a, DVB-S2) and
render an example spectrogram montage.

Run (inside the torchsig environment):
    python structured_signals_example.py

Outputs:
    structured_signals_spectrograms.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.dsp import compute_spectrogram

# Structured signal classes added on the structured-signals branch.
STRUCTURED_CLASSES = [
    "dmr",
    "dvbs2",
    "80211a",
    "80211a_rts",
    "80211a_cts",
    "80211a_ack",
]

NUM_IQ_SAMPLES = 256 * 256
FFT_SIZE = 256


def dataset_metadata() -> dict:
    """Dataset metadata shared by every structured-signal class."""
    return {
        "num_iq_samples_dataset": NUM_IQ_SAMPLES,
        "num_signals_min": 1,
        "num_signals_max": 1,
        "fft_size": FFT_SIZE,
        "fft_stride": FFT_SIZE,
        "sample_rate": 10_000_000,
        "noise_power_db": 0.0,
        "snr_db_min": 25.0,
        "snr_db_max": 35.0,
        "cochannel_overlap_probability": 0.0,
        "signal_duration_in_samples_min": int(NUM_IQ_SAMPLES * 0.9),
        "signal_duration_in_samples_max": NUM_IQ_SAMPLES,
        "bandwidth_min": 2_500_000,
        "bandwidth_max": 3_000_000,
        "signal_center_freq_min": -500_000,
        "signal_center_freq_max": 500_000,
        "frequency_min": -5_000_000,
        "frequency_max": 5_000_000,
    }


def sample_one(class_name: str) -> np.ndarray:
    """Builds a one-class dataset and returns a single IQ example."""
    dataset = TorchSigIterableDataset(
        signal_generators=[class_name],
        metadata=dataset_metadata(),
        target_labels=["class_name"],
        seed=42,
    )
    data, _label = next(iter(dataset))
    return np.asarray(data)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, class_name in zip(axes.ravel(), STRUCTURED_CLASSES):
        iq = sample_one(class_name)
        spectrogram = compute_spectrogram(iq, FFT_SIZE, FFT_SIZE)
        ax.imshow(spectrogram, aspect="auto", cmap="viridis", origin="upper")
        ax.set_title(class_name, fontsize=13)
        ax.set_xlabel("time")
        ax.set_ylabel("frequency")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("TorchSig structured signals — example spectrograms", fontsize=16)
    fig.tight_layout()
    out_path = "examples/structured_signals/structured_signals_spectrograms.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
