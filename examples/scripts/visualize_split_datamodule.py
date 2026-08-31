"""Generate a small TorchSig dataset and visualize its spectrograms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsig.datasets.datamodules import SplitTorchSigDataModule
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


def build_config(
    *,
    dataset_id: str,
    dataset_length: int,
    seed: int,
    fft_size: int,
) -> TorchSigDatasetConfig:
    """Create a small spectrogram-output TorchSig dataset config."""
    num_iq_samples = fft_size**2

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": num_iq_samples,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 2,
            "signal_duration_in_samples_min": int(
                0.25 * num_iq_samples
            ),
            "signal_duration_in_samples_max": int(
                0.75 * num_iq_samples
            ),
        }
    )

    return TorchSigDatasetConfig(
        dataset_id=dataset_id,
        dataset_length=dataset_length,
        dataset_metadata=metadata,
        output_representation="spectrogram",
        output_spectrogram_fft=fft_size,
        signal_sampling_mode="random",
        impairment_level=0,
        seed=seed,
    )


def extract_sample(sample: Any) -> tuple[np.ndarray, Any]:
    """Extract spectrogram data and a target from a dataset sample.

    This accommodates several common TorchSig/static-dataset return formats.
    """
    if hasattr(sample, "data"):
        return np.asarray(sample.data), getattr(sample, "metadata", None)

    if isinstance(sample, dict):
        data = sample.get("data", sample.get("samples"))
        target = sample.get(
            "target",
            sample.get("targets", sample.get("metadata")),
        )

        if data is None:
            raise TypeError(
                f"Could not locate sample data in dictionary keys: "
                f"{list(sample)}"
            )

        return np.asarray(data), target

    if isinstance(sample, (tuple, list)):
        if not sample:
            raise ValueError("Received an empty sample.")

        data = sample[0]
        target = sample[1] if len(sample) > 1 else None

        if hasattr(data, "data") and not isinstance(data, np.ndarray):
            data = data.data

        return np.asarray(data), target

    raise TypeError(
        f"Unsupported dataset sample type: {type(sample).__name__}"
    )


def normalize_spectrogram_shape(
    spectrogram: np.ndarray,
) -> np.ndarray:
    """Convert a spectrogram to a two-dimensional array for plotting."""
    spectrogram = np.asarray(spectrogram)

    # Remove singleton batch/channel dimensions.
    spectrogram = np.squeeze(spectrogram)

    if spectrogram.ndim == 2:
        return spectrogram

    # Some transforms return channel-first arrays such as (1, H, W).
    if spectrogram.ndim == 3 and spectrogram.shape[0] in (1, 2, 3):
        return spectrogram[0]

    # Some transforms may return channel-last arrays.
    if spectrogram.ndim == 3 and spectrogram.shape[-1] in (1, 2, 3):
        return spectrogram[..., 0]

    raise ValueError(
        "Expected a two-dimensional spectrogram or an array with a "
        f"singleton/channel dimension, but received shape "
        f"{spectrogram.shape}."
    )


def target_summary(target: Any) -> str:
    """Create a compact target description for a plot title."""
    if target is None:
        return ""

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if isinstance(target, np.ndarray):
        return f"target shape={target.shape}"

    if isinstance(target, dict):
        return f"target keys={list(target)}"

    text = str(target)
    return text if len(text) <= 80 else f"{text[:77]}..."


def visualize_split(
    dataset: Any,
    split_name: str,
    *,
    num_samples: int,
    save_directory: Path | None,
) -> None:
    """Plot several spectrograms from one dataset split."""
    count = min(num_samples, len(dataset))

    if count == 0:
        print(f"{split_name}: no samples available")
        return

    figure, axes = plt.subplots(
        1,
        count,
        figsize=(5 * count, 4),
        squeeze=False,
    )

    for index in range(count):
        sample = dataset[index]
        data, target = extract_sample(sample)
        spectrogram = normalize_spectrogram_shape(data)

        axis = axes[0, index]
        image = axis.imshow(
            spectrogram,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        title = f"{split_name} sample {index}"
        summary = target_summary(target)

        if summary:
            title = f"{title}\n{summary}"

        axis.set_title(title)
        axis.set_xlabel("Time bin")
        axis.set_ylabel("Frequency bin")
        figure.colorbar(image, ax=axis, shrink=0.8)

    figure.tight_layout()

    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            save_directory
            / f"{split_name.lower()}_spectrograms.png"
        )
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")

    plt.show()
    plt.close(figure)


def main() -> None:
    """Generate datasets and display sample spectrograms."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small split TorchSig dataset and visualize "
            "sample spectrograms."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/split_datamodule_demo"),
        help="Parent directory for generated datasets.",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=64,
        help="FFT size used to create spectrograms.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=3,
        help="Number of spectrograms to visualize per split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated datasets.",
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Optional directory in which to save plot images.",
    )
    args = parser.parse_args()

    dataset_id = "small_spectrogram_demo"

    train_cfg = build_config(
        dataset_id=dataset_id,
        dataset_length=8,
        seed=11,
        fft_size=args.fft_size,
    )
    val_cfg = build_config(
        dataset_id=dataset_id,
        dataset_length=4,
        seed=22,
        fft_size=args.fft_size,
    )
    test_cfg = build_config(
        dataset_id=dataset_id,
        dataset_length=4,
        seed=33,
        fft_size=args.fft_size,
    )

    datamodule = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=args.root,
        batch_size=2,
        num_workers=0,
        create_batch_size=2,
        create_num_workers=0,
        overwrite=args.overwrite,
        collate_fn=identity_collate_fn,
        target_labels=["yolo_label"],
    )

    print(f"Generating dataset under {datamodule.root}")
    datamodule.prepare_data()
    datamodule.setup(None)

    assert datamodule.train is not None
    assert datamodule.val is not None
    assert datamodule.test is not None

    print(
        "Dataset lengths:",
        f"train={len(datamodule.train)},",
        f"val={len(datamodule.val)},",
        f"test={len(datamodule.test)}",
    )

    visualize_split(
        datamodule.train,
        "Train",
        num_samples=args.samples_per_split,
        save_directory=args.save_plots,
    )
    visualize_split(
        datamodule.val,
        "Validation",
        num_samples=args.samples_per_split,
        save_directory=args.save_plots,
    )
    visualize_split(
        datamodule.test,
        "Test",
        num_samples=args.samples_per_split,
        save_directory=args.save_plots,
    )


if __name__ == "__main__":
    main()
