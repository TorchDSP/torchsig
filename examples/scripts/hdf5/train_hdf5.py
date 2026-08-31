"""Generate HDF5 data, read it back, and train a small classifier.

The defaults are intentionally small and CPU-friendly. Run from the repository
root with either::

    python examples/scripts/train_hdf5.py --file-type packed
    python examples/scripts/train_hdf5.py --file-type homogeneous
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigIterableDataset
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)
from torchsig.utils.signal_building import lookup_signal_generator_by_string
from torchsig.utils.writer import DatasetCreator, identity_collate_fn

CLASS_NAMES = ("tone", "bpsk", "qpsk", "2fsk")
MIN_SAMPLES = 16

FileType = Literal["packed", "homogeneous"]


class TinyIQClassifier(nn.Module):
    """Small convolutional classifier for two-channel I/Q arrays."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of two-channel I/Q arrays."""
        return self.network(inputs)


def parse_args() -> argparse.Namespace:
    """Parse generation, storage, and training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-type",
        choices=("packed", "homogeneous"),
        default="homogeneous",
        help="HDF5 storage format to use (default: homogeneous).",
    )
    parser.add_argument("--root", type=Path, help="Keep the generated dataset here.")
    parser.add_argument("--samples", type=int, default=1_024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.samples < MIN_SAMPLES:
        parser.error(f"--samples must be at least {MIN_SAMPLES}")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.root is not None and args.root.exists() and not args.overwrite:
        parser.error("--root exists; pass --overwrite to replace it")

    return args


def dataset_metadata() -> dict:
    """Return metadata for a small four-class narrowband dataset."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "sample_rate": 1_000_000,
            "num_iq_samples_dataset": 1_024,
            "fft_size": 32,
            "fft_stride": 32,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 768,
            "signal_duration_in_samples_max": 1_024,
            "bandwidth_min": 100_000,
            "bandwidth_max": 200_000,
            "signal_center_freq_min": -200_000,
            "signal_center_freq_max": 200_000,
            "frequency_min": -500_000,
            "frequency_max": 499_999,
        }
    )
    return metadata


def file_handler_config(
    file_type: FileType,
    batch_size: int,
) -> tuple[type, type, dict[str, Any]]:
    """Return reader, writer, and creation options for an HDF5 format."""
    if file_type == "packed":
        return (
            PackedHDF5Reader,
            PackedHDF5Writer,
            {"max_batches_in_memory": 1},
        )

    return (
        HomogeneousHDF5Reader,
        HomogeneousHDF5Writer,
        {"chunk_samples": batch_size},
    )


def write_dataset(
    root: Path,
    sample_count: int,
    batch_size: int,
    file_type: FileType,
) -> None:
    """Generate signals and write them using the selected HDF5 format."""
    generators = [
        lookup_signal_generator_by_string(name) for name in CLASS_NAMES
    ]
    source = TorchSigIterableDataset(
        signal_generators=generators,
        metadata=dataset_metadata(),
        transforms=[ComplexTo2D()],
        target_labels=None,
        seed=42,
    )
    creation_loader = WorkerSeedingDataLoader(
        source,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=identity_collate_fn,
        seed=42,
    )
    reader_class, writer_class, writer_options = file_handler_config(
        file_type,
        batch_size,
    )
    DatasetCreator(
        dataloader=creation_loader,
        dataset_length=sample_count,
        root=root,
        overwrite=True,
        file_handler=writer_class,
        file_reader=reader_class,
        compression=None,
        **writer_options,
    ).create()


def make_loaders(
    root: Path,
    sample_count: int,
    batch_size: int,
    file_type: FileType,
) -> tuple[DataLoader, DataLoader, DataLoader, StaticTorchSigDataset]:
    """Reopen the HDF5 dataset and create deterministic data splits."""
    reader_class, _, _ = file_handler_config(file_type, batch_size)
    dataset = StaticTorchSigDataset(
        root=root,
        file_handler_class=reader_class,
        target_labels=["class_index"],
    )

    train_size = int(sample_count * 0.75)
    validation_size = max(1, int(sample_count * 0.125))
    test_size = sample_count - train_size - validation_size

    splits = random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    loaders = tuple(
        DataLoader(split, batch_size=batch_size, shuffle=index == 0)
        for index, split in enumerate(splits)
    )
    return (*loaders, dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    """Return mean cross-entropy loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.float()
            targets = targets.long()
            logits = model(inputs)
            total_loss += nn.functional.cross_entropy(
                logits,
                targets,
                reduction="sum",
            ).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.numel()

    return total_loss / count, correct / count


def train(
    root: Path,
    sample_count: int,
    epochs: int,
    batch_size: int,
    file_type: FileType,
) -> None:
    """Write, reopen, train, and evaluate an HDF5 dataset."""
    torch.manual_seed(42)
    print(f"Writing {sample_count} samples to {file_type} HDF5 at {root}")

    write_dataset(root, sample_count, batch_size, file_type)
    train_loader, validation_loader, test_loader, dataset = make_loaders(
        root,
        sample_count,
        batch_size,
        file_type,
    )
    print(f"Read {len(dataset)} samples with {type(dataset.reader).__name__}")

    model = TinyIQClassifier(len(CLASS_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(
                    model(inputs.float()),
                    targets.long(),
                )
                loss.backward()
                optimizer.step()

            validation_loss, validation_accuracy = evaluate(
                model,
                validation_loader,
            )
            print(
                f"epoch={epoch} validation_loss={validation_loss:.4f} "
                f"validation_accuracy={validation_accuracy:.3f}"
            )

        test_loss, test_accuracy = evaluate(model, test_loader)
        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.3f}")
    finally:
        dataset.reader.teardown()


def main() -> None:
    """Run the complete HDF5 training pipeline."""
    args = parse_args()
    file_type: FileType = args.file_type

    root_context = (
        nullcontext(args.root)
        if args.root is not None
        else TemporaryDirectory(prefix=f"torchsig-{file_type}-training-")
    )

    with root_context as root:
        train(
            Path(root),
            args.samples,
            args.epochs,
            args.batch_size,
            file_type,
        )


if __name__ == "__main__":
    main()