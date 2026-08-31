"""Train a tiny classifier from a TorchSIG loader and string grouping YAML.

Run from the repository root:

    python examples/scripts/train_grouping_string_classifier.py
"""

# ruff: noqa: INP001

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.metadata_transforms import GroupingLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults

CONFIG_PATH = Path(__file__).with_name("grouping_string.yaml")
FFT_SIZE = 32
BATCH_SIZE = 8
TRAINING_STEPS = 60
VALIDATION_STEPS = 5


def dataset_metadata() -> dict:
    """Return lightweight metadata suitable for a quick CPU example."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": FFT_SIZE**2,
            "fft_size": FFT_SIZE,
            "fft_stride": FFT_SIZE,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "sample_rate": 20_000,
            "frequency_min": -10_000,
            "frequency_max": 9_999,
            "signal_center_freq_min": -4_000,
            "signal_center_freq_max": 4_000,
            "signal_duration_in_samples_min": FFT_SIZE**2,
            "signal_duration_in_samples_max": FFT_SIZE**2,
            "bandwidth_min": 1_000,
            "bandwidth_max": 4_000,
            "snr_db_min": 15,
            "snr_db_max": 30,
        }
    )
    return metadata


def build_dataloader(seed: int) -> tuple[WorkerSeedingDataLoader, GroupingLabel]:
    """Build a TorchSIG dataset whose target is produced by GroupingLabel."""
    grouping = GroupingLabel(CONFIG_PATH)
    dataset = TorchSigIterableDataset(
        signal_generators=[
            "bpsk",
            "qpsk",
            "2fsk",
            "4gfsk",
            "tone",
        ],
        metadata=dataset_metadata(),
        transforms=[
            Spectrogram(fft_size=FFT_SIZE),
            grouping,
        ],
        target_labels=[grouping.index_label],
        sampling_grouping=grouping,
    )
    dataloader = WorkerSeedingDataLoader(
        dataset,
        seed=seed,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    return dataloader, grouping


def prepare_features(spectrograms: torch.Tensor) -> torch.Tensor:
    """Normalize and flatten a batch of TorchSIG spectrograms."""
    features = spectrograms.float().flatten(start_dim=1)
    means = features.mean(dim=1, keepdim=True)
    standard_deviations = features.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (features - means) / standard_deviations


def train_classifier(
    dataloader: WorkerSeedingDataLoader,
    num_groups: int,
) -> nn.Module:
    """Train a small classifier on batches produced by the TorchSIG loader."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(FFT_SIZE**2, 32),
        nn.ReLU(),
        nn.Linear(32, num_groups),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    batches = iter(dataloader)

    for _ in range(TRAINING_STEPS):
        spectrograms, group_indices = next(batches)
        logits = model(prepare_features(spectrograms))
        loss = nn.functional.cross_entropy(logits, group_indices.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def evaluate_classifier(
    model: nn.Module,
    dataloader: WorkerSeedingDataLoader,
) -> float:
    """Evaluate on fresh TorchSIG batches."""
    correct = 0
    total = 0
    batches = iter(dataloader)

    with torch.no_grad():
        for _ in range(VALIDATION_STEPS):
            spectrograms, group_indices = next(batches)
            predictions = model(prepare_features(spectrograms)).argmax(dim=1)
            correct += int((predictions == group_indices).sum())
            total += len(group_indices)

    return correct / total


def main() -> None:
    """Train using YAML-produced labels returned by TorchSIG data loaders."""
    train_loader, grouping = build_dataloader(seed=123)
    validation_loader, _ = build_dataloader(seed=456)
    group_names = [group["name"] for group in grouping.groups]

    first_spectrograms, first_targets = next(iter(train_loader))
    print(f"Loader data shape: {tuple(first_spectrograms.shape)}")
    print(f"Loader grouped targets: {first_targets.tolist()}")
    print(f"Target groups: {group_names}")

    model = train_classifier(train_loader, len(group_names))
    accuracy = evaluate_classifier(model, validation_loader)
    print(f"Validation accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
