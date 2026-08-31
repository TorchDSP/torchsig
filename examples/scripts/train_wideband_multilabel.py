"""Train a small PyTorch multilabel classifier on generated wideband data.

Each wideband spectrogram can contain several signal classes. ``MultiHotLabel``
converts their component-level class indices into one sample-level target vector,
and the model is trained with binary cross entropy.

This is a compact usage example, not a reference model or training recipe.

Example:
    python examples/scripts/train_wideband_multilabel.py --steps 20
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse

import torch
from torch import nn

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.metadata_transforms import MultiHotLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.signal_building import lookup_signal_generator_by_string

CLASS_NAMES = ["bpsk", "qpsk", "8psk", "16qam", "fm"]
LOG_INTERVAL = 5
DEFAULT_PREDICTION_THRESHOLD = 0.45


class SpectrogramClassifier(nn.Module):
    """Small convolutional network that predicts independent class logits."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the classifier.

        Args:
            num_classes: Number of independently predicted signal classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Return one logit per class for each spectrogram.

        Args:
            spectrograms: Batch with shape ``(batch, frequency, time)``.

        Returns:
            Logits with shape ``(batch, num_classes)``.
        """
        means = spectrograms.mean(dim=(-2, -1), keepdim=True)
        standard_deviations = spectrograms.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized = (spectrograms - means) / standard_deviations
        features = self.features(normalized.unsqueeze(1)).flatten(1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help=("Probability at or above which a class is predicted. This should normally be tuned on a validation set."),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def build_dataloader(
    batch_size: int,
    num_workers: int,
    seed: int,
) -> WorkerSeedingDataLoader:
    """Build a loader of spectrograms and sample-level multi-hot targets."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": 4096,
            "num_signals_min": 2,
            "num_signals_max": 4,
            "fft_size": 64,
            "fft_stride": 64,
            "signal_duration_in_samples_min": 512,
            "signal_duration_in_samples_max": 1024,
            "bandwidth_min": 250_000,
            "bandwidth_max": 1_000_000,
            "signal_center_freq_min": -4_000_000,
            "signal_center_freq_max": 4_000_000,
            "frequency_min": -5_000_000,
            "frequency_max": 4_999_999,
        }
    )
    dataset = TorchSigIterableDataset(
        signal_generators=[],
        metadata=metadata,
        transforms=[
            Spectrogram(fft_size=64, fft_stride=64),
            MultiHotLabel(),
        ],
        target_labels=["multi_hot_label"],
        seed=seed,
    )
    for class_name in CLASS_NAMES:
        dataset.add_signal_generator(
            lookup_signal_generator_by_string(class_name),
            class_name=class_name,
        )
    return WorkerSeedingDataLoader(
        dataset,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def train(
    model: nn.Module,
    dataloader: WorkerSeedingDataLoader,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    """Train the model for a fixed number of batches."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    batches = iter(dataloader)

    for step in range(1, steps + 1):
        spectrograms, targets = next(batches)
        spectrograms = spectrograms.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        loss = criterion(model(spectrograms), targets)
        loss.backward()
        optimizer.step()

        if step in (1, steps) or step % LOG_INTERVAL == 0:
            print(f"step {step:>3}/{steps}: loss={loss.item():.4f}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: WorkerSeedingDataLoader,
    num_batches: int,
    threshold: float,
    device: torch.device,
) -> None:
    """Report label accuracy and display one multilabel prediction."""
    model.eval()
    correct_labels = 0
    total_labels = 0
    example_targets = None
    example_predictions = None
    example_probabilities = None
    batches = iter(dataloader)

    for _ in range(num_batches):
        spectrograms, targets = next(batches)
        spectrograms = spectrograms.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        probabilities = model(spectrograms).sigmoid()
        predictions = probabilities >= threshold
        correct_labels += int((predictions == targets.bool()).sum())
        total_labels += targets.numel()
        example_targets = targets[0].bool().cpu()
        example_predictions = predictions[0].cpu()
        example_probabilities = probabilities[0].cpu()

    accuracy = correct_labels / total_labels
    true_names = [name for name, present in zip(CLASS_NAMES, example_targets, strict=True) if present]
    predicted_names = [name for name, present in zip(CLASS_NAMES, example_predictions, strict=True) if present]
    probability_text = ", ".join(
        f"{name}={probability:.3f}"
        for name, probability in zip(
            CLASS_NAMES,
            example_probabilities,
            strict=True,
        )
    )
    print(f"evaluation label accuracy: {accuracy:.3f}")
    print(f"prediction threshold:      {threshold:.2f}")
    print(f"example true classes:      {true_names}")
    print(f"example predicted classes: {predicted_names}")
    print(f"example probabilities:     {probability_text}")


def main() -> None:
    """Train and evaluate the example multilabel classifier."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = SpectrogramClassifier(num_classes=len(CLASS_NAMES)).to(device)

    print(f"device: {device}")
    print(f"classes: {CLASS_NAMES}")
    train(model, dataloader, args.steps, args.learning_rate, device)
    evaluate(model, dataloader, args.eval_batches, args.threshold, device)


if __name__ == "__main__":
    main()
