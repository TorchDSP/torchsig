"""Train small models directly from TorchSIG target-label batches.

Run from the repository root:

    python examples/scripts/train_modrec_metadata_targets.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse

import torch
from torch import nn

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.builders.constellation import ConstellationSignalGenerator
from torchsig.signals.builders.ofdm import OFDMSignalGenerator
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults

FFT_SIZE = 32
NUM_IQ_SAMPLES = FFT_SIZE**2
NUM_SUBCARRIERS = 32
MAX_CYCLIC_PREFIX_LEN = NUM_SUBCARRIERS // 2 - 1
BATCH_SIZE = 16


class TwoHeadModel(nn.Module):
    """Small classifier/regressor sharing one feature network."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(FFT_SIZE**2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classification_head = nn.Linear(64, 1)
        self.regression_head = nn.Linear(64, 1)

    def forward(self, spectrogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary-classification logits and a regression prediction."""
        features = spectrogram.float().flatten(start_dim=1)
        features = (features - features.mean(dim=1, keepdim=True)) / features.std(dim=1, keepdim=True).clamp_min(1e-6)
        features = self.features(features)
        return (
            self.classification_head(features).squeeze(1),
            self.regression_head(features).squeeze(1),
        )


def dataset_metadata() -> dict:
    """Return lightweight metadata for a clean training example."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": NUM_IQ_SAMPLES,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "fft_size": FFT_SIZE,
            "fft_stride": FFT_SIZE,
            "sample_rate": 4096,
            "noise_power_db": 0.0,
            "snr_db_min": 40.0,
            "snr_db_max": 40.0,
            "signal_duration_in_samples_min": NUM_IQ_SAMPLES,
            "signal_duration_in_samples_max": NUM_IQ_SAMPLES,
            "bandwidth_min": 1024,
            "bandwidth_max": 1024,
            "signal_center_freq_min": 0.0,
            "signal_center_freq_max": 0.0,
            "frequency_min": -2048.0,
            "frequency_max": 2047.0,
        }
    )
    return metadata


def build_dataloader(
    generator: OFDMSignalGenerator | ConstellationSignalGenerator,
    target_labels: list[str],
    seed: int,
) -> WorkerSeedingDataLoader:
    """Build a TorchSIG loader that returns the requested labels as tensors."""
    dataset = TorchSigIterableDataset(
        signal_generators=[generator],
        metadata=dataset_metadata(),
        transforms=[Spectrogram(fft_size=FFT_SIZE)],
        target_labels=target_labels,
    )
    return WorkerSeedingDataLoader(
        dataset,
        seed=seed,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )


def build_ofdm_dataloader(seed: int) -> WorkerSeedingDataLoader:
    """Return spectrogram, has_cyclic_prefix, and cyclic_prefix_len batches."""
    metadata = dataset_metadata()
    return build_dataloader(
        OFDMSignalGenerator(**metadata, num_subcarriers=NUM_SUBCARRIERS),
        ["has_cyclic_prefix", "cyclic_prefix_len"],
        seed,
    )


def build_constellation_dataloader(seed: int) -> WorkerSeedingDataLoader:
    """Return spectrogram and tensor-ready pulse-shape target batches."""
    metadata = dataset_metadata()
    return build_dataloader(
        ConstellationSignalGenerator(**metadata, constellation_name="qpsk"),
        ["pulse_shape_index", "alpha_rolloff_target"],
        seed,
    )


def train_ofdm(steps: int) -> TwoHeadModel:
    """Train directly on OFDM targets returned by the loader."""
    model = TwoHeadModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    batches = iter(build_ofdm_dataloader(seed=123))
    for _ in range(steps):
        spectrogram, (has_cp, cp_len) = next(batches)
        has_cp_logits, cp_len_prediction = model(spectrogram)
        loss = nn.functional.binary_cross_entropy_with_logits(has_cp_logits, has_cp.float())
        loss += nn.functional.mse_loss(
            cp_len_prediction,
            cp_len.float() / MAX_CYCLIC_PREFIX_LEN,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_constellation(steps: int) -> TwoHeadModel:
    """Train directly on constellation targets returned by the loader."""
    model = TwoHeadModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    batches = iter(build_constellation_dataloader(seed=456))
    for _ in range(steps):
        spectrogram, (pulse_shape, alpha) = next(batches)
        pulse_logits, alpha_prediction = model(spectrogram)
        loss = nn.functional.binary_cross_entropy_with_logits(pulse_logits, pulse_shape.float())
        srrc = pulse_shape.bool()
        if srrc.any():
            loss += nn.functional.mse_loss(alpha_prediction[srrc], alpha.float()[srrc])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def evaluate_ofdm(model: TwoHeadModel, steps: int = 8) -> None:
    """Report metrics from fresh OFDM loader batches."""
    batches = iter(build_ofdm_dataloader(seed=789))
    correct = total = 0
    length_error = 0.0
    with torch.no_grad():
        for _ in range(steps):
            spectrogram, (has_cp, cp_len) = next(batches)
            logits, length = model(spectrogram)
            correct += int(((logits >= 0) == has_cp.bool()).sum())
            total += len(has_cp)
            predicted = (length.clamp(0, 1) * MAX_CYCLIC_PREFIX_LEN).round()
            length_error += (predicted - cp_len).abs().sum().item()
    print(f"cyclic prefix accuracy: {correct / total:.1%}")
    print(f"cyclic prefix length MAE: {length_error / total:.2f} samples")


def evaluate_constellation(model: TwoHeadModel, steps: int = 8) -> None:
    """Report metrics from fresh constellation loader batches."""
    batches = iter(build_constellation_dataloader(seed=987))
    correct = total = alpha_total = 0
    alpha_error = 0.0
    with torch.no_grad():
        for _ in range(steps):
            spectrogram, (pulse_shape, alpha) = next(batches)
            logits, predicted_alpha = model(spectrogram)
            correct += int(((logits >= 0) == pulse_shape.bool()).sum())
            total += len(pulse_shape)
            srrc = pulse_shape.bool()
            alpha_error += (predicted_alpha[srrc] - alpha.float()[srrc]).abs().sum().item()
            alpha_total += int(srrc.sum())
    print(f"pulse shape accuracy: {correct / total:.1%}")
    print(f"alpha roll-off MAE: {alpha_error / alpha_total:.3f}")


def main() -> None:
    """Train both models using only TorchSIG loader outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    torch.manual_seed(0)
    print("Training OFDM targets from target_labels...")
    evaluate_ofdm(train_ofdm(args.steps))
    print("Training constellation targets from target_labels...")
    evaluate_constellation(train_constellation(args.steps))


if __name__ == "__main__":
    main()
