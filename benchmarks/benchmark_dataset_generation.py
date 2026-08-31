"""End-to-end benchmarks for TorchSIG's default training-data pipelines."""

# ruff: noqa: INP001

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from torchsig.datasets.datasets import SafeTorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

DEFAULT_CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "torchsig" / "datasets" / "default_configs"
DEFAULT_TRAINING_CONFIGS = (
    "narrowband_clean_train_all.yaml",
    "narrowband_impaired_train_all.yaml",
    "wideband_clean_train_all.yaml",
    "wideband_impaired_train_all.yaml",
)
BENCHMARK_ROUNDS = 10


def _build_training_dataloader(
    config_name: str,
) -> tuple[WorkerSeedingDataLoader, str]:
    """Build the normal in-memory generation pipeline for one default config."""
    cfg = load_config_from_yaml(DEFAULT_CONFIG_DIRECTORY / config_name)
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(cfg.dataset_metadata)

    impairments = Impairments(level=cfg.impairment_level)
    transforms = [impairments.dataset_transforms]
    target_labels = []

    if cfg.output_representation == "iq":
        transforms.append(ComplexTo2D())
    else:
        transforms.extend(
            [
                Spectrogram(fft_size=int(metadata["fft_size"])),
                YOLOLabel(),
            ]
        )
        target_labels = ["yolo_label"]

    dataset = SafeTorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=transforms,
        component_transforms=[impairments.signal_transforms],
        target_labels=target_labels,
        sampling_grouping=("family" if cfg.signal_sampling_mode == "per_family" else None),
    )
    dataloader = WorkerSeedingDataLoader(
        dataset,
        seed=cfg.seed,
        batch_size=1,
        num_workers=0,
        collate_fn=identity_collate_fn,
    )
    dataloader.seed(cfg.seed)
    return dataloader, cfg.output_representation


def _extract_data(sample: Any) -> Any:
    """Extract generated sample data from identity-collated loader output."""
    item = sample[0]
    return item[0] if isinstance(item, tuple) else item


@pytest.mark.benchmark
@pytest.mark.parametrize("config_name", DEFAULT_TRAINING_CONFIGS)
def benchmark_default_training_data_generation(
    benchmark,
    config_name: str,
) -> None:
    """Benchmark one generated sample through the default training pipeline."""
    dataloader, output_representation = _build_training_dataloader(config_name)
    batches = iter(dataloader)

    result = benchmark.pedantic(
        next,
        args=(batches,),
        iterations=1,
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=1,
    )

    data = _extract_data(result)
    if output_representation == "iq":
        assert data.shape == (2, 4096)
    else:
        assert data.shape == (512, 512)
