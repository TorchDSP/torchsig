"""Generate and write a TorchSig dataset using a configuration YAML file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

from torchsig.utils.defaults import TorchSigDefaults
from torchsig.signals.signal_lists import FAMILY_SHARED_LIST
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.signal_building import lookup_signal_generator_by_string
from torchsig.utils.writer import DatasetCreator, identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml


def configure_signal_generators(
    dataset: TorchSigIterableDataset,
    mode: Literal["per_signal", "per_family"],
) -> None:
    """Configure dataset signal placement probabilities. This function adjusts the signal generator
    probabilities in-place within the dataset based on the specified signal sampling mode. The two 
    modes are defined as follows:

    "per_signal":
    - equal probability per individual signal generator
    - implemented by initializing dataset with signal_generators="all"
        which expands to all base generators

    "per_family":
    - equal probability per family, uniform signal weights inside family
    - implemented by adding one ConcatSignalGenerator per family
        and assigning equal top-level likelihood. ConcatSignalGenerator
        is uniform across wrapped generators

    Args:
        dataset: The TorchSigIterableDataset instance to configure.
        mode: The signal sampling mode, which can be either "per_signal" or "per_family".
    """
    if mode == "per_signal":
        # If dataset was created with signal_generators="all", nothing more to do.
        return

    # per_family
    dataset.signal_generators = []
    dataset.signal_likelihoods = []
    dataset.signal_probabilities = []
    dataset.total_likelihood = 0

    for fam in FAMILY_SHARED_LIST:
        fam_gen = lookup_signal_generator_by_string(fam)    # returns ConcatSignalGenerator
        dataset.add_signal_generator(fam_gen, likelihood=1) # equal likelihood per family


def generate_dataset() -> None:
    """Generate and write the specified dataset.

    Example usage:
        python3 generate_official_dataset.py --root data/ --config narrowband_all_clean_train.yaml --overwrite --batch_size 64

    """
    p = argparse.ArgumentParser(description="TorchSig dataset generator script.")
    p.add_argument("--root", required=True, type=Path, help="Output directory for the generated dataset.")
    p.add_argument("--config", required=True, type=Path, help="Path to a TorchSig dataset YAML config file.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--multithreading", action="store_true")
    p.add_argument(
        "--signal_weighting",
        choices=["per_signal", "per_family"],
        default=None,
        help="Override signal_sampling.mode from YAML.",
    )
    p.add_argument("--save_config_copy", action="store_true",
        help="Save a copy of the YAML used into <root>/original_config.yaml")
    args = p.parse_args()

    # load dataset configuration from yaml file
    cfg = load_config_from_yaml(args.config)
    mode = args.signal_weighting or cfg.signal_sampling_mode # allow command-line override of mode

    # filepaths
    root = os.path.join(args.root, cfg.dataset_id)

    # build metadata from TorchSigDefaults plus YAML configuration overrides
    base = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base)
    dataset_metadata.update(cfg.dataset_metadata)

    # transforms, based on Impairment level and output format
    impairments = Impairments(level=cfg.impairment_level)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms
    transforms = [whole_signal_impairments]

    target_labels = None
    if cfg.output_representation == "spectrogram": # typical wideband
        transforms.append(Spectrogram(fft_size=int(dataset_metadata["fft_size"])))
        transforms.append(YOLOLabel())
        target_labels=["yolo_label"],  # yolo labels
    elif cfg.output_representation == "iq": # typical narrowband
        transforms.append(ComplexTo2D())

    # Dataset construction:
    # - per_signal: initialize with signal_generators="all"
    # - per_family: initialize empty, then add family generators
    signal_generators = "all" if mode == "per_signal" else []
    dataset = TorchSigIterableDataset(
        signal_generators=signal_generators,
        metadata=dataset_metadata,
        transforms=transforms,
        component_transforms=[burst_impairments],
        target_labels=target_labels,
    )
    configure_signal_generators(dataset, mode)

    # DataLoader: identity_collate_fn is stable for Signal objects.
    # NOTE: num_workers>0 may be constrained by platform/pickling; default 0.
    dataloader = WorkerSeedingDataLoader(
        dataset,
        seed=cfg.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=identity_collate_fn,
    )

    creator = DatasetCreator(
        dataloader=dataloader,
        dataset_length=cfg.dataset_length,
        root=root,
        overwrite=args.overwrite,
        multithreading=args.multithreading,
    )
    creator.create()

    if args.save_config_copy:
        (root / "original_config.yaml").write_text(args.config.read_text())

    print(f"Generated dataset '{cfg.dataset_id}' into: {root}")


if __name__ == "__main__":
    generate_dataset()
