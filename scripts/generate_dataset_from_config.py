# ruff: noqa: INP001
"""Generate and write a TorchSig dataset using a configuration YAML file."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

from torchsig.datasets.datasets import SafeTorchSigIterableDataset
from torchsig.signals.signal_lists import FAMILY_SHARED_LIST
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers import (
    HDF5Writer,
    HomogeneousHDF5Writer,
    PackedHDF5Writer,
)
from torchsig.utils.signal_building import lookup_signal_generator_by_string
from torchsig.utils.writer import DatasetCreator, identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

FILE_WRITERS = {
    "legacy": HDF5Writer,
    "packed": PackedHDF5Writer,
    "homogeneous": HomogeneousHDF5Writer,
}


def parse_writer_options(values: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` writer options using YAML scalar syntax."""
    options: dict[str, Any] = {}
    for value in values or []:
        key, separator, raw_value = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid writer option {value!r}; expected KEY=VALUE")
        key = key.strip()
        if key in options:
            raise ValueError(f"Writer option {key!r} was provided more than once")
        options[key] = yaml.safe_load(raw_value)
    return options


def resolve_file_writer(
    config_name: str,
    config_options: dict[str, Any],
    cli_name: str | None,
    cli_options: list[str] | None,
) -> tuple[type, dict[str, Any]]:
    """Resolve writer selection and validate its merged constructor options."""
    writer_name = config_name if cli_name is None else cli_name
    writer = FILE_WRITERS[writer_name]
    options = {**config_options, **parse_writer_options(cli_options)}
    try:
        inspect.signature(writer).bind_partial(root=Path(), **options)
    except TypeError as error:
        raise ValueError(f"Invalid options for {writer.__name__}: {error}") from error
    return writer, options


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-generator command-line parser."""
    parser = argparse.ArgumentParser(description="TorchSig dataset generator script.")
    parser.add_argument("--root", required=True, type=Path, help="Output directory for the generated dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a TorchSig dataset YAML config file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--multithreading", action="store_true")
    parser.add_argument(
        "--signal-weighting",
        "--signal_weighting",
        dest="signal_weighting",
        choices=["per_signal", "per_family"],
        default=None,
        help="Override signal_sampling.mode from YAML.",
    )
    parser.add_argument(
        "--file-writer",
        choices=FILE_WRITERS,
        default=None,
        help="Storage backend; overrides storage.writer from YAML.",
    )
    parser.add_argument(
        "--writer-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Writer constructor option. Repeat to provide multiple options; overrides YAML storage.options.",
    )
    parser.add_argument(
        "--save-config-copy",
        "--save_config_copy",
        dest="save_config_copy",
        action="store_true",
        help="Save a copy of the YAML used into <root>/original_config.yaml",
    )
    return parser


def configure_signal_generators(
    dataset: SafeTorchSigIterableDataset,
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
        fam_gen = lookup_signal_generator_by_string(fam)  # returns ConcatSignalGenerator
        dataset.add_signal_generator(fam_gen, likelihood=1)  # equal likelihood per family


def generate_dataset(argv: Sequence[str] | None = None) -> None:
    """Generate and write the specified dataset.

    Example usage:
        python scripts/generate_dataset_from_config.py \
            --root data/ \
            --config narrowband_all_clean_train.yaml \
            --overwrite \
            --batch-size 64
    """
    args = build_parser().parse_args(argv)

    # load dataset configuration from yaml file
    cfg = load_config_from_yaml(args.config)
    mode = args.signal_weighting or cfg.signal_sampling_mode  # allow command-line override of mode

    # filepaths
    root = args.root / cfg.dataset_id

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
    if cfg.output_representation == "spectrogram":  # typical wideband
        transforms.append(
            Spectrogram(
                fft_size=int(dataset_metadata["fft_size"]),
                fft_stride=int(dataset_metadata.get("fft_stride", dataset_metadata["fft_size"])),
            )
        )
        transforms.append(YOLOLabel())
        target_labels = ["yolo_label"]
    elif cfg.output_representation == "iq":  # typical narrowband
        transforms.append(ComplexTo2D())

    # Dataset construction:
    # - per_signal: initialize with signal_generators="all"
    # - per_family: initialize empty, then add family generators
    signal_generators = "all" if mode == "per_signal" else []
    dataset = SafeTorchSigIterableDataset(
        signal_generators=signal_generators,
        metadata=dataset_metadata,
        transforms=transforms,
        component_transforms=[burst_impairments],
        target_labels=target_labels,
        seed=cfg.seed,
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

    file_writer, writer_options = resolve_file_writer(
        cfg.file_writer_name,
        cfg.file_writer_kwargs,
        args.file_writer,
        args.writer_option,
    )
    creator = DatasetCreator(
        dataloader=dataloader,
        dataset_length=cfg.dataset_length,
        root=root,
        overwrite=args.overwrite,
        multithreading=args.multithreading,
        file_handler=file_writer,
        **writer_options,
    )
    creator.create()

    if args.save_config_copy:
        (root / "original_config.yaml").write_text(args.config.read_text())

    print(f"Generated dataset '{cfg.dataset_id}' into: {root}")


if __name__ == "__main__":
    generate_dataset()
