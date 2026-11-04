"""Materialize a static TorchSig dataset through TorchSigDataModule."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.transforms.metadata_transforms import GroupingLabel
from torchsig.utils.file_handlers import HDF5Writer, HomogeneousHDF5Writer, PackedHDF5Writer, PackedNPYWriter
from torchsig.utils.yaml import load_config_from_yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["build_parser", "generate_dataset", "main", "parse_writer_options", "resolve_file_writer"]

FILE_WRITERS = {
    "legacy": HDF5Writer,
    "packed": PackedHDF5Writer,
    "homogeneous": HomogeneousHDF5Writer,
    "packed-npy": PackedNPYWriter,
}


def parse_writer_options(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` options using YAML scalar syntax."""
    options: dict[str, Any] = {}
    for value in values or ():
        key, separator, raw_value = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"invalid writer option {value!r}; expected KEY=VALUE")
        if key in options:
            raise ValueError(f"writer option {key!r} was provided more than once")
        options[key] = yaml.safe_load(raw_value)
    return options


def resolve_file_writer(config_name: str, config_options: dict[str, Any], name: str | None, values: Sequence[str] | None) -> tuple[type, dict[str, Any]]:
    """Resolve a configured or overridden writer and validate its options."""
    writer_name = name or config_name
    writer = FILE_WRITERS[writer_name]
    options = {**config_options, **parse_writer_options(values)}
    try:
        inspect.signature(writer).bind_partial(root=Path(), **options)
    except TypeError as error:
        raise ValueError(f"invalid options for {writer.__name__}: {error}") from error
    return writer, options


def build_parser() -> argparse.ArgumentParser:
    """Build the installed dataset-generation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="TorchSig dataset YAML configuration.")
    parser.add_argument("--output", required=True, type=Path, help="Dataset output directory.")
    parser.add_argument("--dataset-length", type=int, help="Override the configured dataset length.")
    parser.add_argument("--batch-size", type=int, default=1, help="DataModule read batch size.")
    parser.add_argument("--create-batch-size", type=int, default=8, help="Dataset creation batch size.")
    parser.add_argument("--creation-workers", type=int, default=0, help="Dataset creation worker count.")
    parser.add_argument("--file-handler", choices=FILE_WRITERS, help="Override the configured static writer backend.")
    parser.add_argument("--writer-option", action="append", metavar="KEY=VALUE", help="Override a writer constructor option; repeat as needed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset at the output path.")
    parser.add_argument("--signal-generator", action="append", help="Select a signal generator; repeat to select a subset.")
    parser.add_argument("--sampling-grouping", help="GroupingLabel preset or YAML path used for balanced sampling.")
    parser.add_argument("--metadata-grouping", action="append", help="GroupingLabel preset or YAML path applied before target extraction.")
    parser.add_argument("--target-label", action="append", help="Requested generated target label; repeat for multiple labels.")
    return parser


def generate_dataset(argv: Sequence[str] | None = None) -> Path:
    """Parse arguments and materialize a dataset through the DataModule API."""
    args = build_parser().parse_args(argv)
    cfg = load_config_from_yaml(args.config)
    writer, writer_options = resolve_file_writer(cfg.file_writer_name, cfg.file_writer_kwargs, args.file_handler, args.writer_option)
    signal_generators: str | list[str] = args.signal_generator or "all"
    sampling_grouping = None
    if args.sampling_grouping is not None:
        grouping_path = Path(args.sampling_grouping)
        sampling_grouping = grouping_path if grouping_path.is_file() else args.sampling_grouping
    metadata_transforms = []
    for grouping in args.metadata_grouping or ():
        grouping_path = Path(grouping)
        metadata_transforms.append(GroupingLabel(grouping_path if grouping_path.is_file() else grouping))

    datamodule = TorchSigDataModule.from_config(
        cfg,
        root=args.output,
        dataset_size=args.dataset_length,
        batch_size=args.batch_size,
        create_batch_size=args.create_batch_size,
        create_num_workers=args.creation_workers,
        file_writer=writer,
        file_writer_kwargs=writer_options,
        overwrite=args.overwrite,
        target_labels=args.target_label,
        signal_generators=signal_generators,
        sampling_grouping=sampling_grouping,
        metadata_transforms=metadata_transforms,
    )
    datamodule.prepare_data()
    return args.output


def main() -> int:
    """Run the console entry point."""
    output = generate_dataset()
    print(f"Generated dataset at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
