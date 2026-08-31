"""Debug metadata access while generating wideband spectrogram samples.

This example uses TorchSIG's default clean wideband training configuration and
the integrated DataModule creation pipeline. It prints structured metadata
records correlated by session, sample, worker, and pipeline stage. It is
intended for diagnosing metadata inheritance and access; it is not a
general-purpose performance profiler.

Example:
    python examples/scripts/dev/debug_wideband_pipeline.py --num-samples 2
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.metadata_logging import MetadataDebugFormatter
from torchsig.utils.writer import identity_collate_fn

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
)
DEBUG_KEYS = {
    "bandwidth_max",
    "bandwidth_min",
    "cochannel_overlap_probability",
    "fft_size",
    "fft_stride",
    "frequency_max",
    "frequency_min",
    "num_iq_samples_dataset",
    "num_signals_max",
    "num_signals_min",
    "sample_rate",
    "signal_center_freq_max",
    "signal_center_freq_min",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the wideband debug run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Wideband dataset YAML configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of spectrogram samples to generate.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum number of metadata records to emit.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory in which to keep the generated static dataset.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_events < 0:
        parser.error("--max-events cannot be negative")
    return args


def configure_metadata_logger() -> None:
    """Send structured TorchSIG metadata records to the console."""
    handler = logging.StreamHandler()
    handler.setFormatter(MetadataDebugFormatter())

    logger = logging.getLogger("torchsig.metadata")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def build_wideband_datamodule(
    config_path: Path,
    root: Path,
    num_samples: int,
    max_events: int,
) -> TorchSigDataModule:
    """Build a debug-enabled DataModule from a wideband YAML config."""
    return TorchSigDataModule.from_config(
        cfg=config_path,
        root=root,
        dataset_size=num_samples,
        dataset_splits=[1.0, 0.0, 0.0],
        create_batch_size=1,
        create_num_workers=0,
        overwrite=True,
        collate_fn=identity_collate_fn,
        metadata_debug={
            "keys": DEBUG_KEYS,
            "events": {"lookup"},
            "max_events": max_events,
            "include_values": True,
            "value_repr_limit": 80,
        },
    )


def main() -> None:
    """Generate wideband samples while reporting metadata resolution."""
    args = parse_args()
    configure_metadata_logger()
    root_context = (
        nullcontext(args.root)
        if args.root is not None
        else TemporaryDirectory(prefix="torchsig-wideband-debug-")
    )
    with root_context as root:
        datamodule = build_wideband_datamodule(
            args.config,
            Path(root),
            args.num_samples,
            args.max_events,
        )
        print(f"Creating debug dataset at {root}", flush=True)
        datamodule.prepare_data()


if __name__ == "__main__":
    main()
