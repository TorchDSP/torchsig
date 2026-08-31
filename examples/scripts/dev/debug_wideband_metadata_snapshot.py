"""Log completed wideband metadata once per generated spectrogram.

Unlike event-oriented metadata debugging, this example uses the integrated
DataModule creation pipeline and emits no lookup, set, or delete records. Each
``snapshot`` record contains the completed sample metadata and a separate
metadata mapping for every component signal. Sample arrays are represented
only by shape and dtype.

Example:
    python examples/scripts/dev/debug_wideband_metadata_snapshot.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.metadata_logging import MetadataSnapshotFormatter
from torchsig.utils.writer import identity_collate_fn

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse options for the snapshot-only wideband debug run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Wideband spectrogram YAML configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completed metadata snapshots to emit.",
    )
    parser.add_argument(
        "--value-repr-limit",
        type=int,
        default=160,
        help="Maximum representation length for each metadata value.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory in which to keep the generated static dataset.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.value_repr_limit < 1:
        parser.error("--value-repr-limit must be positive")
    return args


def configure_snapshot_logger() -> None:
    """Configure console output for structured metadata snapshots."""
    handler = logging.StreamHandler()
    handler.setFormatter(MetadataSnapshotFormatter())

    logger = logging.getLogger("torchsig.metadata")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def build_wideband_datamodule(
    config_path: Path,
    root: Path,
    num_samples: int,
    value_repr_limit: int,
) -> TorchSigDataModule:
    """Build a snapshot-enabled DataModule from a wideband YAML config."""
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
            "events": {"snapshot"},
            "max_events": num_samples,
            "include_values": True,
            "value_repr_limit": value_repr_limit,
        },
    )


def main() -> None:
    """Generate samples and emit one completed metadata snapshot per sample."""
    args = parse_args()
    configure_snapshot_logger()
    root_context = (
        nullcontext(args.root)
        if args.root is not None
        else TemporaryDirectory(prefix="torchsig-wideband-snapshot-")
    )
    with root_context as root:
        datamodule = build_wideband_datamodule(
            args.config,
            Path(root),
            args.num_samples,
            args.value_repr_limit,
        )
        print(f"Creating snapshot dataset at {root}", flush=True)
        datamodule.prepare_data()


if __name__ == "__main__":
    main()
