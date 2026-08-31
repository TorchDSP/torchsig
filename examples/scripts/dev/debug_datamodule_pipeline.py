"""Demonstrate metadata debugging through a TorchSig DataModule pipeline.

Run from the repository root:

    python examples/scripts/dev/debug_datamodule_pipeline.py

Pass ``--root PATH`` to keep the generated static dataset for inspection.
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


class MetadataLogFormatter(logging.Formatter):
    """Render the useful structured fields attached to metadata log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one JSON object per metadata event."""
        correlation = getattr(record, "metadata_correlation_fields", {})
        payload = {
            "event": getattr(record, "metadata_event", None),
            "dataset_id": getattr(record, "metadata_dataset_id", None),
            "stage": correlation.get("stage"),
            "sample_index": getattr(record, "metadata_sample_index", None),
            "worker_id": getattr(record, "metadata_worker_id", None),
            "message": record.getMessage(),
        }
        snapshot = getattr(record, "metadata_snapshot", None)
        components = getattr(record, "metadata_component_snapshots", None)
        if snapshot is not None:
            payload["metadata"] = snapshot
        if components is not None:
            payload["components"] = components
        return json.dumps(payload, default=str)


def configure_metadata_logging() -> None:
    """Send structured TorchSig metadata records to the console."""
    logger = logging.getLogger("torchsig.metadata")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(MetadataLogFormatter())
    logger.handlers[:] = [handler]


def small_dataset_metadata() -> dict:
    """Return metadata for a quick, deterministic demonstration dataset."""
    fft_size = 32
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "sample_rate": 20_000,
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": fft_size**2,
            "signal_duration_in_samples_max": fft_size**2,
            "bandwidth_min": 500,
            "bandwidth_max": 4_000,
            "signal_center_freq_min": -4_000,
            "signal_center_freq_max": 4_000,
            "frequency_min": -10_000,
            "frequency_max": 9_999,
        }
    )
    return metadata


def run(root: Path) -> None:
    """Create a debug-enabled static dataset and consume one loader batch."""
    datamodule = TorchSigDataModule(
        root=str(root),
        metadata=small_dataset_metadata(),
        dataset_size=2,
        dataset_splits=[1, 1, 0],
        batch_size=1,
        num_workers=0,
        create_batch_size=1,
        create_num_workers=0,
        overwrite=True,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        target_labels=["class_name"],
        seed=42,
        metadata_debug={
            "events": {"snapshot"},
            "include_values": True,
            "max_events": 10,
            "value_repr_limit": 120,
        },
    )

    print(f"Creating debug dataset at {root}")
    datamodule.prepare_data()
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    print(f"Loaded one training batch containing {len(batch)} sample(s)")


def main() -> None:
    """Parse command-line arguments and run the demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory in which to keep the generated dataset.",
    )
    args = parser.parse_args()

    root_context = nullcontext(args.root) if args.root is not None else TemporaryDirectory(prefix="torchsig-metadata-debug-")
    with root_context as root:
        configure_metadata_logging()
        run(Path(root))


if __name__ == "__main__":
    main()
