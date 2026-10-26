"""Demonstrate programmatic and YAML per-signal configuration.

Run from the repository root:

    python examples/scripts/demo_signal_generation_config.py

By default, the YAML configuration and generated dataset are retained under
``examples/datasets/signal_generation_config_demo``. Pass ``--output-dir`` to
choose another location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.experiment_config import (
    ExperimentConfig,
    FixedValue,
    SignalConfig,
    load_experiment_config,
)
from torchsig.utils.writer import DatasetCreator, identity_collate_fn

FIXED_ALPHA = 0.35
NUM_EXAMPLES = 5
DEFAULT_OUTPUT_DIR = Path("examples/datasets/signal_generation_config_demo")


def dataset_metadata() -> dict:
    """Return small metadata values that keep the demonstration quick."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": 256,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 256,
            "signal_duration_in_samples_max": 256,
            "sample_rate": 100,
            "bandwidth_min": 10,
            "bandwidth_max": 10,
        }
    )
    return metadata


def run_demo(output_dir: Path, *, overwrite: bool = False) -> None:
    """Generate configured QPSK examples and persist their effective config."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; choose another directory or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        signals={
            "qpsk": SignalConfig(
                parameters={"alpha": FixedValue(FIXED_ALPHA)},
            )
        }
    )
    config_path = output_dir / "experiment_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    yaml_config = load_experiment_config(config_path)
    if yaml_config.to_dict() != config.to_dict():
        raise RuntimeError("YAML and programmatic configurations did not match")

    dataset = TorchSigIterableDataset(
        signal_generators=["qpsk"],
        metadata=dataset_metadata(),
        target_labels=None,
        experiment_config=config,
        seed=123,
    )

    generator = dataset.signal_generators[0]
    signals = [generator() for _ in range(NUM_EXAMPLES)]
    observed_alphas = [signal.alpha_rolloff for signal in signals]
    observed_shapes = [signal.pulse_shape_name for signal in signals]

    if observed_alphas != [FIXED_ALPHA] * NUM_EXAMPLES:
        raise RuntimeError(f"fixed alpha was not applied: {observed_alphas}")
    if observed_shapes != ["srrc"] * NUM_EXAMPLES:
        raise RuntimeError(f"fixed alpha did not select SRRC: {observed_shapes}")

    generated_root = output_dir / "generated_dataset"
    dataloader = WorkerSeedingDataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=identity_collate_fn,
        seed=123,
    )
    DatasetCreator(
        dataloader=dataloader,
        dataset_length=NUM_EXAMPLES,
        root=generated_root,
        overwrite=overwrite,
        multithreading=False,
    ).create()

    artifact_path = generated_root / "dataset_info.yaml"
    artifact = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    recorded_alpha = artifact["experiment_config"]["signals"]["qpsk"]["parameters"]["alpha_rolloff"]["value"]
    if recorded_alpha != FIXED_ALPHA:
        raise RuntimeError(f"artifact recorded unexpected alpha: {recorded_alpha}")

    print(f"Created programmatic config: {config.to_dict()}")
    print(f"Equivalent YAML configuration: {config_path}")
    print(f"Generated pulse shapes: {observed_shapes}")
    print(f"Generated alpha values: {observed_alphas}")
    print(f"Recorded effective alpha: {recorded_alpha}")
    print(f"Dataset artifact: {artifact_path}")


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(f"Retain configuration and dataset artifacts in this directory (default: {DEFAULT_OUTPUT_DIR})."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in --output-dir.",
    )
    args = parser.parse_args(argv)

    run_demo(args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
