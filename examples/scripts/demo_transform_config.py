"""Run a small impairment ablation study from declarative YAML.

Run from the repository root:

    python examples/scripts/demo_transform_config.py

The demo compares a clean control and the full configured impairment pipeline
with leave-one-transform-out variants.  Real studies can use the same pattern
to generate named, fingerprinted experimental conditions from configuration.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from torchsig.signals.signal_types import Signal
from torchsig.transforms.base_transforms import Compose
from torchsig.transforms.config import TransformSpec
from torchsig.utils.dsp import TorchSigComplexDataType

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "transforms" / "declarative_transform_config.yaml"


def load_specs(config_path: Path) -> list[TransformSpec]:
    """Load and validate transform specifications from a YAML file."""
    config: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("transforms"), list):
        raise TypeError("configuration must contain a 'transforms' list")
    return [TransformSpec.from_mapping(value) for value in config["transforms"]]


def configuration_fingerprint(specs: list[TransformSpec]) -> str:
    """Return a stable fingerprint for an ordered transform configuration."""
    canonical_config = "[" + ",".join(spec.canonical_json() for spec in specs) + "]"
    return hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()


def ablation_conditions(specs: list[TransformSpec]) -> list[tuple[str, list[TransformSpec]]]:
    """Build control, full, and leave-one-transform-out study conditions."""
    conditions = [("control", []), ("all_impairments", specs)]
    conditions.extend(
        (
            f"without_{index + 1}_{spec.transform}",
            specs[:index] + specs[index + 1 :],
        )
        for index, spec in enumerate(specs)
    )
    return conditions


def run_demo(config_path: Path) -> Signal:
    """Run configured impairment ablations on a shared IQ signal."""
    specs = load_specs(config_path)
    source = Signal(data=np.exp(1j * np.linspace(0, np.pi, 8)).astype(TorchSigComplexDataType))

    print(f"Validated {len(specs)} transform specifications from {config_path}")
    print("Ablation conditions:")

    all_impairments_result = source.copy()
    for name, condition_specs in ablation_conditions(specs):
        # Each condition receives fresh transform instances and the same root
        # seed so repeated runs of a condition are reproducible.
        pipeline = Compose([spec.build() for spec in condition_specs], seed=123)
        transformed = pipeline(source.copy())
        error_power = float(np.mean(np.abs(transformed.data - source.data) ** 2))
        transform_names = [spec.transform for spec in condition_specs]
        print(f"  {name}: transforms={transform_names}, fingerprint={configuration_fingerprint(condition_specs)}, mean_squared_error={error_power:.6g}")
        if name == "all_impairments":
            all_impairments_result = transformed

    return all_impairments_result


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML transform configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)
    run_demo(args.config)


if __name__ == "__main__":
    main()
