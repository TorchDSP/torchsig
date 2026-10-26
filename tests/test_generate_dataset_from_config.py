"""Tests for the YAML-driven dataset generation script."""

from pathlib import Path

import pytest
import yaml

from scripts import generate_dataset_from_config as script
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.file_handlers import (
    HDF5Writer,
    HomogeneousHDF5Writer,
    PackedHDF5Writer,
)


def test_parse_writer_options_uses_yaml_types() -> None:
    """CLI writer values should support booleans, nulls, and integers."""
    assert script.parse_writer_options(["compression=null", "shuffle=false", "chunk_samples=8"]) == {
        "compression": None,
        "shuffle": False,
        "chunk_samples": 8,
    }


@pytest.mark.parametrize("value", ["missing-separator", "=value"])
def test_parse_writer_options_rejects_invalid_assignment(value: str) -> None:
    """Every writer option requires a nonempty key and equals separator."""
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        script.parse_writer_options([value])


def test_parse_writer_options_rejects_duplicate_keys() -> None:
    """Duplicate CLI options should not silently replace one another."""
    with pytest.raises(ValueError, match="provided more than once"):
        script.parse_writer_options(["compression=lzf", "compression=null"])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("legacy", HDF5Writer),
        ("packed", PackedHDF5Writer),
        ("homogeneous", HomogeneousHDF5Writer),
    ],
)
def test_resolve_file_writer_selects_registered_backend(name: str, expected: type) -> None:
    """Each public writer name resolves to its concrete FileWriter class."""
    writer, options = script.resolve_file_writer(name, {}, None, None)
    assert writer is expected
    assert options == {}


def test_resolve_file_writer_applies_cli_precedence() -> None:
    """CLI backend and individual options override YAML storage settings."""
    writer, options = script.resolve_file_writer(
        "legacy",
        {"compression": "gzip", "shuffle": True},
        "packed",
        ["compression=lzf"],
    )
    assert writer is PackedHDF5Writer
    assert options == {"compression": "lzf", "shuffle": True}


def test_resolve_file_writer_rejects_unknown_constructor_option() -> None:
    """Invalid writer options fail before dataset generation begins."""
    with pytest.raises(ValueError, match="Invalid options for PackedHDF5Writer"):
        script.resolve_file_writer("packed", {}, None, ["not_an_option=1"])


def test_generate_dataset_forwards_resolved_configuration(tmp_path, monkeypatch) -> None:
    """Generation forwards seed, stride, output path, writer, and writer options."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "example",
                "dataset_length": 2,
                "seed": 123,
                "impairment_level": 0,
                "output": {"representation": "spectrogram"},
                "signal_sampling": {"mode": "per_signal"},
                "storage": {
                    "writer": "legacy",
                    "options": {"shuffle": True},
                },
                "dataset_metadata": {
                    "sample_rate": 8.0,
                    "num_iq_samples_dataset": 16,
                    "fft_size": 4,
                    "fft_stride": 2,
                },
            }
        )
    )
    observed: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            observed["dataset"] = kwargs

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            observed["loader"] = {"dataset": dataset, **kwargs}

    class FakeCreator:
        def __init__(self, **kwargs):
            observed["creator"] = kwargs
            self.root = Path(kwargs["root"])

        def create(self):
            self.root.mkdir(parents=True)

    monkeypatch.setattr(script, "SafeTorchSigIterableDataset", FakeDataset)
    monkeypatch.setattr(script, "WorkerSeedingDataLoader", FakeLoader)
    monkeypatch.setattr(script, "DatasetCreator", FakeCreator)

    script.generate_dataset(
        [
            "--root",
            str(tmp_path / "output"),
            "--config",
            str(config_path),
            "--file-writer",
            "packed",
            "--writer-option",
            "compression=null",
            "--save-config-copy",
        ]
    )

    dataset_options = observed["dataset"]
    assert dataset_options["seed"] == 123
    assert dataset_options["target_labels"] == ["yolo_label"]
    spectrogram = next(transform for transform in dataset_options["transforms"] if isinstance(transform, Spectrogram))
    assert spectrogram.fft_stride == 2

    creator_options = observed["creator"]
    assert creator_options["root"] == tmp_path / "output" / "example"
    assert creator_options["file_handler"] is PackedHDF5Writer
    assert creator_options["compression"] is None
    assert creator_options["shuffle"] is True
    assert (tmp_path / "output" / "example" / "original_config.yaml").read_text() == config_path.read_text()
