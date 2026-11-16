"""Tests for the DataModule-backed dataset generation CLI."""

from __future__ import annotations

from pathlib import Path

import yaml

from torchsig.cli import generate
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.utils.file_handlers import HDF5Writer, PackedNPYWriter


def _config(path: Path, *, length: int = 3) -> Path:
    source = Path("torchsig/datasets/default_configs/narrowband_toy_dataset.yaml")
    value = yaml.safe_load(source.read_text())
    value["dataset_length"] = length
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_cli_delegates_overrides_to_datamodule(tmp_path, monkeypatch) -> None:
    """Common CLI overrides are forwarded to from_config and prepare_data."""
    config = _config(tmp_path / "config.yaml")
    observed = {}

    class FakeDataModule:
        def prepare_data(self) -> None:
            observed["prepared"] = True

    def fake_from_config(cfg, **kwargs):
        observed["config"] = cfg
        observed["kwargs"] = kwargs
        return FakeDataModule()

    monkeypatch.setattr(generate.TorchSigDataModule, "from_config", fake_from_config)
    output = generate.generate_dataset(
        [
            "--config",
            str(config),
            "--output",
            str(tmp_path / "dataset"),
            "--dataset-length",
            "6",
            "--create-batch-size",
            "2",
            "--creation-workers",
            "0",
            "--file-handler",
            "packed-npy",
            "--signal-generator",
            "bpsk",
            "--signal-generator",
            "qpsk",
            "--metadata-grouping",
            "family",
            "--target-label",
            "family_name",
            "--overwrite",
        ]
    )

    assert output == tmp_path / "dataset"
    assert observed["prepared"]
    assert observed["kwargs"]["dataset_size"] == 6
    assert observed["kwargs"]["create_batch_size"] == 2
    assert observed["kwargs"]["create_num_workers"] == 0
    assert observed["kwargs"]["file_writer"] is PackedNPYWriter
    assert observed["kwargs"]["signal_generators"] == ["bpsk", "qpsk"]
    assert "samples_per_generator" not in observed["kwargs"]
    assert observed["kwargs"]["target_labels"] == ["family_name"]
    assert observed["kwargs"]["overwrite"] is True


def test_cli_materializes_small_dataset_end_to_end(tmp_path) -> None:
    """A YAML config is materialized without a custom Python generation script."""
    config = _config(tmp_path / "config.yaml", length=2)
    output = tmp_path / "dataset"
    result = generate.generate_dataset(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--file-handler",
            "legacy",
            "--dataset-length",
            "2",
            "--signal-generator",
            "bpsk",
            "--create-batch-size",
            "1",
            "--creation-workers",
            "0",
        ]
    )

    assert result == output
    assert (output / "data.h5").is_file()
    dataset = StaticTorchSigDataset(output, target_labels=None)
    try:
        assert len(dataset) == 2
    finally:
        dataset.reader.teardown()


def test_writer_options_use_yaml_values_and_cli_precedence() -> None:
    """Writer overrides preserve booleans, nulls, and numeric values."""
    writer, options = generate.resolve_file_writer("legacy", {"compression": "gzip", "shuffle": True}, "legacy", ["compression=null", "shuffle=false"])
    assert writer is HDF5Writer
    assert options == {"compression": None, "shuffle": False}
