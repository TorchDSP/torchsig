import sys
import types
from unittest.mock import Mock

import pytest
import yaml

from torchsig.utils.yaml import (
    custom_representer,
    dataset_from_yaml_dict,
    dataset_metadata_to_yaml_dict,
    load_config_from_yaml,
    load_dataset_yaml,
    save_dataset_yaml,
    write_dict_to_yaml,
)


def install_fake_datasets_module(monkeypatch):
    datasets_mod = types.ModuleType("torchsig.datasets.datasets")

    class FakeTorchSigDatasetConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTorchSigIterableDataset:
        def __init__(self, metadata, transforms, target_labels, seed):
            self.metadata = metadata
            self.transforms = transforms
            self.target_labels = target_labels
            self.rng_seed = seed

    datasets_mod.TorchSigDatasetConfig = FakeTorchSigDatasetConfig
    datasets_mod.TorchSigIterableDataset = FakeTorchSigIterableDataset

    monkeypatch.setitem(sys.modules, "torchsig.datasets.datasets", datasets_mod)

    return FakeTorchSigDatasetConfig, FakeTorchSigIterableDataset


def make_valid_config_dict():
    return {
        "dataset_id": "example_dataset",
        "dataset_length": 100,
        "seed": 123,
        "impairment_level": 2,
        "output": {"representation": "iq"},
        "signal_sampling": {"mode": "per_signal"},
        "dataset_metadata": {
            "sample_rate": 1_000_000,
            "fft_size": 256,
            "class_list": ["bpsk", "qpsk"],
        },
    }


def test_custom_representer_writes_lists_in_flow_style():
    yaml.add_representer(list, custom_representer)

    result = yaml.dump({"values": [1, 2, 3]}, default_flow_style=False)

    assert "values: [1, 2, 3]" in result


def test_load_config_from_yaml_builds_dataset_config(tmp_path, monkeypatch):
    FakeTorchSigDatasetConfig, _ = install_fake_datasets_module(monkeypatch)

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(make_valid_config_dict()))

    config = load_config_from_yaml(path)

    assert isinstance(config, FakeTorchSigDatasetConfig)
    assert config.kwargs == {
        "dataset_id": "example_dataset",
        "dataset_length": 100,
        "seed": 123,
        "impairment_level": 2,
        "output_representation": "iq",
        "output_spectrogram_fft": 256,
        "signal_sampling_mode": "per_signal",
        "dataset_metadata": {
            "sample_rate": 1_000_000,
            "fft_size": 256,
            "class_list": ["bpsk", "qpsk"],
        },
    }


def test_load_config_from_yaml_defaults_dataset_id_to_file_stem(tmp_path, monkeypatch):
    install_fake_datasets_module(monkeypatch)

    cfg = make_valid_config_dict()
    cfg.pop("dataset_id")

    path = tmp_path / "fallback_name.yaml"
    path.write_text(yaml.safe_dump(cfg))

    config = load_config_from_yaml(path)

    assert config.kwargs["dataset_id"] == "fallback_name"


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda cfg: cfg.pop("dataset_metadata"), "dataset_metadata must be a dict"),
        (lambda cfg: cfg.__setitem__("dataset_metadata", []), "dataset_metadata must be a dict"),
        (lambda cfg: cfg.__setitem__("output", ["not-a-dict"]), "output must be a dict"),
        (lambda cfg: cfg.__setitem__("signal_sampling", ["not-a-dict"]), "signal_sampling must be a dict"),
        (
            lambda cfg: cfg.__setitem__("output", {"representation": "bad"}),
            "output.representation must be 'iq' or 'spectrogram'",
        ),
        (
            lambda cfg: cfg.__setitem__("signal_sampling", {"mode": "bad"}),
            "signal_sampling.mode must be 'per_signal' or 'per_family'",
        ),
    ],
)
def test_load_config_from_yaml_rejects_invalid_config(tmp_path, monkeypatch, mutation, expected_message):
    install_fake_datasets_module(monkeypatch)

    cfg = make_valid_config_dict()
    mutation(cfg)

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match=expected_message):
        load_config_from_yaml(path)


def test_load_config_from_yaml_rejects_non_mapping_root(tmp_path, monkeypatch):
    install_fake_datasets_module(monkeypatch)

    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- dict\n")

    with pytest.raises(ValueError, match="YAML root must be a mapping/dict"):
        load_config_from_yaml(path)


def test_dataset_from_yaml_dict_constructs_iterable_dataset(monkeypatch):
    _, FakeTorchSigIterableDataset = install_fake_datasets_module(monkeypatch)

    metadata = {"sample_rate": 1_000_000}
    yaml_dict = {
        "dataset_metadata": metadata,
        "target_labels": ["class_name", "snr"],
        "seed": 42,
    }

    dataset = dataset_from_yaml_dict(yaml_dict)

    assert isinstance(dataset, FakeTorchSigIterableDataset)
    assert dataset.metadata is metadata
    assert dataset.transforms == []
    assert dataset.target_labels == ["class_name", "snr"]
    assert dataset.rng_seed == 42


def test_load_dataset_yaml_reads_file_and_constructs_dataset(tmp_path, monkeypatch):
    install_fake_datasets_module(monkeypatch)

    path = tmp_path / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 99,
                "target_labels": ["class_name"],
                "dataset_metadata": {"class_list": ["bpsk"]},
            }
        )
    )

    dataset = load_dataset_yaml(str(path))

    assert dataset.rng_seed == 99
    assert dataset.target_labels == ["class_name"]
    assert dataset.metadata == {"class_list": ["bpsk"]}


def test_dataset_metadata_to_yaml_dict_returns_full_metadata():
    metadata = Mock()
    metadata.get_full_metadata.return_value = {
        "sample_rate": 1_000_000,
        "class_list": ["bpsk", "qpsk"],
    }

    result = dataset_metadata_to_yaml_dict(metadata)

    assert result == {
        "sample_rate": 1_000_000,
        "class_list": ["bpsk", "qpsk"],
    }
    metadata.get_full_metadata.assert_called_once_with()


def test_write_dict_to_yaml_preserves_key_order_and_writes_inline_lists(tmp_path):
    path = tmp_path / "out.yaml"

    write_dict_to_yaml(
        str(path),
        {
            "seed": 123,
            "target_labels": ["class_name", "snr"],
            "dataset_metadata": {"class_list": ["bpsk", "qpsk"]},
        },
    )

    result = path.read_text()

    assert result.index("seed:") < result.index("target_labels:") < result.index("dataset_metadata:")
    assert "target_labels: [class_name, snr]" in result
    assert "class_list: [bpsk, qpsk]" in result


def test_save_dataset_yaml_writes_dataset_configuration(tmp_path):
    path = tmp_path / "saved.yaml"

    dataset = Mock()
    dataset.rng_seed = 7
    dataset.target_labels = ["class_name"]
    dataset.get_full_metadata.return_value = {"class_list": ["bpsk"]}

    save_dataset_yaml(str(path), dataset)

    loaded = yaml.safe_load(path.read_text())

    assert loaded == {
        "seed": 7,
        "target_labels": ["class_name"],
        "dataset_metadata": {"class_list": ["bpsk"]},
    }
