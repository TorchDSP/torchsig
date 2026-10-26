"""Tests for YAML-backed signal-generation configuration."""

import pickle
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.experiment_config import (
    ExperimentConfig,
    FixedValue,
    SignalConfig,
    load_experiment_config,
)
from torchsig.utils.writer import DatasetCreator


def test_load_fixed_qpsk_alpha_rolloff_from_yaml(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump({"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 0.35}}}}}),
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.parameter_value("qpsk", "alpha_rolloff") == 0.35
    assert config.to_dict()["signals"]["qpsk"]["parameters"]["alpha_rolloff"] == {"value": 0.35}


def test_programmatic_experiment_config_matches_yaml_schema():
    config = ExperimentConfig(
        signals={
            "qpsk": SignalConfig(
                parameters={"alpha_rolloff": FixedValue(0.35)},
            )
        }
    )

    assert config.parameter_value("qpsk", "alpha_rolloff") == 0.35
    assert config.to_dict() == {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 0.35}}}}}
    assert load_experiment_config(config) is config


@pytest.mark.parametrize(
    ("signals", "exception", "message"),
    [
        ({"not-a-signal": SignalConfig(parameters={})}, ValueError, "unknown signal class"),
        ({"qpsk": SignalConfig(parameters={"bogus": FixedValue(1)})}, ValueError, "unknown parameters"),
        ({"qpsk": SignalConfig(parameters={"alpha_rolloff": FixedValue("0.35")})}, TypeError, "must be a real number"),
    ],
)
def test_programmatic_experiment_config_uses_same_validation(signals, exception, message):
    with pytest.raises(exception, match=message):
        ExperimentConfig(signals=signals)


@pytest.mark.parametrize(
    ("config", "exception", "message"),
    [
        ({"signals": {"not-a-signal": {"parameters": {}}}}, ValueError, "unknown signal class"),
        (
            {"signals": {"qpsk": {"parameters": {"bogus": {"value": 1}}}}},
            ValueError,
            "unknown parameters",
        ),
        (
            {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": "0.35"}}}}},
            TypeError,
            "must be a real number",
        ),
        (
            {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 1.0}}}}},
            ValueError,
            "between 0 and 1",
        ),
        (
            {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"min": 0.2}}}}},
            ValueError,
            "unknown settings",
        ),
    ],
)
def test_invalid_experiment_configuration_is_actionable(config, exception, message):
    with pytest.raises(exception, match=message):
        load_experiment_config(config)


def _qpsk_dataset(experiment_config=None):
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": 256,
            "signal_duration_in_samples_min": 256,
            "signal_duration_in_samples_max": 256,
            "bandwidth_min": 10,
            "bandwidth_max": 10,
            "sample_rate": 100,
        }
    )
    return TorchSigIterableDataset(
        signal_generators=["qpsk"],
        metadata=metadata,
        target_labels=None,
        experiment_config=experiment_config,
        seed=7,
    )


def test_fixed_qpsk_alpha_rolloff_is_applied_to_every_generated_signal():
    config = ExperimentConfig(signals={"qpsk": SignalConfig(parameters={"alpha_rolloff": FixedValue(0.35)})})
    dataset = _qpsk_dataset(config)
    generator = dataset.signal_generators[0]

    generated = [generator() for _ in range(3)]

    assert all(signal.pulse_shape_name == "srrc" for signal in generated)
    assert all(signal.alpha_rolloff == pytest.approx(0.35) for signal in generated)


def test_unconfigured_qpsk_preserves_randomized_fallback():
    generator = _qpsk_dataset().signal_generators[0]

    assert not hasattr(generator, "alpha_rolloff")
    generated = [generator() for _ in range(12)]
    assert {signal.pulse_shape_name for signal in generated} == {
        "rectangular",
        "srrc",
    }


def test_effective_configuration_is_exposed_in_dataset_artifact():
    dataset = _qpsk_dataset({"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 0.35}}}}})
    creator = DatasetCreator.__new__(DatasetCreator)
    creator.dataloader = SimpleNamespace(dataset=dataset)

    info = creator.get_dataset_info_dict(dataset_length=1, original_target_labels=None)

    assert info["experiment_config"] == {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 0.35}}}}}


def test_legacy_qpsk_alpha_is_deprecated_and_normalized():
    with pytest.deprecated_call(match="alpha_rolloff"):
        config = load_experiment_config({"signals": {"qpsk": {"parameters": {"alpha": {"value": 0.35}}}}})

    assert config.parameter_value("qpsk", "alpha_rolloff") == 0.35
    assert config.to_dict() == {"signals": {"qpsk": {"parameters": {"alpha_rolloff": {"value": 0.35}}}}}

    with pytest.deprecated_call(match="alpha_rolloff"):
        assert config.parameter_value("qpsk", "alpha") == 0.35


def test_alpha_and_alpha_rolloff_cannot_both_be_configured():
    with pytest.raises(ValueError, match="cannot both be configured"):
        load_experiment_config(
            {
                "signals": {
                    "qpsk": {
                        "parameters": {
                            "alpha": {"value": 0.35},
                            "alpha_rolloff": {"value": 0.35},
                        }
                    }
                }
            }
        )


@pytest.mark.parametrize("class_name", ["bpsk", "16qam", "dvbs2"])
def test_fixed_alpha_rolloff_is_supported_by_rolloff_generators(class_name):
    config = ExperimentConfig(signals={class_name: SignalConfig(parameters={"alpha_rolloff": FixedValue(0.25)})})

    assert config.parameter_value(class_name, "alpha_rolloff") == 0.25


@pytest.mark.parametrize("class_name", ["bpsk", "dvbs2"])
def test_dataset_applies_fixed_alpha_rolloff_to_supported_generator(class_name):
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update(
        {
            "num_iq_samples_dataset": 256,
            "signal_duration_in_samples_min": 256,
            "signal_duration_in_samples_max": 256,
            "bandwidth_min": 10,
            "bandwidth_max": 10,
            "sample_rate": 100,
        }
    )
    dataset = TorchSigIterableDataset(
        signal_generators=[class_name],
        metadata=metadata,
        target_labels=None,
        experiment_config={"signals": {class_name: {"parameters": {"alpha_rolloff": {"value": 0.25}}}}},
        seed=7,
    )

    assert dataset.signal_generators[0]["alpha_rolloff"] == pytest.approx(0.25)


def test_signal_config_pickle_round_trip():
    config = SignalConfig(
        parameters={
            "alpha_rolloff": FixedValue(0.35),
        },
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert isinstance(restored.parameters, MappingProxyType)


def test_experiment_config_pickle_round_trip():
    config = ExperimentConfig(
        signals={
            "qpsk": SignalConfig(
                parameters={
                    "alpha_rolloff": FixedValue(0.35),
                },
            ),
        },
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored.to_dict() == config.to_dict()
    assert isinstance(restored.signals, MappingProxyType)
    assert isinstance(
        restored.signals["qpsk"].parameters,
        MappingProxyType,
    )


def test_empty_experiment_config_is_pickleable():
    config = ExperimentConfig()

    restored = pickle.loads(pickle.dumps(config))

    assert restored.to_dict() == {"signals": {}}
