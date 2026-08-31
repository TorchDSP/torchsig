from unittest.mock import Mock

import pytest

import torchsig.utils.defaults as defaults_module
from torchsig.utils.defaults import TorchSigDefaults, default_dataloader, default_dataset


def test_default_dataset_metadata_returns_copy():
    defaults = TorchSigDefaults()

    metadata = defaults.default_dataset_metadata
    metadata["fft_size"] = 123

    assert defaults.default_dataset_metadata["fft_size"] == 512


def test_default_dataset_metadata_contains_expected_defaults():
    metadata = TorchSigDefaults().default_dataset_metadata

    assert metadata["num_iq_samples_dataset"] == 262144
    assert metadata["fft_size"] == 512
    assert metadata["sample_rate"] == 10_000_000
    assert metadata["num_signals_min"] == 1
    assert metadata["num_signals_max"] == 1


def test_default_dataset_without_impairments_uses_provided_transforms(monkeypatch):
    dataset = Mock()
    signal_generator = {}
    dataset.signal_generators = [signal_generator]

    dataset_cls = Mock(return_value=dataset)
    monkeypatch.setattr(
        "torchsig.datasets.datasets.TorchSigIterableDataset",
        dataset_cls,
    )

    transforms = [Mock(name="dataset_transform")]
    component_transforms = [Mock(name="component_transform")]

    result = default_dataset(
        transforms=transforms,
        component_transforms=component_transforms,
        custom_arg="value",
    )

    assert result is dataset
    dataset_cls.assert_called_once()
    _, kwargs = dataset_cls.call_args
    assert kwargs["metadata"] == TorchSigDefaults().default_dataset_metadata
    assert kwargs["transforms"] is transforms
    assert kwargs["custom_arg"] == "value"
    assert signal_generator["transforms"] is component_transforms


def test_default_dataset_with_impairments_prepends_impairment_transforms(monkeypatch):
    dataset = Mock()
    signal_generator = {}
    dataset.signal_generators = [signal_generator]

    dataset_cls = Mock(return_value=dataset)
    monkeypatch.setattr(
        "torchsig.datasets.datasets.TorchSigIterableDataset",
        dataset_cls,
    )

    burst_impairments = Mock(name="burst_impairments")
    signal_impairments = Mock(name="signal_impairments")
    impairments = Mock(
        signal_transforms=burst_impairments,
        dataset_transforms=signal_impairments,
    )
    impairments_cls = Mock(return_value=impairments)
    monkeypatch.setattr(defaults_module, "Impairments", impairments_cls)

    dataset_transform = Mock(name="dataset_transform")
    component_transform = Mock(name="component_transform")

    result = default_dataset(
        impairment_level=2,
        transforms=[dataset_transform],
        component_transforms=[component_transform],
    )

    assert result is dataset
    impairments_cls.assert_called_once_with(2)

    _, kwargs = dataset_cls.call_args
    assert kwargs["transforms"] == [signal_impairments, dataset_transform]
    assert signal_generator["transforms"] == [burst_impairments, component_transform]


def test_default_dataset_ignores_signal_generators_without_transform_assignment(monkeypatch):
    class NoSetItem:
        def __setitem__(self, key, value):
            raise TypeError("does not support item assignment")

    dataset = Mock()
    mutable_generator = {}
    immutable_generator = NoSetItem()
    dataset.signal_generators = [mutable_generator, immutable_generator]

    dataset_cls = Mock(return_value=dataset)
    monkeypatch.setattr(
        "torchsig.datasets.datasets.TorchSigIterableDataset",
        dataset_cls,
    )

    component_transforms = [Mock()]

    result = default_dataset(component_transforms=component_transforms)

    assert result is dataset
    assert mutable_generator["transforms"] is component_transforms


def test_default_dataloader_creates_dataset_and_loader(monkeypatch):
    dataset = Mock(name="dataset")
    loader = Mock(name="loader")

    default_dataset_mock = Mock(return_value=dataset)
    dataloader_cls = Mock(return_value=loader)

    monkeypatch.setattr(defaults_module, "default_dataset", default_dataset_mock)
    monkeypatch.setattr(defaults_module, "WorkerSeedingDataLoader", dataloader_cls)

    collate_fn = Mock(name="collate_fn")

    result = default_dataloader(
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=2,
        impairment_level=1,
    )

    assert result is loader
    default_dataset_mock.assert_called_once_with(impairment_level=1)
    dataloader_cls.assert_called_once_with(
        dataset,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=2,
    )
    loader.seed.assert_not_called()


@pytest.mark.parametrize("seed", [123, 999])
def test_default_dataloader_seeds_loader_when_seed_is_truthy(monkeypatch, seed):
    dataset = Mock(name="dataset")
    loader = Mock(name="loader")

    monkeypatch.setattr(defaults_module, "default_dataset", Mock(return_value=dataset))
    monkeypatch.setattr(
        defaults_module,
        "WorkerSeedingDataLoader",
        Mock(return_value=loader),
    )

    result = default_dataloader(seed=seed)

    assert result is loader
    loader.seed.assert_called_once_with(seed)


@pytest.mark.parametrize("seed", [False, 0, None])
def test_default_dataloader_does_not_seed_loader_when_seed_is_falsy(monkeypatch, seed):
    dataset = Mock(name="dataset")
    loader = Mock(name="loader")

    monkeypatch.setattr(defaults_module, "default_dataset", Mock(return_value=dataset))
    monkeypatch.setattr(
        defaults_module,
        "WorkerSeedingDataLoader",
        Mock(return_value=loader),
    )

    result = default_dataloader(seed=seed)

    assert result is loader
    loader.seed.assert_not_called()
