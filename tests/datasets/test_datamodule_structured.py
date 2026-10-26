"""Focused tests for opt-in DataModule structured materialization."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from torch.utils.data import Subset

from torchsig.datasets import StructuredHDF5Dataset
from torchsig.datasets.datamodules import (
    SplitTorchSigDataModule,
    TorchSigDataModule,
    _OnlineTransformDataset,
)
from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigDatasetConfig
from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_source(root: Path, count: int) -> None:
    signals = [Signal(data=np.full(16, index, dtype=np.complex64), class_index=index) for index in range(count)]
    with HomogeneousHDF5Writer(root, compression=None, shuffle=False, fletcher32=False, chunk_samples=1) as writer:
        writer.write(0, signals)


def _single_module(source_root: Path, **kwargs) -> TorchSigDataModule:
    target_labels = kwargs.pop("target_labels", ["class_index"])
    kwargs.setdefault("structured_validation", "full")
    kwargs.setdefault(
        "structured_writer_kwargs",
        {"compression": None, "shuffle": False, "fletcher32": False, "chunk_samples": 1},
    )
    return TorchSigDataModule(
        root=source_root,
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        dataset_size=10,
        dataset_splits=[6, 2, 2],
        file_writer=HomogeneousHDF5Writer,
        file_reader=HomogeneousHDF5Reader,
        target_labels=target_labels,
        structured_materialization=True,
        num_workers=0,
        **kwargs,
    )


def _materialize_single(module: TorchSigDataModule) -> None:
    dataset = StaticTorchSigDataset(module.root, file_handler_class=module.file_reader, target_labels=module.target_labels)
    manifest = module._structured_manifest(
        module.root,
        module.structured_root,
        split="full",
        expected_length=module.dataset_size,
        target_labels=module.target_labels,
    )
    try:
        module._ensure_structured_dataset(dataset, module.structured_root, manifest)
    finally:
        dataset.reader.teardown()


def _mock_config(*, dataset_length: int, seed: int) -> MagicMock:
    config = MagicMock(spec=TorchSigDatasetConfig)
    config.dataset_id = "test_dataset"
    config.dataset_length = dataset_length
    config.seed = seed
    config.output_representation = "iq"
    config.dataset_metadata = {}
    config.experiment_config = MagicMock()
    return config


def test_torchsig_datamodule_selects_structured_dataset_and_preserves_splits(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, 10)
    module = _single_module(source_root)
    _materialize_single(module)

    module.setup()

    assert isinstance(module.train, Subset)
    assert isinstance(module.train.dataset, StructuredHDF5Dataset)
    assert tuple(map(len, (module.train, module.val, module.test))) == (6, 2, 2)
    all_indices = [*module.train.indices, *module.val.indices, *module.test.indices]
    assert sorted(all_indices) == list(range(10))


def test_compatible_materialization_is_reused_and_manifest_mismatch_fails(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, 10)
    module = _single_module(source_root)
    _materialize_single(module)
    dataset = StructuredHDF5Dataset(module.structured_root)
    manifest = module._structured_manifest(source_root, module.structured_root, split="full", expected_length=10, target_labels=module.target_labels)

    def unexpected(*args, **kwargs):
        raise AssertionError("compatible output should be reused")

    monkeypatch.setattr("torchsig.datasets.datamodules.materialize_structured_dataset", unexpected)
    try:
        module._ensure_structured_dataset(dataset, module.structured_root, manifest)
    finally:
        dataset.close()

    incompatible = _single_module(source_root, target_labels=None)
    with pytest.raises(ValueError, match="Incompatible structured materialization"):
        incompatible.setup()


def test_explicit_overwrite_replaces_an_incompatible_materialization(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, 10)
    module = _single_module(source_root)
    _materialize_single(module)
    manifest_path = module.structured_root.with_name(f"{module.structured_root.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["validation"] = "sampled"
    manifest_path.write_text(json.dumps(manifest))

    replacement = _single_module(source_root, structured_overwrite=True)
    _materialize_single(replacement)
    replacement.setup()

    assert isinstance(replacement.train, Subset)
    assert json.loads(manifest_path.read_text())["validation"] == "full"


def test_split_datamodule_uses_physically_separate_structured_roots(tmp_path: Path) -> None:
    configs = (
        _mock_config(dataset_length=5, seed=11),
        _mock_config(dataset_length=3, seed=22),
        _mock_config(dataset_length=2, seed=33),
    )
    module = SplitTorchSigDataModule(
        train_cfg=configs[0],
        val_cfg=configs[1],
        test_cfg=configs[2],
        root=tmp_path,
        file_writer=HomogeneousHDF5Writer,
        file_reader=HomogeneousHDF5Reader,
        structured_materialization=True,
        structured_validation="full",
        structured_writer_kwargs={"compression": None},
        num_workers=0,
    )
    for split, config in zip(("train", "val", "test"), configs, strict=True):
        source_root = module.root / split
        _write_source(source_root, config.dataset_length)
        source = StaticTorchSigDataset(source_root, file_handler_class=module.file_reader, target_labels=module.target_labels)
        output_root = module.structured_root / split
        manifest = module._structured_manifest(
            source_root,
            output_root,
            split=split,
            expected_length=config.dataset_length,
            target_labels=module.target_labels,
        )
        try:
            module._ensure_structured_dataset(source, output_root, manifest)
        finally:
            source.reader.teardown()

    module.setup(None)

    assert all(isinstance(dataset, StructuredHDF5Dataset) for dataset in (module.train, module.val, module.test))
    assert tuple(map(len, (module.train, module.val, module.test))) == (5, 3, 2)
    assert (module.structured_root / "train" / "data.h5").exists()
    assert (module.structured_root / "val" / "data.h5").exists()
    assert (module.structured_root / "test" / "data.h5").exists()


def test_training_only_transform_remains_online(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, 10)

    class Increment:
        def __init__(self) -> None:
            self.count = 0

        def __call__(self, sample):
            self.count += 1
            values, target = sample
            return values + self.count, target

    module = _single_module(source_root, structured_train_transforms=[Increment()])
    _materialize_single(module)
    module.setup()

    assert isinstance(module.train, _OnlineTransformDataset)
    first = module.train[0][0]
    second = module.train[0][0]
    assert not np.array_equal(first, second)
    assert isinstance(module.val, Subset)


def test_structured_dataset_loads_with_multiple_workers(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root, 10)
    module = _single_module(source_root)
    module.num_workers = 2
    _materialize_single(module)
    module.setup()

    samples, targets = next(iter(module.train_dataloader()))

    assert len(samples) == len(targets) > 0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"same_root": True}, ValueError, "separate"),
        ({"structured_validation": "invalid"}, ValueError, "structured_validation"),
        ({"structured_num_workers": -1}, ValueError, "structured_num_workers"),
    ],
)
def test_structured_configuration_validation(tmp_path: Path, kwargs, error, message) -> None:
    same_root = kwargs.pop("same_root", False)
    source_root = tmp_path / "source"
    if same_root:
        kwargs["structured_root"] = source_root
    with pytest.raises(error, match=message):
        _single_module(source_root, **kwargs)
