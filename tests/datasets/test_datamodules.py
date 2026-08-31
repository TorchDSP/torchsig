"""Unit Tests for datamodules"""

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import h5py

from torchsig.datasets.datamodules import (
    SplitTorchSigDataModule,
    TorchSigDataModule,
)
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.file_handlers import (
    PackedHDF5Reader,
    PackedHDF5Writer,
    HDF5Reader,
    HDF5Writer,
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.writer import identity_collate_fn


def _signal_summary_collate(batch):
    return [
        (
            signal["duration_in_samples"],
            signal.data.shape,
            signal.data.dtype.str,
        )
        for signal in batch
    ]


@pytest.fixture
def split_configs():
    """Return distinct train, validation, and test dataset configs."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3276,
            "signal_duration_in_samples_max": 4096,
        }
    )

    common = {
        "dataset_id": "test_dataset",
        "dataset_metadata": metadata,
        "output_representation": "iq",
        "output_spectrogram_fft": None,
        "signal_sampling_mode": "random",
        "impairment_level": 0,
    }

    return (
        TorchSigDatasetConfig(
            **common,
            dataset_length=12,
            seed=11,
        ),
        TorchSigDatasetConfig(
            **common,
            dataset_length=6,
            seed=22,
        ),
        TorchSigDatasetConfig(
            **common,
            dataset_length=4,
            seed=33,
        ),
    )


def _mock_config(
    *,
    dataset_id: str = "test_dataset",
    dataset_length: int,
    seed: int,
    output_representation: str = "iq",
) -> MagicMock:
    cfg = MagicMock(spec=TorchSigDatasetConfig)
    cfg.dataset_id = dataset_id
    cfg.dataset_length = dataset_length
    cfg.seed = seed
    cfg.output_representation = output_representation
    cfg.dataset_metadata = {}
    return cfg


@pytest.mark.parametrize("overwrite", [True, False])
def test_datamodule_prepare_data_creates_dataset(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    """Verify prepare_data creates artifacts and respects overwrite."""
    fft_size = 16
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )

    initial = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=4,
        overwrite=True,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=0,
        create_num_workers=0,
        seed=42,
    )
    initial.prepare_data()

    assert any(tmp_path.iterdir())

    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("existing data")

    datamodule = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=4,
        overwrite=overwrite,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=0,
        create_num_workers=0,
        seed=42,
    )
    datamodule.prepare_data()

    assert any(tmp_path.iterdir())
    assert sentinel.exists() is not overwrite


def test_datamodule_dataloaders_return_batches(
    tmp_path: Path,
) -> None:
    """Verify train, validation, and test dataloaders return batches."""
    fft_size = 16
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )

    datamodule = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[0.5, 0.25, 0.25],
        batch_size=1,
        overwrite=True,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=0,
        create_num_workers=0,
        seed=42,
    )

    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.impairment_level == 0

    for loader in (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ):
        batch = next(iter(loader))
        assert len(batch) > 0


def test_datamodule_dataloaders_with_workers_return_batches(
    tmp_path: Path,
) -> None:
    """Verify multiprocess loaders in an isolated, bounded process."""
    fft_size = 16
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )
    datamodule = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[0.5, 0.25, 0.25],
        overwrite=True,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=0,
        create_num_workers=0,
        seed=42,
    )
    datamodule.prepare_data()

    script = textwrap.dedent(
        """
        import sys

        from torchsig.datasets.datamodules import TorchSigDataModule
        from torchsig.utils.writer import identity_collate_fn

        datamodule = TorchSigDataModule(
            root=sys.argv[1],
            metadata={},
            dataset_size=6,
            dataset_splits=[0.5, 0.25, 0.25],
            batch_size=1,
            collate_fn=identity_collate_fn,
            num_workers=2,
            seed=42,
        )
        datamodule.setup()
        for loader_factory in (
            datamodule.train_dataloader,
            datamodule.val_dataloader,
            datamodule.test_dataloader,
        ):
            loader = loader_factory()
            assert len(next(iter(loader))) > 0
            del loader
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("multiprocess DataLoaders did not finish within 30 seconds\n" + stdout + stderr)

    assert process.returncode == 0, stdout + stderr


def _first_n_batches(loader, n: int):
    """Return up to ``n`` batches from ``loader``.
    If the loader has fewer than ``n`` batches (e.g. because the split is tiny),
    the function simply returns the available ones instead of raising StopIteration.
    """
    it = iter(loader)
    batches = []
    for _ in range(n):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches


def _tensors_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return ``True`` iff *both* tensors contain exactly the same samples,
    irrespective of their shape.

    * If the shapes differ → they cannot be identical → ``False``.
    * If the shapes are the same → we compare element-wise with a tolerant
      ``torch.allclose`` (the default tolerance is fine for the synthetic
      signals used in the test suite).

    The function works for 2-D tensors of shape ``(N, …)`` where the first
    dimension is the *batch* (i.e. the number of samples).
    """
    # Different number of samples → definitely not identical.
    if a.shape != b.shape:
        return False

    # Same shape → do a normal allclose check.
    return torch.allclose(a, b)


@pytest.fixture(scope="module")
def reproducibility_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict]:
    """Create one small static dataset for reproducibility tests."""
    root = tmp_path_factory.mktemp("dataloader_reproducibility")
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 256,
            "fft_size": 16,
            "fft_stride": 16,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 16,
            "signal_duration_in_samples_max": 256,
        }
    )

    data_module = TorchSigDataModule(
        root=root,
        metadata=metadata,
        dataset_size=10,
        dataset_splits=[0.6, 0.2, 0.2],
        batch_size=2,
        num_workers=0,
        create_num_workers=0,
        seed=42,
        collate_fn=list,
    )
    data_module.prepare_data()

    return root, metadata


def _loader_data(loader: DataLoader) -> list[np.ndarray]:
    """Return copied sample arrays from a complete dataloader iteration."""
    return [signal.data.copy() for batch in loader for signal in batch]


def _assert_same_samples(
    actual: list[np.ndarray],
    expected: list[np.ndarray],
) -> None:
    """Assert that two loaders returned identical samples in identical order."""
    assert len(actual) == len(expected)

    for actual_sample, expected_sample in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_sample, expected_sample)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_reproducibility(
    reproducibility_dataset: tuple[Path, dict],
    num_workers: int,
) -> None:
    """Verify seeded data modules produce identical splits and loader order."""
    root, metadata = reproducibility_dataset

    shared_kwargs = {
        "root": root,
        "metadata": metadata.copy(),
        "dataset_size": 10,
        "dataset_splits": [0.6, 0.2, 0.2],
        "batch_size": 2,
        "num_workers": num_workers,
        "seed": 42,
        "collate_fn": list,
    }

    first = TorchSigDataModule(**shared_kwargs)
    second = TorchSigDataModule(**shared_kwargs)

    first.setup()
    second.setup()

    loader_pairs = (
        (first.train_dataloader(), second.train_dataloader()),
        (first.val_dataloader(), second.val_dataloader()),
        (first.test_dataloader(), second.test_dataloader()),
    )

    for first_loader, second_loader in loader_pairs:
        _assert_same_samples(
            _loader_data(first_loader),
            _loader_data(second_loader),
        )


def test_split_datamodule_initializes_from_three_configs(tmp_path):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        batch_size=8,
        num_workers=None,
        create_batch_size=4,
        create_num_workers=2,
        signal_generators=["bpsk"],
    )

    assert dm.train_cfg is train_cfg
    assert dm.val_cfg is val_cfg
    assert dm.test_cfg is test_cfg

    assert dm.root == tmp_path / "test_dataset"
    assert dm.batch_size == 8
    assert dm.num_workers == 0
    assert dm.create_batch_size == 4
    assert dm.create_num_workers == 2
    assert dm.signal_generators == ["bpsk"]

    assert dm.train is None
    assert dm.val is None
    assert dm.test is None


def test_split_datamodule_rejects_mismatched_output_representations(tmp_path):
    train_cfg = _mock_config(
        dataset_length=12,
        seed=11,
        output_representation="iq",
    )
    val_cfg = _mock_config(
        dataset_length=6,
        seed=22,
        output_representation="spectrogram",
    )
    test_cfg = _mock_config(
        dataset_length=4,
        seed=33,
        output_representation="iq",
    )

    with pytest.raises(
        ValueError,
        match="same output representation",
    ):
        SplitTorchSigDataModule(
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            root=tmp_path,
        )


@patch("torchsig.datasets.datamodules.DatasetCreator")
@patch("torchsig.datasets.datamodules.WorkerSeedingDataLoader")
@patch("torchsig.datasets.datamodules.TorchSigIterableDataset")
def test_split_datamodule_prepare_data_creates_all_splits(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    datasets = [MagicMock(), MagicMock(), MagicMock()]
    iterable_dataset_cls.side_effect = datasets

    loaders = [MagicMock(), MagicMock(), MagicMock()]
    dataloader_cls.side_effect = loaders

    creators = [MagicMock(), MagicMock(), MagicMock()]
    dataset_creator_cls.side_effect = creators

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        create_batch_size=4,
        create_num_workers=2,
        overwrite=True,
        signal_generators=["bpsk", "qpsk"],
    )

    dm.prepare_data()

    assert iterable_dataset_cls.call_count == 3
    assert dataloader_cls.call_count == 3
    assert dataset_creator_cls.call_count == 3

    assert dataset_creator_cls.call_args_list[0].kwargs["dataset_length"] == 12
    assert dataset_creator_cls.call_args_list[1].kwargs["dataset_length"] == 6
    assert dataset_creator_cls.call_args_list[2].kwargs["dataset_length"] == 4

    assert Path(dataset_creator_cls.call_args_list[0].kwargs["root"]) == (tmp_path / "test_dataset" / "train")
    assert Path(dataset_creator_cls.call_args_list[1].kwargs["root"]) == (tmp_path / "test_dataset" / "val")
    assert Path(dataset_creator_cls.call_args_list[2].kwargs["root"]) == (tmp_path / "test_dataset" / "test")

    for creator in creators:
        creator.create.assert_called_once_with()


@patch("torchsig.datasets.datamodules.DatasetCreator")
@patch("torchsig.datasets.datamodules.WorkerSeedingDataLoader")
@patch("torchsig.datasets.datamodules.TorchSigIterableDataset")
def test_split_datamodule_uses_split_specific_seeds(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.prepare_data()

    dataset_seeds = [call_args.kwargs["seed"] for call_args in iterable_dataset_cls.call_args_list]
    loader_seeds = [call_args.kwargs["seed"] for call_args in dataloader_cls.call_args_list]

    assert dataset_seeds == [11, 22, 33]
    assert loader_seeds == [11, 22, 33]


@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_fit_loads_train_and_val_only(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    train_dataset = MagicMock()
    val_dataset = MagicMock()
    static_dataset_cls.side_effect = [train_dataset, val_dataset]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup("fit")

    assert dm.train is train_dataset
    assert dm.val is val_dataset
    assert dm.test is None

    assert static_dataset_cls.call_count == 2

    assert Path(static_dataset_cls.call_args_list[0].kwargs["root"]) == (tmp_path / "test_dataset" / "train")
    assert Path(static_dataset_cls.call_args_list[1].kwargs["root"]) == (tmp_path / "test_dataset" / "val")


@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_test_loads_test_only(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    test_dataset = MagicMock()
    static_dataset_cls.return_value = test_dataset

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup("test")

    assert dm.train is None
    assert dm.val is None
    assert dm.test is test_dataset

    static_dataset_cls.assert_called_once()

    assert Path(static_dataset_cls.call_args.kwargs["root"]) == (tmp_path / "test_dataset" / "test")


@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_none_loads_all_splits(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    datasets = [MagicMock(), MagicMock(), MagicMock()]
    static_dataset_cls.side_effect = datasets

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup(None)

    assert dm.train is datasets[0]
    assert dm.val is datasets[1]
    assert dm.test is datasets[2]


@patch("torchsig.datasets.datamodules.random_split")
@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_does_not_use_random_split(
    static_dataset_cls,
    random_split_mock,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    static_dataset_cls.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup(None)

    random_split_mock.assert_not_called()


@patch("torchsig.datasets.datamodules.DataLoader")
def test_split_datamodule_dataloader_shuffle_behavior(
    dataloader_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dataloader_cls.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        shuffle=True,
    )
    dm.train = MagicMock()
    dm.val = MagicMock()
    dm.test = MagicMock()

    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()

    assert dataloader_cls.call_args_list[0].kwargs["shuffle"] is True
    assert dataloader_cls.call_args_list[1].kwargs["shuffle"] is False
    assert dataloader_cls.call_args_list[2].kwargs["shuffle"] is False


@pytest.mark.parametrize(
    "method_name",
    [
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ],
)
def test_split_datamodule_dataloader_requires_setup(
    method_name,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="setup"):
        getattr(dm, method_name)()


@pytest.mark.slow_no_gpu
def test_split_datamodule_smoke(tmp_path, split_configs):
    train_cfg, val_cfg, test_cfg = split_configs

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        batch_size=2,
        num_workers=0,
        create_batch_size=2,
        create_num_workers=0,
        overwrite=True,
        collate_fn=identity_collate_fn,
    )

    dm.prepare_data()
    dm.setup(None)

    assert len(dm.train) == train_cfg.dataset_length
    assert len(dm.val) == val_cfg.dataset_length
    assert len(dm.test) == test_cfg.dataset_length

    assert next(iter(dm.train_dataloader()))
    assert next(iter(dm.val_dataloader()))
    assert next(iter(dm.test_dataloader()))


@pytest.mark.parametrize(
    ("transforms", "expected_ndim"),
    [([], 1), ([Spectrogram(fft_size=64)], 2)],
    ids=["iq", "spectrogram"],
)
def test_torchsig_datamodule_infers_packed_reader_end_to_end(tmp_path, transforms, expected_ndim):
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4_096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3_276,
            "signal_duration_in_samples_max": 4_096,
        }
    )
    writer_options = {
        "compression": None,
        "shuffle": False,
        "fletcher32": False,
        "max_batches_in_memory": 1,
    }
    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[4, 1, 1],
        create_batch_size=2,
        file_writer=PackedHDF5Writer,
        file_writer_kwargs=writer_options,
        overwrite=True,
        impairment_level=0,
        transforms=transforms,
        collate_fn=identity_collate_fn,
        num_workers=0,
    )
    writer_options["compression"] = "lzf"

    assert dm.file_reader is PackedHDF5Reader
    assert dm.file_writer_kwargs["compression"] is None
    dm.prepare_data()
    dm.setup()

    full_dataset = dm.train.dataset
    assert isinstance(full_dataset.reader, PackedHDF5Reader)
    assert full_dataset[0].data.ndim == expected_ndim
    full_dataset.reader.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data/0"].compression is None
        assert not handle["data/0"].shuffle
        assert not handle["data/0"].fletcher32
    writer_info = (tmp_path / "writer_info.yaml").read_text()
    assert "torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Writer" in writer_info
    assert "torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Reader" in writer_info


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    ("transforms", "expected_ndim"),
    [([], 1), ([Spectrogram(fft_size=64)], 2)],
    ids=["iq", "spectrogram"],
)
def test_torchsig_datamodule_infers_homogeneous_reader_end_to_end(
    tmp_path,
    transforms,
    expected_ndim,
    num_workers,
):
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4_096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3_276,
            "signal_duration_in_samples_max": 4_096,
        }
    )
    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[4, 1, 1],
        create_batch_size=2,
        create_num_workers=0,
        file_writer=HomogeneousHDF5Writer,
        file_writer_kwargs={
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "chunk_samples": 2,
        },
        overwrite=True,
        impairment_level=0,
        transforms=transforms,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
    )

    assert dm.file_reader is HomogeneousHDF5Reader
    dm.prepare_data()
    dm.setup()
    full_dataset = dm.train.dataset
    assert isinstance(full_dataset.reader, HomogeneousHDF5Reader)
    assert full_dataset[0].data.ndim == expected_ndim
    dm.collate_fn = _signal_summary_collate
    batch = next(iter(dm.train_dataloader()))
    assert batch
    assert all(len(item[1]) == expected_ndim for item in batch)
    full_dataset.reader.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data"].compression is None
        assert handle["data"].chunks[0] == 2


@pytest.mark.parametrize(
    ("file_writer", "file_reader"),
    [
        (PackedHDF5Writer, HDF5Reader),
        (HDF5Writer, PackedHDF5Reader),
        (HomogeneousHDF5Writer, PackedHDF5Reader),
        (PackedHDF5Writer, HomogeneousHDF5Reader),
    ],
)
def test_torchsig_datamodule_rejects_incompatible_handler_pair(tmp_path, file_writer, file_reader):
    with pytest.raises(ValueError, match="Incompatible file handler pair"):
        TorchSigDataModule(
            root=tmp_path,
            metadata=TorchSigDefaults().default_dataset_metadata,
            dataset_size=1,
            file_writer=file_writer,
            file_reader=file_reader,
        )


@pytest.mark.parametrize(
    "file_writer_kwargs",
    [{"unknown_option": True}, ["not", "a", "dictionary"]],
)
def test_torchsig_datamodule_rejects_invalid_writer_options(tmp_path, file_writer_kwargs):
    with pytest.raises(TypeError, match="file_writer_kwargs|Invalid options"):
        TorchSigDataModule(
            root=tmp_path,
            metadata=TorchSigDefaults().default_dataset_metadata,
            dataset_size=1,
            file_writer=PackedHDF5Writer,
            file_writer_kwargs=file_writer_kwargs,
        )


def test_split_datamodule_infers_packed_reader_end_to_end(tmp_path, split_configs):
    train_cfg, val_cfg, test_cfg = split_configs
    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        create_batch_size=2,
        create_num_workers=0,
        file_writer=PackedHDF5Writer,
        file_writer_kwargs={
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "max_batches_in_memory": 1,
        },
        overwrite=True,
        collate_fn=identity_collate_fn,
    )

    assert dm.file_reader is PackedHDF5Reader
    dm.prepare_data()
    dm.setup(None)

    assert isinstance(dm.train.reader, PackedHDF5Reader)
    assert isinstance(dm.val.reader, PackedHDF5Reader)
    assert isinstance(dm.test.reader, PackedHDF5Reader)
    dm.train.reader.teardown()
    dm.val.reader.teardown()
    dm.test.reader.teardown()
    with h5py.File(dm.root / "train" / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data/0"].compression is None
