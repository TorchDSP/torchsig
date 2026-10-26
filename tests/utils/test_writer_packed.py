"""Integration tests for DatasetCreator with packed HDF5 storage."""

from typing import ClassVar

import h5py
import numpy as np
import pytest
import yaml
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.file_handlers import (
    FileWriter,
    PackedHDF5Reader,
    PackedHDF5Writer,
)
from torchsig.utils.writer import DatasetCreator

DATASET_LENGTH = 7
NUM_SAMPLES = 64


class _SignalDataset(Dataset):
    """Small deterministic map-style dataset yielding Signal objects."""

    def __init__(self, *, spectrogram: bool = False) -> None:
        self.spectrogram = spectrogram

    def __len__(self) -> int:
        return DATASET_LENGTH

    def __getitem__(self, idx: int) -> Signal:
        data = np.exp(2j * np.pi * (idx + 1) * np.arange(NUM_SAMPLES) / NUM_SAMPLES).astype(np.complex64)
        signal = Signal(data=data, sample_index=idx)
        return Spectrogram(fft_size=8)(signal) if self.spectrogram else signal


class _FailingSignalDataset(_SignalDataset):
    """Raise after the first complete DataLoader batch."""

    def __getitem__(self, idx: int) -> Signal:
        if idx == 2:
            raise RuntimeError("injected generation failure")
        return super().__getitem__(idx)


class _OptionRecordingWriter(FileWriter):
    """Minimal writer recording options received by each construction."""

    observed_options: ClassVar[list[object]] = []

    def __init__(self, root, *, required_option: object) -> None:
        super().__init__(root)
        self.observed_options.append(required_option)

    def write(self, batch_idx, data) -> None:
        del batch_idx, data


@pytest.mark.parametrize("multithreading", [False, True])
@pytest.mark.parametrize("spectrogram", [False, True], ids=["iq", "spectrogram"])
def test_dataset_creator_round_trips_packed_signals(tmp_path, multithreading, spectrogram) -> None:
    source = _SignalDataset(spectrogram=spectrogram)
    dataloader = DataLoader(source, batch_size=3)
    DatasetCreator(
        dataloader=dataloader,
        root=tmp_path,
        file_handler=PackedHDF5Writer,
        multithreading=multithreading,
    ).create()

    dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=PackedHDF5Reader,
        target_labels=None,
    )
    assert len(dataset) == DATASET_LENGTH
    for idx in range(DATASET_LENGTH):
        expected = source[idx]
        actual = dataset[idx]
        assert actual.data.dtype == expected.data.dtype
        assert actual.data.shape == expected.data.shape
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual["sample_index"] == idx

    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler"] == "PackedHDF5Writer"
    assert writer_info["complete"] is True


@pytest.mark.parametrize("multithreading", [False, True])
def test_dataset_creator_failure_leaves_packed_file_incomplete(tmp_path, multithreading) -> None:
    dataloader = DataLoader(_FailingSignalDataset(), batch_size=2)
    creator = DatasetCreator(
        dataloader=dataloader,
        root=tmp_path,
        file_handler=PackedHDF5Writer,
        multithreading=multithreading,
    )

    with pytest.raises(RuntimeError, match="injected generation failure"):
        creator.create()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])
    with pytest.raises(ValueError, match="file is incomplete"):
        PackedHDF5Reader(tmp_path).read(0)
    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["complete"] is False


def test_dataset_creator_forwards_and_records_packed_writer_options(
    tmp_path,
) -> None:
    options = {
        "compression": None,
        "shuffle": False,
        "fletcher32": False,
        "max_batches_in_memory": 1,
    }
    DatasetCreator(
        dataloader=DataLoader(_SignalDataset(), batch_size=3),
        root=tmp_path,
        file_handler=PackedHDF5Writer,
        multithreading=False,
        **options,
    ).create()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data/0"].compression is None
        assert not handle["data/0"].shuffle
        assert not handle["data/0"].fletcher32
    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler_kwargs"] == options


def test_dataset_creator_forwards_options_to_every_writer_construction(
    tmp_path,
) -> None:
    _OptionRecordingWriter.observed_options.clear()
    DatasetCreator(
        dataloader=DataLoader(_SignalDataset(), batch_size=3),
        root=tmp_path,
        file_handler=_OptionRecordingWriter,
        multithreading=False,
        required_option="forwarded",
    ).create()

    assert _OptionRecordingWriter.observed_options == [
        "forwarded",
        "forwarded",
    ]
    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler_kwargs"] == {"required_option": "forwarded"}


def test_dataset_creator_normalizes_legacy_writer_alias_and_class_option(
    tmp_path,
) -> None:
    _OptionRecordingWriter.observed_options.clear()
    DatasetCreator(
        dataloader=DataLoader(_SignalDataset(), batch_size=3),
        root=tmp_path,
        file_writer=_OptionRecordingWriter,
        multithreading=False,
        required_option=PackedHDF5Writer,
    ).create()

    assert _OptionRecordingWriter.observed_options == [
        PackedHDF5Writer,
        PackedHDF5Writer,
    ]
    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler"] == "_OptionRecordingWriter"
    assert writer_info["file_handler_kwargs"] == {"required_option": ("torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Writer")}
