"""Integration tests for DatasetCreator with homogeneous HDF5 storage."""

import h5py
import numpy as np
import pytest
import yaml
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
    PackedHDF5Reader,
)
from torchsig.utils.writer import DatasetCreator

_DATASET_LENGTH = 8
_NUM_SAMPLES = 64


def _signal_summary_collate(
    batch: list[Signal],
) -> list[tuple[int, tuple[int, ...], str]]:
    return [
        (
            signal["sample_index"],
            signal.data.shape,
            signal.data.dtype.str,
        )
        for signal in batch
    ]


class _HomogeneousSignalDataset(Dataset):
    def __init__(self, *, spectrogram: bool = False) -> None:
        self.spectrogram = spectrogram
        self.parent = HierarchicalMetadataObject(
            metadata={"sample_rate": 1.0},
        )

    def __len__(self) -> int:
        return _DATASET_LENGTH

    def __getitem__(self, idx: int) -> Signal:
        data = np.full(_NUM_SAMPLES, idx, dtype=np.complex64)
        components = [
            Signal(
                data=np.arange(component_idx + 2, dtype=np.float32),
                parent=self.parent,
                component_index=component_idx,
            )
            for component_idx in range(idx % 4)
        ]
        signal = Signal(
            data=data,
            component_signals=components,
            parent=self.parent,
            sample_index=idx,
        )
        return Spectrogram(fft_size=8)(signal) if self.spectrogram else signal


@pytest.mark.parametrize("multithreading", [False, True])
@pytest.mark.parametrize(
    "spectrogram",
    [False, True],
    ids=["iq", "spectrogram"],
)
def test_dataset_creator_round_trips_homogeneous_signals(
    tmp_path,
    multithreading,
    spectrogram,
) -> None:
    source = _HomogeneousSignalDataset(spectrogram=spectrogram)
    DatasetCreator(
        dataloader=DataLoader(source, batch_size=3),
        root=tmp_path,
        file_handler=HomogeneousHDF5Writer,
        multithreading=multithreading,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ).create()

    dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=HomogeneousHDF5Reader,
        target_labels=None,
    )
    assert len(dataset) == _DATASET_LENGTH
    for idx in range(_DATASET_LENGTH):
        expected = source[idx]
        actual = dataset[idx]
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.data.dtype == expected.data.dtype
        assert actual.data.shape == expected.data.shape
        assert actual["sample_index"] == idx
        assert actual["sample_rate"] == 1.0
        assert actual.parent is None
        assert len(actual.component_signals) == idx % 4
        for component in actual.component_signals:
            assert component["sample_rate"] == 1.0
            assert component.parent is None
    dataset.reader.teardown()

    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler"] == "HomogeneousHDF5Writer"
    assert writer_info["file_reader_qualified"].endswith(".HomogeneousHDF5Reader")


def test_dataset_creator_homogeneous_dataset_supports_workers(tmp_path) -> None:
    source = _HomogeneousSignalDataset(spectrogram=True)
    DatasetCreator(
        dataloader=DataLoader(source, batch_size=4),
        root=tmp_path,
        file_handler=HomogeneousHDF5Writer,
        multithreading=False,
    ).create()
    dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=HomogeneousHDF5Reader,
        target_labels=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=_signal_summary_collate,
    )

    actual = [summary for batch in loader for summary in batch]
    assert [summary[0] for summary in actual] == list(range(_DATASET_LENGTH))
    assert all(summary[1] == (8, 8) for summary in actual)
    assert all(summary[2] == np.dtype(np.float32).str for summary in actual)
    dataset.reader.teardown()


def test_dataset_creator_forwards_homogeneous_writer_options(
    tmp_path,
) -> None:
    options = {
        "compression": None,
        "shuffle": False,
        "fletcher32": False,
        "chunk_samples": 4,
    }
    DatasetCreator(
        dataloader=DataLoader(
            _HomogeneousSignalDataset(),
            batch_size=4,
        ),
        root=tmp_path,
        file_handler=HomogeneousHDF5Writer,
        multithreading=False,
        **options,
    ).create()

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file["data"].compression is None
        assert file["data"].chunks[0] == 4
    writer_info = yaml.safe_load((tmp_path / "writer_info.yaml").read_text())
    assert writer_info["file_handler_kwargs"] == options


def test_dataset_creator_rejects_incompatible_homogeneous_reader(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="Incompatible file handler pair"):
        DatasetCreator(
            dataloader=DataLoader(
                _HomogeneousSignalDataset(),
                batch_size=4,
            ),
            root=tmp_path,
            file_handler=HomogeneousHDF5Writer,
            file_reader=PackedHDF5Reader,
        )
