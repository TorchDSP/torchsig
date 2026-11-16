"""Tests for the variable-length packed NPY file handler."""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets.datamodules import _resolve_file_reader as resolve_datamodule_reader
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import NPYReader
from torchsig.utils.file_handlers.packed_npy import PackedNPYReader, PackedNPYWriter
from torchsig.utils.writer import _resolve_file_reader as resolve_creator_reader

if TYPE_CHECKING:
    from pathlib import Path

_COMPLEX64 = np.dtype("complex64")


def _identity_collate(batch):
    return batch


class _PackedDataset(Dataset):
    """Pickleable adapter used to exercise DataLoader worker processes."""

    def __init__(self, root: Path) -> None:
        self.reader = PackedNPYReader(root)

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        signal = self.reader.read(index)
        return index, np.asarray(signal.data).copy()


def _signals(dtype: np.dtype = _COMPLEX64) -> list[Signal]:
    lengths = (1, 5, 0, 9)
    return [
        Signal(
            data=(np.arange(length, dtype=np.float32) + index * 1j if np.issubdtype(dtype, np.complexfloating) else np.arange(length, dtype=np.float32) + index).astype(dtype),
            record_name=f"record-{index}",
        )
        for index, length in enumerate(lengths)
    ]


def _write(root: Path, signals: list[Signal]) -> None:
    with PackedNPYWriter(root) as writer:
        writer.write(1, signals[2:])
        writer.write(0, signals[:2])


def test_round_trip_variable_lengths_boundary_and_empty_record(tmp_path: Path) -> None:
    signals = _signals()
    _write(tmp_path, signals)

    reader = PackedNPYReader(tmp_path)

    assert len(reader) == len(signals)
    for index in (0, 2, len(signals) - 1):
        actual = reader.read(index)
        np.testing.assert_array_equal(actual.data, signals[index].data)
        assert actual.data.shape == signals[index].data.shape
        assert actual.data.dtype == signals[index].data.dtype
        assert actual.metadata["record_name"] == f"record-{index}"
    assert isinstance(reader.read(0).data.base, np.memmap)


@pytest.mark.parametrize("dtype", [np.dtype("<c8"), np.dtype(">c8"), np.dtype("<f8")])
def test_round_trip_preserves_supported_dtype_and_byte_order(tmp_path: Path, dtype: np.dtype) -> None:
    signals = _signals(dtype)
    _write(tmp_path, signals)

    reader = PackedNPYReader(tmp_path)

    assert reader.dtype == dtype
    assert reader.read(1).data.dtype == dtype
    np.testing.assert_array_equal(reader.read(1).data, signals[1].data)


def test_round_trip_multidimensional_shapes(tmp_path: Path) -> None:
    signals = [
        Signal(data=np.arange(6, dtype=np.float32).reshape(2, 3), name="matrix"),
        Signal(data=np.arange(8, dtype=np.float32).reshape(2, 2, 2), name="cube"),
    ]
    with PackedNPYWriter(tmp_path) as writer:
        writer.write(0, signals)

    reader = PackedNPYReader(tmp_path)
    np.testing.assert_array_equal(reader.read(0).data, signals[0].data)
    np.testing.assert_array_equal(reader.read(1).data, signals[1].data)


def test_rejects_mixed_dtypes_without_publishing_dataset(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="requires one dtype"), PackedNPYWriter(tmp_path) as writer:
        writer.write(
            0,
            [
                Signal(data=np.ones(2, dtype=np.complex64)),
                Signal(data=np.ones(2, dtype=np.complex128)),
            ],
        )

    assert not (tmp_path / "packed_npy.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records["sample_offset"].__setitem__(1, 0), "overlaps"),
        (lambda records: records["sample_count"].__setitem__(0, 10_000), "out of bounds"),
        (lambda records: records["shape_rank"].__setitem__(0, 10_000), "shape range"),
        (lambda records: records["metadata_index"].__setitem__(0, 10_000), "metadata index"),
        (lambda records: records["record_index"].__setitem__(0, 7), "record_index"),
    ],
)
def test_rejects_corrupt_descriptors(tmp_path: Path, mutation, message: str) -> None:
    _write(tmp_path, _signals())
    path = tmp_path / "records.npy"
    records = np.load(path, allow_pickle=False)
    mutation(records)
    np.save(path, records, allow_pickle=False)

    with pytest.raises(ValueError, match=message):
        PackedNPYReader(tmp_path)


def test_rejects_shape_that_does_not_match_length(tmp_path: Path) -> None:
    _write(tmp_path, _signals())
    path = tmp_path / "shapes.npy"
    shapes = np.load(path, allow_pickle=False)
    shapes[0] = 2
    np.save(path, shapes, allow_pickle=False)

    with pytest.raises(ValueError, match="shape does not match"):
        PackedNPYReader(tmp_path)


def test_bounds_and_incomplete_batch_sequence(tmp_path: Path) -> None:
    _write(tmp_path, _signals())
    reader = PackedNPYReader(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        reader.read(-1)
    with pytest.raises(IndexError, match="out of range"):
        reader.read(len(reader))

    incomplete = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="missing batch index 0"), PackedNPYWriter(incomplete) as writer:
        writer.write(1, _signals())


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_shuffled_multiworker_access(tmp_path: Path, context: str) -> None:
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} multiprocessing is unavailable")
    signals = [Signal(data=np.full(index % 7 + 1, index, dtype=np.float32)) for index in range(24)]
    with PackedNPYWriter(tmp_path) as writer:
        writer.write(0, signals)

    generator = np.random.default_rng(41)
    expected_order = generator.permutation(len(signals)).tolist()
    sampler = expected_order
    loader = DataLoader(
        _PackedDataset(tmp_path),
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        multiprocessing_context=context,
        collate_fn=_identity_collate,
    )

    observed = []
    for batch in loader:
        for index, data in batch:
            observed.append(int(index))
            np.testing.assert_array_equal(data, signals[int(index)].data)
    assert observed == expected_order


def test_legacy_one_file_per_record_reader_remains_available() -> None:
    assert NPYReader.__name__ == "NPYReader"


def test_dataset_creation_integrations_resolve_matching_reader() -> None:
    assert resolve_creator_reader(PackedNPYWriter, None) is PackedNPYReader
    assert resolve_datamodule_reader(PackedNPYWriter, None) is PackedNPYReader
