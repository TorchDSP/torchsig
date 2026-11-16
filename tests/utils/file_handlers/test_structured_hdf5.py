"""Tests for structured homogeneous HDF5 storage."""

import json
import pickle

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
    StructuredHDF5Reader,
    StructuredHDF5Writer,
    infer_structured_sample_schema,
)


def _samples() -> list[dict]:
    return [
        {
            "input": (
                np.full((2, 3), index + 1j * index, dtype=np.complex64),
                np.array([index, index + 1], dtype=np.float32),
            ),
            "target": {"class": np.int16(index), "valid": np.bool_(index % 2)},
        }
        for index in range(5)
    ]


def _assert_sample_equal(actual, expected) -> None:
    assert isinstance(actual, dict)
    assert isinstance(actual["input"], tuple)
    assert isinstance(actual["target"], dict)
    for key in (0, 1):
        np.testing.assert_array_equal(actual["input"][key], expected["input"][key])
        assert actual["input"][key].dtype == expected["input"][key].dtype
    for key in ("class", "valid"):
        np.testing.assert_array_equal(actual["target"][key], expected["target"][key])
        assert actual["target"][key].dtype == expected["target"][key].dtype


def _write_samples(root) -> None:
    samples = _samples()
    with StructuredHDF5Writer(root, compression=None, shuffle=False, fletcher32=False) as writer:
        writer.write(0, samples[:2])
        writer.write(1, samples[2:])


def test_round_trip_mapping_tuple_fields_and_random_access(tmp_path) -> None:
    samples = _samples()
    _write_samples(tmp_path)

    with StructuredHDF5Reader(tmp_path) as reader:
        assert len(reader) == len(samples)
        for index in (4, 0, 2, 1):
            _assert_sample_equal(reader.read(index), samples[index])


def test_contiguous_batch_read_round_trips_samples(tmp_path) -> None:
    samples = _samples()
    _write_samples(tmp_path)

    with StructuredHDF5Reader(tmp_path) as reader:
        actual = reader.read_batch(1, 4)
        assert reader.read_batch(2, 2) == []

    for item, expected in zip(actual, samples[1:4], strict=True):
        _assert_sample_equal(item, expected)


def test_index_batch_restores_shuffled_order_and_duplicates(tmp_path) -> None:
    samples = _samples()
    _write_samples(tmp_path)

    with StructuredHDF5Reader(tmp_path) as reader:
        actual = reader.read_indices([4, 1, 2, 1, 0])

    for item, expected_index in zip(actual, [4, 1, 2, 1, 0], strict=True):
        _assert_sample_equal(item, samples[expected_index])


def test_index_batch_coalesces_contiguous_runs_per_leaf(tmp_path) -> None:
    _write_samples(tmp_path)
    reader = StructuredHDF5Reader(tmp_path)
    assert len(reader) == 5
    keys: list[list[object]] = [[] for _ in reader._datasets]

    class TrackingDataset:
        def __init__(self, dataset, field_keys) -> None:
            self.dataset = dataset
            self.field_keys = field_keys

        def __getitem__(self, key):
            self.field_keys.append(key)
            return self.dataset[key]

    reader._datasets = [TrackingDataset(dataset, field_keys) for dataset, field_keys in zip(reader._datasets, keys, strict=True)]
    try:
        reader.read_indices([4, 0, 2, 1])
    finally:
        reader.close()

    assert keys
    assert all(field_keys == [slice(0, 3), slice(4, 5)] for field_keys in keys)


def test_index_batch_avoids_slice_overhead_for_sparse_indices(tmp_path) -> None:
    _write_samples(tmp_path)
    reader = StructuredHDF5Reader(tmp_path)
    assert len(reader) == 5
    keys: list[list[object]] = [[] for _ in reader._datasets]

    class TrackingDataset:
        def __init__(self, dataset, field_keys) -> None:
            self.dataset = dataset
            self.field_keys = field_keys

        def __getitem__(self, key):
            self.field_keys.append(key)
            return self.dataset[key]

    reader._datasets = [TrackingDataset(dataset, field_keys) for dataset, field_keys in zip(reader._datasets, keys, strict=True)]
    try:
        reader.read_indices([4, 2, 0])
    finally:
        reader.close()

    assert keys
    assert all(field_keys == [4, 2, 0] for field_keys in keys)


def test_writer_persists_schema_metadata_and_leaf_datasets(tmp_path) -> None:
    _write_samples(tmp_path)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["format"] == "torchsig-structured-homogeneous"
        assert handle.attrs["schema_major"] == 1
        assert handle.attrs["schema_minor"] == 0
        assert handle.attrs["length"] == 5
        assert bool(handle.attrs["complete"])
        assert json.loads(handle["schema"].asstr()[()])["fields"][0]["path"] == ["input", 0]
        assert tuple(handle["fields/0"].shape) == (5, 2, 3)


def test_explicit_schema_supports_empty_dataset(tmp_path) -> None:
    schema = infer_structured_sample_schema((np.empty((0,), dtype=np.float32), np.int64(0)))
    with StructuredHDF5Writer(tmp_path, schema=schema):
        pass

    with StructuredHDF5Reader(tmp_path) as reader:
        assert len(reader) == 0
        assert reader.read_batch(0, 0) == []


def test_writer_rejects_empty_finalization_without_schema(tmp_path) -> None:
    writer = StructuredHDF5Writer(tmp_path)
    writer.setup()
    with pytest.raises(ValueError, match="without a schema"):
        writer.teardown()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda handle: handle.attrs.update({"complete": False}), "incomplete"),
        (lambda handle: handle.attrs.__setitem__("schema_major", 99), "major version"),
        (lambda handle: handle.__delitem__("fields/0"), "missing required field"),
        (lambda handle: handle.attrs.__setitem__("length", 4), "has shape"),
    ],
)
def test_reader_rejects_malformed_files(tmp_path, mutation, message) -> None:
    _write_samples(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        mutation(handle)

    with pytest.raises(ValueError, match=message):
        len(StructuredHDF5Reader(tmp_path))


def test_reader_rejects_field_dtype_mismatch(tmp_path) -> None:
    _write_samples(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        values = handle["fields/1"][:].astype(np.float64)
        del handle["fields/1"]
        handle.create_dataset("fields/1", data=values)

    with pytest.raises(ValueError, match="has dtype"):
        len(StructuredHDF5Reader(tmp_path))


def test_invalid_indices_and_ranges_are_rejected(tmp_path) -> None:
    _write_samples(tmp_path)
    with StructuredHDF5Reader(tmp_path) as reader:
        with pytest.raises(IndexError, match="out of range"):
            reader.read(-1)
        with pytest.raises(IndexError, match="out of bounds"):
            reader.read_batch(3, 6)
        assert reader.read_indices([]) == []
        with pytest.raises(TypeError, match="sequence"):
            reader.read_indices(1)
        with pytest.raises(TypeError, match="integers"):
            reader.read_indices([1.5])
        with pytest.raises(TypeError, match="integers"):
            reader.read_indices([True])
        with pytest.raises(IndexError, match="out of range"):
            reader.read_indices([5])


def test_existing_homogeneous_version_one_remains_compatible(tmp_path) -> None:
    signal = Signal(data=np.arange(8, dtype=np.complex64))
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, [signal])

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        np.testing.assert_array_equal(reader.read(0).data, signal.data)
    finally:
        reader.teardown()


def test_reader_is_lazy_and_reopens_after_explicit_close(tmp_path) -> None:
    _write_samples(tmp_path)
    reader = StructuredHDF5Reader(tmp_path)

    assert reader._file is None
    _assert_sample_equal(reader.read(0), _samples()[0])
    assert reader._file is not None
    reader.close()
    assert reader._file is None
    assert reader._datasets == []
    _assert_sample_equal(reader.read(1), _samples()[1])
    reader.close()


def test_reader_pickle_state_excludes_hdf5_handles(tmp_path) -> None:
    _write_samples(tmp_path)
    reader = StructuredHDF5Reader(tmp_path)
    assert len(reader) == 5

    restored = pickle.loads(pickle.dumps(reader))
    try:
        assert restored._file is None
        assert restored._pid is None
        assert restored._datasets == []
        _assert_sample_equal(restored.read(4), _samples()[4])
    finally:
        reader.close()
        restored.close()
