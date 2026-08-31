"""Tests for the self-describing packed HDF5 schema."""

import json

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5_schema import (
    default_packed_schema,
    read_schema,
)
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)


def _write_file(root) -> None:
    with PackedHDF5Writer(root, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])


def _update_schema(root, update) -> None:
    with h5py.File(root / "data.h5", "r+") as handle:
        payload = handle["schema"][()]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        update(value)
        handle["schema"][()] = json.dumps(value, separators=(",", ":"))


def test_packed_writer_embeds_frozen_schema(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        schema = read_schema(handle)

    assert schema == default_packed_schema()
    assert schema.format == "torchsig-packed"
    assert schema.schema_major == 1
    assert schema.schema_minor == 0


def test_packed_reader_rejects_unsupported_schema_major(tmp_path) -> None:
    _write_file(tmp_path)
    _update_schema(tmp_path, lambda value: value.update(schema_major=999))

    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="schema major version"):
        reader.read(0)
    assert reader._file is None  # noqa: SLF001


def test_packed_reader_rejects_other_format(tmp_path) -> None:
    _write_file(tmp_path)
    _update_schema(tmp_path, lambda value: value.update(format="other-format"))

    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="Unsupported HDF5 format"):
        reader.read(0)


def test_packed_reader_rejects_unknown_required_feature(tmp_path) -> None:
    _write_file(tmp_path)

    def add_feature(value) -> None:
        value["required_features"].append("future_required_feature")

    _update_schema(tmp_path, add_feature)
    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="Unsupported required"):
        reader.read(0)


def test_packed_reader_rejects_missing_declared_path(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        del handle["shapes"]

    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="missing declared paths"):
        reader.read(0)


def test_packed_reader_rejects_colliding_schema_paths(tmp_path) -> None:
    _write_file(tmp_path)

    def collide_paths(value) -> None:
        value["datasets"]["shapes"]["path"] = value["datasets"]["index"]["path"]

    _update_schema(tmp_path, collide_paths)
    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="share path"):
        reader.read(0)


@pytest.mark.parametrize("sentinel", [-1, 1 << 64])
def test_packed_reader_rejects_out_of_range_parent_sentinel(tmp_path, sentinel) -> None:
    _write_file(tmp_path)

    def update_sentinel(value) -> None:
        value["sentinels"]["no_parent"] = sentinel

    _update_schema(tmp_path, update_sentinel)
    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="sentinel does not fit uint64"):
        reader.read(0)


def test_packed_reader_rejects_incomplete_file(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        handle.attrs["complete"] = False

    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="file is incomplete"):
        reader.read(0)
    assert reader._file is None  # noqa: SLF001


def test_packed_reader_rejects_missing_completeness_marker(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        del handle.attrs["complete"]

    reader = PackedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="missing completeness marker"):
        reader.read(0)
    assert reader._file is None  # noqa: SLF001
