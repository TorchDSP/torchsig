"""Compatibility tests for committed packed and homogeneous HDF5 files."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
)
from torchsig.utils.file_handlers.packed_hdf5 import PackedHDF5Reader

_GOLDEN_ROOT = Path(__file__).with_name("golden")


def test_reads_packed_v1_golden_file() -> None:
    root = _GOLDEN_ROOT / "packed_v1"
    golden_file = root / "data.h5"

    if not golden_file.exists():
        pytest.skip(f"Golden file not available: {golden_file}")

    with h5py.File(golden_file, "r") as file:
        schema = file["schema"].asstr()[()]
        assert '"format":"torchsig-packed"' in schema
        assert '"schema_major":1' in schema
        assert '"schema_minor":0' in schema

    reader = PackedHDF5Reader(root)
    try:
        assert len(reader) == 2
        first = reader.read(0)
        second = reader.read(1)
    finally:
        reader.teardown()

    np.testing.assert_array_equal(
        first.data,
        np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
    )
    assert first.data.dtype == np.dtype(np.complex64)
    assert first["name"] == "packed-first"
    assert first["bounds"] == (1, 3)
    assert first["scalar"] == np.int16(7)
    assert type(first["scalar"]) is np.int16
    assert first["dataset"] == "golden"
    assert first["sample_rate"] == 8.0
    assert first.parent is not None
    assert len(first.component_signals) == 1
    np.testing.assert_array_equal(
        first.component_signals[0].data,
        np.array([9, 10], dtype=np.int16),
    )
    assert first.component_signals[0]["kind"] == "pulse"

    np.testing.assert_array_equal(
        second.data,
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    assert second.data.dtype == np.dtype(np.float32)
    assert second["name"] == "packed-second"
    assert second["payload"] == b"golden"
    assert second.parent is not None
    assert second.parent["dataset"] == first.parent["dataset"]
    assert second.parent["sample_rate"] == first.parent["sample_rate"]
    assert second.component_signals == []


def test_reads_homogeneous_v1_golden_file() -> None:
    root = _GOLDEN_ROOT / "homogeneous_v1"
    golden_file = root / "data.h5"

    if not golden_file.exists():
        pytest.skip(f"Golden file not available: {golden_file}")

    with h5py.File(golden_file, "r") as file:
        assert file.attrs["format"] == "torchsig-homogeneous"
        assert file.attrs["schema_version"] == 1

    reader = HomogeneousHDF5Reader(root)
    try:
        assert len(reader) == 2
        batch = reader.read_batch(0, 2)
        signals = [reader.read(0), reader.read(1)]
    finally:
        reader.teardown()

    np.testing.assert_array_equal(
        batch,
        np.stack(
            [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                (np.arange(6, dtype=np.float32) + 10).reshape(2, 3),
            ]
        ),
    )
    np.testing.assert_array_equal(
        signals[0].data,
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    assert signals[0].data.dtype == np.dtype(np.float32)
    assert signals[0]["name"] == "homogeneous-first"
    assert signals[0]["dataset"] == "golden"
    assert signals[0]["sample_rate"] == 8.0
    assert signals[0].parent is None
    assert len(signals[0].component_signals) == 1
    np.testing.assert_array_equal(
        signals[0].component_signals[0].data,
        np.array([1 + 1j], dtype=np.complex64),
    )
    assert signals[0].component_signals[0]["kind"] == "tone"

    np.testing.assert_array_equal(
        signals[1].data,
        (np.arange(6, dtype=np.float32) + 10).reshape(2, 3),
    )
    assert signals[1].data.dtype == np.dtype(np.float32)
    assert signals[1]["name"] == "homogeneous-second"
    assert signals[1].parent is None
    assert len(signals[1].component_signals) == 2
    np.testing.assert_array_equal(
        signals[1].component_signals[0].data,
        np.array([2, 3], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        signals[1].component_signals[1].data,
        np.array([[4.0, 5.0]], dtype=np.float64),
    )
    assert signals[1].component_signals[0]["kind"] == "left"
    assert signals[1].component_signals[1]["kind"] == "right"
