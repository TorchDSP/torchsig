"""Tests for the homogeneous-top-level HDF5 format."""

import h5py
import numpy as np
import pytest

import torchsig.utils.file_handlers.homogeneous_hdf5 as homogeneous_module
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)


def _signals() -> list[Signal]:
    result = []
    for idx, component_count in enumerate((0, 1, 3)):
        components = [
            Signal(
                data=np.arange(
                    2 + component_idx,
                    dtype=np.float32 if component_idx % 2 == 0 else np.int16,
                ).reshape((2, 2) if component_idx == 2 else (-1,)),
                component_index=component_idx,
            )
            for component_idx in range(component_count)
        ]
        result.append(
            Signal(
                data=np.full((2, 8), idx, dtype=np.complex64),
                component_signals=components,
                sample_index=idx,
            )
        )
    return result


def _write_signals(root) -> None:
    with HomogeneousHDF5Writer(root) as writer:
        writer.write(0, _signals())


def test_homogeneous_hdf5_round_trip_variable_components(tmp_path) -> None:
    signals = _signals()
    with HomogeneousHDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ) as writer:
        writer.write(0, signals[:2])
        writer.write(1, signals[2:])

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        assert len(reader) == len(signals)
        for idx, expected in enumerate(signals):
            actual = reader.read(idx)
            np.testing.assert_array_equal(actual.data, expected.data)
            assert actual.data.dtype == expected.data.dtype
            assert actual["sample_index"] == idx
            assert len(actual.component_signals) == len(expected.component_signals)
            for actual_component, expected_component in zip(
                actual.component_signals,
                expected.component_signals,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    actual_component.data,
                    expected_component.data,
                )
                assert actual_component["component_index"] == expected_component["component_index"]
    finally:
        reader.teardown()


def test_homogeneous_hdf5_round_trips_scalar_and_empty_components(
    tmp_path,
) -> None:
    scalar_component = Signal(data=np.array([3.5], dtype=np.float32))
    scalar_component.data = np.array(3.5, dtype=np.float32)
    components = [
        scalar_component,
        Signal(data=np.empty((2, 0), dtype=np.int16)),
    ]
    signal = Signal(
        data=np.ones(8, dtype=np.complex64),
        component_signals=components,
    )
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, [signal])

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)
        for actual_component, expected_component in zip(
            actual.component_signals,
            components,
            strict=True,
        ):
            np.testing.assert_array_equal(
                actual_component.data,
                expected_component.data,
            )
            assert actual_component.data.shape == expected_component.data.shape
            assert actual_component.data.dtype == expected_component.data.dtype
    finally:
        reader.teardown()


def test_homogeneous_hdf5_rejects_empty_batch_without_consuming_index(
    tmp_path,
) -> None:
    with HomogeneousHDF5Writer(tmp_path) as writer:
        with pytest.raises(ValueError, match="must not be empty"):
            writer.write(0, [])
        writer.write(0, [_signals()[0]])


@pytest.mark.parametrize(
    ("batch_idx", "error", "message"),
    [
        (True, TypeError, "must be an integer"),
        (0.0, TypeError, "must be an integer"),
        (-1, ValueError, "must be non-negative"),
        (1, ValueError, "requires sequential"),
    ],
)
def test_homogeneous_hdf5_rejects_invalid_batch_indices(
    tmp_path,
    batch_idx,
    error,
    message,
) -> None:
    with HomogeneousHDF5Writer(tmp_path) as writer:
        with pytest.raises(error, match=message):
            writer.write(batch_idx, [_signals()[0]])
        writer.write(0, [_signals()[0]])


def test_homogeneous_hdf5_rejects_empty_finalization(tmp_path) -> None:
    writer = HomogeneousHDF5Writer(tmp_path)
    writer.setup()

    with pytest.raises(ValueError, match="cannot finalize an empty dataset"):
        writer.teardown()
    writer.teardown()

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert not file.attrs["complete"]


def test_homogeneous_hdf5_writer_can_be_reused_after_teardown(tmp_path) -> None:
    writer = HomogeneousHDF5Writer(tmp_path)
    writer.setup()
    writer.write(0, [_signals()[0]])
    writer.teardown()

    replacement = Signal(data=np.arange(4, dtype=np.float32))
    writer.setup()
    writer.write(0, [replacement])
    writer.teardown()
    writer.teardown()

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        assert len(reader) == 1
        np.testing.assert_array_equal(reader.read(0).data, replacement.data)
    finally:
        reader.teardown()


def test_homogeneous_hdf5_rejects_setup_while_open(tmp_path) -> None:
    with HomogeneousHDF5Writer(tmp_path) as writer:
        with pytest.raises(RuntimeError, match="already open"):
            writer.setup()
        writer.write(0, [_signals()[0]])


def test_homogeneous_hdf5_writes_frozen_format_version(tmp_path) -> None:
    _write_signals(tmp_path)

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file.attrs["format"] == "torchsig-homogeneous"
        assert file.attrs["schema_version"] == 1


def test_homogeneous_hdf5_rejects_unsupported_schema_version(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file.attrs["schema_version"] = 2

    with pytest.raises(ValueError, match=r"Unsupported.*schema version"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_other_format(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file.attrs["format"] = "other-format"

    with pytest.raises(ValueError, match="Not a homogeneous"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_incomplete_file(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file.attrs["complete"] = False

    with pytest.raises(ValueError, match="file is incomplete"):
        len(HomogeneousHDF5Reader(tmp_path))


@pytest.mark.parametrize(
    "name",
    ["data", "component_offsets", "component_data"],
)
def test_homogeneous_hdf5_rejects_missing_required_storage(
    tmp_path,
    name,
) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        del file[name]

    with pytest.raises(ValueError, match="missing required"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_invalid_component_offsets(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["component_offsets"][-1] += 1

    with pytest.raises(ValueError, match="component offsets are invalid"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_top_level_length_mismatch(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["metadata"].resize(len(file["metadata"]) - 1, axis=0)

    with pytest.raises(ValueError, match="data and metadata lengths differ"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_missing_component_stream(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        del file["component_data"]["0"]

    with pytest.raises(ValueError, match="data stream is missing"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_invalid_component_data_range(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        record = file["components"][0]
        record["data_offset"] = len(file["component_data"]["0"])
        file["components"][0] = record

    with pytest.raises(ValueError, match="data range is out of bounds"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_component_shape_mismatch(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        record = file["components"][0]
        record["data_length"] = 1
        file["components"][0] = record

    with pytest.raises(ValueError, match="shape does not match"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_validation_crosses_chunk_boundaries(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        homogeneous_module,
        "_VALIDATION_CHUNK_RECORDS",
        2,
    )
    signals = [
        Signal(
            data=np.ones(8, dtype=np.complex64),
            component_signals=[Signal(data=np.array([idx], dtype=np.float32))],
        )
        for idx in range(6)
    ]
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, signals)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["component_offsets"][4] = 1

    with pytest.raises(ValueError, match="component offsets are invalid"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_caches_successful_validation(
    tmp_path,
    monkeypatch,
) -> None:
    _write_signals(tmp_path)
    calls = 0
    original = HomogeneousHDF5Reader._validate_integrity

    def count_validation(reader) -> None:
        nonlocal calls
        calls += 1
        original(reader)

    monkeypatch.setattr(
        HomogeneousHDF5Reader,
        "_validate_integrity",
        count_validation,
    )
    first = HomogeneousHDF5Reader(tmp_path)
    assert len(first) == 3
    first.teardown()
    second = HomogeneousHDF5Reader(tmp_path)
    assert len(second) == 3
    second.teardown()
    assert calls == 1


def test_homogeneous_hdf5_invalidates_validation_cache(tmp_path) -> None:
    _write_signals(tmp_path)
    reader = HomogeneousHDF5Reader(tmp_path)
    assert len(reader) == 3
    reader.teardown()

    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["component_offsets"][-1] += 1
        file.attrs["cache_buster"] = 1

    with pytest.raises(ValueError, match="component offsets are invalid"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_reads_native_contiguous_batch(tmp_path) -> None:
    signals = _signals()
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, signals)

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        actual = reader.read_batch(0, len(signals))
        expected = np.stack([signal.data for signal in signals])
        np.testing.assert_array_equal(actual, expected)
    finally:
        reader.teardown()


def test_homogeneous_hdf5_reads_contiguous_signal_batch(tmp_path) -> None:
    signals = _signals()
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, signals)

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        actual = reader.read_signals_batch(0, len(signals))
        assert len(actual) == len(signals)
        for actual_signal, expected_signal in zip(actual, signals, strict=True):
            np.testing.assert_array_equal(actual_signal.data, expected_signal.data)
            assert actual_signal.metadata == expected_signal.metadata
            assert len(actual_signal.component_signals) == len(expected_signal.component_signals)
            for actual_component, expected_component in zip(
                actual_signal.component_signals,
                expected_signal.component_signals,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    actual_component.data,
                    expected_component.data,
                )
                assert actual_component.metadata == expected_component.metadata
    finally:
        reader.teardown()


@pytest.mark.parametrize(
    ("start", "stop"),
    [(-1, 1), (0, 4), (2, 1)],
)
def test_homogeneous_hdf5_rejects_invalid_signal_batch_range(
    tmp_path,
    start,
    stop,
) -> None:
    _write_signals(tmp_path)
    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        with pytest.raises(IndexError, match="batch range out of bounds"):
            reader.read_signals_batch(start, stop)
    finally:
        reader.teardown()


@pytest.mark.parametrize(
    ("shape", "dtype", "expected_chunks"),
    [
        ((65_536,), np.complex64, (1, 65_536)),
        ((128, 256), np.float32, (1, 128, 256)),
        ((2_048,), np.complex64, (32, 2_048)),
    ],
    ids=["wideband", "spectrogram", "narrowband"],
)
def test_homogeneous_hdf5_selects_top_level_chunks(
    tmp_path,
    shape,
    dtype,
    expected_chunks,
) -> None:
    signal = Signal(data=np.ones(shape, dtype=dtype))
    with HomogeneousHDF5Writer(
        tmp_path,
        compression="lzf",
        chunk_samples=32,
    ) as writer:
        writer.write(0, [signal])

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file["data"].chunks == expected_chunks


@pytest.mark.parametrize(
    "signal",
    [
        Signal(data=np.ones((2, 7), dtype=np.complex64)),
        Signal(data=np.ones((2, 8), dtype=np.float32)),
    ],
    ids=["shape", "dtype"],
)
def test_homogeneous_hdf5_rejects_heterogeneous_top_level_data(tmp_path, signal) -> None:
    writer = HomogeneousHDF5Writer(tmp_path)
    writer.setup()
    writer.write(0, [_signals()[0]])
    with pytest.raises(ValueError, match="share one dtype and shape"):
        writer.write(1, [signal])
    writer.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert not file.attrs["complete"]


def test_homogeneous_hdf5_flattens_parent_metadata(tmp_path) -> None:
    parent = HierarchicalMetadataObject(metadata={"sample_rate": 1.0})
    signal = Signal(
        data=np.ones((2, 8), dtype=np.complex64),
        parent=parent,
        sample_index=3,
    )
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, [signal])

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)
        assert actual.parent is None
        assert actual["sample_rate"] == 1.0
        assert actual["sample_index"] == 3
    finally:
        reader.teardown()


def test_homogeneous_hdf5_rejects_parent_metadata_cycle(tmp_path) -> None:
    signal = Signal(data=np.ones((2, 8), dtype=np.complex64))
    signal.parent = signal
    with (
        HomogeneousHDF5Writer(tmp_path) as writer,
        pytest.raises(ValueError, match="parent metadata cycle"),
    ):
        writer.write(0, [signal])


def test_homogeneous_hdf5_rejects_nested_components(tmp_path) -> None:
    nested = Signal(data=np.ones(2, dtype=np.complex64))
    component = Signal(
        data=np.ones(3, dtype=np.complex64),
        component_signals=[nested],
    )
    signal = Signal(
        data=np.ones((2, 8), dtype=np.complex64),
        component_signals=[component],
    )
    with (
        HomogeneousHDF5Writer(tmp_path) as writer,
        pytest.raises(ValueError, match="does not support nested components"),
    ):
        writer.write(0, [signal])
