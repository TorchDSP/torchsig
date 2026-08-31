"""Tests for the packed HDF5 format."""

import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers import packed_hdf5
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)


def test_packed_hdf5_round_trip_components_and_metadata(tmp_path) -> None:
    grandparent = HierarchicalMetadataObject(metadata={"sample_rate": 2_000_000.0, "labels": np.array([1, 2])})
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={"split": "train"},
    )
    component = Signal(
        data=np.arange(7, dtype=np.complex64),
        parent=parent,
        class_name="component",
        bounds=(1, 4),
    )
    signals = [
        Signal(
            data=(np.arange(16 + idx) + 1j * idx).astype(np.complex64),
            component_signals=[component.copy()],
            parent=parent,
            class_name="sample",
            sample_index=np.int64(idx),
        )
        for idx in range(3)
    ]

    with PackedHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=2,
    ) as writer:
        writer.write(1, signals[2:])
        writer.write(0, signals[:2])

    reader = PackedHDF5Reader(tmp_path)
    try:
        assert len(reader) == len(signals)
        for idx, expected in enumerate(signals):
            actual = reader.read(idx)
            np.testing.assert_array_equal(actual.data, expected.data)
            assert actual.metadata == expected.metadata
            assert actual["split"] == "train"
            np.testing.assert_array_equal(actual["labels"], np.array([1, 2]))
            assert len(actual.component_signals) == 1
            np.testing.assert_array_equal(
                actual.component_signals[0].data,
                expected.component_signals[0].data,
            )
            assert actual.component_signals[0]["bounds"] == (1, 4)
    finally:
        reader.teardown()


def test_packed_hdf5_preserves_mixed_signal_dtypes(tmp_path) -> None:
    signals = [
        Signal(data=np.ones(4, dtype=np.complex64)),
        Signal(data=np.ones((2, 4), dtype=np.float32)),
    ]
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, signals)

    reader = PackedHDF5Reader(tmp_path)
    try:
        complex_signal = reader.read(0)
        real_signal = reader.read(1)
        assert complex_signal.data.dtype == np.complex64
        assert real_signal.data.dtype == np.float32
        assert real_signal.data.shape == (2, 4)
    finally:
        reader.teardown()


def test_packed_hdf5_orders_batches_across_flush_boundaries(tmp_path) -> None:
    batches = {idx: [Signal(data=np.array([idx], dtype=np.int64))] for idx in range(4)}
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=2) as writer:
        writer.write(2, batches[2])
        writer.write(3, batches[3])
        writer.write(0, batches[0])
        writer.write(1, batches[1])

    reader = PackedHDF5Reader(tmp_path)
    try:
        assert [int(reader.read(idx).data[0]) for idx in range(4)] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        reader.teardown()


def test_packed_hdf5_rejects_duplicate_batch_index(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=2)
    writer.setup()
    writer.write(0, [])
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write(0, [])
    writer.teardown()


def test_packed_hdf5_rejects_operations_before_setup(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path)

    with pytest.raises(RuntimeError, match=r"not open.*state is new"):
        writer.write(0, [])
    with pytest.raises(RuntimeError, match=r"length.*state is new"):
        len(writer)

    writer.teardown()


def test_packed_hdf5_rejects_repeated_setup_without_overwriting(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(0, [Signal(data=np.ones(2, dtype=np.complex64))])

    with pytest.raises(RuntimeError, match=r"setup.*state is open"):
        writer.setup()

    assert len(writer) == 1
    writer.teardown()
    reader = PackedHDF5Reader(tmp_path)
    try:
        assert len(reader) == 1
    finally:
        reader.teardown()


def test_packed_hdf5_rejects_operations_after_teardown(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path)
    writer.setup()
    writer.teardown()
    writer.teardown()

    with pytest.raises(RuntimeError, match=r"not open.*state is closed"):
        writer.write(0, [])
    with pytest.raises(RuntimeError, match=r"length.*state is closed"):
        len(writer)
    with pytest.raises(RuntimeError, match=r"setup.*state is closed"):
        writer.setup()


def test_packed_hdf5_rejects_missing_batch_at_teardown(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(1, [])

    with pytest.raises(ValueError, match="missing batch index 0"):
        writer.teardown()
    assert writer._file is None  # noqa: SLF001
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


@pytest.mark.parametrize("max_batches", [0, -1])
def test_packed_hdf5_rejects_non_positive_batch_buffer_limit(tmp_path, max_batches) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PackedHDF5Writer(tmp_path, max_batches_in_memory=max_batches)


@pytest.mark.parametrize("max_batches", [1.5, True])
def test_packed_hdf5_rejects_non_integer_batch_buffer_limit(tmp_path, max_batches) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        PackedHDF5Writer(tmp_path, max_batches_in_memory=max_batches)


def test_packed_hdf5_enforces_out_of_order_batch_buffer_limit(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=2)
    writer.setup()
    writer.write(2, [])
    writer.write(1, [])

    with pytest.raises(BufferError, match=r"buffer is full.*expected batch index 0"):
        writer.write(3, [])

    writer.write(0, [])
    writer.write(3, [])
    writer.teardown()


def test_packed_hdf5_snapshots_buffered_batch_container(tmp_path) -> None:
    expected = Signal(data=np.array([1], dtype=np.int64))
    replacement = Signal(data=np.array([2], dtype=np.int64))
    batch = [expected]
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=2) as writer:
        writer.write(0, batch)
        batch[0] = replacement
        writer.write(1, [])

    reader = PackedHDF5Reader(tmp_path)
    try:
        np.testing.assert_array_equal(reader.read(0).data, expected.data)
    finally:
        reader.teardown()


def test_packed_hdf5_orders_concurrent_batch_writes(tmp_path) -> None:
    batch_count = 16
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=batch_count)
    writer.setup()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                writer.write,
                batch_idx,
                [Signal(data=np.array([batch_idx], dtype=np.int64))],
            )
            for batch_idx in reversed(range(batch_count))
        ]
        for future in futures:
            future.result()
    writer.teardown()

    reader = PackedHDF5Reader(tmp_path)
    try:
        assert [int(reader.read(idx).data[0]) for idx in range(batch_count)] == list(range(batch_count))
    finally:
        reader.teardown()


@pytest.mark.parametrize("batch_idx", [-1, 1.5, True])
def test_packed_hdf5_rejects_invalid_batch_index(tmp_path, batch_idx) -> None:
    with (
        PackedHDF5Writer(tmp_path) as writer,
        pytest.raises((TypeError, ValueError), match="batch index"),
    ):
        writer.write(batch_idx, [])


def test_packed_hdf5_preserves_reserved_metadata_tags(tmp_path) -> None:
    metadata = {
        "__torchsig_type__": "complex",
        "nested": {
            "__torchsig_type__": "ndarray",
            "data": "ordinary user metadata",
        },
    }
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(
            0,
            [Signal(data=np.ones(4, dtype=np.complex64), payload=metadata)],
        )

    reader = PackedHDF5Reader(tmp_path)
    try:
        assert reader.read(0)["payload"] == metadata
    finally:
        reader.teardown()


@pytest.mark.parametrize(
    "value",
    [
        b"\x00\xff",
        1 + 2j,
        (1, "two"),
        np.int16(3),
        np.array([[1, 2], [3, 4]], dtype=np.int32),
    ],
    ids=["bytes", "complex", "tuple", "numpy-scalar", "numpy-array"],
)
def test_packed_hdf5_round_trips_encoded_metadata_types(tmp_path, value) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(
            0,
            [Signal(data=np.ones(4, dtype=np.complex64), value=value)],
        )

    reader = PackedHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)["value"]
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual, value)
        else:
            assert actual == value
            assert type(actual) is type(value)
    finally:
        reader.teardown()


def test_packed_hdf5_rejects_non_string_metadata_dictionary_key(
    tmp_path,
) -> None:
    signal = Signal(
        data=np.ones(4, dtype=np.complex64),
        payload={1: "not allowed"},
    )
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    with pytest.raises(TypeError, match="keys must be strings"):
        writer.write(0, [signal])
    writer.teardown()


def test_packed_hdf5_invalid_batch_does_not_append_partial_data(
    tmp_path,
) -> None:
    valid = Signal(data=np.ones(4, dtype=np.complex64), label="valid")
    invalid = Signal(
        data=np.ones(4, dtype=np.float32),
        unsupported=object(),
    )
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(0, [valid])
    lengths_before = {
        "records": len(writer._records),  # noqa: SLF001
        "metadata": len(writer._metadata),  # noqa: SLF001
        "shapes": len(writer._shapes),  # noqa: SLF001
        "components": len(writer._components),  # noqa: SLF001
        "index": len(writer._index),  # noqa: SLF001
        "dtypes": len(writer._dtypes),  # noqa: SLF001
        "parents": len(writer._parent_records),  # noqa: SLF001
    }
    data_streams_before = set(writer._data_group)  # noqa: SLF001

    with pytest.raises(TypeError, match="Unsupported TorchSig metadata"):
        writer.write(1, [valid, invalid])

    lengths_after = {
        "records": len(writer._records),  # noqa: SLF001
        "metadata": len(writer._metadata),  # noqa: SLF001
        "shapes": len(writer._shapes),  # noqa: SLF001
        "components": len(writer._components),  # noqa: SLF001
        "index": len(writer._index),  # noqa: SLF001
        "dtypes": len(writer._dtypes),  # noqa: SLF001
        "parents": len(writer._parent_records),  # noqa: SLF001
    }
    assert lengths_after == lengths_before
    assert set(writer._data_group) == data_streams_before  # noqa: SLF001
    writer.teardown()


@pytest.mark.parametrize("fail_on_append", [1, 3, 6, 9])
def test_packed_hdf5_rolls_back_failed_batch_commit(tmp_path, monkeypatch, fail_on_append) -> None:
    original_append = packed_hdf5._append  # noqa: SLF001
    append_count = 0

    def failing_append(dataset, values):
        nonlocal append_count
        append_count += 1
        result = original_append(dataset, values)
        if append_count == fail_on_append:
            raise OSError("injected HDF5 append failure")
        return result

    monkeypatch.setattr(packed_hdf5, "_append", failing_append)
    parent = HierarchicalMetadataObject(metadata={"split": "train"})
    component = Signal(data=np.ones(2, dtype=np.complex64))
    signal = Signal(
        data=np.ones(4, dtype=np.complex64),
        component_signals=[component],
        parent=parent,
    )
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()

    with pytest.raises(OSError, match="injected HDF5 append failure"):
        writer.write(0, [signal])

    assert len(writer._dtypes) == 0  # noqa: SLF001
    assert len(writer._records) == 0  # noqa: SLF001
    assert len(writer._metadata) == 0  # noqa: SLF001
    assert len(writer._shapes) == 0  # noqa: SLF001
    assert len(writer._components) == 0  # noqa: SLF001
    assert len(writer._index) == 0  # noqa: SLF001
    assert len(writer._parent_records) == 0  # noqa: SLF001
    assert len(writer._parent_metadata) == 0  # noqa: SLF001
    assert not writer._data  # noqa: SLF001
    assert not writer._dtype_ids  # noqa: SLF001
    assert not writer._parent_ids  # noqa: SLF001
    with pytest.raises(RuntimeError, match="cannot continue"):
        writer.write(1, [signal])

    writer.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


def test_packed_hdf5_rollback_preserves_committed_batches(tmp_path, monkeypatch) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    original_append = packed_hdf5._append  # noqa: SLF001
    append_count = 0

    def failing_append(dataset, values):
        nonlocal append_count
        append_count += 1
        result = original_append(dataset, values)
        if append_count == 8:
            raise OSError("injected HDF5 append failure")
        return result

    monkeypatch.setattr(packed_hdf5, "_append", failing_append)
    parent = HierarchicalMetadataObject(metadata={"split": "test"})
    signal = Signal(
        data=np.ones(3, dtype=np.float32),
        parent=parent,
    )

    with pytest.raises(OSError, match="injected HDF5 append failure"):
        writer.write(1, [signal])

    assert len(writer._dtypes) == 1  # noqa: SLF001
    assert len(writer._records) == 1  # noqa: SLF001
    assert len(writer._metadata) == 1  # noqa: SLF001
    assert len(writer._shapes) == 1  # noqa: SLF001
    assert len(writer._index) == 1  # noqa: SLF001
    assert len(writer._parent_records) == 0  # noqa: SLF001
    assert set(writer._data_group) == {"0"}  # noqa: SLF001
    assert len(writer._data[0]) == 4  # noqa: SLF001
    assert writer._dtype_ids == {np.dtype(np.complex64).str: 0}  # noqa: SLF001
    assert not writer._parent_ids  # noqa: SLF001

    writer.teardown()


def test_packed_hdf5_rejects_component_cycle_before_appending(
    tmp_path,
) -> None:
    signal = Signal(data=np.ones(4, dtype=np.complex64))
    signal.component_signals.append(signal)
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()

    with pytest.raises(ValueError, match="component signal cycle"):
        writer.write(0, [signal])

    assert len(writer._records) == 0  # noqa: SLF001
    assert len(writer._index) == 0  # noqa: SLF001
    writer.teardown()


def test_packed_hdf5_rejects_parent_cycle_before_appending(tmp_path) -> None:
    parent = HierarchicalMetadataObject(metadata={"name": "parent"})
    signal = Signal(data=np.ones(4, dtype=np.complex64), parent=parent)
    parent.parent = parent
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()

    with pytest.raises(ValueError, match="parent metadata cycle"):
        writer.write(0, [signal])

    assert len(writer._records) == 0  # noqa: SLF001
    assert len(writer._index) == 0  # noqa: SLF001
    writer.teardown()


def test_packed_hdf5_marks_successful_file_complete(tmp_path) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert bool(handle.attrs["complete"])


def test_packed_hdf5_context_exception_leaves_file_incomplete(tmp_path) -> None:
    writer = PackedHDF5Writer(tmp_path, max_batches_in_memory=1)

    def fail_during_generation() -> None:
        with writer:
            writer.write(1, [Signal(data=np.ones(4, dtype=np.complex64))])
            raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        fail_during_generation()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])
    with pytest.raises(RuntimeError, match=r"not open.*state is closed"):
        writer.write(0, [])


@pytest.mark.parametrize(
    "transform",
    [ComplexTo2D(), Spectrogram(fft_size=8)],
    ids=["complex-to-2d", "spectrogram"],
)
def test_packed_hdf5_preserves_transformed_2d_shape(tmp_path, transform) -> None:
    source = Signal(data=np.exp(2j * np.pi * np.arange(64) / 8).astype(np.complex64))
    expected = transform(source)
    assert expected.data.ndim == 2

    with PackedHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, [expected])

    reader = PackedHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)
        assert actual.data.shape == expected.data.shape
        assert actual.data.dtype == expected.data.dtype
        np.testing.assert_array_equal(actual.data, expected.data)
    finally:
        reader.teardown()


def test_packed_hdf5_reads_component_hierarchy_beyond_recursion_limit(
    tmp_path,
) -> None:
    depth = sys.getrecursionlimit() + 100
    signal = Signal(data=np.array([depth - 1], dtype=np.int16))
    for idx in reversed(range(depth - 1)):
        signal = Signal(
            data=np.array([idx], dtype=np.int16),
            component_signals=[signal],
        )
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])

    reader = PackedHDF5Reader(tmp_path)
    try:
        signal = reader.read(0)
        for expected in range(depth):
            assert int(signal.data[0]) == expected
            if expected < depth - 1:
                signal = signal.component_signals[0]
        assert not signal.component_signals
    finally:
        reader.teardown()


def test_packed_hdf5_reads_parent_hierarchy_beyond_recursion_limit(
    tmp_path,
) -> None:
    depth = sys.getrecursionlimit() + 100
    parent = None
    for idx in reversed(range(depth)):
        parent = HierarchicalMetadataObject(
            parent=parent,
            metadata={"depth": idx},
        )
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(
            0,
            [
                Signal(
                    data=np.ones(1, dtype=np.complex64),
                    parent=parent,
                )
            ],
        )

    reader = PackedHDF5Reader(tmp_path)
    try:
        parent = reader.read(0).parent
        actual_depth = 0
        while parent is not None:
            assert parent["depth"] == actual_depth
            actual_depth += 1
            parent = parent.parent
        assert actual_depth == depth
    finally:
        reader.teardown()


def test_packed_hdf5_reader_rejects_mismatched_record_metadata_lengths(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        handle["metadata"].resize(0, axis=0)

    with pytest.raises(ValueError, match="records and metadata lengths differ"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_wrong_table_rank(tmp_path) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        values = handle["shapes"][:]
        del handle["shapes"]
        handle.create_dataset("shapes", data=values.reshape(1, -1))

    with pytest.raises(ValueError, match="shapes must be a one-dimensional"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_signed_index_dtype(tmp_path) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        values = handle["index"][:].astype(np.int64)
        del handle["index"]
        handle.create_dataset("index", data=values)

    with pytest.raises(ValueError, match="index must have dtype uint64"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_incompatible_record_field_dtype(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        original = handle["records"]
        dtype = np.dtype(
            [
                (
                    name,
                    np.int64 if name == "data_offset" else original.dtype.fields[name][0],
                )
                for name in original.dtype.names
            ]
        )
        values = original[:].astype(dtype)
        del handle["records"]
        handle.create_dataset("records", data=values)

    with pytest.raises(ValueError, match=r"data_offset.*incompatible dtype"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_non_utf8_metadata_table(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        value = handle["metadata"][0]
        del handle["metadata"]
        handle.create_dataset("metadata", data=[value], dtype="S256")

    with pytest.raises(ValueError, match="metadata must contain UTF-8"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_data_dataset_instead_of_group(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        del handle["data"]
        handle.create_dataset("data", data=np.ones(4, dtype=np.complex64))

    with pytest.raises(ValueError, match="data path must be a group"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_multidimensional_data_stream(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        values = handle["data/0"][:]
        del handle["data/0"]
        handle["data"].create_dataset("0", data=values.reshape(2, 2))

    with pytest.raises(ValueError, match=r"data stream '0'.*one-dimensional"):
        PackedHDF5Reader(tmp_path).read(0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype_id", 99, "invalid dtype ID"),
        ("data_offset", 99, "data slice out of bounds"),
        ("shape_offset", 99, "shape slice out of bounds"),
        ("component_offset", 99, "component slice out of bounds"),
        ("parent_id", 99, "invalid parent ID"),
    ],
)
def test_packed_hdf5_reader_rejects_invalid_record_references(tmp_path, field, value, message) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        record = handle["records"][0]
        record[field] = value
        handle["records"][0] = record

    with pytest.raises(ValueError, match=message):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_shape_data_length_mismatch(
    tmp_path,
) -> None:
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        handle["shapes"][0] = 3

    with pytest.raises(ValueError, match="shape does not match data length"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_component_cycle(tmp_path) -> None:
    component = Signal(data=np.ones(2, dtype=np.complex64))
    signal = Signal(
        data=np.ones(4, dtype=np.complex64),
        component_signals=[component],
    )
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        handle["components"][0] = 0

    with pytest.raises(ValueError, match="component relationship cycle"):
        PackedHDF5Reader(tmp_path).read(0)


def test_packed_hdf5_reader_rejects_parent_cycle(tmp_path) -> None:
    parent = HierarchicalMetadataObject(metadata={"name": "parent"})
    signal = Signal(data=np.ones(4, dtype=np.complex64), parent=parent)
    with PackedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        record = handle["parent_records"][0]
        record["parent_id"] = 0
        handle["parent_records"][0] = record

    with pytest.raises(ValueError, match="parent relationship cycle"):
        PackedHDF5Reader(tmp_path).read(0)
