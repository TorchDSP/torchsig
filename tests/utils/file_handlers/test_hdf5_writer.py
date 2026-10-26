"""Tests for standard HDF5 writer options."""

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import Spectrogram, reconstruct_spectrogram_support_vectors
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer, populate_hdf5_group_with_metadata


def test_standard_hdf5_writer_supports_no_compression(tmp_path) -> None:
    """Record disabled compression without passing None to HDF5 attributes."""
    signals = [Signal(data=np.ones(4, dtype=np.complex64))]

    with HDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ) as writer:
        writer.write(0, signals)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data"]["0"].compression is None


def test_standard_hdf5_writer_round_trips_array_metadata(tmp_path) -> None:
    """Supported scalar, sequence, and array metadata retain their values."""
    array_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    signal = Signal(
        data=np.ones(4, dtype=np.complex64),
        scalar_value=7,
        sequence_value=[1, 2, 3],
        array_value=array_value,
        omitted_value=None,
    )

    with HDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])

    reader = HDF5Reader(tmp_path)
    try:
        loaded = reader.read(0)
    finally:
        reader.teardown()

    assert loaded.scalar_value == 7
    np.testing.assert_array_equal(loaded.sequence_value, [1, 2, 3])
    np.testing.assert_array_equal(loaded.array_value, array_value)
    assert "omitted_value" not in loaded.keys()  # noqa: SIM118 - metadata objects are not iterable


def test_standard_hdf5_writer_rejects_unsupported_array_metadata(tmp_path) -> None:
    """Unsupported arrays identify the metadata field, dtype, and shape."""
    metadata = HierarchicalMetadataObject(
        metadata={"custom_objects": np.array([{"value": 1}, {"value": 2}], dtype=object)},
    )

    with (
        h5py.File(tmp_path / "metadata.h5", "w") as handle,
        pytest.raises(
            TypeError,
            match=r"HDF5 metadata field 'custom_objects' does not support NumPy array dtype object and shape \(2,\)",
        ),
    ):
        populate_hdf5_group_with_metadata(handle.create_group("metadata"), metadata)


def test_standard_hdf5_writer_assigns_stable_keys_to_metadata_parents(tmp_path) -> None:
    """Distinct, shared, and nested metadata parents receive stable writer keys."""
    root_parent = HierarchicalMetadataObject(metadata={"sample_rate": 1.0})
    first_parent = HierarchicalMetadataObject(
        metadata={"class_index": 1},
        parent=root_parent,
    )
    second_parent = HierarchicalMetadataObject(
        metadata={"class_index": 2},
        parent=root_parent,
    )
    signals = [
        Signal(data=np.ones(4, dtype=np.complex64), parent=first_parent),
        Signal(data=np.ones(4, dtype=np.complex64), parent=second_parent),
    ]

    with HDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, signals)

    first_parent_key = vars(first_parent)["_hdf5_key"]
    second_parent_key = vars(second_parent)["_hdf5_key"]
    root_parent_key = vars(root_parent)["_hdf5_key"]
    parent_keys = {first_parent_key, second_parent_key, root_parent_key}
    assert len(parent_keys) == 3
    assert parent_keys.isdisjoint(vars(signal)["_hdf5_key"] for signal in signals)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        metadata = handle["metadata"]
        assert metadata[first_parent_key]["class_index"][()] == 1
        assert metadata[second_parent_key]["class_index"][()] == 2
        assert metadata[first_parent_key]["parent_metadata_id"].asstr()[()] == root_parent_key
        assert metadata[second_parent_key]["parent_metadata_id"].asstr()[()] == root_parent_key


def test_standard_hdf5_writer_refreshes_parent_keys_between_sessions(tmp_path) -> None:
    """A parent reused by another writer must not retain its old key namespace."""
    parent = HierarchicalMetadataObject(metadata={"class_index": 3})
    first_signal = Signal(data=np.ones(4, dtype=np.complex64), parent=parent)
    with HDF5Writer(tmp_path / "first", max_batches_in_memory=1) as writer:
        writer.write(0, [first_signal])
    first_token = vars(parent)["_hdf5_writer_token"]

    second_signal = Signal(data=np.ones(4, dtype=np.complex64), parent=parent)
    with HDF5Writer(tmp_path / "second", max_batches_in_memory=1) as writer:
        writer.write(0, [second_signal])

    assert vars(parent)["_hdf5_writer_token"] is not first_token
    assert vars(parent)["_hdf5_key"] != vars(second_signal)["_hdf5_key"]


def test_standard_hdf5_writer_persists_compact_spectrogram_support(tmp_path) -> None:
    """Spectrogram support must be stored as scalar affine coordinates."""
    signal = Spectrogram(fft_size=4, fft_stride=2)(Signal(data=np.ones(16, dtype=np.complex64), sample_rate=8.0))
    frequency, time = reconstruct_spectrogram_support_vectors(signal)

    with HDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        metadata = handle["metadata"]["0"]
        assert "spectrogram_frequency" not in metadata
        assert "spectrogram_time" not in metadata
        assert metadata["spectrogram_frequency_start"][()] == frequency[0]
        assert metadata["spectrogram_frequency_step"][()] == frequency[1] - frequency[0]
        assert metadata["spectrogram_time_start"][()] == time[0]
        assert metadata["spectrogram_time_step"][()] == time[1] - time[0]
        assert metadata["spectrogram_frequency_units"].asstr()[()] == "Hz"
        assert metadata["spectrogram_time_units"].asstr()[()] == "seconds"
