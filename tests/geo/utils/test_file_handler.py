"""Tests for GeoDataset FileHandler functionality.

This module tests:
- GeoDatasetWriter: Writing .yaml and .dat files
- GeoDatasetReader: Reading .yaml and .dat files
- GeoDatasetFileHandler: Factory class
- Round-trip write/read operations
- Field mapping and filtering
- Different data types
- Error handling and edge cases
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from torchsig.datasets.datasets import TorchSigIterableDataset

from torchsig.geo.datasets import Receiver, TorchSigGeoDataset, Transmitter
from torchsig.geo.types import GeoPoint
from torchsig.geo.utils.file_handler import (
    GeoDatasetFileHandler,
    GeoDatasetReader,
    GeoDatasetWriter,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import MetadataAttributeError
from torchsig.utils.defaults import TorchSigDefaults


# Helper functions moved from file_handler.py since they're only used in tests
DATA_TYPE_TO_NP_DTYPE = {
    "short": np.dtype(np.int16),
    "int16": np.dtype(np.int16),
    "int": np.dtype(np.int32),
    "int32": np.dtype(np.int32),
    "float": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
}


def _get_numpy_dtype(data_type: str) -> np.dtype:
    """Get numpy dtype from human-readable data type string."""
    data_type = data_type.lower()
    if data_type not in DATA_TYPE_TO_NP_DTYPE:
        raise ValueError(f"Unsupported data_type: {data_type}. Supported types: {list(DATA_TYPE_TO_NP_DTYPE.keys())}")
    return DATA_TYPE_TO_NP_DTYPE[data_type]


def _get_item_size(data_type: str) -> int:
    """Get item size in bytes for interleaved I/Q pair."""
    dtype = _get_numpy_dtype(data_type)
    return 2 * dtype.itemsize


def _complex_to_interleaved(data: np.ndarray) -> np.ndarray:
    """Convert complex array to interleaved I/Q format."""
    if not np.iscomplexobj(data):
        raise ValueError("Input data must be complex-valued")
    i_samples = np.real(data)
    q_samples = np.imag(data)
    interleaved = np.empty(len(data) * 2, dtype=i_samples.dtype)
    interleaved[::2] = i_samples
    interleaved[1::2] = q_samples
    return interleaved


def _interleaved_to_complex(data: np.ndarray) -> np.ndarray:
    """Convert interleaved I/Q array to complex format."""
    i_samples = data[::2]
    q_samples = data[1::2]
    return i_samples + 1j * q_samples


def make_signal(data: np.ndarray, **metadata) -> Signal:
    """Create a minimal Signal for file-handler tests."""
    return Signal(
        data=data,
        center_freq=2.4e9,
        sample_rate=1.0e6,
        **metadata,
    )


def write_signal(
    root: Path,
    signal: Signal,
    *,
    data_type: str = "float32",
    field_mapping: dict[str, str] | None = None,
) -> None:
    """Write one signal using the geo file handler."""
    writer = GeoDatasetWriter(
        root=str(root),
        data_type=data_type,
        field_mapping=field_mapping,
    )
    with writer:
        writer.write(0, signal)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing."""
    data = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    return Signal(
        data=data,
        rx_id="rx_0",
        rx_lat=37.7749,
        rx_lon=-122.4194,
        rx_alt=10.0,
        sample_rate=1000000.0,
        tx_id="tx_0",
        tx_lat=37.7759,
        tx_lon=-122.4195,
        tx_alt=15.0,
    )


@pytest.fixture
def geo_dataset():
    """Create a simple GeoDataset for testing."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["num_iq_samples_dataset"] = 64
    metadata["signal_duration_in_samples_min"] = 32
    metadata["signal_duration_in_samples_max"] = 63

    source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["bpsk"])

    tx_pos = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
    rx_pos = GeoPoint(lat=37.7759, lon=-122.4194, alt=15)

    transmitter = Transmitter(source_ds, tx_pos, identifier="tx_0")
    receiver = Receiver(rx_pos, sample_rate=metadata["sample_rate"], identifier="rx_0")

    return TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])


# =============================================================================
# Helper Functions
# =============================================================================


def assert_interleaved_equal(interleaved, expected):
    """Helper to assert interleaved arrays are equal."""
    np.testing.assert_array_equal(interleaved, expected)


def assert_complex_equal(complex_data, expected):
    """Helper to assert complex arrays are equal."""
    np.testing.assert_array_equal(complex_data, expected)


def create_sample_files(temp_dir, index=0, lat=37.7749, lon=-122.4194, alt=10.0):
    """Helper to create sample .yaml and .dat files for testing."""
    metadata = {
        "rx_id": "rx_0",
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "sample_rate": 1000000.0,
        "data_type": "float32",
        "complex_type": True,
        "item_size": 8,
        "swapped": False,
    }

    yaml_file = temp_dir / f"{index}.yaml"
    with Path.open(yaml_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    dat_file = temp_dir / f"{index}.dat"
    interleaved = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    with Path.open(dat_file, "wb") as f:
        f.write(interleaved.tobytes())

    return yaml_file, dat_file


# =============================================================================
# Dtype Utility Tests
# =============================================================================


class TestDtypeUtils:
    """Tests for dtype utility functions."""

    @pytest.mark.parametrize(
        ("data_type", "expected_dtype"),
        [
            ("float32", np.float32),
            ("float64", np.float64),
            ("short", np.int16),
            ("int16", np.int16),
            ("int32", np.int32),
            ("double", np.float64),
            ("int", np.int32),
        ],
    )
    def test_get_numpy_dtype(self, data_type, expected_dtype):
        """Test getting numpy dtype for various data types."""
        dtype = _get_numpy_dtype(data_type)
        assert dtype == expected_dtype

    @pytest.mark.parametrize("data_type", ["uint8", "invalid_type", "float128"])
    def test_get_numpy_dtype_invalid(self, data_type):
        """Test that invalid data type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data_type"):
            _get_numpy_dtype(data_type)

    @pytest.mark.parametrize(
        ("data_type", "expected_size"),
        [
            ("float32", 8),
            ("float64", 16),
            ("short", 4),
            ("int16", 4),
            ("int32", 8),
        ],
    )
    def test_get_item_size(self, data_type, expected_size):
        """Test item size for various data types."""
        assert _get_item_size(data_type) == expected_size


# =============================================================================
# Interleaving Function Tests
# =============================================================================


class TestInterleaving:
    """Tests for I/Q interleaving functions."""

    def test_complex_to_interleaved_basic(self):
        """Test converting complex array to interleaved I/Q."""
        complex_data = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        interleaved = _complex_to_interleaved(complex_data)
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        assert_interleaved_equal(interleaved, expected)

    def test_interleaved_to_complex_basic(self):
        """Test converting interleaved I/Q to complex array."""
        interleaved = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        complex_data = _interleaved_to_complex(interleaved)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        assert_complex_equal(complex_data, expected)

    @pytest.mark.parametrize("size", [1, 10, 100, 10000])
    def test_round_trip_interleaving(self, size):
        """Test that complex -> interleaved -> complex preserves data."""
        rng = np.random.default_rng(42)
        real = rng.standard_normal(size).astype(np.float32)
        imag = rng.standard_normal(size).astype(np.float32)
        complex_data = real + 1j * imag

        interleaved = _complex_to_interleaved(complex_data)
        result = _interleaved_to_complex(interleaved)
        np.testing.assert_array_equal(result, complex_data)

    def test_complex_to_interleaved_single_element(self):
        """Test with single complex element."""
        complex_data = np.array([1 + 2j], dtype=np.complex64)
        interleaved = _complex_to_interleaved(complex_data)
        expected = np.array([1, 2], dtype=np.float32)
        assert_interleaved_equal(interleaved, expected)

    def test_complex_to_interleaved_empty(self):
        """Test with empty array."""
        complex_data = np.array([], dtype=np.complex64)
        interleaved = _complex_to_interleaved(complex_data)
        expected = np.array([], dtype=np.float32)
        assert_interleaved_equal(interleaved, expected)

    def test_complex_to_interleaved_large_array(self):
        """Test with large array."""
        rng = np.random.default_rng(42)
        complex_data = rng.standard_normal(10000) + 1j * rng.standard_normal(10000)
        complex_data = complex_data.astype(np.complex64)
        interleaved = _complex_to_interleaved(complex_data)
        assert len(interleaved) == len(complex_data) * 2
        assert interleaved.dtype == np.float32

    def test_interleaved_to_complex_single_element(self):
        """Test with single interleaved pair."""
        interleaved = np.array([1, 2], dtype=np.float32)
        complex_data = _interleaved_to_complex(interleaved)
        expected = np.array([1 + 2j], dtype=np.complex64)
        assert_complex_equal(complex_data, expected)

    def test_interleaved_to_complex_empty(self):
        """Test with empty interleaved array."""
        interleaved = np.array([], dtype=np.float32)
        complex_data = _interleaved_to_complex(interleaved)
        expected = np.array([], dtype=np.complex64)
        assert_complex_equal(complex_data, expected)

    def test_interleaved_to_complex_odd_length(self):
        """Test with odd-length interleaved array."""
        interleaved = np.array([1, 2, 3], dtype=np.float32)
        complex_data = _interleaved_to_complex(interleaved)
        assert len(complex_data) == 2

    @pytest.mark.parametrize(
        "input_data",
        [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.int32),
        ],
    )
    def test_complex_to_interleaved_non_complex_raises(self, input_data):
        """Test that non-complex input raises ValueError."""
        with pytest.raises(ValueError, match="complex-valued"):
            _complex_to_interleaved(input_data)

    def test_interleaved_output_dtype(self):
        """Test that interleaved output has correct dtype."""
        complex_data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        interleaved = _complex_to_interleaved(complex_data)
        assert interleaved.dtype == np.float32

    def test_complex_output_dtype(self):
        """Test that complex output has correct dtype."""
        interleaved = np.array([1, 2, 3, 4], dtype=np.float32)
        complex_data = _interleaved_to_complex(interleaved)
        assert complex_data.dtype == np.complex64


# =============================================================================
# GeoDatasetWriter Tests
# =============================================================================


class TestGeoDatasetWriter:
    """Tests for GeoDatasetWriter."""

    def test_write_single_signal(self, temp_dir, sample_signal):
        """Test writing a single signal to files."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        dat_file = temp_dir / "0.dat"
        assert yaml_file.exists()
        assert dat_file.exists()

    @pytest.mark.parametrize("num_signals", [1, 2, 3])
    def test_write_multiple_signals(self, temp_dir, sample_signal, num_signals):
        """Test writing multiple signals."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            for i in range(num_signals):
                writer.write(i, sample_signal)

        for i in range(num_signals):
            yaml_file = temp_dir / f"{i}.yaml"
            dat_file = temp_dir / f"{i}.dat"
            assert yaml_file.exists()
            assert dat_file.exists()

    def test_write_list_of_signals(self, temp_dir, sample_signal):
        """Test writing a list of signals."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            writer.write(0, [sample_signal, sample_signal])

        assert (temp_dir / "0.yaml").exists()
        assert (temp_dir / "0.dat").exists()
        assert (temp_dir / "1.yaml").exists()
        assert (temp_dir / "1.dat").exists()

    def test_yaml_content(self, temp_dir, sample_signal):
        """Test YAML metadata content with default field mapping."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file) as f:
            metadata = yaml.safe_load(f)

        assert "lat" in metadata
        assert "lon" in metadata
        assert "alt" in metadata
        assert metadata["lat"] == pytest.approx(37.7749)
        assert metadata["lon"] == pytest.approx(-122.4194)
        assert metadata["alt"] == pytest.approx(10.0)
        assert metadata["data_type"] == "float32"
        assert metadata["complex_type"] is True
        assert metadata["item_size"] == 8

    @pytest.mark.parametrize("data_type", ["float32", "float64", "int16"])
    def test_dat_content(self, temp_dir, data_type):
        """Test DAT file content with various data types."""
        # Use normalized data for integer types
        if data_type in ["int16", "short"]:
            data = np.array([0.1 + 0.2j, 0.3 + 0.4j, 0.5 + 0.6j, 0.7 + 0.8j], dtype=np.complex64)
        else:
            data = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
        signal = Signal(data=data, rx_id="rx_0")

        writer = GeoDatasetWriter(root=temp_dir, data_type=data_type)

        with writer:
            writer.write(0, signal)

        dat_file = temp_dir / "0.dat"
        with Path.open(dat_file, "rb") as f:
            raw_data = f.read()

        np_dtype = _get_numpy_dtype(data_type)
        interleaved = np.frombuffer(raw_data, dtype=np_dtype)
        if data_type in ["int16", "short"]:
            # For integer types, values are scaled by int16_max
            scale = 32767
            expected = np.array(
                [
                    int(0.1 * scale),
                    int(0.2 * scale),
                    int(0.3 * scale),
                    int(0.4 * scale),
                    int(0.5 * scale),
                    int(0.6 * scale),
                    int(0.7 * scale),
                    int(0.8 * scale),
                ],
                dtype=np_dtype,
            )
        else:
            expected = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np_dtype)
        np.testing.assert_array_equal(interleaved, expected)

    @pytest.mark.parametrize(
        "field_mapping",
        [
            {"rx_lat": "latitude", "rx_lon": "longitude", "rx_alt": "altitude"},
            {"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
        ],
    )
    def test_field_mapping(self, temp_dir, sample_signal, field_mapping):
        """Test field mapping functionality."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32", field_mapping=field_mapping)

        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file) as f:
            metadata = yaml.safe_load(f)

        for orig_key, mapped_key in field_mapping.items():
            assert mapped_key in metadata
            assert orig_key not in metadata

    def test_allowlist(self, temp_dir, sample_signal):
        """Test allowlist filtering."""
        writer = GeoDatasetWriter(
            root=temp_dir,
            data_type="float32",
            allowlist=["lat", "lon", "alt", "data_type", "complex_type", "item_size"],
        )

        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file) as f:
            metadata = yaml.safe_load(f)

        assert "sample_rate" not in metadata
        assert "tx_id" not in metadata

    def test_blocklist(self, temp_dir, sample_signal):
        """Test blocklist filtering."""
        writer = GeoDatasetWriter(
            root=temp_dir,
            data_type="float32",
            blocklist=["tx_id", "tx_lat", "tx_lon", "tx_alt"],
        )

        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file) as f:
            metadata = yaml.safe_load(f)

        assert "tx_id" not in metadata
        assert "id" in metadata or "UID" in metadata

    def test_len(self, temp_dir, sample_signal):
        """Test __len__ method."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            writer.write(0, sample_signal)
            writer.write(1, sample_signal)
            assert len(writer) == 2

    def test_write_without_context_manager(self, temp_dir, sample_signal):
        """Test writing without using context manager."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")
        writer.setup()
        writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        dat_file = temp_dir / "0.dat"
        assert yaml_file.exists()
        assert dat_file.exists()

    def test_multiple_rx_ids(self, temp_dir):
        """Test writing signals with different rx_ids."""
        signal1 = Signal(data=np.ones(10, dtype=np.complex64), rx_id="rx_0", rx_lat=37.7749)
        signal2 = Signal(data=np.ones(10, dtype=np.complex64), rx_id="rx_1", rx_lat=37.7759)

        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")

        with writer:
            writer.write(0, signal1)
            writer.write(0, signal2)

        assert (temp_dir / "0.yaml").exists()
        assert (temp_dir / "1.yaml").exists()

    def test_to_yaml_dat_pairs_overwrites_existing_dataset(
        self,
        tmp_path,
        simple_geo_ds,
    ):
        simple_geo_ds.to_yaml_dat_pairs(
            root=str(tmp_path),
            dataset_length=2,
            overwrite=True,
            multithreading=False,
        )
        simple_geo_ds.to_yaml_dat_pairs(
            root=str(tmp_path),
            dataset_length=1,
            overwrite=True,
            multithreading=False,
        )
        assert (tmp_path / "0.yaml").exists()
        assert (tmp_path / "0.dat").exists()
        assert not (tmp_path / "1.yaml").exists()
        assert not (tmp_path / "1.dat").exists()

    def test_to_yaml_dat_pairs_preserves_existing_dataset_without_overwrite(
        self,
        tmp_path,
        simple_geo_ds,
    ):
        """overwrite=False reuses the existing dataset without replacing it."""
        simple_geo_ds.to_yaml_dat_pairs(
            root=str(tmp_path),
            dataset_length=1,
            overwrite=True,
            multithreading=False,
        )

        original_yaml = (tmp_path / "0.yaml").read_bytes()
        original_dat = (tmp_path / "0.dat").read_bytes()

        simple_geo_ds.to_yaml_dat_pairs(
            root=str(tmp_path),
            dataset_length=1,
            overwrite=False,
            multithreading=False,
        )

        assert (tmp_path / "0.yaml").read_bytes() == original_yaml
        assert (tmp_path / "0.dat").read_bytes() == original_dat

    def test_overwrite_removes_stale_indexed_files(
        self,
        tmp_path,
        sample_signal,
    ):
        writer = GeoDatasetWriter(root=tmp_path, overwrite=True)
        with writer:
            writer.write(0, sample_signal)
            writer.write(1, sample_signal)

        replacement = GeoDatasetWriter(root=tmp_path, overwrite=True)
        with replacement:
            replacement.write(0, sample_signal)

        assert (tmp_path / "0.yaml").exists()
        assert (tmp_path / "0.dat").exists()
        assert not (tmp_path / "1.yaml").exists()
        assert not (tmp_path / "1.dat").exists()


# =============================================================================
# GeoDatasetReader Tests
# =============================================================================


class TestGeoDatasetReader:
    """Tests for GeoDatasetReader."""

    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample .yaml and .dat files for testing."""
        return create_sample_files(temp_dir)

    def test_read_single_signal(self, temp_dir, sample_files):  # noqa: ARG002
        """Test reading a single signal."""
        reader = GeoDatasetReader(root=temp_dir)
        signal = reader.read(0)
        assert isinstance(signal, Signal)
        assert len(signal.data) == 4

    def test_read_metadata(self, temp_dir, sample_files):  # noqa: ARG002
        """Test reading metadata from YAML."""
        reader = GeoDatasetReader(root=temp_dir)
        signal = reader.read(0)
        assert signal["rx_lat"] == pytest.approx(37.7749)
        assert signal["rx_lon"] == pytest.approx(-122.4194)
        assert signal["rx_alt"] == pytest.approx(10.0)
        assert signal["sample_rate"] == pytest.approx(1000000.0)

    def test_read_iq_data(self, temp_dir, sample_files):  # noqa: ARG002
        """Test reading IQ data from DAT file."""
        reader = GeoDatasetReader(root=temp_dir)
        signal = reader.read(0)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
        np.testing.assert_array_equal(signal.data, expected)

    @pytest.mark.parametrize("num_files", [1, 2, 3])
    def test_read_multiple_signals(self, temp_dir, num_files):
        """Test reading multiple signals."""
        for i in range(num_files):
            create_sample_files(temp_dir, index=i, lat=37.7749 + i * 0.001)

        reader = GeoDatasetReader(root=temp_dir)

        for i in range(num_files):
            signal = reader.read(i)
            assert isinstance(signal, Signal)
            assert len(signal.data) == 4

    def test_len(self, temp_dir, sample_files):  # noqa: ARG002
        """Test __len__ method."""
        reader = GeoDatasetReader(root=temp_dir)
        assert len(reader) == 1

    def test_index_out_of_range(self, temp_dir):
        """Test that out of range index raises IndexError."""
        reader = GeoDatasetReader(root=temp_dir)
        with pytest.raises(IndexError):
            reader.read(10)

    def test_field_mapping_reverse(self, temp_dir):
        """Test reverse field mapping when reading.

        Files have latitude/longitude/altitude keys, we want internal keys to be lat/lon/alt.
        With the unified mapping direction (internal -> file), we specify:
        internal lat -> file latitude, so reader will map file latitude back to internal lat.
        """
        metadata = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10.0,
            "data_type": "float32",
            "complex_type": True,
            "item_size": 8,
        }

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        dat_file = temp_dir / "0.dat"
        interleaved = np.array([1, 2, 3, 4], dtype=np.float32)
        with Path.open(dat_file, "wb") as f:
            f.write(interleaved.tobytes())

        reader = GeoDatasetReader(
            root=temp_dir,
            field_mapping={"lat": "latitude", "lon": "longitude", "alt": "altitude"},
        )
        signal = reader.read(0)

        signal_keys = signal.keys()
        assert "lat" in signal_keys
        assert "lon" in signal_keys
        assert "alt" in signal_keys

    def test_read_without_field_mapping(self, temp_dir):
        """Test reading without field mapping preserves original keys."""
        metadata = {
            "rx_lat": 37.7749,
            "rx_lon": -122.4194,
            "rx_alt": 10.0,
            "data_type": "float32",
            "complex_type": True,
            "item_size": 8,
        }

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        dat_file = temp_dir / "0.dat"
        interleaved = np.array([1, 2, 3, 4], dtype=np.float32)
        with Path.open(dat_file, "wb") as f:
            f.write(interleaved.tobytes())

        reader = GeoDatasetReader(root=temp_dir)
        signal = reader.read(0)

        signal_keys = signal.keys()
        assert "rx_lat" in signal_keys
        assert "rx_lon" in signal_keys
        assert "rx_alt" in signal_keys


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for round-trip write/read operations."""

    def test_round_trip_single_signal(self, temp_dir):
        """Test round-trip for a single signal.

        With the unified mapping direction (internal -> file):
        - Writer: rx_lat -> lat (internal -> file)
        - Reader: lat -> rx_lat (file -> internal, automatically reversed)
        Both use the same mapping direction, reader reverses internally.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        original_signal = Signal(
            data=data.astype(np.complex64),
            rx_id="rx_0",
            rx_lat=37.7749,
            rx_lon=-122.4194,
            rx_alt=10.0,
            sample_rate=1000000.0,
        )

        writer = GeoDatasetWriter(
            root=temp_dir,
            data_type="float32",
            field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
        )

        with writer:
            writer.write(0, original_signal)

        reader = GeoDatasetReader(
            root=temp_dir,
            field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
        )
        read_signal = reader.read(0)

        np.testing.assert_array_equal(read_signal.data.astype(np.complex64), original_signal.data.astype(np.complex64))
        assert read_signal["rx_lat"] == 37.7749
        assert read_signal["rx_lon"] == -122.4194
        assert read_signal["rx_alt"] == 10.0

    def test_round_trip_with_geo_dataset(self, temp_dir, geo_dataset):
        """Test round-trip using GeoDataset."""
        geo_dataset.to_yaml_dat_pairs(
            root=temp_dir,
            dataset_length=5,
            data_type="float32",
            field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
        )

        reader = GeoDatasetReader(
            root=temp_dir,
            field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
        )
        for i in range(5):
            signal = reader.read(i)
            assert isinstance(signal, Signal)
            assert len(signal.data) > 0

    @pytest.mark.parametrize("data_type", ["float32", "float64"])
    def test_round_trip_different_dtypes(self, temp_dir, data_type):
        """Test round-trip with different floating point data types."""
        data = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        original_signal = Signal(data=data, rx_id="rx_0")

        for f in temp_dir.glob("*"):
            f.unlink()

        writer = GeoDatasetWriter(root=temp_dir, data_type=data_type)
        with writer:
            writer.write(0, original_signal)

        reader = GeoDatasetReader(root=temp_dir)
        read_signal = reader.read(0)

        np.testing.assert_array_almost_equal(
            read_signal.data.astype(np.complex64),
            original_signal.data.astype(np.complex64),
            decimal=5,
        )

    def test_round_trip_preserves_all_metadata(self, temp_dir):
        """Test that round-trip preserves all metadata fields."""
        original_signal = Signal(
            data=np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            rx_id="rx_0",
            rx_lat=37.7749,
            rx_lon=-122.4194,
            rx_alt=10.0,
            sample_rate=1000000.0,
            tx_id="tx_0",
            custom_field="custom_value",
        )

        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")
        with writer:
            writer.write(0, original_signal)

        reader = GeoDatasetReader(root=temp_dir)
        read_signal = reader.read(0)

        assert read_signal["rx_id"] == "rx_0"
        assert read_signal["rx_lat"] == pytest.approx(37.7749)
        assert read_signal["sample_rate"] == pytest.approx(1000000.0)
        assert read_signal["tx_id"] == "tx_0"
        assert read_signal["custom_field"] == "custom_value"

    def test_float64_real_data_round_trip_preserves_precision(
        self,
        tmp_path: Path,
    ) -> None:
        """Real float64 input must not be downcast through complex64."""
        data = np.array(
            [
                1.0 + 2.0**-40,
                -0.5 + 2.0**-42,
                np.pi,
            ],
            dtype=np.float64,
        )
        signal = make_signal(data)

        write_signal(tmp_path, signal, data_type="float64")
        restored = GeoDatasetReader(root=str(tmp_path)).read(0)

        assert restored.data.dtype == np.complex128
        np.testing.assert_array_equal(restored.data.real, data)
        np.testing.assert_array_equal(
            restored.data.imag,
            np.zeros_like(data),
        )


# =============================================================================
# GeoDatasetFileHandler Tests
# =============================================================================


class TestGeoDatasetFileHandler:
    """Tests for GeoDatasetFileHandler factory."""

    @pytest.mark.parametrize("mode", ["w", "r"])
    def test_create_handler(self, temp_dir, mode):
        """Test creating handlers via factory."""
        if mode == "w":
            handler = GeoDatasetFileHandler.create_handler(mode=mode, root=temp_dir, data_type="float32")
            assert isinstance(handler, GeoDatasetWriter)
        else:
            handler = GeoDatasetFileHandler.create_handler(mode=mode, root=temp_dir)
            assert isinstance(handler, GeoDatasetReader)

    @pytest.mark.parametrize("mode", ["x", "W", ""])
    def test_invalid_mode(self, temp_dir, mode):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid File Handler mode"):
            GeoDatasetFileHandler.create_handler(mode, root=temp_dir)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_write_to_nonexistent_directory(self, sample_signal):
        """Test writing to a non-existent directory creates it."""
        temp_path = Path(tempfile.mkdtemp()) / "subdir" / "nested"
        temp_path.mkdir(parents=True, exist_ok=True)

        try:
            writer = GeoDatasetWriter(root=temp_path, data_type="float32")
            with writer:
                writer.write(0, sample_signal)
            assert (temp_path / "0.yaml").exists()
            assert (temp_path / "0.dat").exists()
        finally:
            shutil.rmtree(temp_path.parent, ignore_errors=True)

    def test_read_from_empty_directory(self, temp_dir):
        """Test reading from empty directory."""
        reader = GeoDatasetReader(root=temp_dir)
        assert len(reader) == 0

    def test_write_signal_with_empty_data(self, temp_dir):
        """Test writing a signal with empty data array."""
        signal = Signal(data=np.array([], dtype=np.complex64), rx_id="rx_0")
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32")
        with writer:
            writer.write(0, signal)
        assert (temp_dir / "0.yaml").exists()
        assert (temp_dir / "0.dat").exists()

    def test_read_corrupted_yaml_file(self, temp_dir):
        """Test reading with corrupted YAML file."""
        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file, "w") as f:
            f.write("invalid: yaml: content:")

        dat_file = temp_dir / "0.dat"
        with Path.open(dat_file, "wb") as f:
            f.write(np.array([1, 2, 3, 4], dtype=np.float32).tobytes())

        reader = GeoDatasetReader(root=temp_dir)
        with pytest.raises(yaml.YAMLError):
            reader.read(0)

    def test_read_missing_dat_file(self, temp_dir):
        """Test reading with missing DAT file."""
        metadata = {"rx_id": "rx_0", "data_type": "float32", "complex_type": True, "item_size": 8}
        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file, "w") as f:
            yaml.dump(metadata, f)

        reader = GeoDatasetReader(root=temp_dir)
        with pytest.raises(IndexError, match="Index 0 not found"):
            reader.read(0)

    def test_write_overwrite_existing(self, temp_dir, sample_signal):
        """Test that overwrite=True allows overwriting existing files."""
        writer1 = GeoDatasetWriter(root=temp_dir, data_type="float32", overwrite=True)
        with writer1:
            writer1.write(0, sample_signal)

        signal2 = Signal(data=np.array([9 + 8j, 7 + 6j], dtype=np.complex64), rx_id="rx_0")
        writer2 = GeoDatasetWriter(root=temp_dir, data_type="float32", overwrite=True)
        with writer2:
            writer2.write(0, signal2)

        reader = GeoDatasetReader(root=temp_dir)
        read_signal = reader.read(0)
        expected = np.array([9 + 8j, 7 + 6j], dtype=np.complex64)
        np.testing.assert_array_equal(read_signal.data, expected)

    @pytest.mark.parametrize(
        ("field_mapping", "missing_key"),
        [
            ({"rx_lat": "lat", "nonexistent": "missing"}, "missing"),
            ({"rx_lat": "lat"}, "rx_lon"),
        ],
    )
    def test_field_mapping_with_missing_keys(self, temp_dir, sample_signal, field_mapping, missing_key):
        """Test field mapping with keys that don't exist in signal."""
        writer = GeoDatasetWriter(root=temp_dir, data_type="float32", field_mapping=field_mapping)
        with writer:
            writer.write(0, sample_signal)

        yaml_file = temp_dir / "0.yaml"
        with Path.open(yaml_file) as f:
            metadata = yaml.safe_load(f)
        assert "lat" in metadata
        assert missing_key not in metadata

    @pytest.mark.parametrize("nonexistent_key", ["nonexistent", "missing_key"])
    def test_allowlist_with_nonexistent_keys(self, temp_dir, sample_signal, nonexistent_key):
        """Test allowlist with keys that don't exist."""
        writer = GeoDatasetWriter(
            root=temp_dir,
            data_type="float32",
            allowlist=["lat", "lon", nonexistent_key],
        )
        with writer:
            writer.write(0, sample_signal)
        assert (temp_dir / "0.yaml").exists()

    @pytest.mark.parametrize("nonexistent_key", ["nonexistent_key", "missing"])
    def test_blocklist_with_nonexistent_keys(self, temp_dir, sample_signal, nonexistent_key):
        """Test blocklist with keys that don't exist."""
        writer = GeoDatasetWriter(
            root=temp_dir,
            data_type="float32",
            blocklist=[nonexistent_key],
        )
        with writer:
            writer.write(0, sample_signal)
        assert (temp_dir / "0.yaml").exists()


class TestGeoDatasetWriterValidation:
    """Tests for writer input validation and failure behavior."""

    def test_failed_integer_write_leaves_no_partial_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Validation failure must not leave an orphaned YAML or DAT file."""
        signal = make_signal(np.array([0.0 + 0.0j, 1.1 + 0.0j], dtype=np.complex64))

        writer = GeoDatasetWriter(
            root=str(tmp_path),
            data_type="int16",
        )

        with writer:
            with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
                writer.write(0, signal)

        assert list(tmp_path.glob("*.yaml")) == []
        assert list(tmp_path.glob("*.dat")) == []
        assert len(writer) == 0

    @pytest.mark.parametrize(
        "bad_value",
        [
            np.nan + 0.0j,
            np.inf + 0.0j,
            -np.inf + 0.0j,
            0.0 + np.nan * 1j,
        ],
    )
    def test_integer_writer_rejects_nonfinite_iq_data(
        self,
        tmp_path: Path,
        bad_value: complex,
    ) -> None:
        """NaN and infinity must not be silently converted to integers."""
        signal = make_signal(np.array([0.0 + 0.0j, bad_value], dtype=np.complex128))

        writer = GeoDatasetWriter(
            root=str(tmp_path),
            data_type="int16",
        )

        with writer:
            with pytest.raises(ValueError, match="finite"):
                writer.write(0, signal)

        assert list(tmp_path.glob("*.yaml")) == []
        assert list(tmp_path.glob("*.dat")) == []

    def test_writer_rejects_nonempty_output_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Starting a new writer must not mix new files with an old dataset."""
        first = make_signal(np.array([1.0 + 0.0j], dtype=np.complex64))
        second = make_signal(np.array([2.0 + 0.0j], dtype=np.complex64))

        writer = GeoDatasetWriter(root=str(tmp_path))
        with writer:
            writer.write(0, first)
            writer.write(1, second)

        replacement = GeoDatasetWriter(root=str(tmp_path))

        with pytest.raises(
            FileExistsError,
            match="output directory.*not empty",
        ):
            with replacement:
                pass

        # The first dataset must remain intact.
        assert (tmp_path / "0.yaml").exists()
        assert (tmp_path / "0.dat").exists()
        assert (tmp_path / "1.yaml").exists()
        assert (tmp_path / "1.dat").exists()

    def test_writer_rejects_field_mapping_collision(
        self,
        tmp_path: Path,
    ) -> None:
        """Two internal metadata fields must not map to one file field."""
        with pytest.raises(
            ValueError,
            match=r"mapping.*lat|duplicate.*lat|collision.*lat",
        ):
            GeoDatasetWriter(
                root=str(tmp_path),
                field_mapping={
                    # Conflicts with the default rx_lat -> lat mapping.
                    "custom_latitude": "lat",
                },
            )


class TestGeoDatasetReaderValidation:
    """Tests for malformed or incomplete input datasets."""

    def test_reader_rejects_yaml_without_matching_dat(
        self,
        tmp_path: Path,
    ) -> None:
        """An incomplete YAML/DAT pair must not be silently ignored."""
        with (tmp_path / "0.yaml").open("w") as file:
            yaml.safe_dump({"data_type": "float32"}, file)

        reader = GeoDatasetReader(root=str(tmp_path))

        with pytest.raises(
            ValueError,
            match=r"0\.yaml.*matching.*0\.dat",
        ):
            len(reader)

    def test_reader_rejects_dat_without_matching_yaml(
        self,
        tmp_path: Path,
    ) -> None:
        """An incomplete DAT/YAML pair must not be silently ignored."""
        np.array([1.0, 0.0], dtype=np.float32).tofile(tmp_path / "0.dat")

        reader = GeoDatasetReader(root=str(tmp_path))

        with pytest.raises(
            ValueError,
            match=r"0\.dat.*matching.*0\.yaml",
        ):
            len(reader)

    def test_reader_rejects_noncontiguous_indexes(
        self,
        tmp_path: Path,
    ) -> None:
        """Reader indexes must agree with range(len(reader))."""
        signal = make_signal(np.array([1.0 + 2.0j], dtype=np.complex64))
        write_signal(tmp_path, signal)

        (tmp_path / "0.yaml").rename(tmp_path / "1.yaml")
        (tmp_path / "0.dat").rename(tmp_path / "1.dat")

        reader = GeoDatasetReader(root=str(tmp_path))

        with pytest.raises(
            ValueError,
            match="contiguous",
        ):
            len(reader)

    def test_reader_rejects_odd_number_of_iq_components(
        self,
        tmp_path: Path,
    ) -> None:
        """A DAT file must contain complete interleaved I/Q pairs."""
        with (tmp_path / "0.yaml").open("w") as file:
            yaml.safe_dump({"data_type": "float32"}, file)

        # I0, Q0, I1 -- Q1 is missing.
        np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(tmp_path / "0.dat")

        reader = GeoDatasetReader(root=str(tmp_path))

        with pytest.raises(
            ValueError,
            match=r"even number|complete.*I/Q",
        ):
            reader.read(0)
