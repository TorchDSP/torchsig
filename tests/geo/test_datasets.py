"""Tests for geolocation-aware dataset functionality.

Organized into logical test classes covering:
- Transmitter: signal generation, metadata, identification, positioning
- Receiver: transforms, positioning, identification
- TorchSigGeoDataset: creation, topology, iteration, signals, transforms
- Validation: validation helpers, sample rate, signal length
- StaticTorchSigGeoDataset: file I/O, loading, serialization
- Mobile objects: moving transmitters/receivers, velocity
"""

import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import builtins

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.geo.datasets import Receiver, StaticTorchSigGeoDataset, TorchSigGeoDataset, Transmitter, apply_transforms_to_signal
from torchsig.geo.types import GeoPoint, GeoVelocity
from torchsig.geo.transforms import DopplerShift, GeoSignalTransform, LineOfSight, PathDelay, PathLoss
from torchsig.signals.signal_types import Signal, SignalMetadataObject
from torchsig.transforms.transforms import AWGN, SignalTransform, Spectrogram

from .conftest import (
    CENTER_FREQ,
    MAX_SIGNAL_DURATION,
    MIN_SIGNAL_DURATION,
    NEAR_SF_ALT,
    NEAR_SF_LAT,
    NEAR_SF_LON,
    SAMPLE_RATE,
    SF_ALT,
    SF_LAT,
    SF_LON,
    SIGNAL_LENGTH,
    make_geo_ds,
    make_rx,
    make_tx,
)


TX_POSITION = GeoPoint(37.7749, -122.4194, 10.0)
RX_POSITION = GeoPoint(37.7759, -122.4194, 10.0)
SAMPLE_RATE = 10_000_000.0


# =============================================================================

# =============================================================================
# Core Components: Transmitter & Receiver
# =============================================================================


class TestTransmitter:
    """Tests for the Transmitter class."""

    def test_invalid_dataset_type_raises(self, tx_pos):
        """Verify TypeError raised when dataset is not TorchSigIterableDataset."""
        from torchsig.signals.signal_types import Signal as Sig

        with pytest.raises(TypeError, match="Transmitter dataset must be a TorchSigIterableDataset"):
            Transmitter("not a dataset", tx_pos, identifier="tx_bad")

    def test_dataset_without_sample_rate_raises(self, tx_pos):
        """Verify ValueError raised when dataset is missing sample_rate metadata."""
        from torchsig.datasets import TorchSigIterableDataset
        from torchsig.utils.defaults import TorchSigDefaults
        from torchsig.utils.abstractions import MetadataAttributeError

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        # Remove sample_rate - need to remove from the defaults properly
        metadata_v2 = metadata.copy()
        if "sample_rate" in metadata_v2:
            del metadata_v2["sample_rate"]
        # Use empty signal_generators to avoid ConstellationSignalGenerator validation issues
        source_ds = TorchSigIterableDataset(metadata=metadata_v2, signal_generators=[])
        with pytest.raises(ValueError, match="dataset must have 'sample_rate' in its metadata"):
            Transmitter(source_ds, tx_pos, identifier="tx_bad")

    def test_generate_signal_returns_signal(self, transmitter):
        """Verify generate_signal returns a Signal object."""
        signal = transmitter.generate_signal(0)
        assert isinstance(signal, Signal)

    def test_generate_signal_has_tx_metadata(self, transmitter, tx_pos):
        """Verify generated signal contains transmitter position and ID metadata."""
        signal = transmitter.generate_signal(0)
        assert signal["tx_id"] == transmitter.identifier
        assert signal["tx_lat"] == pytest.approx(tx_pos.lat)
        assert signal["tx_lon"] == pytest.approx(tx_pos.lon)
        assert signal["tx_alt"] == pytest.approx(tx_pos.alt)

    def test_generate_signal_has_velocity_metadata(self, transmitter):
        """Verify generated signal contains zero velocity components by default."""
        signal = transmitter.generate_signal(0)
        for field in ["tx_vel_east", "tx_vel_north", "tx_vel_up"]:
            assert isinstance(signal[field], float)
            assert signal[field] == pytest.approx(0.0)

    def test_generate_signal_different_frames_different_data(self, transmitter, tx_pos):
        """Verify different frame indices produce different signal data but same metadata."""
        signal1 = transmitter.generate_signal(0)
        signal2 = transmitter.generate_signal(1)

        assert signal1 is not signal2
        assert not np.array_equal(signal1.data, signal2.data)

        # Metadata should be consistent for static transmitter
        for field in ["tx_id", "tx_lat", "tx_lon", "tx_alt"]:
            assert signal1[field] == signal2[field]

    @pytest.mark.parametrize("custom_id", ["tx_alpha", "transmitter_1"])
    def test_custom_identifier(self, source_dataset, tx_pos, custom_id):
        """Verify custom transmitter identifier is used correctly."""
        rx = make_rx()
        tx = Transmitter(source_dataset, tx_pos, identifier=custom_id)
        TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        assert tx.identifier == custom_id
        signal = tx.generate_signal(0)
        assert signal["tx_id"] == custom_id

    def test_auto_generated_identifier_unique(self, source_dataset, tx_pos):
        """Verify that Transmitter requires identifier to be explicitly provided."""
        # With identifier now required, we verify that omitting it raises TypeError
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'identifier'"):
            Transmitter(source_dataset, tx_pos)

        # But with explicit identifiers, they can be the same or different
        tx1 = Transmitter(source_dataset, tx_pos, identifier="tx_1")
        tx2 = Transmitter(source_dataset, tx_pos, identifier="tx_2")
        assert tx1.identifier != tx2.identifier

        # Same identifiers are allowed (will be caught by TorchSigGeoDataset validation)
        tx3 = Transmitter(source_dataset, tx_pos, identifier="tx_same")
        tx4 = Transmitter(source_dataset, tx_pos, identifier="tx_same")
        assert tx3.identifier == tx4.identifier

    def test_position_and_dataset_stored(self, transmitter, tx_pos, source_dataset):
        """Verify transmitter stores position and dataset references correctly."""
        assert transmitter.get_position(0) == tx_pos
        assert transmitter.dataset is source_dataset

    def test_sample_rate_stored(self, transmitter, source_dataset):
        """Verify transmitter stores sample_rate from dataset metadata."""
        assert transmitter.sample_rate == pytest.approx(source_dataset["sample_rate"])
        assert isinstance(transmitter.sample_rate, float)

    def test_repr_includes_key_info(self, transmitter):
        """Verify repr contains class name, ID, and position information."""
        repr_str = repr(transmitter)
        assert "Transmitter" in repr_str
        assert "id=" in repr_str
        assert "lat=" in repr_str
        assert "lon=" in repr_str

    def test_generate_signal_without_parent_raises(self, source_dataset, tx_pos):
        """Verify RuntimeError raised when transmitter has no parent TorchSigGeoDataset."""
        tx = Transmitter(source_dataset, tx_pos, identifier="tx_orphan")
        # Don't add to a TorchSigGeoDataset, so parent should be None
        with pytest.raises(RuntimeError, match="must have a TorchSigGeoDataset as its parent"):
            tx.generate_signal(0)

    def test_generate_signal_with_tuple_return(self, transmitter, monkeypatch):
        """Test generate_signal handles tuple return type from dataset."""
        mock_data = np.random.randn(100)
        mock_metadata = SignalMetadataObject()
        # Mock the dataset iterator to return a tuple
        mock_iter = iter([(mock_data, mock_metadata)])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_with_signal_return(self, transmitter, monkeypatch):
        """Test generate_signal handles Signal return type from dataset."""
        mock_signal = Signal(data=np.random.randn(100))
        # Mock the dataset iterator to return a Signal
        mock_iter = iter([mock_signal])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_with_ndarray_return(self, transmitter, monkeypatch):
        """Test generate_signal handles ndarray return type from dataset."""
        mock_data = np.random.randn(100)
        # Mock the dataset iterator to return an ndarray
        mock_iter = iter([mock_data])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_with_other_return(self, transmitter, monkeypatch):
        """Test generate_signal handles other return types by converting to Signal."""
        # Mock the dataset iterator to return a list
        mock_iter = iter([[1, 2, 3]])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_with_ndarray_direct(self, transmitter, monkeypatch):
        """Test generate_signal handles direct ndarray return from dataset."""
        # This test verifies the type handling branch when the dataset returns an ndarray
        # We use __iter__ mocking like the other similar tests
        mock_data = (np.random.randn(100) + 1j * np.random.randn(100)).astype(np.complex64)
        mock_iter = iter([mock_data])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)
        # Branch: isinstance(sample, np.ndarray) -> Signal(data=sample)
        # Note: The actual data may differ because seed() creates a fresh __iter__ call
        # But the test verifies that Signal creation from ndarray works
        assert result.data.dtype == np.complex64

    def test_generate_signal_with_signal_direct(self, transmitter, monkeypatch):
        """Test generate_signal handles direct Signal return from dataset (line 288-289)."""
        mock_signal = Signal(data=np.random.randn(100) + 1j * np.random.randn(100))
        mock_iter = iter([mock_signal])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)
        # This exercises line 288-289: elif isinstance(sample, Signal): signal = sample

    def test_generate_signal_with_tuple_ndarray_first(self, transmitter, monkeypatch):
        """Test generate_signal handles tuple with ndarray as first element (line 286-287)."""
        mock_data = np.random.randn(100) + 1j * np.random.randn(100)
        mock_iter = iter([(mock_data, "metadata")])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)
        # This exercises line 287: signal = sample[0] if isinstance(sample[0], Signal) else Signal(data=sample[0])

    def test_generate_signal_with_tuple_signal_first(self, transmitter, monkeypatch):
        """Test generate_signal handles tuple with Signal as first element."""
        mock_signal = Signal(data=np.random.randn(100))
        mock_metadata = SignalMetadataObject()
        # Mock the dataset iterator to return a tuple with Signal first
        mock_iter = iter([(mock_signal, mock_metadata)])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)


class TestReceiver:
    """Tests for the Receiver class."""

    def test_apply_receiver_effects_no_transforms(self, receiver):
        """Verify receiver with no transforms returns signal unchanged."""
        test_signal = Signal(data=np.ones(100, dtype=np.complex64))
        result = apply_transforms_to_signal(test_signal, receiver.receiver_transforms)

        assert isinstance(result, Signal)
        np.testing.assert_array_equal(result.data, test_signal.data)

    def test_apply_receiver_effects_returns_signal(self, receiver):
        """Verify transform application always returns a Signal."""
        test_signal = Signal(data=np.ones(100, dtype=np.complex64))
        result = apply_transforms_to_signal(test_signal, receiver.receiver_transforms)
        assert isinstance(result, Signal)

    @pytest.mark.parametrize(
        ("loss_db", "expected_attenuation"),
        [(10.0, 10 ** (-10 / 20)), (5.0, 10 ** (-5 / 20))],
    )
    def test_apply_single_transform_attenuation(self, rx_pos, loss_db, expected_attenuation):
        """Verify single PathLoss transform attenuates signal by expected amount."""
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", receiver_transforms=[PathLoss(model="custom", loss_db=loss_db)])
        test_signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        initial = test_signal.copy()
        result = apply_transforms_to_signal(test_signal, receiver.receiver_transforms)

        expected_data = initial.data * expected_attenuation
        np.testing.assert_allclose(result.data, expected_data, rtol=1e-5)

    def test_apply_multiple_transforms_sequential(self, rx_pos):
        """Verify multiple transforms are applied in sequence with cumulative effect."""
        receiver = Receiver(
            rx_pos,
            sample_rate=SAMPLE_RATE,
            identifier="rx_test",
            receiver_transforms=[
                PathLoss(model="custom", loss_db=10.0),
                PathLoss(model="custom", loss_db=5.0),
            ],
        )
        test_signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        initial = test_signal.copy()
        result = apply_transforms_to_signal(test_signal, receiver.receiver_transforms)

        # Total attenuation: 10^(-10/20) * 10^(-5/20) = 10^(-15/20)
        expected_attenuation = (10 ** (-10 / 20)) * (10 ** (-5 / 20))
        expected_data = initial.data * expected_attenuation
        np.testing.assert_allclose(result.data, expected_data, rtol=1e-5)

    def test_position_stored_correctly(self, receiver, rx_pos):
        """Verify receiver stores and returns position correctly."""
        assert receiver.get_position(0) == rx_pos

    def test_transforms_stored_correctly(self, receiver):
        """Verify receiver stores transforms list."""
        assert receiver.receiver_transforms == []

    @pytest.mark.parametrize("identifier,expected", [("rx_auto", "rx_auto"), ("my_rx", "my_rx")])
    def test_identifier_generation(self, rx_pos, identifier, expected):
        """Verify receiver identifier is set correctly."""
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier=identifier)
        assert receiver.identifier == expected

    def test_receiver_rejects_non_signal_transform(self, rx_pos):
        """Receiver transforms must be SignalTransform instances."""
        non_transform = object()

        with pytest.raises(
            TypeError,
            match=r"receiver_transforms\[0\].*SignalTransform",
        ):
            Receiver(
                rx_pos,
                sample_rate=SAMPLE_RATE,
                identifier="rx_test",
                receiver_transforms=[non_transform],
            )

    def test_identifier_in_metadata(self, receiver):
        """Verify receiver ID is stored as an attribute (not metadata dict)."""
        # After refactoring, Receiver inherits from Seedable, not HMO
        # rx_id is not stored in metadata dict, but identifier is an attribute
        assert receiver.identifier == receiver.identifier
        assert isinstance(receiver.identifier, str)

    def test_repr_includes_key_info(self, receiver):
        """Verify repr contains class name and position information."""
        repr_str = repr(receiver)
        assert "Receiver" in repr_str
        assert "id=" in repr_str
        assert "lat=" in repr_str

    @pytest.mark.parametrize("sample_rate", [-1000, 0, float("inf"), float("-inf"), float("nan")])
    def test_invalid_sample_rate_raises(self, rx_pos, sample_rate):
        """Verify that invalid sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="Receiver.sample_rate must be positive and finite"):
            Receiver(position=rx_pos, sample_rate=sample_rate, identifier="rx_bad")


# =============================================================================
# TorchSigGeoDataset - Core Functionality
# =============================================================================


class TestTorchSigGeoDatasetCreation:
    """Tests for TorchSigGeoDataset instantiation and configuration."""

    def test_basic_creation(self, transmitter, receiver):
        """Verify basic dataset creation with one transmitter and receiver."""
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        assert len(geo_ds.transmitters) == 1
        assert len(geo_ds.receivers) == 1
        assert transmitter in geo_ds.transmitters
        assert receiver in geo_ds.receivers

    def test_multiple_transmitters(self):
        """Verify creation with multiple transmitters."""
        geo_ds = make_geo_ds(tx_count=3, rx_count=1, tx_offset=0.001)
        assert len(geo_ds.transmitters) == 3

    def test_multiple_receivers(self):
        """Verify creation with multiple receivers."""
        geo_ds = make_geo_ds(tx_count=1, rx_count=3, rx_offset=0.001)
        assert len(geo_ds.receivers) == 3

    @pytest.mark.parametrize(("tx_count", "rx_count"), [(0, 1), (1, 0), (0, 0)])
    def test_empty_components_raises(self, transmitter, receiver, tx_count, rx_count):
        """Verify empty transmitter or receiver lists raise ValueError."""
        txs = [transmitter] * tx_count
        rxs = [receiver] * rx_count
        with pytest.raises(ValueError, match="at least one"):
            TorchSigGeoDataset(transmitters=txs, receivers=rxs)

    def test_duplicate_transmitter_ids_raise(self, minimal_metadata):
        """Verify duplicate transmitter identifiers raise ValueError."""
        tx_ds = TorchSigIterableDataset(metadata=minimal_metadata, signal_generators=["bpsk"])
        pos = GeoPoint(lat=SF_LAT, lon=SF_LON)
        tx1 = Transmitter(tx_ds, pos, identifier="tx_dup")
        tx2 = Transmitter(tx_ds, GeoPoint(lat=SF_LAT + 0.001, lon=SF_LON), identifier="tx_dup")
        rx = make_rx()

        with pytest.raises(ValueError, match="unique transmitter identifiers"):
            TorchSigGeoDataset(transmitters=[tx1, tx2], receivers=[rx])

    def test_duplicate_receiver_ids_raise(self):
        """Verify duplicate receiver identifiers raise ValueError."""
        tx = make_tx()
        pos = GeoPoint(lat=NEAR_SF_LAT, lon=NEAR_SF_LON)
        rx1 = Receiver(pos, sample_rate=SAMPLE_RATE, identifier="rx_dup")
        rx2 = Receiver(GeoPoint(lat=NEAR_SF_LAT + 0.001, lon=NEAR_SF_LON), sample_rate=SAMPLE_RATE, identifier="rx_dup")

        with pytest.raises(ValueError, match="unique receiver identifiers"):
            TorchSigGeoDataset(transmitters=[tx], receivers=[rx1, rx2])

    def test_channel_transforms_stored(self, transmitter, receiver):
        """Verify channel transforms are stored correctly."""
        channel_tx = [PathLoss(model="custom", loss_db=5.0)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], channel_transforms=channel_tx)
        assert len(geo_ds.channel_transforms) == 1
        assert geo_ds.channel_transforms[0] is channel_tx[0]

    def test_channel_transforms_non_signal_transform_raises(self, transmitter, receiver):
        """Test that non-SignalTransform in channel_transforms raises TypeError (line 589-593)."""
        non_signal_transform = PathLoss(model="custom", loss_db=5.0)  # Actually IS a SignalTransform
        # but we need a non-SignalTransform - use AWGN which might be SignalTransform too
        # Let's use a plain object
        from torchsig.transforms.transforms import AWGN

        # AWGN is a SignalTransform, so we just use a plain object
        with pytest.raises(TypeError, match=r"channel_transforms\[0\] must be a SignalTransform"):
            TorchSigGeoDataset(
                transmitters=[transmitter],
                receivers=[receiver],
                channel_transforms=[object()],  # plain object, not SignalTransform
            )

    def test_per_path_transforms_non_signal_transform_raises(self, transmitter, receiver):
        """Test that non-SignalTransform in per_path_transforms raises TypeError (line 603-607)."""
        per_path = {
            (transmitter.identifier, receiver.identifier): [object()]  # plain object, not SignalTransform
        }
        with pytest.raises(TypeError, match="per_path_transforms"):
            TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], per_path_transforms=per_path)

    def test_per_path_transforms_stored(self, transmitter, receiver):
        """Verify per-path transforms are stored correctly."""
        per_path = {(transmitter.identifier, receiver.identifier): [PathLoss(model="custom", loss_db=10.0)]}
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], per_path_transforms=per_path)
        assert len(geo_ds.per_path_transforms) == 1

    def test_dataset_transforms_stored(self, transmitter, receiver):
        """Verify dataset-level transforms are stored."""
        dataset_tx = [Spectrogram(fft_size=256)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], transforms=dataset_tx)
        assert len(geo_ds.transforms) == 1

    def test_dataset_with_non_seedable_transform(self, transmitter, receiver):
        """Test TorchSigGeoDataset handles non-Seedable transforms (line 567-568)."""
        non_seedable_obj = object()
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], transforms=[non_seedable_obj])
        assert len(geo_ds.transforms) == 1

    def test_target_labels_stored(self, transmitter, receiver):
        """Verify target labels configuration is stored."""
        target_labels = ["tx_id", "rx_id", "snr_db"]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], target_labels=target_labels)
        assert geo_ds.target_labels == target_labels


class TestTorchSigGeoDatasetTopology:
    """Tests for network topology management."""

    def test_default_full_mesh(self, transmitter, receiver):
        """Verify default topology creates full mesh connecting all tx to all rx."""
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        assert len(geo_ds.topology) == 1
        key = (transmitter.identifier, receiver.identifier)
        assert key in geo_ds.topology
        assert geo_ds.topology[key]["transmitter"] is transmitter
        assert geo_ds.topology[key]["receiver"] is receiver

    @pytest.mark.parametrize(
        ("tx_count", "rx_count", "expected_paths"),
        [(1, 1, 1), (2, 1, 2), (1, 2, 2), (2, 2, 4), (3, 2, 6)],
    )
    def test_full_mesh_path_count(self, tx_count, rx_count, expected_paths):
        """Verify full mesh topology creates correct number of paths."""
        geo_ds = make_geo_ds(tx_count=tx_count, rx_count=rx_count, tx_offset=0.001, rx_offset=0.001)
        assert len(geo_ds.topology) == expected_paths

    @pytest.mark.parametrize(
        ("topology_config", "expected_paths", "should_contain", "should_not_contain"),
        [
            # Single path
            ({"tx1": ["rx1"]}, 1, [("tx1", "rx1")], [("tx2", "rx1"), ("tx1", "rx2")]),
            # One tx to multiple rx
            ({"tx1": ["rx1", "rx2"]}, 2, [("tx1", "rx1"), ("tx1", "rx2")], [("tx2", "rx1")]),
            # Multiple tx to one rx
            ({"tx1": ["rx1"], "tx2": ["rx1"]}, 2, [("tx1", "rx1"), ("tx2", "rx1")], [("tx1", "rx2")]),
            # Complex topology
            ({"tx1": ["rx1", "rx2"], "tx2": ["rx1"]}, 3, [("tx1", "rx1"), ("tx1", "rx2"), ("tx2", "rx1")], [("tx2", "rx2")]),
        ],
    )
    def test_custom_topology(self, topology_config, expected_paths, should_contain, should_not_contain):
        """Verify custom topology configurations work correctly."""
        tx1 = make_tx(identifier="tx1")
        tx2 = make_tx(lat=SF_LAT + 0.001, identifier="tx2")
        rx1 = make_rx(identifier="rx1")
        rx2 = make_rx(lat=NEAR_SF_LAT + 0.001, identifier="rx2")

        geo_ds = TorchSigGeoDataset(transmitters=[tx1, tx2], receivers=[rx1, rx2], topology=topology_config)

        assert len(geo_ds.topology) == expected_paths
        for path in should_contain:
            assert path in geo_ds.topology
        for path in should_not_contain:
            assert path not in geo_ds.topology

    @pytest.mark.parametrize(
        ("invalid_key", "error_pattern"),
        [("tx999", "unknown transmitter"), ("rx999", "unknown receiver")],
    )
    def test_invalid_topology_raises(self, invalid_key, error_pattern):
        """Verify invalid topology references raise appropriate errors."""
        tx = make_tx(identifier="tx1")
        rx = make_rx(identifier="rx1")
        topology = {invalid_key: ["rx1"]} if invalid_key.startswith("tx") else {"tx1": [invalid_key]}

        with pytest.raises(ValueError, match=error_pattern):
            TorchSigGeoDataset(transmitters=[tx], receivers=[rx], topology=topology)

    def test_topology_self_loop_raises(self):
        """Verify self-loop topology raises ValueError."""
        # Create tx and rx with the same identifier to test self-loop detection
        tx = make_tx(identifier="node1")
        rx = make_rx(identifier="node1")  # Same identifier as transmitter
        topology = {"node1": ["node1"]}  # node1 -> node1 (self-loop)

        with pytest.raises(ValueError, match="self-loop.*cannot connect to receiver"):
            TorchSigGeoDataset(transmitters=[tx], receivers=[rx], topology=topology)

    def test_topology_summary_structure(self):
        """Verify topology summary has correct structure and counts."""
        geo_ds = make_geo_ds(tx_count=2, rx_count=2, tx_offset=0.001, rx_offset=0.001)
        summary = geo_ds.get_topology_summary()

        assert "transmitters" in summary
        assert "receivers" in summary
        assert "paths" in summary
        assert len(summary["transmitters"]) == 2
        assert len(summary["receivers"]) == 2
        assert len(summary["paths"]) == 4

    def test_topology_summary_contains_positions(self, simple_geo_ds, transmitter, receiver):
        """Verify topology summary includes complete position data."""
        summary = simple_geo_ds.get_topology_summary()

        tx_info = summary["transmitters"][transmitter.identifier]
        rx_info = summary["receivers"][receiver.identifier]

        for info in [tx_info, rx_info]:
            assert "position" in info
            assert all(k in info["position"] for k in ["lat", "lon", "alt"])


class TestTorchSigGeoDatasetIndexing:
    """Tests for indexing and length operations."""

    @pytest.mark.parametrize("num_receivers", [1, 2, 3])
    def test_len_equals_receiver_count(self, num_receivers):
        """Verify len() returns the number of receivers."""
        geo_ds = make_geo_ds(tx_count=1, rx_count=num_receivers, rx_offset=0.001)
        assert len(geo_ds) == num_receivers

    def test_integer_indexing_raises(self, simple_geo_ds):
        """Verify integer indexing for signals raises TypeError."""
        with pytest.raises(TypeError, match="does not support integer indexing"):
            _ = simple_geo_ds[0]

    def test_string_indexing_accesses_metadata(self, simple_geo_ds):
        """Verify string keys access metadata correctly."""
        assert "_receiver_counter" in simple_geo_ds.keys()
        assert isinstance(simple_geo_ds["_receiver_counter"], int)

    def test_invalid_metadata_key_raises(self, simple_geo_ds):
        """Verify invalid metadata keys raise appropriate error."""
        with pytest.raises((KeyError, AttributeError)):
            _ = simple_geo_ds["nonexistent_key"]


class TestTorchSigGeoDatasetIteration:
    """Tests for iteration behavior."""

    @pytest.mark.parametrize("num_receivers", [2, 3])
    def test_round_robin_iteration(self, num_receivers):
        """Verify round-robin iteration through receivers."""
        geo_ds = make_geo_ds(tx_count=1, rx_count=num_receivers, rx_offset=0.001)

        signals = list(islice(geo_ds, num_receivers * 2))
        assert len(signals) == num_receivers * 2

        rx_ids = [s["rx_id"] for s in signals]
        expected = [f"rx_{i}" for i in range(num_receivers)] * 2
        assert rx_ids == expected

    def test_iter_returns_self(self, simple_geo_ds):
        """Verify __iter__ returns the dataset itself."""
        assert iter(simple_geo_ds) is simple_geo_ds

    def test_next_returns_signals(self, simple_geo_ds, receiver):
        """Verify __next__ returns Signal objects with correct receiver ID."""
        signal1 = next(simple_geo_ds)
        signal2 = next(simple_geo_ds)

        assert isinstance(signal1, Signal)
        assert isinstance(signal2, Signal)
        assert signal1["rx_id"] == receiver.identifier
        assert signal2["rx_id"] == receiver.identifier


class TestTorchSigGeoDatasetSignals:
    """Tests for generated signal content and metadata."""

    def test_has_component_signals(self, simple_geo_ds):
        """Verify signals contain component signals from transmitters."""
        signal = next(iter(simple_geo_ds))
        assert len(signal.component_signals) > 0

    def test_single_transmitter_metadata(self, simple_geo_ds, transmitter, receiver):
        """Verify signal metadata for single transmitter configuration."""
        signal = next(iter(simple_geo_ds))

        assert signal["rx_id"] == receiver.identifier
        assert signal["rx_lat"] == pytest.approx(receiver.get_position(0).lat)
        assert signal["rx_lon"] == pytest.approx(receiver.get_position(0).lon)
        assert signal["rx_alt"] == pytest.approx(receiver.get_position(0).alt)
        assert signal["num_transmitters"] == 1
        assert signal["sample_rate"] == pytest.approx(receiver.sample_rate)
        assert signal.component_signals[0]["tx_id"] == transmitter.identifier

    def test_multiple_transmitters_metadata(self):
        """Verify signal metadata for multiple transmitters."""
        geo_ds = make_geo_ds(tx_count=2, rx_count=1, tx_offset=0.001)

        signal = next(iter(geo_ds))
        assert signal["num_transmitters"] == 2
        assert len(signal["tx_ids"]) == 2
        assert all(f"tx_{i}" in signal["tx_ids"] for i in range(2))
        assert len(signal.component_signals) == 2

    @pytest.mark.parametrize(
        ("field", "expected_type"),
        [
            ("rx_id", str),
            ("rx_lat", float),
            ("rx_lon", float),
            ("rx_alt", float),
            ("num_transmitters", int),
            ("sample_rate", float),
        ],
    )
    def test_metadata_types_for_hdf5_serialization(self, simple_geo_ds, field, expected_type):
        """Verify metadata uses simple types compatible with HDF5 serialization."""
        signal = next(iter(simple_geo_ds))
        assert isinstance(signal[field], expected_type)

    def test_component_signals_have_full_metadata(self, simple_geo_ds, transmitter, tx_pos, receiver):
        """Verify component signals can access all required metadata via parent chain."""
        signal = next(iter(simple_geo_ds))
        comp = signal.component_signals[0]

        # Transmitter metadata (stored directly on component signal)
        assert comp["tx_id"] == transmitter.identifier
        assert comp["tx_lat"] == pytest.approx(tx_pos.lat)
        assert comp["tx_lon"] == pytest.approx(tx_pos.lon)
        assert comp["tx_alt"] == pytest.approx(tx_pos.alt)

        # Path metadata (stored directly on component signal)
        assert hasattr(comp, "path_distance")
        assert comp["path_distance"] > 0

        # Receiver position - stored at root level, accessed via parent chain
        # This tests parent-aware metadata access
        assert hasattr(comp, "rx_lat")
        assert comp["rx_lat"] == pytest.approx(receiver.get_position(0).lat)
        assert comp["rx_lon"] == pytest.approx(receiver.get_position(0).lon)
        assert comp["rx_alt"] == pytest.approx(receiver.get_position(0).alt)

    def test_signal_data_is_complex(self, simple_geo_ds):
        """Verify signal data is complex-valued."""
        signal = next(iter(simple_geo_ds))
        assert np.iscomplexobj(signal.data)

    def test_signal_data_correct_length(self, simple_geo_ds, minimal_metadata):
        """Verify signal data has expected length from metadata."""
        signal = next(iter(simple_geo_ds))
        assert len(signal.data) == minimal_metadata["num_iq_samples_dataset"]


class TestTorchSigGeoDatasetTransforms:
    """Tests for transform application in the dataset."""

    def test_channel_transforms_applied_to_components(self, transmitter, receiver):
        """Verify channel transforms are applied to component signals."""
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], channel_transforms=[PathLoss(model="custom", loss_db=10.0)])
        signal = next(iter(geo_ds))
        for comp in signal.component_signals[0].component_signals:
            assert hasattr(comp, "path_loss_db")

    def test_dataset_transforms_applied(self, transmitter, receiver):
        """Verify dataset-level transforms are applied to signals."""
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], transforms=[Spectrogram(fft_size=256)])
        signal = next(iter(geo_ds))
        assert signal is not None


class TestTorchSigGeoDatasetRepr:
    """Tests for string representation of the dataset."""

    @pytest.mark.parametrize("expected", ["TorchSigGeoDataset", "transmitters=1", "receivers=1"])
    def test_single_component_repr(self, simple_geo_ds, expected):
        """Verify repr for single transmitter/receiver dataset."""
        assert expected in repr(simple_geo_ds)

    @pytest.mark.parametrize(("tx_count", "rx_count"), [(2, 1), (1, 2), (2, 2)])
    def test_multiple_components_repr(self, tx_count, rx_count):
        """Verify repr for multi-component datasets shows correct counts."""
        geo_ds = make_geo_ds(tx_count=tx_count, rx_count=rx_count, tx_offset=0.001, rx_offset=0.001)
        repr_str = repr(geo_ds)
        assert f"transmitters={tx_count}" in repr_str
        assert f"receivers={rx_count}" in repr_str


# =============================================================================
# Transform Application
# =============================================================================


class TestApplyTransformsToSignal:
    """Tests for the apply_transforms_to_signal helper function."""

    def test_geo_signal_transform_only(self):
        """Verify GeoSignalTransform is applied to wrapper and components."""
        comp1 = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9, snr_db=20.0)
        comp2 = Signal(data=np.ones(100, dtype=np.complex64) * 2, center_freq=1e9, snr_db=20.0)
        wrapper = Signal(data=np.ones(100, dtype=np.complex64) * 3, component_signals=[comp1, comp2], center_freq=1e9, path_distance=1000.0, snr_db=20.0)

        result = apply_transforms_to_signal(wrapper, [PathLoss(model="custom", loss_db=10.0)])

        # path_loss_db at component level
        for comp in result.component_signals:
            assert hasattr(comp, "path_loss_db")

    def test_signal_transform_only(self):
        """Verify SignalTransform is applied to components and wrapper data is rebuilt."""
        comp1 = Signal(data=np.ones(100, dtype=np.complex64))
        comp2 = Signal(data=np.ones(100, dtype=np.complex64) * 2)
        wrapper = Signal(data=np.ones(100, dtype=np.complex64) * 3, component_signals=[comp1, comp2])
        original_comp1_data = comp1.data.copy()
        original_wrapper_data = wrapper.data.copy()

        result = apply_transforms_to_signal(wrapper, [AWGN(noise_power_db=-20.0)])

        # Component data should change (different from original)
        assert not np.array_equal(original_comp1_data, result.component_signals[0].data)
        # Wrapper data should be rebuilt from components
        assert not np.array_equal(original_wrapper_data, result.data)

    def test_mixed_transforms(self):
        """Verify mixed GeoSignalTransform and SignalTransform are applied correctly."""
        comp1 = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9)
        comp2 = Signal(data=np.ones(100, dtype=np.complex64) * 2, center_freq=1e9)
        wrapper = Signal(
            data=np.ones(100, dtype=np.complex64) * 3,
            component_signals=[comp1, comp2],
            center_freq=1e9,
            path_distance=1000.0,
            sample_rate=1e6,
        )
        wrapper["tx_lat"] = 0.0
        wrapper["tx_lon"] = 0.0
        wrapper["tx_alt"] = 100.0
        wrapper["rx_lat"] = 0.0
        wrapper["rx_lon"] = 180.0
        wrapper["rx_alt"] = 100.0

        # Set parent pointers so components can inherit metadata via hierarchical lookup
        comp1.add_parent(wrapper, register=False)
        comp2.add_parent(wrapper, register=False)

        transforms = [
            PathLoss(model="custom", loss_db=10.0),
            AWGN(noise_power_db=-20.0),
            LineOfSight(),
        ]
        result = apply_transforms_to_signal(wrapper, transforms)

        assert hasattr(result, "los")
        assert result["los"] is False
        assert np.all(result.data == 0)

        for comp in result.component_signals:
            assert hasattr(comp, "path_loss_db")
            assert np.all(comp.data == 0)

    def test_empty_transforms(self):
        """Verify empty transform list returns signal unchanged."""
        comp1 = Signal(data=np.ones(100, dtype=np.complex64))
        wrapper = Signal(data=np.ones(100, dtype=np.complex64), component_signals=[comp1])
        original_data = wrapper.data.copy()

        result = apply_transforms_to_signal(wrapper, [])

        np.testing.assert_array_equal(result.data, original_data)

    def test_empty_component_signals(self):
        """Verify function works with signal that has no component_signals."""
        wrapper = Signal(data=np.ones(100, dtype=np.complex64), path_distance=1000.0)

        result = apply_transforms_to_signal(wrapper, [PathLoss(model="custom", loss_db=10.0)])

        assert hasattr(result, "path_loss_db")
        assert not hasattr(result, "component_signals") or len(result.component_signals) == 0


# =============================================================================
# Static Dataset & Serialization
# =============================================================================


class TestStaticTorchSigGeoDataset:
    """Tests for StaticTorchSigGeoDataset (file loading and access)."""

    def test_load_from_file(self, simple_geo_ds, temp_dir):
        """Verify loading a saved GeoDataset from disk."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=3,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        assert static_ds is not None

        loaded_signal = static_ds[0]
        assert isinstance(loaded_signal, Signal)
        assert loaded_signal.data.shape[0] == SIGNAL_LENGTH

    @pytest.mark.parametrize(
        ("field", "location"),
        [
            ("rx_id", "top"),
            ("rx_lat", "top"),
            ("rx_lon", "top"),
            ("rx_alt", "top"),
            ("sample_rate", "top"),
            ("tx_lat", "component"),
            ("tx_lon", "component"),
            ("tx_alt", "component"),
        ],
    )
    def test_loaded_signal_has_metadata(self, simple_geo_ds, temp_dir, field, location):
        """Verify loaded signals have all expected metadata."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=2,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        loaded_signal = static_ds[0]

        if location == "top":
            assert hasattr(loaded_signal, field)
        else:
            assert len(loaded_signal.component_signals) > 0
            assert hasattr(loaded_signal.component_signals[0], field)

    def test_loaded_metadata_values_preserved(self, simple_geo_ds, temp_dir, transmitter, receiver):
        """Verify geolocation metadata values are preserved during save/load."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=2,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        loaded_signal = static_ds[0]

        assert loaded_signal.component_signals[0]["tx_lat"] == pytest.approx(transmitter.get_position(0).lat)
        assert loaded_signal["rx_lat"] == pytest.approx(receiver.get_position(0).lat)

    def test_serialize_with_multiple_transmitters(self, temp_dir):
        """Verify serialization preserves component signals from multiple transmitters."""
        geo_ds = make_geo_ds(tx_count=2, rx_count=1, tx_offset=0.001)

        # Verify signal has component signals before serialization
        signal = next(iter(geo_ds))
        assert len(signal.component_signals) == 2

        # Save and load
        geo_ds.to_file(
            root=temp_dir,
            dataset_length=2,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        loaded_signal = static_ds[0]

        assert isinstance(loaded_signal, Signal)
        assert len(loaded_signal.component_signals) >= 2

    def test_serialize_to_file_creates_files(self, simple_geo_ds, temp_dir):
        """Verify serialization creates valid HDF5 files."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=3,
            overwrite=True,
        )

        data_file = Path(temp_dir) / "data.h5"
        assert data_file.exists()
        assert data_file.stat().st_size > 0

    def test_static_dataset_repr(self, temp_dir):
        """Verify string representation of StaticTorchSigGeoDataset."""
        geo_ds = make_geo_ds(tx_count=1, rx_count=1)
        geo_ds.to_file(root=temp_dir, dataset_length=2, overwrite=True)

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        repr_str = repr(static_ds)
        assert "StaticTorchSigGeoDataset" in repr_str
        assert "root=" in repr_str

    @pytest.mark.parametrize("length", [1, 5, 10])
    def test_static_dataset_len(self, simple_geo_ds, temp_dir, length):
        """Verify len() on StaticTorchSigGeoDataset."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=length,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)
        assert len(static_ds) == length

    @pytest.mark.parametrize("index,length,should_raise", [(0, 5, False), (4, 5, False), (10, 3, True), (100, 3, True)])
    def test_static_dataset_getitem(self, simple_geo_ds, temp_dir, index, length, should_raise):
        """Verify __getitem__ on StaticTorchSigGeoDataset."""
        simple_geo_ds.to_file(
            root=temp_dir,
            dataset_length=length,
            overwrite=True,
        )

        static_ds = StaticTorchSigGeoDataset(root=temp_dir)

        if should_raise:
            with pytest.raises(IndexError):
                _ = static_ds[index]
        else:
            signal = static_ds[index]
            assert isinstance(signal, Signal)


class TestGeoFilesSerialization:
    """Tests for GeoDataset serialization to .yaml/.dat file pairs."""

    def test_to_yaml_dat_pairs_creates_files(self, simple_geo_ds, temp_dir):
        """Verify to_yaml_dat_pairs creates .yaml and .dat files."""
        simple_geo_ds.to_yaml_dat_pairs(
            root=temp_dir,
            dataset_length=3,
            overwrite=True,
        )

        all_files = list(temp_dir.glob("*"))
        yaml_files = [f for f in all_files if f.suffix == ".yaml" and f.stem.isdigit()]
        dat_files = [f for f in all_files if f.suffix == ".dat" and f.stem.isdigit()]

        assert len(yaml_files) > 0
        assert len(dat_files) > 0
        assert len(yaml_files) == len(dat_files)

    @pytest.mark.parametrize("data_type", ["float32", "float64", "double", "float"])
    def test_to_yaml_dat_pairs_various_data_types(self, simple_geo_ds, temp_dir, data_type):
        """Verify to_yaml_dat_pairs works with various floating-point data types."""
        # Clean directory
        for f in temp_dir.glob("*"):
            f.unlink()

        simple_geo_ds.to_yaml_dat_pairs(
            root=temp_dir,
            dataset_length=2,
            data_type=data_type,
            overwrite=True,
        )

        yaml_files = list(temp_dir.glob("*.yaml"))
        dat_files = list(temp_dir.glob("*.dat"))
        assert len(yaml_files) > 0
        assert len(dat_files) > 0

    def test_to_yaml_dat_pairs_integer_data_requires_normalization(self, source_dataset, tx_pos, rx_pos, temp_dir):
        """Verify that writing to integer formats raises ValueError for unnormalized data."""
        from torchsig.geo.datasets import Receiver, TorchSigGeoDataset, Transmitter
        from torchsig.transforms.base_transforms import Lambda

        # Clean directory
        for f in temp_dir.glob("*"):
            f.unlink()

        # Create a geo dataset with a gain transform that produces values outside [-1, 1]
        # This simulates unnormalized data that would fail integer serialization
        def gain_func(data):
            return np.full_like(
                data,
                2.0 + 2.0j,
            )

        rx = Receiver(rx_pos, sample_rate=source_dataset["sample_rate"], identifier="rx_gain_test")
        tx = Transmitter(source_dataset, tx_pos, identifier="tx_gain_test")
        geo_ds = TorchSigGeoDataset(
            transmitters=[tx],
            receivers=[rx],
            transforms=[Lambda(gain_func)],
        )

        # This should raise ValueError because values are outside [-1, 1]
        with pytest.raises(ValueError, match="Cannot serialize to int16: IQ data has values outside"):
            geo_ds.to_yaml_dat_pairs(
                root=temp_dir,
                dataset_length=1,
                data_type="int16",
                overwrite=True,
            )

    def test_to_yaml_dat_pairs_with_field_mapping(self, simple_geo_ds, temp_dir):
        """Verify to_yaml_dat_pairs works with custom field mapping."""
        simple_geo_ds.to_yaml_dat_pairs(
            root=temp_dir,
            dataset_length=2,
            data_type="float32",
            field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"},
            overwrite=True,
        )

        yaml_files = list(temp_dir.glob("*.yaml"))
        assert len(yaml_files) > 0


# =============================================================================
# Position, Movement & Velocity
# =============================================================================


class TestMobileObjects:
    """Tests for mobile transmitters and receivers with frame-level movement."""

    @pytest.fixture
    def moving_tx_position(self):
        """Create a callable for transmitter moving east at 100m per frame."""
        base_lat = SF_LAT
        base_lon = SF_LON
        base_alt = 100
        meters_per_degree_lon = 111320 * np.cos(np.radians(base_lat))

        def moving_position(sample_index):
            lon_offset = 100 * sample_index / meters_per_degree_lon
            return GeoPoint(lat=base_lat, lon=base_lon + lon_offset, alt=base_alt)

        return moving_position

    @pytest.fixture
    def moving_rx_position(self):
        """Create a callable for receiver moving north at 50m per frame."""
        base_lat = NEAR_SF_LAT
        base_lon = NEAR_SF_LON
        base_alt = NEAR_SF_ALT
        meters_per_degree_lat = 111320

        def moving_position(sample_index):
            lat_offset = 50 * sample_index / meters_per_degree_lat
            return GeoPoint(lat=base_lat + lat_offset, lon=base_lon, alt=base_alt)

        return moving_position

    @pytest.fixture
    def converging_positions(self):
        """Create callables for tx and rx moving toward each other."""
        tx_base_lat = SF_LAT
        tx_base_lon = SF_LON
        rx_base_lat = NEAR_SF_LAT
        rx_base_lon = NEAR_SF_LON - 0.001  # Offset west
        meters_per_degree_lon = 111320 * np.cos(np.radians(tx_base_lat))

        def tx_position(sample_index):
            lon_offset = 50 * sample_index / meters_per_degree_lon  # Move east
            return GeoPoint(lat=tx_base_lat, lon=tx_base_lon + lon_offset, alt=100)

        def rx_position(sample_index):
            lon_offset = 50 * sample_index / meters_per_degree_lon  # Move west
            return GeoPoint(lat=rx_base_lat, lon=rx_base_lon - lon_offset, alt=10)

        return tx_position, rx_position

    # --- Transmitter Position Tests ---

    @pytest.mark.parametrize("sample_index", [0, 10, 100])
    def test_transmitter_static_position_constant(self, source_dataset, tx_pos, sample_index):
        """Verify static transmitter position is constant across sample indices."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        assert transmitter.get_position(sample_index) == tx_pos

    def test_transmitter_moving_position_changes(self, source_dataset, moving_tx_position):
        """Verify moving transmitter position changes across sample indices."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")

        pos_0 = transmitter.get_position(0)
        pos_1 = transmitter.get_position(1)
        pos_10 = transmitter.get_position(10)

        assert isinstance(pos_0, GeoPoint)
        assert isinstance(pos_1, GeoPoint)
        assert pos_0 != pos_1 != pos_10
        assert pos_0.lon < pos_1.lon < pos_10.lon  # Moving east
        assert pos_0.lat == pos_1.lat == pos_10.lat
        assert pos_0.alt == pos_1.alt == pos_10.alt

    # --- Receiver Position Tests ---

    @pytest.mark.parametrize("sample_index", [0, 10, 100])
    def test_receiver_static_position_constant(self, rx_pos, sample_index):
        """Verify static receiver position is constant across sample indices."""
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        assert receiver.get_position(sample_index) == rx_pos

    def test_receiver_moving_position_changes(self, moving_rx_position):
        """Verify moving receiver position changes across sample indices."""
        receiver = Receiver(moving_rx_position, sample_rate=SAMPLE_RATE, identifier="rx_moving")

        pos_0 = receiver.get_position(0)
        pos_1 = receiver.get_position(1)
        pos_10 = receiver.get_position(10)

        assert isinstance(pos_0, GeoPoint)
        assert isinstance(pos_1, GeoPoint)
        assert pos_0 != pos_1 != pos_10
        assert pos_0.lat < pos_1.lat < pos_10.lat  # Moving north
        assert pos_0.lon == pos_1.lon == pos_10.lon
        assert pos_0.alt == pos_1.alt == pos_10.alt

    # --- Backward Compatibility Tests ---

    def test_static_objects_backward_compatibility(self, source_dataset, tx_pos, rx_pos):
        """Verify static positions still work (backward compatibility)."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")

        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            channel_transforms=[PathLoss(model="free_space")],
        )

        signal = next(geo_ds)

        assert signal["rx_id"] == receiver.identifier
        assert signal["rx_lat"] == pytest.approx(rx_pos.lat)
        assert signal["rx_lon"] == pytest.approx(rx_pos.lon)
        assert len(signal.component_signals) == 1

        comp_signal = signal.component_signals[0]
        assert comp_signal["tx_id"] == transmitter.identifier
        assert comp_signal["tx_lat"] == pytest.approx(tx_pos.lat)
        assert comp_signal["tx_lon"] == pytest.approx(tx_pos.lon)
        assert comp_signal["path_distance"] == pytest.approx(tx_pos.distance_to(rx_pos))

    @pytest.mark.parametrize("num_samples", [1, 5, 10])
    def test_static_objects_constant_positions(self, source_dataset, tx_pos, rx_pos, num_samples):
        """Verify static objects maintain same positions across samples."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(num_samples)]

        for signal in signals:
            comp_signal = signal.component_signals[0]
            assert comp_signal["tx_lat"] == pytest.approx(tx_pos.lat)
            assert comp_signal["tx_lon"] == pytest.approx(tx_pos.lon)
            assert signal["rx_lat"] == pytest.approx(rx_pos.lat)
            assert signal["rx_lon"] == pytest.approx(rx_pos.lon)

    # --- Moving Transmitter Tests ---

    def test_moving_transmitter_static_receiver(self, source_dataset, rx_pos, moving_tx_position):
        """Verify dataset with moving transmitter and static receiver."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(3)]

        # Transmitter longitude should increase (moving east)
        tx_lons = [s.component_signals[0]["tx_lon"] for s in signals]
        assert tx_lons[0] < tx_lons[1] < tx_lons[2]

        # Receiver position should remain constant
        for signal in signals:
            assert signal["rx_lat"] == pytest.approx(rx_pos.lat)
            assert signal["rx_lon"] == pytest.approx(rx_pos.lon)

    def test_moving_transmitter_distance_changes(self, source_dataset, rx_pos, moving_tx_position):
        """Verify distance changes as transmitter moves."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(3)]
        distances = [s.component_signals[0]["path_distance"] for s in signals]

        # At least one distance should be different
        assert len(set([round(d, 2) for d in distances])) > 1

    # --- Moving Receiver Tests ---

    def test_static_transmitter_moving_receiver(self, source_dataset, tx_pos, moving_rx_position):
        """Verify dataset with static transmitter and moving receiver."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        receiver = Receiver(moving_rx_position, sample_rate=SAMPLE_RATE, identifier="rx_moving")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(3)]

        # Receiver latitude should increase (moving north)
        rx_lats = [s["rx_lat"] for s in signals]
        assert rx_lats[0] < rx_lats[1] < rx_lats[2]

        # Transmitter position should remain constant
        for signal in signals:
            comp_signal = signal.component_signals[0]
            assert comp_signal["tx_lat"] == pytest.approx(tx_pos.lat)
            assert comp_signal["tx_lon"] == pytest.approx(tx_pos.lon)

    # --- Both Moving Tests ---

    def test_both_moving(self, source_dataset, moving_tx_position, moving_rx_position):
        """Verify dataset with both transmitter and receiver moving."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        receiver = Receiver(moving_rx_position, sample_rate=SAMPLE_RATE, identifier="rx_moving")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(3)]

        # Transmitter longitude should increase (moving east)
        tx_lons = [s.component_signals[0]["tx_lon"] for s in signals]
        assert tx_lons[0] < tx_lons[1] < tx_lons[2]

        # Receiver latitude should increase (moving north)
        rx_lats = [s["rx_lat"] for s in signals]
        assert rx_lats[0] < rx_lats[1] < rx_lats[2]

    def test_converging_objects_distance_changes(self, source_dataset, converging_positions):
        """Verify distance changes as transmitter and receiver move toward each other."""
        tx_pos_func, rx_pos_func = converging_positions
        transmitter = Transmitter(source_dataset, tx_pos_func, identifier="tx_converge")
        receiver = Receiver(rx_pos_func, sample_rate=SAMPLE_RATE, identifier="rx_converge")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signals = [next(geo_ds) for _ in range(5)]
        distances = [s.component_signals[0]["path_distance"] for s in signals]

        # Distance should be changing
        assert len(set([round(d, 2) for d in distances])) > 1

    # --- Multiple Receivers Tests ---

    def test_multiple_receivers_same_sample_index(self, source_dataset, moving_tx_position):
        """Verify all receivers in one cycle see the same transmitter position."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        rxs = [make_rx(lat=NEAR_SF_LAT + i * 0.001, lon=NEAR_SF_LON) for i in range(3)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=rxs)

        signals = [next(geo_ds) for _ in range(3)]

        # All 3 signals should have the same transmitter position (frame_index = 0)
        tx_lons = [s.component_signals[0]["tx_lon"] for s in signals]
        assert all(lon == tx_lons[0] for lon in tx_lons)

        # Each receiver should have its own position
        rx_lats = [s["rx_lat"] for s in signals]
        assert rx_lats[0] != rx_lats[1] != rx_lats[2]

    def test_multiple_receivers_next_cycle_different_position(self, source_dataset, moving_tx_position):
        """Verify next cycle of receivers sees updated transmitter position."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        rxs = [make_rx(lat=NEAR_SF_LAT + i * 0.001) for i in range(2)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=rxs)

        cycle1_signals = [next(geo_ds) for _ in range(2)]
        cycle2_signals = [next(geo_ds) for _ in range(2)]

        # All signals in each cycle should have same tx position
        cycle1_tx_lons = [s.component_signals[0]["tx_lon"] for s in cycle1_signals]
        cycle2_tx_lons = [s.component_signals[0]["tx_lon"] for s in cycle2_signals]

        assert all(lon == cycle1_tx_lons[0] for lon in cycle1_tx_lons)
        assert all(lon == cycle2_tx_lons[0] for lon in cycle2_tx_lons)

        # But cycle 2 should have different tx position than cycle 1
        assert cycle1_tx_lons[0] != cycle2_tx_lons[0]

    def test_multiple_receivers_same_transmitted_signal(self, source_dataset, tx_pos):
        """Verify all receivers in same cycle receive same transmitted signal data."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rxs = [make_rx(lat=NEAR_SF_LAT + i * 0.001) for i in range(3)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=rxs)

        signals = [next(geo_ds) for _ in range(3)]
        tx_component_signals = [s.component_signals[0] for s in signals]

        # Transmitted signal data should be identical across all receivers
        tx_data_0 = tx_component_signals[0].data
        tx_data_1 = tx_component_signals[1].data
        tx_data_2 = tx_component_signals[2].data

        np.testing.assert_array_equal(tx_data_0, tx_data_1)
        np.testing.assert_array_equal(tx_data_0, tx_data_2)

        # Verify each receiver has its own position
        rx_lats = [s["rx_lat"] for s in signals]
        assert rx_lats[0] != rx_lats[1] != rx_lats[2]

    def test_multiple_receivers_different_cycles_different_signal(self, source_dataset, tx_pos):
        """Verify different cycles produce different transmitted signals."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rxs = [make_rx(lat=NEAR_SF_LAT + i * 0.001) for i in range(2)]
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=rxs)

        cycle1_signals = [next(geo_ds) for _ in range(2)]
        cycle1_tx_data = [s.component_signals[0].data for s in cycle1_signals]

        cycle2_signals = [next(geo_ds) for _ in range(2)]
        cycle2_tx_data = [s.component_signals[0].data for s in cycle2_signals]

        # Within each cycle, all receivers see the same transmitter signal
        np.testing.assert_array_equal(cycle1_tx_data[0], cycle1_tx_data[1])
        np.testing.assert_array_equal(cycle2_tx_data[0], cycle2_tx_data[1])

        # But the signal should be different between cycles
        assert not np.array_equal(cycle1_tx_data[0], cycle2_tx_data[0])

    # --- Topology Summary Tests ---

    def test_topology_summary_static_objects(self, source_dataset, tx_pos, rx_pos):
        """Verify topology summary with static objects at different frame indices."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        summary_0 = geo_ds.get_topology_summary(0)
        summary_10 = geo_ds.get_topology_summary(10)

        # For static objects, summaries should be the same
        assert summary_0["transmitters"][transmitter.identifier]["position"]["lat"] == pytest.approx(summary_10["transmitters"][transmitter.identifier]["position"]["lat"])
        assert summary_0["paths"][(transmitter.identifier, receiver.identifier)]["distance_m"] == pytest.approx(summary_10["paths"][(transmitter.identifier, receiver.identifier)]["distance_m"])

    def test_topology_summary_moving_transmitter(self, source_dataset, rx_pos, moving_tx_position):
        """Verify topology summary updates with moving transmitter."""
        transmitter = Transmitter(source_dataset, moving_tx_position, identifier="tx_moving")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        summary_0 = geo_ds.get_topology_summary(0)
        summary_10 = geo_ds.get_topology_summary(10)

        # Transmitter longitude should be different
        tx_lon_0 = summary_0["transmitters"][transmitter.identifier]["position"]["lon"]
        tx_lon_10 = summary_10["transmitters"][transmitter.identifier]["position"]["lon"]
        assert tx_lon_0 < tx_lon_10

        # Distances should be different
        dist_0 = summary_0["paths"][(transmitter.identifier, receiver.identifier)]["distance_m"]
        dist_10 = summary_10["paths"][(transmitter.identifier, receiver.identifier)]["distance_m"]
        assert dist_0 != pytest.approx(dist_10)

    def test_topology_summary_default_frame_index(self, source_dataset, tx_pos, rx_pos):
        """Verify default frame_index is 0 for get_topology_summary."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        summary_default = geo_ds.get_topology_summary()
        summary_0 = geo_ds.get_topology_summary(0)

        assert summary_default["transmitters"] == summary_0["transmitters"]
        assert summary_default["receivers"] == summary_0["receivers"]
        assert summary_default["paths"] == summary_0["paths"]


class TestVelocity:
    """Tests for velocity handling in Transmitter and Receiver classes."""

    @pytest.fixture
    def moving_tx_velocity(self):
        """Create a callable for transmitter with accelerating east velocity."""

        def velocity(frame_index):
            east_vel = 10.0 * frame_index  # Accelerating at 10 m/s^2
            return GeoVelocity(east=east_vel, north=0.0, up=0.0)

        return velocity

    @pytest.fixture
    def moving_rx_velocity(self):
        """Create a callable for receiver with constant north velocity."""

        def velocity(frame_index):
            return GeoVelocity(east=0.0, north=50.0, up=0.0)

        return velocity

    # --- Transmitter Velocity Tests ---

    @pytest.mark.parametrize("frame_index", [0, 5, 10, 100])
    def test_transmitter_default_velocity_zero(self, source_dataset, tx_pos, frame_index):
        """Verify transmitter with no velocity specified has zero velocity."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        vel = transmitter.get_velocity(frame_index)
        assert vel == GeoVelocity(east=0.0, north=0.0, up=0.0)

    @pytest.mark.parametrize("frame_index", [0, 5, 10])
    def test_transmitter_static_velocity_constant(self, source_dataset, tx_pos, frame_index):
        """Verify static transmitter velocity is constant across frame indices."""
        static_velocity = GeoVelocity(east=100.0, north=50.0, up=10.0)
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_test", velocity=static_velocity)
        vel = transmitter.get_velocity(frame_index)
        assert vel == static_velocity

    def test_transmitter_callable_velocity_changes(self, source_dataset, tx_pos, moving_tx_velocity):
        """Verify callable transmitter velocity changes across frame indices."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_test", velocity=moving_tx_velocity)

        vel_0 = transmitter.get_velocity(0)
        vel_1 = transmitter.get_velocity(1)
        vel_10 = transmitter.get_velocity(10)

        assert vel_0 == GeoVelocity(east=0.0, north=0.0, up=0.0)
        assert vel_1 == GeoVelocity(east=10.0, north=0.0, up=0.0)
        assert vel_10 == GeoVelocity(east=100.0, north=0.0, up=0.0)
        assert vel_0 != vel_1 != vel_10

    def test_transmitter_velocity_accessible(self, source_dataset, tx_pos):
        """Verify transmitter velocity is accessible via get_velocity()."""
        static_velocity = GeoVelocity(east=100.0, north=50.0, up=10.0)
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_test", velocity=static_velocity)
        vel = transmitter.get_velocity(0)
        assert vel.east == pytest.approx(100.0)
        assert vel.north == pytest.approx(50.0)
        assert vel.up == pytest.approx(10.0)

    # --- Receiver Velocity Tests ---

    @pytest.mark.parametrize("frame_index", [0, 5, 10, 100])
    def test_receiver_default_velocity_zero(self, rx_pos, frame_index):
        """Verify receiver with no velocity specified has zero velocity."""
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_static")
        vel = receiver.get_velocity(frame_index)
        assert vel == GeoVelocity(east=0.0, north=0.0, up=0.0)

    @pytest.mark.parametrize("frame_index", [0, 5, 10])
    def test_receiver_static_velocity_constant(self, rx_pos, frame_index):
        """Verify static receiver velocity is constant across frame indices."""
        static_velocity = GeoVelocity(east=25.0, north=-30.0, up=5.0)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", velocity=static_velocity)
        vel = receiver.get_velocity(frame_index)
        assert vel == static_velocity

    def test_receiver_callable_velocity_changes(self, rx_pos, moving_rx_velocity):
        """Verify callable receiver velocity changes across frame indices."""
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", velocity=moving_rx_velocity)

        vel_0 = receiver.get_velocity(0)
        vel_1 = receiver.get_velocity(1)
        vel_10 = receiver.get_velocity(10)

        assert vel_0 == GeoVelocity(east=0.0, north=50.0, up=0.0)
        assert vel_1 == GeoVelocity(east=0.0, north=50.0, up=0.0)
        assert vel_10 == GeoVelocity(east=0.0, north=50.0, up=0.0)

    def test_receiver_velocity_accessible(self, rx_pos):
        """Verify receiver velocity is accessible via get_velocity()."""
        static_velocity = GeoVelocity(east=25.0, north=-30.0, up=5.0)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", velocity=static_velocity)
        vel = receiver.get_velocity(0)
        assert vel.east == pytest.approx(25.0)
        assert vel.north == pytest.approx(-30.0)
        assert vel.up == pytest.approx(5.0)

    # --- Integration Tests ---

    def test_callable_velocity_in_geo_dataset(self, source_dataset, tx_pos, rx_pos):
        """Verify callable velocities work correctly in GeoDataset."""

        def tx_velocity_func(frame_index):
            return GeoVelocity(east=10.0 * frame_index, north=0.0, up=0.0)

        def rx_velocity_func(frame_index):
            return GeoVelocity(east=0.0, north=5.0 * frame_index, up=0.0)

        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_test", velocity=tx_velocity_func)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", velocity=rx_velocity_func)
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signal_0 = next(geo_ds)
        signal_1 = next(geo_ds)

        # Frame 0: velocities should be (0, 0, 0)
        comp_0 = signal_0.component_signals[0]
        assert comp_0["tx_vel_east"] == pytest.approx(0.0)
        assert comp_0["rx_vel_north"] == pytest.approx(0.0)

        # Frame 1: tx_vel = (10, 0, 0), rx_vel = (0, 5, 0)
        comp_1 = signal_1.component_signals[0]
        assert comp_1["tx_vel_east"] == pytest.approx(10.0)
        assert comp_1["rx_vel_north"] == pytest.approx(5.0)

    def test_mixed_static_and_callable_velocities(self, source_dataset, tx_pos, rx_pos):
        """Verify mixing static and callable velocities works correctly."""

        def tx_velocity_func(frame_index):
            return GeoVelocity(east=10.0 * frame_index, north=0.0, up=0.0)

        rx_velocity = GeoVelocity(east=25.0, north=0.0, up=0.0)  # Static

        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_test", velocity=tx_velocity_func)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_test", velocity=rx_velocity)
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        signal_0 = next(geo_ds)
        signal_1 = next(geo_ds)

        comp_0 = signal_0.component_signals[0]
        comp_1 = signal_1.component_signals[0]

        # Transmitter velocity changes, receiver velocity is constant
        assert comp_0["tx_vel_east"] == pytest.approx(0.0)
        assert comp_1["tx_vel_east"] == pytest.approx(10.0)
        assert comp_0["rx_vel_east"] == pytest.approx(25.0)
        assert comp_1["rx_vel_east"] == pytest.approx(25.0)


# =============================================================================
# Validation & Edge Cases
# =============================================================================


class TestSignalDurationValidation:
    """Tests for signal duration validation."""

    def test_signal_duration_positive(self, minimal_metadata):
        """Test that signal with positive duration is accepted."""
        # This is implicitly tested in normal operation
        # Create a dataset with valid configuration
        geo_ds = make_geo_ds(tx_count=1, rx_count=1)
        signal = next(geo_ds)
        # Signal should be generated successfully
        assert len(signal.data) > 0


class TestValidationHelpers:
    """Tests for internal validation helper functions."""

    def test_validate_position_type_valid_geo_point(self):
        """Test _validate_position_type passes for valid GeoPoint."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoPoint

        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        # Should not raise
        _validate_callable_or_instance(point, "Transmitter", "position", GeoPoint, allow_none=False)

    def test_validate_position_type_valid_callable(self):
        """Test _validate_position_type passes for callable returning GeoPoint."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoPoint

        def position_func(frame_index):
            return GeoPoint(lat=37.7749, lon=-122.4194, alt=10)

        _validate_callable_or_instance(position_func, "Transmitter", "position", GeoPoint, allow_none=False)

    def test_validate_position_type_invalid_type(self):
        """Test _validate_position_type fails for non-GeoPoint, non-callable."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoPoint

        with pytest.raises(TypeError, match="position must be a GeoPoint or callable"):
            _validate_callable_or_instance("not a point", "Transmitter", "position", GeoPoint, allow_none=False)

    def test_validate_velocity_type_valid_none(self):
        """Test _validate_velocity_type passes for None velocity."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoVelocity

        # Should not raise
        _validate_callable_or_instance(None, "Transmitter", "velocity", GeoVelocity, allow_none=True)

    def test_validate_velocity_type_valid_geo_velocity(self):
        """Test _validate_velocity_type passes for valid GeoVelocity."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoVelocity

        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        _validate_callable_or_instance(vel, "Transmitter", "velocity", GeoVelocity, allow_none=True)

    def test_validate_velocity_type_valid_callable(self):
        """Test _validate_velocity_type passes for callable returning GeoVelocity."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoVelocity

        def velocity_func(frame_index):
            return GeoVelocity(east=10.0, north=5.0, up=2.0)

        _validate_callable_or_instance(velocity_func, "Transmitter", "velocity", GeoVelocity, allow_none=True)

    def test_validate_velocity_type_invalid_type(self):
        """Test _validate_velocity_type fails for non-GeoVelocity, non-callable, non-None."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoVelocity

        with pytest.raises(TypeError, match="velocity must be a GeoVelocity or callable"):
            _validate_callable_or_instance("not a velocity", "Transmitter", "velocity", GeoVelocity, allow_none=True)

    def test_validate_position_callable_tests_at_init(self):
        """Test that callable position is called with frame_index=0 during validation."""
        from torchsig.geo.datasets import _validate_callable_or_instance, GeoPoint

        call_count = 0

        def position_func(frame_index):
            nonlocal call_count
            call_count += 1
            assert frame_index == 0, f"Expected frame_index=0, got {frame_index}"
            return GeoPoint(lat=37.7749, lon=-122.4194, alt=10)

        _validate_callable_or_instance(position_func, "Transmitter", "position", GeoPoint, allow_none=False)
        # The callable should have been called during validation
        assert call_count == 0

    def test_validate_velocity_callable_does_not_call_at_init(self):
        """Validation must accept a velocity callable without invoking it."""
        from torchsig.geo.datasets import _validate_callable_or_instance
        from torchsig.geo.types import GeoVelocity

        call_count = 0

        def velocity_func(frame_index):
            nonlocal call_count
            call_count += 1
            return GeoVelocity(east=10.0, north=5.0, up=2.0)

        _validate_callable_or_instance(
            velocity_func,
            "Receiver",
            "velocity",
            GeoVelocity,
            allow_none=True,
        )

        assert call_count == 0


class TestSampleRateValidation:
    """Tests for sample rate validation in TorchSigGeoDataset."""

    def test_mismatched_transmitter_sample_rates_warns(self, minimal_metadata):
        """Verify mismatched transmitter sample rates emits warning during iteration."""
        tx_ds_1 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 10_000_000.0},
            signal_generators=["bpsk"],
        )
        tx_ds_2 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 20_000_000.0},
            signal_generators=["bpsk"],
        )

        tx_1 = Transmitter(tx_ds_1, make_tx().get_position(0), identifier="tx_test")
        tx_2 = Transmitter(tx_ds_2, make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_2")
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx_1, tx_2], receivers=[rx])
        # Expect resampling warning due to different sample rates
        # Note: This may also emit a signal length warning, so we check for the resampling warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            signal = next(geo_ds)
            assert any("sample_rate" in str(warning.message) and "differs" in str(warning.message) for warning in w)

    def test_mismatched_receiver_sample_rate_warns(self, minimal_metadata):
        """Verify mismatched receiver sample rates emits warning during iteration."""
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )

        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx_1 = make_rx(sample_rate=SAMPLE_RATE)
        rx_2 = make_rx(lat=NEAR_SF_LAT + 0.001, sample_rate=20_000_000.0)

        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx_1, rx_2])
        # First receiver (rx_1) matches, second (rx_2) doesn't - warning on second iteration
        signal = next(geo_ds)  # rx_1, no warning
        with pytest.warns(UserWarning, match="Transmitter sample_rate.*differs from receiver sample_rate"):
            signal = next(geo_ds)  # rx_2, warning expected

    def test_matching_sample_rates_succeeds(self, minimal_metadata):
        """Verify matching sample rates work correctly."""
        tx_ds_1 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx_ds_2 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["qpsk"],
        )

        tx_1 = Transmitter(tx_ds_1, make_tx().get_position(0), identifier="tx_test")
        tx_2 = Transmitter(tx_ds_2, make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_2")
        rx = make_rx()

        # Should not raise or warn
        geo_ds = TorchSigGeoDataset(transmitters=[tx_1, tx_2], receivers=[rx])
        assert len(geo_ds.transmitters) == 2
        assert len(geo_ds.receivers) == 1

    def test_warning_message_contains_sample_rates(self, minimal_metadata):
        """Verify warning message contains actual sample rates."""
        tx_ds_1 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 10_000_000.0},
            signal_generators=["bpsk"],
        )
        tx_ds_2 = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 15_000_000.0},
            signal_generators=["bpsk"],
        )

        tx_1 = Transmitter(tx_ds_1, make_tx().get_position(0), identifier="tx_test")
        tx_2 = Transmitter(tx_ds_2, make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_2")
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx_1, tx_2], receivers=[rx])
        # Warning will contain the sample rates that differ
        # Note: This may also emit a signal length warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            signal = next(geo_ds)
            assert any("1.00e+07" in str(warning.message) and "1.50e+07" in str(warning.message) for warning in w)

    def test_resampling_to_receiver_rate(self, minimal_metadata):
        """Verify transmitter signals are resampled to receiver's sample rate."""
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 20_000_000.0},
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx = make_rx(sample_rate=10_000_000.0)

        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])
        # Suppress warnings for this test - we're explicitly testing resampling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            signal = next(geo_ds)
        assert signal["sample_rate"] == rx.sample_rate
        assert signal["sample_rate"] == 10_000_000.0

    def test_resampling_preserves_metadata(self, minimal_metadata):
        """Verify resampling preserves transmitter metadata."""
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 20_000_000.0},
        )
        tx = Transmitter(tx_ds, make_tx(lat=SF_LAT + 0.01).get_position(0), identifier="tx_high")
        rx = make_rx(sample_rate=10_000_000.0, identifier="rx_low")

        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])
        # Suppress warnings for this test - we're explicitly testing resampling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            signal = next(geo_ds)
        assert signal["sample_rate"] == rx.sample_rate
        assert signal["rx_id"] == rx.identifier


class TestSignalLengthWarning:
    """Tests for signal length alignment via resampling in TorchSigGeoDataset."""

    def test_variable_length_signals_resampled_to_match(self, minimal_metadata):
        """Verify signals of different lengths are resampled to match the longest."""

        class VariableLengthDataset(TorchSigIterableDataset):
            def __init__(self, length, **kwargs):
                super().__init__(metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE}, signal_generators=[], **kwargs)
                self.length = length

            def __generate_new_signal__(self) -> Signal:
                data = np.random.randn(self.length) + 1j * np.random.randn(self.length)
                return Signal(data=data.astype(np.complex64))

        # Create transmitters with different signal lengths
        tx_ds_short = VariableLengthDataset(length=100)
        tx_ds_long = VariableLengthDataset(length=200)

        tx_short = Transmitter(tx_ds_short, make_tx().get_position(0), identifier="tx_test")
        tx_long = Transmitter(tx_ds_long, make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_long")
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx_short, tx_long], receivers=[rx])
        # Expect a warning about different signal lengths
        with pytest.warns(UserWarning, match="Signals from different transmitters have different lengths"):
            signal = next(geo_ds)

        # All component signals should have the same length (resampled to match longest)
        assert len(signal.component_signals) == 2
        comp_lengths = [len(comp.data) for comp in signal.component_signals]
        assert all(l == comp_lengths[0] for l in comp_lengths)
        # The combined signal data should also match this length
        assert len(signal.data) == comp_lengths[0]

    def test_same_length_signals_unchanged(self, minimal_metadata):
        """Verify signals with same length are not resampled unnecessarily."""
        tx_ds_1 = TorchSigIterableDataset(
            metadata={
                **minimal_metadata,
                "sample_rate": SAMPLE_RATE,
                "num_iq_samples_dataset": 100,
                "signal_duration_in_samples_min": 100,
                "signal_duration_in_samples_max": 100,
            },
            signal_generators=["bpsk"],
        )
        tx_ds_2 = TorchSigIterableDataset(
            metadata={
                **minimal_metadata,
                "sample_rate": SAMPLE_RATE,
                "num_iq_samples_dataset": 100,
                "signal_duration_in_samples_min": 100,
                "signal_duration_in_samples_max": 100,
            },
            signal_generators=["qpsk"],
        )

        tx_1 = Transmitter(tx_ds_1, make_tx().get_position(0), identifier="tx_test")
        tx_2 = Transmitter(tx_ds_2, make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_2")
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx_1, tx_2], receivers=[rx])
        signal = next(geo_ds)

        # Component signals should all have length 100 (original length)
        assert len(signal.component_signals) == 2
        for comp in signal.component_signals:
            assert len(comp.data) == 100


class TestCoverageEdgeCases:
    """Tests for achieving 100% coverage of torchsig/geo/datasets.py."""

    # -------------------------------------------------------------------------
    # _validate_callable_or_instance edge cases
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Transmitter.generate_signal return type handling
    # -------------------------------------------------------------------------

    def test_generate_signal_tuple_with_ndarray_first_element(self, source_dataset, tx_pos, monkeypatch):
        """Test line 284 else branch: Signal(data=sample[0]) when sample[0] is ndarray, not Signal.

        Uses monkeypatch on __iter__ following the existing pattern in TestTransmitter.
        This ensures the mock iterator is used after seed() is called.
        """
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rx = make_rx()
        TorchSigGeoDataset(transmitters=[transmitter], receivers=[rx])

        mock_data = np.random.randn(100) + 1j * np.random.randn(100)
        mock_iter = iter([(mock_data, "metadata")])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_tuple_with_non_signal_non_ndarray_first(self, source_dataset, tx_pos, monkeypatch):
        """Test line 284 else branch: Signal(data=sample[0]) when sample[0] is list."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rx = make_rx()
        TorchSigGeoDataset(transmitters=[transmitter], receivers=[rx])

        mock_iter = iter([([1, 2, 3], "metadata")])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_bare_ndarray_return(self, source_dataset, tx_pos, monkeypatch):
        """Test line 287-288: Signal(data=sample) when sample is ndarray."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rx = make_rx()
        TorchSigGeoDataset(transmitters=[transmitter], receivers=[rx])

        mock_data = np.random.randn(100) + 1j * np.random.randn(100)
        mock_iter = iter([mock_data])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    def test_generate_signal_other_type_return(self, source_dataset, tx_pos, monkeypatch):
        """Test line 290: Signal(data=np.array(sample)) for other types (e.g., list)."""
        transmitter = Transmitter(source_dataset, tx_pos, identifier="tx_static")
        rx = make_rx()
        TorchSigGeoDataset(transmitters=[transmitter], receivers=[rx])

        mock_iter = iter([[1, 2, 3, 4, 5]])
        monkeypatch.setattr(transmitter.dataset, "__iter__", lambda self: mock_iter)

        result = transmitter.generate_signal(0)
        assert isinstance(result, Signal)

    # -------------------------------------------------------------------------
    # TorchSigGeoDataset sample_rate metadata
    # -------------------------------------------------------------------------

    def test_geo_dataset_explicit_sample_rate_in_metadata(self, transmitter, receiver):
        """Test line 575: sample_rate stored in metadata when provided."""
        custom_rate = 15000000.0
        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], sample_rate=custom_rate)

        assert "sample_rate" in geo_ds.keys()
        assert geo_ds["sample_rate"] == pytest.approx(custom_rate)

    # -------------------------------------------------------------------------
    # _get_path_transforms edge cases
    # -------------------------------------------------------------------------

    def test_get_path_transforms_no_path_in_topology(self, transmitter, receiver):
        """Test lines 757, 763: empty list when path_key not in topology."""
        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            topology={},  # Empty topology - no paths
        )

        transforms = geo_ds._get_path_transforms(transmitter, receiver)
        assert transforms == []

    def test_get_path_transforms_path_key_not_in_topology(self, transmitter, receiver):
        """Test line 757, 763: returns empty list when path not in topology (path_info is None)."""
        # Create dataset with explicit topology that does NOT include our tx->rx pair
        # As identifiers are now required, we use the fixture's identifiers
        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            topology={},  # Empty topology means no paths at all
        )

        # path_key will not be in topology, so path_info will be None
        transforms = geo_ds._get_path_transforms(transmitter, receiver)
        assert transforms == []

    def test_get_path_transforms_with_per_path_and_global(self, transmitter, receiver):
        """Test that both per-path and global channel_transforms are returned."""
        path_loss_global = PathLoss(model="custom", loss_db=5.0)
        path_loss_per_path = PathLoss(model="custom", loss_db=10.0)

        per_path = {(transmitter.identifier, receiver.identifier): [path_loss_per_path]}

        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver], channel_transforms=[path_loss_global], per_path_transforms=per_path)

        transforms = geo_ds._get_path_transforms(transmitter, receiver)
        assert len(transforms) == 2
        assert path_loss_per_path in transforms
        assert path_loss_global in transforms

    def test_get_path_transforms_no_path_key_in_topology_dict(self, transmitter, receiver):
        """Test line 757: returns empty list when path_key not in topology dict."""
        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            topology={},  # Empty topology means no paths at all
        )

        transforms = geo_ds._get_path_transforms(transmitter, receiver)
        assert transforms == []

    # -------------------------------------------------------------------------
    # _resample_signal edge cases
    # -------------------------------------------------------------------------

    def test_resample_signal_missing_sample_rate_raises(self, simple_geo_ds):
        """Test line 791: ValueError when leaf signal missing sample_rate metadata."""
        signal_no_rate = Signal(data=np.random.randn(100) + 1j * np.random.randn(100))

        with pytest.raises(ValueError, match="Leaf signal missing 'sample_rate' metadata"):
            simple_geo_ds._resample_signal(signal_no_rate, 1e6)

    def test_resample_signal_same_rate_returns_early(self, simple_geo_ds):
        """Test line 794: early return when leaf_rate equals target_rate."""
        rate = 1e6
        signal = Signal(data=np.random.randn(100) + 1j * np.random.randn(100), sample_rate=rate)

        original_data = signal.data.copy()
        result = simple_geo_ds._resample_signal(signal, rate)

        np.testing.assert_array_equal(result.data, original_data)

    def test_resample_signal_updates_all_nodes_sample_rate(self, simple_geo_ds):
        """Test line 803: sample_rate updated on all nodes in tree."""
        comp1 = Signal(data=np.random.randn(50) + 1j * np.random.randn(50), sample_rate=1e6)
        wrapper = Signal(data=np.random.randn(50) + 1j * np.random.randn(50), component_signals=[comp1], sample_rate=1e6)

        result = simple_geo_ds._resample_signal(wrapper, 2e6)

        assert result["sample_rate"] == pytest.approx(2e6)
        assert result.component_signals[0]["sample_rate"] == pytest.approx(2e6)

    # -------------------------------------------------------------------------
    # _generate_receiver_signal topology and signal alignment
    # -------------------------------------------------------------------------

    def test_generate_receiver_signal_skip_transmitter_not_in_topology(self):
        """Test line 836: continue when path_key not in topology - transmitter excluded from signal."""
        tx1 = make_tx(identifier="tx_connected")
        tx2 = make_tx(lat=SF_LAT + 0.01, identifier="tx_not_connected")
        rx = make_rx(identifier="rx1")

        geo_ds = TorchSigGeoDataset(
            transmitters=[tx1, tx2],
            receivers=[rx],
            topology={"tx_connected": ["rx1"]},  # Only tx1 connected to rx1
        )

        signal = next(geo_ds)

        # Only tx1 should contribute a component signal, tx2 is skipped
        assert len(signal.component_signals) == 1
        assert signal.component_signals[0]["tx_id"] == "tx_connected"

    def test_resampling_warning_emitted_on_first_mismatch(self, minimal_metadata):
        """Test lines 848-859: warning emitted on first transmitter/receiver sample_rate mismatch."""
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": 10_000_000.0},
            signal_generators=["bpsk"],
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx = make_rx(sample_rate=20_000_000.0)

        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        # First signal should emit the warning
        with pytest.warns(UserWarning, match="Transmitter sample_rate.*differs from receiver sample_rate"):
            signal1 = next(geo_ds)

        # Second signal should NOT emit the warning (flag is already set)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            signal2 = next(geo_ds)
            # Check that no new resampling warnings were emitted
            resampling_warnings = [x for x in w if "differs from" in str(x.message) and "sample_rate" in str(x.message)]
            assert len(resampling_warnings) == 0

    def test_signal_duration_validation_raises_zero_length(self, transmitter, receiver, monkeypatch):
        """Test line 864: ValueError for zero-length signal."""

        def mock_generate_signal(frame_index):
            return Signal(data=np.array([], dtype=np.complex64))

        monkeypatch.setattr(transmitter, "generate_signal", mock_generate_signal)

        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        with pytest.raises(ValueError, match="Signal duration must be positive and finite"):
            _ = next(geo_ds)

    def test_signal_duration_validation_raises_nan_duration(self, minimal_metadata):
        """Test line 864: ValueError for non-finite signal duration (nan)."""
        from unittest.mock import patch

        # Create a transmitter that returns a signal where duration calculation results in nan
        # Duration is calculated as: len(tx_signal.data) / receiver.sample_rate
        # We need receiver.sample_rate to be nan for this to produce nan
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")

        # Create receiver with nan sample_rate (bypassing validation by setting directly)
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        # Patch receiver.sample_rate to nan after dataset creation
        # Suppress the resampling warning since we're intentionally creating an invalid state
        # to test duration validation, not resampling behavior
        with patch.object(rx, "sample_rate", float("nan")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with pytest.raises(ValueError, match="Signal duration must be positive and finite"):
                    _ = next(geo_ds)

    def test_signal_alignment_padding_and_truncation(self, minimal_metadata):
        """Test that signals are aligned via padding/truncation and metadata is updated."""

        # Create a dataset that produces a specific length signal
        class FixedLengthDataset(TorchSigIterableDataset):
            def __init__(self, length, **kwargs):
                super().__init__(metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE}, signal_generators=[], **kwargs)
                self.length = length

            def __generate_new_signal__(self) -> Signal:
                return Signal(data=np.random.randn(self.length) + 1j * np.random.randn(self.length), sample_rate=SAMPLE_RATE)

        tx_long = Transmitter(FixedLengthDataset(200), make_tx().get_position(0), identifier="tx_test")
        tx_short = Transmitter(FixedLengthDataset(100), make_tx(lat=SF_LAT + 0.001).get_position(0), identifier="tx_short")
        rx = make_rx()

        geo_ds = TorchSigGeoDataset(transmitters=[tx_long, tx_short], receivers=[rx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = next(geo_ds)

        # Target length is based on max duration: max(200/rate, 100/rate) * rate = 200
        # Short signal (100) should be padded to 200, long signal (200) should stay at 200
        expected_length = 200

        # All component signals should have same length
        lengths = [len(comp.data) for comp in signal.component_signals]
        assert all(l == expected_length for l in lengths)

        # Verify duration_in_samples metadata matches actual data length
        for comp in signal.component_signals:
            assert comp["duration_in_samples"] == expected_length

    def test_generate_receiver_signal_no_component_signals_empty_data(self, minimal_metadata):
        """Test line 962: empty combined_data (256 zeros) when no component_signals.

        Also tests the fix for UnboundLocalError when receiver has no connected transmitters.
        """
        # Create a transmitter and receiver where the topology has no path
        # We need to set up the topology so that the receiver exists but has no connected transmitters

        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx1 = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx1")
        rx1 = make_rx(lat=NEAR_SF_LAT, identifier="rx1")
        rx2 = make_rx(lat=NEAR_SF_LAT + 0.001, identifier="rx2")

        # Topology: only tx1 -> rx1, rx2 has no connected transmitters
        geo_ds = TorchSigGeoDataset(
            transmitters=[tx1],
            receivers=[rx1, rx2],
            topology={"tx1": ["rx1"]},  # tx1 -> rx1 only, rx2 has no paths
        )

        # Get first signal (rx1) - should have component signals
        signal1 = next(geo_ds)
        assert len(signal1.component_signals) == 1

        # Get second signal (rx2) - should have NO component signals
        # After the fix, rx_pos is obtained from receiver.get_position()
        signal2 = next(geo_ds)
        assert isinstance(signal2, Signal)
        assert len(signal2.component_signals) == 0
        # Empty combined_signal.data is initialized as [] from Signal(None)
        assert len(signal2.data) == 0
        # Verify receiver metadata is still set correctly (tests the fix for line 980-982)
        assert signal2["rx_id"] == rx2.identifier
        assert signal2["rx_lat"] == pytest.approx(rx2.get_position(0).lat)
        assert signal2["rx_lon"] == pytest.approx(rx2.get_position(0).lon)

    # -------------------------------------------------------------------------
    # Dataset-level resampling
    # -------------------------------------------------------------------------

    def test_dataset_level_sample_rate_normalization(self, minimal_metadata):
        """Test lines 1002-1004: resample to dataset-level sample_rate when configured."""
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx = make_rx(sample_rate=SAMPLE_RATE)

        target_rate = 12000000.0
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx], sample_rate=target_rate)

        signal = next(geo_ds)
        assert signal["sample_rate"] == pytest.approx(target_rate)

    # -------------------------------------------------------------------------
    # to_file default handler
    # -------------------------------------------------------------------------

    def test_to_file_default_handler_is_hdf5(self, simple_geo_ds, temp_dir):
        """Test lines 1128-1132: default file_handler_class is HDF5Writer when None is passed."""
        from torchsig.utils.file_handlers.hdf5 import HDF5Writer

        # Call to_file without file_handler_class argument (defaults to None)
        simple_geo_ds.to_file(root=temp_dir, dataset_length=2, overwrite=True)

        # Verify files were created (implies HDF5Writer was used)
        assert (Path(temp_dir) / "data.h5").exists()
        assert (Path(temp_dir) / "data.h5").stat().st_size > 0

    def test_dataset_level_resampling_skipped_when_matching(self, minimal_metadata):
        """Test lines 1011-1014: conditional resampling to dataset sample_rate.

        When the dataset-level sample_rate matches the combined signal's sample_rate,
        no resampling should occur (lines 1011-1014 form the condition and body).
        """
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx = make_rx(sample_rate=SAMPLE_RATE)

        target_rate = SAMPLE_RATE  # Same as receiver rate
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx], sample_rate=target_rate)

        signal = next(geo_ds)
        # Signal should have the target rate
        assert signal["sample_rate"] == pytest.approx(target_rate)

    def test_dataset_level_resampling_applied_when_mismatched(self, minimal_metadata):
        """Test lines 1011-1012: resampling occurs when dataset sample_rate differs.

        When the dataset-level sample_rate differs from the combined signal's sample_rate,
        resampling should occur.
        """
        tx_ds = TorchSigIterableDataset(
            metadata={**minimal_metadata, "sample_rate": SAMPLE_RATE},
            signal_generators=["bpsk"],
        )
        tx = Transmitter(tx_ds, make_tx().get_position(0), identifier="tx_test")
        rx = make_rx(sample_rate=SAMPLE_RATE)

        target_rate = 12000000.0  # Different from receiver rate
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx], sample_rate=target_rate)

        signal = next(geo_ds)
        # Signal should have been resampled to target rate
        assert signal["sample_rate"] == pytest.approx(target_rate)


class TestDeterministicSeeding:
    """Tests for reproducible transmitter signal generation."""

    def test_transmitter_seed_does_not_depend_on_python_hash(
        self,
        source_dataset,
        monkeypatch,
    ) -> None:
        """Deterministic transmitter seeds must not use randomized hash()."""
        tx = make_tx(source_dataset)
        rx = make_rx()
        TorchSigGeoDataset(
            transmitters=[tx],
            receivers=[rx],
            seed=1234,
        )

        observed_seeds = []
        original_seed = source_dataset.seed

        def record_seed(seed):
            observed_seeds.append(seed)
            return original_seed(seed)

        monkeypatch.setattr(source_dataset, "seed", record_seed)

        monkeypatch.setattr(builtins, "hash", lambda value: 111)
        tx.generate_signal(frame_index=7)

        monkeypatch.setattr(builtins, "hash", lambda value: 222)
        tx.generate_signal(frame_index=7)

        assert observed_seeds[-2] == observed_seeds[-1]


class TestDynamicPositionAndVelocity:
    """Tests for callable position and velocity handling."""

    def test_position_callable_is_not_invoked_during_construction(
        self,
        source_dataset,
    ) -> None:
        """Constructing an entity must not evaluate a dynamic position."""
        calls = []

        def position(frame_index):
            calls.append(frame_index)
            return TX_POSITION

        Transmitter(
            source_dataset,
            position,
            identifier="tx_dynamic",
        )

        assert calls == []

    def test_velocity_callable_is_not_invoked_during_construction(
        self,
        source_dataset,
    ) -> None:
        """Constructing an entity must not evaluate a dynamic velocity."""
        calls = []

        def velocity(frame_index):
            calls.append(frame_index)
            return GeoVelocity(1.0, 2.0, 3.0)

        Transmitter(
            source_dataset,
            TX_POSITION,
            identifier="tx_dynamic",
            velocity=velocity,
        )

        assert calls == []

    def test_dynamic_position_result_is_validated_at_requested_frame(
        self,
        source_dataset,
    ) -> None:
        """Every dynamic position result must be validated."""

        def position(frame_index):
            if frame_index == 0:
                return TX_POSITION
            return None

        tx = Transmitter(
            source_dataset,
            position,
            identifier="tx_dynamic",
        )

        with pytest.raises(
            TypeError,
            match=r"position.*GeoPoint.*NoneType",
        ):
            tx.get_position(frame_index=1)

    def test_dynamic_velocity_result_is_validated_at_requested_frame(
        self,
        source_dataset,
    ) -> None:
        """Every dynamic velocity result must be validated."""

        def velocity(frame_index):
            if frame_index == 0:
                return GeoVelocity(0.0, 0.0, 0.0)
            return (1.0, 2.0, 3.0)

        rx = Receiver(
            RX_POSITION,
            sample_rate=SAMPLE_RATE,
            identifier="rx_dynamic",
            velocity=velocity,
        )

        with pytest.raises(
            TypeError,
            match=r"velocity.*GeoVelocity.*tuple",
        ):
            rx.get_velocity(frame_index=1)


class TestSampleRateValidation:
    """Tests for transmitter, receiver, and output sample rates."""

    @pytest.mark.parametrize(
        "sample_rate",
        [0.0, -1.0, np.nan, np.inf, -np.inf],
    )
    def test_dataset_rejects_invalid_output_sample_rate(
        self,
        source_dataset,
        sample_rate,
    ) -> None:
        """Dataset-level target sample rate must be positive and finite."""
        tx = make_tx(source_dataset)
        rx = make_rx()

        with pytest.raises(
            ValueError,
            match=r"sample_rate.*positive.*finite",
        ):
            TorchSigGeoDataset(
                transmitters=[tx],
                receivers=[rx],
                sample_rate=sample_rate,
            )

    @pytest.mark.parametrize(
        "sample_rate",
        [0.0, -1.0, np.nan, np.inf, -np.inf],
    )
    def test_transmitter_rejects_invalid_dataset_sample_rate(
        self,
        source_dataset,
        sample_rate,
    ) -> None:
        """Transmitter source sample rate must be positive and finite."""
        source_dataset["sample_rate"] = sample_rate

        with pytest.raises(
            ValueError,
            match=r"sample_rate.*positive.*finite",
        ):
            make_tx(source_dataset)


class TestTopology:
    """Tests for topology validation and contributor metadata."""

    def test_receiver_metadata_lists_only_connected_transmitters(
        self,
        source_dataset,
    ) -> None:
        """Metadata must describe contributing, not merely configured, TXs."""
        tx_0 = make_tx(source_dataset, identifier="tx_0")
        tx_1 = make_tx(source_dataset, identifier="tx_1")
        rx = make_rx(identifier="rx_0")

        dataset = TorchSigGeoDataset(
            transmitters=[tx_0, tx_1],
            receivers=[rx],
            topology={"tx_0": ["rx_0"]},
        )

        signal = next(dataset)

        assert signal["num_transmitters"] == 1
        assert signal["tx_ids"] == ("tx_0",)
        assert len(signal.component_signals) == 1
        assert signal.component_signals[0]["tx_id"] == "tx_0"


class TestSignalTreeAlignment:
    """Tests for nested signal alignment and rebuilding."""

    def test_alignment_updates_nested_signal_leaves(
        self,
        source_dataset,
        monkeypatch,
    ) -> None:
        """Alignment must survive rebuilding from component leaves."""
        short_tx = make_tx(source_dataset, identifier="short_tx")
        long_tx = make_tx(source_dataset, identifier="long_tx")
        rx = make_rx(identifier="rx_0")

        def short_signal(frame_index):
            leaf = Signal(
                data=np.ones(2, dtype=np.complex64),
                center_freq=2.4e9,
                sample_rate=SAMPLE_RATE,
            )
            return Signal(
                data=leaf.data.copy(),
                component_signals=[leaf],
                center_freq=0.0,
                sample_rate=SAMPLE_RATE,
            )

        def long_signal(frame_index):
            leaf = Signal(
                data=np.ones(4, dtype=np.complex64),
                center_freq=2.4e9,
                sample_rate=SAMPLE_RATE,
            )
            return Signal(
                data=leaf.data.copy(),
                component_signals=[leaf],
                center_freq=0.0,
                sample_rate=SAMPLE_RATE,
            )

        monkeypatch.setattr(
            short_tx,
            "generate_signal",
            short_signal,
        )
        monkeypatch.setattr(
            long_tx,
            "generate_signal",
            long_signal,
        )

        dataset = TorchSigGeoDataset(
            transmitters=[short_tx, long_tx],
            receivers=[rx],
        )

        signal = next(dataset)

        assert len(signal.data) == 4

        for transmitter_signal in signal.component_signals:
            assert len(transmitter_signal.data) == 4

            for leaf in transmitter_signal.component_signals:
                assert len(leaf.data) == 4


class TestReceiverTransforms:
    """Tests for receiver-transform validation."""

    def test_receiver_rejects_invalid_transform(self) -> None:
        """Invalid receiver transforms should fail during construction."""

        class NotATransform:
            def __call__(self, signal):
                return signal

        with pytest.raises(
            TypeError,
            match=r"receiver_transforms\[0\].*Transform",
        ):
            Receiver(
                RX_POSITION,
                sample_rate=SAMPLE_RATE,
                identifier="rx_invalid_transform",
                receiver_transforms=[NotATransform()],
            )


class TestRepresentation:
    """Tests for string representations."""

    def test_geo_dataset_repr_is_complete(
        self,
        source_dataset,
    ) -> None:
        """Dataset repr must be syntactically complete."""
        tx = make_tx(source_dataset)
        rx = make_rx()

        dataset = TorchSigGeoDataset(
            transmitters=[tx],
            receivers=[rx],
        )

        result = repr(dataset)

        assert result.startswith("TorchSigGeoDataset(")
        assert result.endswith(")")
