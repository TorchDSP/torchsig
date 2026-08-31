"""Tests for geolocation-related transforms.

Organized into logical test classes covering:
- PathLoss: creation, custom model, free-space model, SNR updates, metadata
- PathDelay: creation, application, formula, metadata
- LineOfSight: creation, application, various scenarios
- DopplerShift: creation, radial velocity, integration
- Integration: transforms with TorchSigGeoDataset
- String representation of all transforms
- Signal tree helpers: map_signal_tree, map_signal_leaves, rebuild_signal_from_leaves
"""

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.geo.datasets import Receiver, TorchSigGeoDataset, Transmitter
from torchsig.geo.types import GeoPoint, GeoVelocity
from torchsig.geo.transforms import (
    DopplerShift,
    GeoSignalTransform,
    LineOfSight,
    PathDelay,
    PathLoss,
    align_signal_length,
    get_absolute_center_freq,
    map_signal_tree,
    map_signal_leaves,
    rebuild_signal_from_leaves,
)
from torchsig.transforms.transforms import AWGN, SignalTransform
from torchsig.geo.utils.propagation import SPEED_OF_LIGHT_M_PER_S
from torchsig.geo.utils.coordinate_system import lla_to_ecef
from torchsig.signals.signal_types import Signal

from .conftest import (
    CENTER_FREQ,
    PATH_DISTANCE,
    NEAR_SF_ALT,
    NEAR_SF_LAT,
    NEAR_SF_LON,
    SAMPLE_RATE,
    SF_ALT,
    SF_LAT,
    SF_LON,
    MIN_SIGNAL_CENTER_FREQ,
    SIGNAL_LENGTH,
    compute_fspl,
    make_rx,
    make_tx,
)


# =============================================================================
# PathLoss Creation Tests
# =============================================================================


class TestPathLossCreation:
    """Tests for PathLoss transform creation and configuration."""

    def test_create_default(self):
        """Verify creating PathLoss with default parameters."""
        transform = PathLoss(model="free_space")
        assert transform.model == "free_space"
        assert transform.loss_db is None

    def test_create_custom_with_loss_db(self):
        """Verify creating PathLoss with custom model and loss_db."""
        transform = PathLoss(model="custom", loss_db=30.0)
        assert transform.loss_db == 30.0
        assert transform.model == "custom"

    @pytest.mark.parametrize("model", ["invalid_model", "Free_Space", ""])
    def test_invalid_model(self, model):
        """Verify invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown path loss model"):
            PathLoss(model=model)

    def test_custom_model_without_loss_db(self):
        """Verify creating custom model without loss_db raises ValueError."""
        with pytest.raises(ValueError, match="PathLoss custom model requires loss_db"):
            PathLoss(model="custom")

    def test_create_free_space_with_propagation_constant(self):
        """Verify creating free-space PathLoss with custom propagation constant."""
        transform = PathLoss(model="free_space", propagation_constant=0.5)
        assert transform.propagation_constant == 0.5

    def test_create_default_propagation_constant(self):
        """Verify default propagation_constant is 1.0."""
        transform = PathLoss(model="free_space")
        assert transform.propagation_constant == 1.0

    @pytest.mark.parametrize("propagation_constant", [0.0, -1.0, -1000.0])
    def test_create_with_invalid_propagation_constant_raises(self, propagation_constant):
        """Verify creating PathLoss with invalid propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be positive"):
            PathLoss(model="free_space", propagation_constant=propagation_constant)

    @pytest.mark.parametrize("propagation_constant", [np.nan, np.inf, -np.inf])
    def test_create_with_non_finite_propagation_constant_raises(self, propagation_constant):
        """Verify creating PathLoss with non-finite propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be finite"):
            PathLoss(model="free_space", propagation_constant=propagation_constant)

    def test_repr_free_space_with_propagation_constant(self):
        """Verify PathLoss.__repr__ includes propagation_constant when != 1.0."""
        transform = PathLoss(model="free_space", propagation_constant=0.5)
        repr_str = repr(transform)
        assert "free_space" in repr_str
        assert "0.500" in repr_str
        assert "propagation_constant" in repr_str

    def test_repr_custom_with_loss_db(self):
        """Verify PathLoss.__repr__ includes loss_db for custom model."""
        transform = PathLoss(model="custom", loss_db=30.0)
        repr_str = repr(transform)
        assert "custom" in repr_str
        assert "30.0dB" in repr_str


# =============================================================================
# PathLoss Application Tests - Custom Model
# =============================================================================


class TestPathLossCustomModel:
    """Tests for PathLoss with custom model."""

    @pytest.mark.parametrize("loss_db", [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, -10.0])
    def test_apply_various_loss_values(self, loss_db):
        """Verify custom model works with various loss values including negative (amplification)."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9)
        transform = PathLoss(model="custom", loss_db=loss_db)
        result = transform(signal.copy())

        expected_attenuation = 10 ** (-loss_db / 20)
        np.testing.assert_allclose(result.data, signal.data * expected_attenuation, rtol=1e-5)
        assert result["path_loss_db"] == pytest.approx(loss_db)

    def test_apply_requires_loss_db(self):
        """Verify custom model without loss_db raises ValueError at instantiation."""
        with pytest.raises(ValueError, match="PathLoss custom model requires loss_db"):
            PathLoss(model="custom")


# =============================================================================
# PathLoss Application Tests - Free Space Model
# =============================================================================


class TestPathLossFreeSpaceModel:
    """Tests for PathLoss with free_space model."""

    def test_apply_free_space_model(self):
        """Verify applying free-space PathLoss with signal metadata."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            center_freq=CENTER_FREQ,
        )
        signal["path_distance"] = 1000.0

        transform = PathLoss(model="free_space")
        result = transform(signal)

        assert hasattr(result, "path_loss_db")
        assert result["path_loss_db"] > 0
        assert result["path_distance"] == pytest.approx(1000.0)

    def test_free_space_path_loss_formula(self):
        """Verify free space path loss formula correctness."""
        distance = 1000.0
        frequency = CENTER_FREQ
        expected_fspl = compute_fspl(distance, frequency)

        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space")
        result = transform(signal)

        assert result["path_loss_db"] == pytest.approx(expected_fspl, rel=1e-5)

    def test_signal_center_freq_required(self):
        """Verify PathLoss requires center_freq in signal metadata for free_space model."""
        distance = 1000.0
        frequency = CENTER_FREQ
        expected_fspl = compute_fspl(distance, frequency)

        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space")
        result = transform(signal)

        assert result["path_loss_db"] == pytest.approx(expected_fspl, rel=1e-5)

    def test_propagation_constant_scaling(self):
        """Verify propagation_constant scales the path loss correctly."""
        distance = 1000.0
        frequency = CENTER_FREQ
        propagation_constant = 2 / 3  # Fiber: n=1.5

        expected_fspl_vacuum = compute_fspl(distance, frequency)
        expected_fspl_fiber = compute_fspl(distance, frequency, propagation_constant)

        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space", propagation_constant=propagation_constant)
        result = transform(signal)

        assert result["path_loss_db"] == pytest.approx(expected_fspl_fiber, rel=1e-5)
        assert result["path_loss_db"] > expected_fspl_vacuum

    @pytest.mark.parametrize("distance", [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    def test_various_distances(self, distance):
        """Verify free-space path loss with various distances."""
        frequency = CENTER_FREQ
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space")
        result = transform(signal.copy())

        assert hasattr(result, "path_loss_db")
        assert result["path_loss_db"] > 0

    @pytest.mark.parametrize("frequency", [1e6, 1e7, 1e8, 1e9, 2.4e9, 5e9, 10e9])
    def test_various_frequencies(self, frequency):
        """Verify free-space path loss with various frequencies."""
        distance = 1000.0
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space")
        result = transform(signal)

        assert hasattr(result, "path_loss_db")
        assert result["path_loss_db"] > 0

    def test_frequency_dependence(self):
        """Verify path loss increases with frequency for fixed distance."""
        distance = 1000.0
        frequencies = [1e8, 1e9, 10e9]
        path_losses = []

        for freq in frequencies:
            signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=freq)
            signal["path_distance"] = distance
            transform = PathLoss(model="free_space")
            result = transform(signal)
            path_losses.append(result["path_loss_db"])

        # Path loss should increase monotonically with frequency
        assert path_losses == sorted(path_losses)
        assert path_losses[-1] > path_losses[0]

    def test_distance_dependence(self):
        """Verify path loss increases with distance for fixed frequency."""
        frequency = CENTER_FREQ
        distances = [100.0, 1000.0, 10000.0]
        path_losses = []

        for dist in distances:
            signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=frequency)
            signal["path_distance"] = dist
            transform = PathLoss(model="free_space")
            result = transform(signal)
            path_losses.append(result["path_loss_db"])

        # Path loss should increase monotonically with distance
        assert path_losses == sorted(path_losses)
        assert path_losses[-1] > path_losses[0]

    def test_missing_distance_raises(self):
        """Verify missing path_distance raises ValueError for free_space model."""
        transform = PathLoss(model="free_space")
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        with pytest.raises(ValueError, match="path_distance"):
            transform(signal)

    def test_missing_frequency_raises(self):
        """Verify missing center_freq raises ValueError for free_space model."""
        transform = PathLoss(model="free_space")
        signal = Signal(data=np.ones(100, dtype=np.complex64))
        signal["path_distance"] = 1000.0
        with pytest.raises(ValueError, match="center_freq"):
            transform(signal)


# =============================================================================
# PathLoss SNR Update Tests
# =============================================================================


class TestPathLossSNRUpdate:
    """Tests for SNR metadata updates in PathLoss."""

    @pytest.mark.parametrize("loss_db", [0.0, 10.0, 20.0])
    def test_snr_update_custom_model(self, loss_db):
        """Verify SNR metadata is updated when present (custom model)."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            center_freq=CENTER_FREQ,
            snr_db=20.0,
        )
        transform = PathLoss(model="custom", loss_db=loss_db)
        result = transform(signal.copy())

        # SNR should be reduced by path loss
        assert result["snr_db"] == pytest.approx(20.0 - loss_db)

    def test_snr_unchanged_when_not_present(self):
        """Verify SNR is unchanged when not present in signal."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            center_freq=CENTER_FREQ,
        )
        signal["path_distance"] = 1000.0
        transform = PathLoss(model="free_space")
        result = transform(signal)
        assert not hasattr(result, "snr_db")

    def test_snr_update_free_space_model(self):
        """Verify SNR metadata is updated when present (free_space model)."""
        distance = 1000.0
        frequency = CENTER_FREQ
        expected_fspl = compute_fspl(distance, frequency)

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            center_freq=frequency,
            snr_db=20.0,
        )
        signal["path_distance"] = distance

        transform = PathLoss(model="free_space")
        result = transform(signal)

        expected_snr = 20.0 - expected_fspl
        assert result["snr_db"] == pytest.approx(expected_snr, rel=1e-5)


# =============================================================================
# PathLoss Metadata Tests
# =============================================================================


class TestPathLossMetadata:
    """Tests for metadata handling in PathLoss."""

    def test_path_loss_db_stored_in_metadata(self):
        """Verify path_loss_db is stored in signal metadata."""
        transform = PathLoss(model="custom", loss_db=30.0)
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        result = transform(signal)
        assert hasattr(result, "path_loss_db")
        assert result["path_loss_db"] == pytest.approx(30.0)

    def test_original_signal_unchanged(self):
        """Verify original signal data is not modified."""
        transform = PathLoss(model="custom", loss_db=30.0)
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        original_data = signal.data.copy()
        _ = transform(signal.copy())
        assert np.array_equal(signal.data, original_data)

    def test_original_signal_unchanged_free_space(self):
        """Verify PathLoss does not modify original signal data."""
        original_data = np.ones(100, dtype=np.complex64)
        signal = Signal(data=original_data.copy(), center_freq=CENTER_FREQ)
        signal["path_distance"] = 1000.0
        signal["snr_db"] = 20.0
        original_data_copy = signal.data.copy()

        transform = PathLoss(model="free_space")
        result = transform(signal.copy())

        # Original signal should not be modified
        np.testing.assert_array_equal(signal.data, original_data_copy)

    def test_returns_signal(self):
        """Verify PathLoss returns a Signal object."""
        transform = PathLoss(model="custom", loss_db=30.0)
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        result = transform(signal)
        assert isinstance(result, Signal)


# =============================================================================
# PathLoss Error Handling Tests
# =============================================================================


class TestPathLossErrorHandling:
    """Tests for error handling in PathLoss."""

    @pytest.mark.parametrize(
        "input_data",
        [
            np.array([1, 2, 3]),
            {"data": np.array([1, 2, 3])},
            None,
            [1, 2, 3],
        ],
    )
    def test_apply_to_non_signal_raises(self, input_data):
        """Verify applying to non-Signal raises TypeError."""
        transform = PathLoss(model="free_space")
        with pytest.raises(TypeError, match="Signal class"):
            transform(input_data)


# =============================================================================
# PathDelay Creation Tests
# =============================================================================


class TestPathDelayCreation:
    """Tests for PathDelay transform creation and configuration."""

    def test_create_default(self):
        """Verify creating PathDelay with default parameters."""
        transform = PathDelay()
        assert transform.propagation_constant == 1.0

    def test_create_custom_propagation_constant(self):
        """Verify creating PathDelay with custom propagation constant."""
        speed = 1500.0  # m/s for underwater acoustics
        scaling_factor = speed / SPEED_OF_LIGHT_M_PER_S
        transform = PathDelay(propagation_constant=scaling_factor)
        assert transform.propagation_constant == pytest.approx(scaling_factor)

    @pytest.mark.parametrize(
        ("propagation_constant", "sample_rate"),
        [(0.5, None), (2.0, None)],
    )
    def test_create_with_various_params(self, propagation_constant, sample_rate):
        """Verify creating PathDelay with various parameter combinations."""
        kwargs = {}
        if propagation_constant is not None:
            kwargs["propagation_constant"] = propagation_constant
        transform = PathDelay(**kwargs)
        if propagation_constant is not None:
            assert transform.propagation_constant == propagation_constant

    @pytest.mark.parametrize("propagation_constant", [0.0, -1.0, -1000.0])
    def test_create_with_invalid_propagation_constant_raises(self, propagation_constant):
        """Verify creating PathDelay with invalid propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="PathDelay.propagation_constant must be positive"):
            PathDelay(propagation_constant=propagation_constant)

    @pytest.mark.parametrize("propagation_constant", [np.nan, np.inf, -np.inf])
    def test_create_with_non_finite_propagation_constant_raises(self, propagation_constant):
        """Verify creating PathDelay with non-finite propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be finite"):
            PathDelay(propagation_constant=propagation_constant)


# =============================================================================
# PathDelay Application Tests
# =============================================================================


class TestPathDelayApplication:
    """Tests for PathDelay transform application."""

    def test_apply_requires_path_distance(self):
        """Verify missing path_distance raises ValueError."""
        transform = PathDelay()
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=1e6)
        with pytest.raises(ValueError, match="path_distance"):
            transform(signal)

    def test_apply_requires_sample_rate(self):
        """Verify missing sample_rate raises ValueError."""
        transform = PathDelay()
        signal = Signal(data=np.ones(100, dtype=np.complex64))
        signal["path_distance"] = 1000.0
        with pytest.raises(ValueError, match="sample_rate"):
            transform(signal)

    def test_apply_to_non_signal_raises(self):
        """Verify applying to non-Signal raises TypeError."""
        transform = PathDelay()
        with pytest.raises(TypeError, match="Signal class"):
            transform(np.array([1, 2, 3]))

    @pytest.mark.parametrize(("distance", "sample_rate"), [(1000.0, 1e6), (2000.0, 1e6), (1000.0, 2e6)])
    def test_apply_basic_delay(self, distance, sample_rate):
        """Verify applying PathDelay with basic parameters."""
        expected_delay_seconds = distance / SPEED_OF_LIGHT_M_PER_S
        expected_delay_samples = round(expected_delay_seconds * sample_rate)

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        # Verify metadata
        assert hasattr(result, "path_delay_seconds")
        assert hasattr(result, "path_delay_samples")
        assert result["path_delay_seconds"] == pytest.approx(expected_delay_seconds)
        assert result["path_delay_samples"] == expected_delay_samples

        # Verify data was shifted (first samples should be zero)
        assert np.all(result.data[:expected_delay_samples] == 0)

        # Verify non-zero data starts at delay position
        if expected_delay_samples < len(result.data):
            assert not np.all(result.data[expected_delay_samples:] == 0)

    def test_apply_zero_distance(self):
        """Verify zero distance results in no delay."""
        transform = PathDelay()
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=1e6)
        signal["path_distance"] = 0.0

        result = transform(signal)

        assert result["path_delay_seconds"] == 0.0
        assert result["path_delay_samples"] == 0
        assert np.array_equal(result.data, signal.data)

    def test_sample_rate_from_metadata(self):
        """Verify sample_rate is obtained from signal metadata."""
        distance = 1000.0
        sample_rate = 1e6

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        expected_delay = distance / SPEED_OF_LIGHT_M_PER_S * sample_rate
        assert result["path_delay_samples"] == pytest.approx(round(expected_delay))

    def test_delay_formula_seconds(self):
        """Verify delay formula in seconds."""
        distance = 2000.0
        expected_delay = distance / SPEED_OF_LIGHT_M_PER_S

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=1e6)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        assert result["path_delay_seconds"] == pytest.approx(expected_delay)

    def test_delay_formula_samples(self):
        """Verify delay formula in samples."""
        distance = 1000.0
        sample_rate = 1e6
        expected_delay_samples = distance / SPEED_OF_LIGHT_M_PER_S * sample_rate

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        assert result["path_delay_samples"] == round(expected_delay_samples)

    def test_integer_rounding(self):
        """Verify fractional samples are rounded correctly."""
        distance = 100.0
        sample_rate = 1000.0
        delay_samples = distance / SPEED_OF_LIGHT_M_PER_S * sample_rate
        expected_rounded = round(delay_samples)

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        assert result["path_delay_samples"] == expected_rounded

    @pytest.mark.parametrize("distance", [0.1, 0.5, 1.0])
    def test_very_small_distance(self, distance):
        """Verify handling of sub-sample distances."""
        sample_rate = 1e6
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        # Should round to 0 or 1 samples
        assert result["path_delay_samples"] in [0, 1]

    def test_large_distance(self):
        """Verify handling of distance exceeding signal length."""
        distance = 1e9
        sample_rate = 1e6
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        # All samples should be zero if delay exceeds signal length
        if result["path_delay_samples"] >= len(signal.data):
            assert np.all(result.data == 0)

    def test_negative_distance(self):
        """Verify that negative path distance is rejected."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=1e6,
            path_distance=-100.0,
        )

        with pytest.raises(
            ValueError,
            match=r"path_distance.*nonnegative.*finite",
        ):
            PathDelay()(signal)

    def test_custom_propagation_constant_formula(self):
        """Verify custom propagation constant (e.g., underwater)."""
        distance = 1000.0
        speed = 1500.0  # m/s for underwater acoustics
        sample_rate = 1000.0
        expected_delay = distance / speed
        expected_samples = round(expected_delay * sample_rate)

        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        scaling_factor = speed / SPEED_OF_LIGHT_M_PER_S
        transform = PathDelay(propagation_constant=scaling_factor)
        result = transform(signal)

        assert result["path_delay_seconds"] == pytest.approx(expected_delay)
        assert result["path_delay_samples"] == expected_samples


# =============================================================================
# PathDelay Metadata Tests
# =============================================================================


class TestPathDelayMetadata:
    """Tests for metadata handling in PathDelay."""

    def test_metadata_updated(self):
        """Verify metadata is updated correctly."""
        distance = 500.0
        sample_rate = 2e6
        signal = Signal(data=np.ones(1000, dtype=np.complex64), sample_rate=sample_rate)
        signal["path_distance"] = distance

        transform = PathDelay()
        result = transform(signal)

        assert hasattr(result, "path_delay_seconds")
        assert hasattr(result, "path_delay_samples")
        assert result["path_delay_seconds"] > 0
        assert result["path_delay_samples"] >= 0

    def test_original_signal_unchanged(self):
        """Verify original signal data is not modified."""
        transform = PathDelay()
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=1e6)
        signal["path_distance"] = 1000.0
        original_data = signal.data.copy()

        _ = transform(signal.copy())

        assert np.array_equal(signal.data, original_data)

    def test_returns_signal(self):
        """Verify PathDelay returns a Signal object."""
        transform = PathDelay()
        signal = Signal(data=np.ones(100, dtype=np.complex64), sample_rate=1e6)
        signal["path_distance"] = 1000.0

        result = transform(signal)
        assert isinstance(result, Signal)


# =============================================================================
# get_absolute_center_freq Tests
# =============================================================================


class TestGetAbsoluteCenterFreq:
    """Tests for get_absolute_center_freq helper function."""

    def test_basic_center_freq(self):
        """Test getting center_freq from a simple signal."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        result = get_absolute_center_freq(signal)
        assert result == pytest.approx(CENTER_FREQ)

    def test_positive_center_freq(self):
        """Test that positive center_freq is returned correctly."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9)
        result = get_absolute_center_freq(signal)
        assert result == pytest.approx(1e9)

    def test_nan_center_freq_raises(self):
        """Test that NaN center_freq raises ValueError."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=np.nan)
        with pytest.raises(ValueError, match="get_absolute_center_freq found non-finite center_freq"):
            get_absolute_center_freq(signal)

    def test_inf_center_freq_raises(self):
        """Test that infinite center_freq raises ValueError."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=np.inf)
        with pytest.raises(ValueError, match="get_absolute_center_freq found non-finite center_freq"):
            get_absolute_center_freq(signal)

    def test_zero_center_freq_raises(self):
        """Test that zero center_freq (with no parent) raises ValueError."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=0.0)
        with pytest.raises(ValueError, match="No non-zero center_freq found"):
            get_absolute_center_freq(signal)

    def test_parent_chain_sum(self):
        """Test that center_freq is summed up the parent chain."""
        parent = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9)
        child = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e8, parent=parent)
        result = get_absolute_center_freq(child)
        assert result == pytest.approx(1.1e9)

    def test_returns_float(self):
        """Test that result is a float."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), center_freq=CENTER_FREQ)
        result = get_absolute_center_freq(signal)
        assert isinstance(result, float)

    def test_circular_reference_detection(self):
        """Test that circular reference in parent chain is detected and breaks loop."""
        signal1 = Signal(data=np.ones(100, dtype=np.complex64), center_freq=1e9)
        signal2 = Signal(data=np.ones(100, dtype=np.complex64), center_freq=2e9, parent=signal1)
        # Create circular reference
        signal1.parent = signal2

        # Should not raise, should break on circular reference
        result = get_absolute_center_freq(signal2)
        # Should sum both frequencies before breaking
        assert result == pytest.approx(3e9)

    def test_inherited_center_frequency_is_counted_once(self) -> None:
        """Inherited metadata must not be added at every hierarchy level."""
        parent = Signal(
            data=np.ones(16, dtype=np.complex64),
            center_freq=2.4e9,
        )
        leaf = Signal(
            data=np.ones(16, dtype=np.complex64),
        )
        leaf.add_parent(parent, register=False)

        result = get_absolute_center_freq(leaf)

        assert result == pytest.approx(2.4e9)

    def test_negative_relative_frequency_offset_is_allowed(self) -> None:
        """A negative child offset may produce a valid absolute frequency."""
        parent = Signal(
            data=np.ones(16, dtype=np.complex64),
            center_freq=2.4e9,
        )
        leaf = Signal(
            data=np.ones(16, dtype=np.complex64),
            center_freq=-1.0e6,
        )
        leaf.add_parent(parent, register=False)

        result = get_absolute_center_freq(leaf)

        assert result == pytest.approx(2.399e9)

    def test_nonpositive_absolute_frequency_is_rejected(self) -> None:
        """The final absolute frequency must be positive."""
        parent = Signal(
            data=np.ones(16, dtype=np.complex64),
            center_freq=1.0e6,
        )
        leaf = Signal(
            data=np.ones(16, dtype=np.complex64),
            center_freq=-2.0e6,
        )
        leaf.add_parent(parent, register=False)

        with pytest.raises(
            ValueError,
            match=r"absolute.*positive|center frequency.*positive",
        ):
            get_absolute_center_freq(leaf)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPathLossIntegration:
    """Integration tests for PathLoss with TorchSigGeoDataset."""

    def test_with_torchsig_geo_dataset(self):
        """Verify PathLoss works in the full TorchSigGeoDataset pipeline."""
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_center_freq_min"] = 1e6  # Use realistic frequency (1 MHz)
        metadata["signal_duration_in_samples_min"] = 128
        metadata["signal_duration_in_samples_max"] = SIGNAL_LENGTH - 1

        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])
        tx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        rx_pos = GeoPoint(lat=SF_LAT + 0.001, lon=SF_LON, alt=SF_ALT)

        transmitter = Transmitter(source_ds, tx_pos, identifier="tx_test")
        receiver = Receiver(rx_pos, sample_rate=metadata["sample_rate"], identifier="rx_test")

        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            channel_transforms=[PathLoss(model="free_space")],
            sample_rate=metadata["sample_rate"],
        )

        rx_signal = next(iter(geo_ds))
        assert len(rx_signal.component_signals) > 0
        # path_loss_db is stored at the leaf level (one per transmitter signal)
        for component in rx_signal.component_signals:
            # Check all leaves of each component
            leaves = []
            stack = [component]
            while stack:
                s = stack.pop()
                if s.component_signals:
                    stack.extend(s.component_signals)
                else:
                    leaves.append(s)
            for leaf in leaves:
                assert "path_loss_db" in leaf.keys()
                assert leaf["path_loss_db"] > 0


class TestPathDelayIntegration:
    """Integration tests for PathDelay with TorchSigGeoDataset."""

    def test_with_torchsig_geo_dataset(self):
        """Verify PathDelay works in the full TorchSigGeoDataset pipeline."""
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_center_freq_min"] = 1e6  # Use realistic frequency (1 MHz)
        metadata["signal_duration_in_samples_min"] = 128
        metadata["signal_duration_in_samples_max"] = SIGNAL_LENGTH - 1
        sample_rate = metadata["sample_rate"]

        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])
        tx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        rx_pos = GeoPoint(lat=SF_LAT + 0.001, lon=SF_LON, alt=SF_ALT)

        transmitter = Transmitter(source_ds, tx_pos, identifier="tx_test")
        receiver = Receiver(rx_pos, sample_rate=sample_rate, identifier="rx_test")

        geo_ds = TorchSigGeoDataset(
            transmitters=[transmitter],
            receivers=[receiver],
            channel_transforms=[PathDelay(sample_rate=sample_rate)],
        )

        rx_signal = next(iter(geo_ds))
        assert len(rx_signal.component_signals) > 0
        for signal in rx_signal.component_signals:
            assert hasattr(signal, "path_delay_seconds")
            assert hasattr(signal, "path_delay_samples")
            assert signal["path_delay_seconds"] > 0


# =============================================================================
# LineOfSight Tests
# =============================================================================


class TestLineOfSightCreation:
    """Tests for LineOfSight transform creation."""

    def test_create_default(self):
        """Verify creating LineOfSight with default parameters."""
        transform = LineOfSight()
        assert transform is not None


class TestLineOfSightApplication:
    """Tests for LineOfSight transform application."""

    @pytest.mark.parametrize(
        ("tx_lat", "tx_lon", "tx_alt", "rx_lat", "rx_lon", "rx_alt", "expected_los"),
        [
            # Short distance at altitude - should have LOS
            (SF_LAT, SF_LON, 100.0, SF_LAT + 0.001, SF_LON, 10.0, True),
            # Opposite sides of Earth - no LOS
            (0.0, 0.0, 100.0, 0.0, 180.0, 100.0, False),
            # Same point - has LOS
            (SF_LAT, SF_LON, 100.0, SF_LAT, SF_LON, 100.0, True),
            # TX inside Earth - no LOS
            (SF_LAT, SF_LON, -1000.0, SF_LAT + 0.001, SF_LON, 100.0, False),
            # RX inside Earth - no LOS
            (SF_LAT, SF_LON, 100.0, SF_LAT + 0.001, SF_LON, -1000.0, False),
            # North Pole to South Pole - no LOS
            (90.0, 0.0, 100.0, -90.0, 0.0, 100.0, False),
            # High altitude satellite - has LOS
            (0.0, 0.0, 400000.0, 10.0, 5.0, 0.0, True),
            # Long distance at surface - no LOS
            (0.0, 0.0, 0.0, 0.0, 90.0, 0.0, False),
        ],
    )
    def test_los_various_scenarios(self, tx_lat, tx_lon, tx_alt, rx_lat, rx_lon, rx_alt, expected_los):
        """Verify LOS for various transmitter/receiver configurations."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": tx_lat,
                "tx_lon": tx_lon,
                "tx_alt": tx_alt,
                "rx_lat": rx_lat,
                "rx_lon": rx_lon,
                "rx_alt": rx_alt,
            },
        )
        transform = LineOfSight()
        result = transform(signal)

        assert hasattr(result, "los")
        assert result["los"] is expected_los
        # Verify signal data is zeroed when LOS is blocked, unchanged when LOS exists
        if expected_los:
            assert np.array_equal(result.data, signal.data)
        else:
            assert np.all(result.data == 0)

    @pytest.mark.parametrize("missing_key", ["tx_lat", "tx_lon", "tx_alt", "rx_lat", "rx_lon", "rx_alt"])
    def test_missing_metadata_raises(self, missing_key):
        """Verify missing required metadata raises ValueError."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=100.0,
            rx_lat=SF_LAT + 0.001,
            rx_lon=SF_LON,
            rx_alt=10.0,
        )
        del signal[missing_key]

        transform = LineOfSight()
        with pytest.raises(ValueError, match=missing_key):
            transform(signal)

    @pytest.mark.parametrize(
        "input_data",
        [
            np.array([1, 2, 3]),
            {"data": np.array([1, 2, 3])},
            None,
            [1, 2, 3],
        ],
    )
    def test_apply_to_non_signal_raises(self, input_data):
        """Verify applying to non-Signal raises TypeError."""
        transform = LineOfSight()
        with pytest.raises(TypeError, match="Signal class"):
            transform(input_data)

    def test_returns_signal(self):
        """Verify LineOfSight returns a Signal object."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": SF_LAT,
                "tx_lon": SF_LON,
                "tx_alt": 100.0,
                "rx_lat": SF_LAT + 0.001,
                "rx_lon": SF_LON,
                "rx_alt": 10.0,
            },
        )
        transform = LineOfSight()
        result = transform(signal)
        assert isinstance(result, Signal)

    def test_signal_zeroed_when_los_blocked(self):
        """Verify signal data is zeroed when LOS is blocked."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": 0.0,
                "tx_lon": 0.0,
                "tx_alt": 100.0,
                "rx_lat": 0.0,
                "rx_lon": 180.0,
                "rx_alt": 100.0,
            },
        )
        transform = LineOfSight()
        result = transform(signal)

        assert result["los"] is False
        assert np.all(result.data == 0)

    def test_signal_unchanged_when_los_exists(self):
        """Verify signal data passes through unchanged when LOS exists."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": SF_LAT,
                "tx_lon": SF_LON,
                "tx_alt": 100.0,
                "rx_lat": SF_LAT + 0.001,
                "rx_lon": SF_LON,
                "rx_alt": 10.0,
            },
        )
        transform = LineOfSight()
        original_data = signal.data.copy()
        result = transform(signal)

        assert result["los"] is True
        assert np.array_equal(result.data, original_data)

    def test_los_tangent_grazes_earth(self):
        """Test LOS when line is tangent to Earth (discriminant = 0)."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": 0.0,
                "tx_lon": 0.0,
                "tx_alt": 1000.0,
                "rx_lat": 0.0,
                "rx_lon": 90.0,
                "rx_alt": 0.0,
            },
        )
        transform = LineOfSight()
        original_data = signal.data.copy()
        result = transform(signal)

        # Line grazes Earth - LOS blocked
        assert result["los"] is False
        assert np.all(result.data == 0)

    def test_los_discriminant_positive_branch(self):
        """Test LOS when discriminant > 0 (quadratic has two real roots)."""
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            metadata={
                "tx_lat": 0.0,
                "tx_lon": 0.0,
                "tx_alt": 100.0,
                "rx_lat": 0.0,
                "rx_lon": 180.0,
                "rx_alt": 100.0,
            },
        )
        transform = LineOfSight()
        result = transform(signal)

        assert result["los"] is False
        assert np.all(result.data == 0)

    def test_vertical_path_to_surface_has_los(self):
        """A vertical path ending at the Earth's surface has LOS."""
        original_data = np.ones(100, dtype=np.complex64)
        signal = Signal(
            data=original_data.copy(),
            metadata={
                "tx_lat": 0.0,
                "tx_lon": 0.0,
                "tx_alt": 10000.0,
                "rx_lat": 0.0,
                "rx_lon": 0.0,
                "rx_alt": 0.0,
            },
        )

        result = LineOfSight()(signal)

        assert result["los"] is True
        np.testing.assert_array_equal(result.data, original_data)

    def test_los_repr(self):
        """Test LineOfSight.__repr__ method."""
        transform = LineOfSight()
        repr_str = repr(transform)
        assert "LineOfSight" in repr_str


# =============================================================================
# Signal Tree Helper Tests
# =============================================================================


class TestMapSignalTree:
    """Tests for map_signal_tree helper function."""

    def test_apply_to_single_signal(self):
        """Verify map_signal_tree applies function to a single signal (no component_signals)."""
        signal = Signal(data=np.ones(100, dtype=np.complex64))
        original_data = signal.data.copy()

        def double_data(s):
            s.data = s.data * 2

        result = map_signal_tree(signal, double_data)

        assert result is signal  # Same object returned
        assert np.array_equal(signal.data, original_data * 2)

    def test_apply_to_signal_with_components(self):
        """Verify map_signal_tree applies function to all signals in tree."""
        comp1 = Signal(data=np.ones(100, dtype=np.complex64))
        comp2 = Signal(data=np.ones(100, dtype=np.complex64) * 2)
        wrapper = Signal(data=np.ones(100, dtype=np.complex64) * 3, component_signals=[comp1, comp2])

        def add_ten(s):
            s.data = s.data + 10

        result = map_signal_tree(wrapper, add_ten)

        assert result is wrapper
        assert np.array_equal(comp1.data, np.ones(100) + 10)
        assert np.array_equal(comp2.data, np.ones(100) * 2 + 10)
        assert np.array_equal(wrapper.data, np.ones(100) * 3 + 10)

    def test_nested_component_signals(self):
        """Verify map_signal_tree handles nested component signals."""
        leaf1 = Signal(data=np.ones(50, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(50, dtype=np.complex64) * 2)
        inner = Signal(data=np.ones(50, dtype=np.complex64) * 3, component_signals=[leaf1, leaf2])
        outer = Signal(data=np.ones(50, dtype=np.complex64) * 4, component_signals=[inner])

        def multiply_by_10(s):
            s.data = s.data * 10

        result = map_signal_tree(outer, multiply_by_10)

        assert result is outer
        assert np.array_equal(leaf1.data, np.ones(50) * 10)
        assert np.array_equal(leaf2.data, np.ones(50) * 2 * 10)
        assert np.array_equal(inner.data, np.ones(50) * 3 * 10)
        assert np.array_equal(outer.data, np.ones(50) * 4 * 10)

    def test_max_depth_allows_root_only(self):
        """Verify max_depth=0 allows only root processing but blocks children."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=0: root (depth=0) is processed (0 > 0 is False),
        # but when recursing to children at depth=1, 1 > 0 is True, raising error
        with pytest.raises(RecursionError, match="Maximum recursion depth 0 exceeded"):
            map_signal_tree(outer, modify, max_depth=0)

    def test_max_depth_allows_one_level(self):
        """Verify max_depth=1 allows root and one level of children."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=1: root (depth=0) and inner (depth=1) are processed,
        # but leaf at depth=2 exceeds limit (2 > 1 is True)
        with pytest.raises(RecursionError, match="Maximum recursion depth 1 exceeded"):
            map_signal_tree(outer, modify, max_depth=1)

    def test_max_depth_allows_all_levels(self):
        """Verify max_depth=2 allows full tree traversal."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=2: all levels processed (0, 1, 2)
        result = map_signal_tree(outer, modify, max_depth=2)

        assert np.array_equal(outer.data, np.ones(50) * 2)
        assert np.array_equal(inner.data, np.ones(50) * 2)
        assert np.array_equal(leaf.data, np.ones(50) * 2)

    def test_max_depth_exceeded_raises(self):
        """Verify RecursionError is raised when max_depth is exceeded."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=1 allows root and one level, but leaf is at depth 2
        with pytest.raises(RecursionError, match="Maximum recursion depth 1 exceeded"):
            map_signal_tree(outer, modify, max_depth=1)

    def test_empty_component_signals(self):
        """Verify map_signal_tree works with signal that has empty component_signals list."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), component_signals=[])
        original_data = signal.data.copy()

        def multiply_by_5(s):
            s.data = s.data * 5

        result = map_signal_tree(signal, multiply_by_5)

        assert result is signal
        assert np.array_equal(signal.data, original_data * 5)

    def test_metadata_modification(self):
        """Verify map_signal_tree can modify metadata."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), snr_db=10.0)

        def increase_snr(s):
            s["snr_db"] = s["snr_db"] + 5.0

        result = map_signal_tree(signal, increase_snr)

        assert result["snr_db"] == pytest.approx(15.0)

    def test_none_signal_returns_none(self):
        """Verify map_signal_tree returns None when input is None."""

        def modify(s):
            s.data = s.data * 2

        result = map_signal_tree(None, modify)
        assert result is None

    def test_already_visited_signal_returns_early(self):
        """Verify map_signal_tree returns early when signal already visited."""
        # Create a signal tree with circular reference
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])
        # Create circular reference
        inner.component_signals.append(outer)

        def modify(s):
            s.data = s.data * 2

        # Should not raise RecursionError, should handle gracefully
        result = map_signal_tree(outer, modify)
        assert result is outer


class TestMapSignalLeaves:
    """Tests for map_signal_leaves helper function."""

    def test_apply_to_leaf_signals_only(self):
        """Verify map_signal_leaves applies function only to leaf signals."""
        leaf1 = Signal(data=np.ones(100, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(100, dtype=np.complex64) * 2)
        wrapper = Signal(data=np.ones(100, dtype=np.complex64) * 3, component_signals=[leaf1, leaf2])

        original_wrapper_data = wrapper.data.copy()

        def multiply_by_10(s):
            s.data = s.data * 10

        result = map_signal_leaves(wrapper, multiply_by_10)

        assert result is wrapper
        assert np.array_equal(leaf1.data, np.ones(100) * 10)
        assert np.array_equal(leaf2.data, np.ones(100) * 2 * 10)
        # Wrapper should NOT be modified (it's not a leaf)
        assert np.array_equal(wrapper.data, original_wrapper_data)

    def test_nested_leaf_signals(self):
        """Verify map_signal_leaves handles nested trees correctly."""
        leaf1 = Signal(data=np.ones(50, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(50, dtype=np.complex64) * 2)
        inner = Signal(data=np.ones(50, dtype=np.complex64) * 3, component_signals=[leaf1, leaf2])
        outer = Signal(data=np.ones(50, dtype=np.complex64) * 4, component_signals=[inner])

        original_inner_data = inner.data.copy()
        original_outer_data = outer.data.copy()

        def add_hundred(s):
            s.data = s.data + 100

        result = map_signal_leaves(outer, add_hundred)

        assert result is outer
        assert np.array_equal(leaf1.data, np.ones(50) + 100)
        assert np.array_equal(leaf2.data, np.ones(50) * 2 + 100)
        # Inner and outer are NOT leaves, should remain unchanged
        assert np.array_equal(inner.data, original_inner_data)
        assert np.array_equal(outer.data, original_outer_data)

    def test_single_leaf_signal(self):
        """Verify map_signal_leaves works with a single leaf signal."""
        signal = Signal(data=np.ones(100, dtype=np.complex64))

        def multiply_by_5(s):
            s.data = s.data * 5

        result = map_signal_leaves(signal, multiply_by_5)

        assert result is signal
        assert np.array_equal(signal.data, np.ones(100) * 5)

    def test_max_depth_allows_leaf_at_depth(self):
        """Verify max_depth parameter works correctly for leaves."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=2 allows outer (depth=0), inner (depth=1), and leaf (depth=2)
        # Note: Leaf is at depth 2, but the check is current_depth > max_depth
        # So at depth=2, 2 > 2 is False, allowing leaf to be processed
        result = map_signal_leaves(outer, modify, max_depth=2)

        # Leaf is at depth 2, which is <= max_depth, so it should be modified
        assert np.array_equal(leaf.data, np.ones(50) * 2)

    def test_max_depth_blocks_leaf(self):
        """Verify max_depth=1 blocks leaf at depth 2."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=1: outer (depth=0) and inner (depth=1) can traverse,
        # but leaf is at depth=2, and 2 > 1 is True
        with pytest.raises(RecursionError, match="Maximum recursion depth 1 exceeded"):
            map_signal_leaves(outer, modify, max_depth=1)

    def test_max_depth_exceeded_raises(self):
        """Verify RecursionError is raised when max_depth is exceeded."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])

        def modify(s):
            s.data = s.data * 2

        # max_depth=0 prevents any traversal beyond root
        with pytest.raises(RecursionError, match="Maximum recursion depth 0 exceeded"):
            map_signal_leaves(outer, modify, max_depth=0)

    def test_single_signal_with_empty_components_is_leaf(self):
        """Verify signal with empty component_signals list is treated as leaf."""
        signal = Signal(data=np.ones(100, dtype=np.complex64), component_signals=[])

        def multiply_by_5(s):
            s.data = s.data * 5

        result = map_signal_leaves(signal, multiply_by_5)

        assert result is signal
        assert np.array_equal(signal.data, np.ones(100) * 5)

    def test_none_signal_returns_early(self):
        """Verify map_signal_leaves handles None input gracefully."""

        def modify(s):
            s.data = s.data * 2

        # Should not raise, should return None
        result = map_signal_leaves(None, modify)
        assert result is None

    def test_circular_component_signals_handled(self):
        """Verify map_signal_leaves handles circular component_signals."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.ones(50, dtype=np.complex64), component_signals=[inner])
        # Create a cycle: inner.component_signals includes outer
        inner.component_signals.append(outer)

        def modify(s):
            s.data = s.data * 2

        # Should not infinite loop, should skip already visited
        result = map_signal_leaves(outer, modify, max_depth=10)
        assert result is outer


class TestAlignSignalLength:
    """Tests for align_signal_length helper function."""

    def test_pad_short_signal(self):
        """Verify padding a short signal to target length."""
        signal = Signal(data=np.ones(50, dtype=np.complex64))
        result = align_signal_length(signal, 100)

        assert result is signal
        assert len(signal.data) == 100
        assert signal["duration_in_samples"] == 100
        # First 50 samples should be original, rest should be zero
        assert np.array_equal(signal.data[:50], np.ones(50))
        assert np.array_equal(signal.data[50:], np.zeros(50))

    def test_truncate_long_signal(self):
        """Verify truncating a long signal to target length."""
        signal = Signal(data=np.arange(100, dtype=np.complex64))
        result = align_signal_length(signal, 50)

        assert result is signal
        assert len(signal.data) == 50
        assert signal["duration_in_samples"] == 50
        # Should contain first 50 samples
        assert np.array_equal(signal.data, np.arange(50, dtype=np.complex64))

    def test_no_change_when_matching_length(self):
        """Verify no change when signal already matches target length."""
        original_data = np.ones(100, dtype=np.complex64)
        signal = Signal(data=original_data.copy())
        signal["duration_in_samples"] = 100

        result = align_signal_length(signal, 100)

        assert result is signal
        assert len(signal.data) == 100
        assert signal["duration_in_samples"] == 100
        assert np.array_equal(signal.data, original_data)

    def test_updates_duration_metadata_on_pad(self):
        """Verify duration_in_samples is updated when padding."""
        signal = Signal(data=np.ones(50, dtype=np.complex64))
        signal["duration_in_samples"] = 50

        align_signal_length(signal, 75)

        assert signal["duration_in_samples"] == 75

    def test_updates_duration_metadata_on_truncate(self):
        """Verify duration_in_samples is updated when truncating."""
        signal = Signal(data=np.ones(100, dtype=np.complex64))
        signal["duration_in_samples"] = 100

        align_signal_length(signal, 25)

        assert signal["duration_in_samples"] == 25


class TestRebuildSignalFromLeaves:
    """Tests for rebuild_signal_from_leaves helper function."""

    def test_rebuild_simple_tree(self):
        """Verify rebuild_signal_from_leaves correctly rebuilds a simple tree."""
        leaf1 = Signal(data=np.ones(100, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(100, dtype=np.complex64) * 2)
        wrapper = Signal(data=np.zeros(100, dtype=np.complex64), component_signals=[leaf1, leaf2])

        result = rebuild_signal_from_leaves(wrapper)

        assert result is wrapper
        expected_data = np.ones(100) + np.ones(100) * 2
        assert np.array_equal(wrapper.data, expected_data)
        assert wrapper["duration_in_samples"] == 100

    def test_rebuild_nested_tree(self):
        """Verify rebuild_signal_from_leaves handles nested trees."""
        leaf1 = Signal(data=np.ones(50, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(50, dtype=np.complex64) * 2)
        inner = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[leaf1, leaf2])
        leaf3 = Signal(data=np.ones(50, dtype=np.complex64) * 3)
        outer = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[inner, leaf3])

        result = rebuild_signal_from_leaves(outer)

        assert result is outer
        # inner should be sum of leaf1 and leaf2
        expected_inner = np.ones(50) + np.ones(50) * 2
        assert np.array_equal(inner.data, expected_inner)
        # outer should be sum of inner and leaf3
        expected_outer = expected_inner + np.ones(50) * 3
        assert np.array_equal(outer.data, expected_outer)

    def test_rebuild_single_leaf(self):
        """Verify rebuild_signal_from_leaves works with single leaf signal."""
        signal = Signal(data=np.ones(100, dtype=np.complex64))
        original_data = signal.data.copy()

        result = rebuild_signal_from_leaves(signal)

        assert result is signal
        assert np.array_equal(signal.data, original_data)

    def test_rebuild_updates_duration(self):
        """Verify duration_in_samples is updated correctly."""
        leaf1 = Signal(data=np.ones(75, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(75, dtype=np.complex64) * 2)
        wrapper = Signal(data=np.zeros(100, dtype=np.complex64), component_signals=[leaf1, leaf2])

        result = rebuild_signal_from_leaves(wrapper)

        assert result["duration_in_samples"] == 75

    def test_rebuild_max_depth_allows_all(self):
        """Verify max_depth=2 allows full rebuild."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[inner])

        result = rebuild_signal_from_leaves(outer, max_depth=2)

        # inner should be rebuilt from leaf
        assert np.array_equal(inner.data, np.ones(50))
        # outer should be rebuilt from inner
        assert np.array_equal(outer.data, np.ones(50))
        assert result is outer

    def test_rebuild_max_depth_blocks_deep(self):
        """Verify max_depth=0 blocks all child processing."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[inner])

        original_outer_data = outer.data.copy()
        # max_depth=0: outer at depth=0 can process, but inner at depth=1 exceeds limit (1 > 0)
        with pytest.raises(RecursionError, match="Maximum recursion depth 0 exceeded"):
            rebuild_signal_from_leaves(outer, max_depth=0)

    def test_rebuild_max_depth_blocks_at_depth_one(self):
        """Verify max_depth=1 allows one level but blocks deeper."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[inner])

        # max_depth=1: outer (depth=0) and inner (depth=1) can process,
        # but leaf at depth=2 exceeds limit (2 > 1)
        with pytest.raises(RecursionError, match="Maximum recursion depth 1 exceeded"):
            rebuild_signal_from_leaves(outer, max_depth=1)

    def test_rebuild_max_depth_exceeded_raises(self):
        """Verify RecursionError is raised when max_depth is exceeded."""
        leaf = Signal(data=np.ones(50, dtype=np.complex64))
        inner = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[leaf])
        outer = Signal(data=np.zeros(50, dtype=np.complex64), component_signals=[inner])

        with pytest.raises(RecursionError, match="Maximum recursion depth 0 exceeded"):
            rebuild_signal_from_leaves(outer, max_depth=0)

    def test_rebuild_leaf_signals_unchanged(self):
        """Verify leaf signals' data is not modified by rebuild."""
        leaf1 = Signal(data=np.ones(100, dtype=np.complex64))
        leaf2 = Signal(data=np.ones(100, dtype=np.complex64) * 2)
        original_leaf1 = leaf1.data.copy()
        original_leaf2 = leaf2.data.copy()

        wrapper = Signal(data=np.zeros(100, dtype=np.complex64), component_signals=[leaf1, leaf2])
        rebuild_signal_from_leaves(wrapper)

        assert np.array_equal(leaf1.data, original_leaf1)
        assert np.array_equal(leaf2.data, original_leaf2)


# =============================================================================
# DopplerShift Tests
# =============================================================================


class TestDopplerShiftCreation:
    """Tests for DopplerShift transform creation and configuration."""

    def test_create_default(self):
        """Verify creating DopplerShift with default parameters."""
        transform = DopplerShift()
        assert transform.propagation_constant == 1.0

    def test_create_with_propagation_constant(self):
        """Verify creating DopplerShift with custom propagation constant."""
        transform = DopplerShift(propagation_constant=2 / 3)
        assert transform.propagation_constant == pytest.approx(2 / 3)

    def test_invalid_propagation_constant_zero(self):
        """Verify creating DopplerShift with zero propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be positive"):
            DopplerShift(propagation_constant=0.0)

    def test_invalid_propagation_constant_negative(self):
        """Verify creating DopplerShift with negative propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be positive"):
            DopplerShift(propagation_constant=-1.0)

    @pytest.mark.parametrize("propagation_constant", [np.nan, np.inf, -np.inf])
    def test_invalid_propagation_constant_non_finite_raises(self, propagation_constant):
        """Verify creating DopplerShift with non-finite propagation_constant raises ValueError."""
        with pytest.raises(ValueError, match="propagation_constant must be finite"):
            DopplerShift(propagation_constant=propagation_constant)


class TestDopplerShiftApplication:
    """Tests for DopplerShift transform application."""

    def test_apply_requires_velocity_metadata(self):
        """Verify DopplerShift requires velocity metadata."""
        transform = DopplerShift()
        # Create a signal without velocity metadata
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            rx_lat=SF_LAT + 0.001,
            rx_lon=SF_LON,
            rx_alt=SF_ALT,
        )

        with pytest.raises(ValueError, match="requires metadata 'tx_vel_east' on signal"):
            transform(signal)

    def test_apply_with_signal_center_freq_and_geo_parent(self):
        """Verify DopplerShift works with center_freq in signal metadata."""
        from torchsig.geo.datasets import Receiver, TorchSigGeoDataset
        from torchsig.geo.types import GeoPoint
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_center_freq_min"] = 1e6  # Use realistic frequency (1 MHz)
        metadata["signal_duration_in_samples_min"] = 128
        metadata["signal_duration_in_samples_max"] = SIGNAL_LENGTH

        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])
        tx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        transmitter = Transmitter(source_ds, tx_pos, identifier="tx_0")
        rx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_0")

        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        transform = DopplerShift()
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=SF_LAT,
            rx_lon=SF_LON,
            rx_alt=SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            parent=geo_ds,
        )

        # Should work with signal metadata
        result = transform(signal)
        assert "doppler_shift_hz" in result.keys()

    def test_apply_with_zero_velocity(self):
        """Verify DopplerShift works with zero velocity (stationary transmitter and receiver)."""
        from torchsig.geo.datasets import Receiver, TorchSigGeoDataset
        from torchsig.geo.types import GeoPoint
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_center_freq_min"] = 1e6  # Use realistic frequency (1 MHz)
        metadata["signal_duration_in_samples_min"] = 128
        metadata["signal_duration_in_samples_max"] = SIGNAL_LENGTH

        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])
        tx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        transmitter = Transmitter(source_ds, tx_pos, identifier="tx_0")
        rx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_0")

        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        # Get a real sample from the dataset which has proper metadata hierarchy
        sample = next(iter(geo_ds))
        # Use the first component signal which has tx_vel_* from transmitter
        # and rx_vel_* inherited from combined signal parent
        signal = sample.component_signals[0]

        transform = DopplerShift()
        result = transform(signal)
        # Doppler shift is applied to leaf signals, not to the wrapper signal
        # Check leaf signal for doppler_shift_hz
        assert len(result.component_signals) > 0
        leaf = result.component_signals[0]
        assert hasattr(leaf, "doppler_shift_hz")
        # With zero velocity, doppler shift should be zero
        assert leaf["doppler_shift_hz"] == pytest.approx(0.0, abs=1e-10)

    def test_apply_requires_position_metadata(self):
        """Verify DopplerShift requires position metadata."""
        from torchsig.geo.datasets import Receiver, TorchSigGeoDataset
        from torchsig.geo.types import GeoPoint
        from torchsig.utils.defaults import TorchSigDefaults

        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_center_freq_min"] = 1e6  # Use realistic frequency (1 MHz)
        metadata["signal_duration_in_samples_min"] = 128
        metadata["signal_duration_in_samples_max"] = SIGNAL_LENGTH

        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])
        tx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        transmitter = Transmitter(source_ds, tx_pos, identifier="tx_0")
        rx_pos = GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)
        receiver = Receiver(rx_pos, sample_rate=SAMPLE_RATE, identifier="rx_0")

        geo_ds = TorchSigGeoDataset(transmitters=[transmitter], receivers=[receiver])

        transform = DopplerShift()
        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            parent=geo_ds,
        )

        with pytest.raises(ValueError, match="requires metadata 'tx_lat' on signal"):
            transform(signal)

    def test_apply_requires_center_freq_or_parameter(self):
        """Verify DopplerShift requires center_freq from signal or parameter."""
        tx = make_tx(identifier="tx_0")
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=0.0,  # Invalid/unset
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        # Without center_freq in signal metadata
        transform = DopplerShift()
        with pytest.raises(ValueError, match="No non-zero center_freq found"):
            transform(signal)

    def test_apply_with_signal_center_freq(self):
        """Verify DopplerShift uses signal center_freq when available."""
        from torchsig.geo.types import GeoVelocity

        tx = make_tx(identifier="tx_0", vel=GeoVelocity(0, 0, 0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift()
        result = transform(signal)

        assert hasattr(result, "doppler_shift_hz")
        assert isinstance(result["doppler_shift_hz"], float)

    def test_doppler_formula_correctness(self):
        """Verify Doppler shift formula is applied correctly."""
        from torchsig.geo.types import GeoVelocity

        # Set up stationary transmitter and receiver (zero radial velocity)
        tx = make_tx(identifier="tx_0", vel=GeoVelocity(0, 0, 0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift(center_freq=CENTER_FREQ)
        result = transform(signal)

        # With zero velocity, Doppler shift should be zero
        assert result["doppler_shift_hz"] == pytest.approx(0.0, abs=1e-10)
        assert result["radial_velocity_mps"] == pytest.approx(0.0, abs=1e-10)

    def test_apply_to_non_signal_raises(self):
        """Verify applying DopplerShift to non-Signal raises TypeError."""
        transform = DopplerShift()
        with pytest.raises(TypeError, match="Signal class"):
            transform(np.array([1, 2, 3]))

    def test_metadata_added(self):
        """Verify DopplerShift adds required metadata."""
        from torchsig.geo.types import GeoVelocity

        tx = make_tx(identifier="tx_0", vel=GeoVelocity(0, 0, 0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift()
        result = transform(signal)

        assert hasattr(result, "doppler_shift_hz")
        assert hasattr(result, "radial_velocity_mps")

    def test_returns_signal(self):
        """Verify DopplerShift returns a Signal object."""
        from torchsig.geo.types import GeoVelocity

        tx = make_tx(identifier="tx_0", vel=GeoVelocity(0, 0, 0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=0.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift()
        result = transform(signal)
        assert isinstance(result, Signal)


class TestDopplerShiftWithVelocity:
    """Tests for DopplerShift with moving transmitter/receiver."""

    def test_moving_transmitter(self):
        """Verify DopplerShift with moving transmitter."""
        from torchsig.geo.types import GeoVelocity

        # Transmitter moving east at 100 m/s
        tx = make_tx(identifier="tx_0", vel=GeoVelocity(east=100.0, north=0.0, up=0.0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift(center_freq=CENTER_FREQ)
        result = transform(signal)

        assert hasattr(result, "doppler_shift_hz")
        assert hasattr(result, "radial_velocity_mps")
        # With eastward motion at ~0.001 degree longitude difference (~111m),
        # the radial velocity should be non-zero
        # The exact value depends on the geometry
        assert isinstance(result["doppler_shift_hz"], float)
        assert isinstance(result["radial_velocity_mps"], float)

    def test_propagation_constant_scaling(self):
        """Verify propagation_constant scales the speed of light correctly."""
        from torchsig.geo.types import GeoVelocity

        # Same setup but with different propagation constants
        tx = make_tx(identifier="tx_0", vel=GeoVelocity(east=100.0, north=0.0, up=0.0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        # Test with different propagation constants (fiber: n=1.5, constant = 2/3)
        transform_vacuum = DopplerShift(center_freq=CENTER_FREQ, propagation_constant=1.0)
        result_vacuum = transform_vacuum(signal.copy())

        transform_fiber = DopplerShift(center_freq=CENTER_FREQ, propagation_constant=2 / 3)
        result_fiber = transform_fiber(signal.copy())

        # In fiber, speed is slower so for same velocity, Doppler shift is larger
        # f_d = (v_radial / (c * propagation_constant)) * f_center
        # With smaller propagation_constant, denominator is smaller, so f_d is larger
        assert result_fiber["doppler_shift_hz"] > result_vacuum["doppler_shift_hz"]

    def test_zero_separation_no_nan(self):
        """Verify DopplerShift handles zero separation (same position) without NaN.

        When transmitter and receiver are at the exact same position, there is
        no defined direction vector. The transform should return zero radial
        velocity and zero Doppler shift instead of NaN.
        """
        from torchsig.geo.types import GeoVelocity

        # Transmitter and receiver at the exact same position
        tx = make_tx(identifier="tx_0", lat=SF_LAT, lon=SF_LON, alt=SF_ALT, vel=GeoVelocity(east=100.0, north=0.0, up=0.0))
        rx = make_rx(lat=SF_LAT, lon=SF_LON, alt=SF_ALT, identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=SF_LAT,
            rx_lon=SF_LON,
            rx_alt=SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift()
        result = transform(signal)

        # Should not produce NaN values
        assert np.isfinite(result["doppler_shift_hz"])
        assert np.isfinite(result["radial_velocity_mps"])
        # With zero separation, radial velocity should be zero
        assert result["radial_velocity_mps"] == pytest.approx(0.0, abs=1e-10)
        # Doppler shift should also be zero (v_radial = 0)
        assert result["doppler_shift_hz"] == pytest.approx(0.0, abs=1e-10)


class TestDopplerShiftIntegration:
    """Integration tests for DopplerShift with TorchSigGeoDataset."""

    def test_with_torchsig_geo_dataset_as_channel_transform(self):
        """Verify DopplerShift works as a channel transform in TorchSigGeoDataset.

        Note: DopplerShift requires receiver-level metadata (rx_id, rx_lat, rx_lon, rx_alt)
        which are only available at the combined signal level, not at the individual
        transmitter component signal level. Therefore, DopplerShift as a channel
        transform needs the metadata to be available through hierarchical parent access.
        """
        from torchsig.utils.defaults import TorchSigDefaults
        from torchsig.geo.types import GeoVelocity

        # Create a simple signal with all required metadata for DopplerShift
        # This simulates a pre-processed signal with proper parent hierarchy
        tx = make_tx(identifier="tx_0", vel=GeoVelocity(east=100.0, north=0.0, up=0.0))
        rx = make_rx(identifier="rx_0")
        geo_ds = TorchSigGeoDataset(transmitters=[tx], receivers=[rx])

        # Manually construct a signal with proper parent hierarchy
        leaf_signal = Signal(
            data=np.ones(100, dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            center_freq=CENTER_FREQ,
            tx_id="tx_0",
            rx_id="rx_0",
            tx_lat=SF_LAT,
            tx_lon=SF_LON,
            tx_alt=SF_ALT,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_lat=NEAR_SF_LAT,
            rx_lon=NEAR_SF_LON,
            rx_alt=NEAR_SF_ALT,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
            frame_index=0,
            parent=geo_ds,
        )

        transform = DopplerShift()
        result = transform(leaf_signal)

        # Verify the transform was applied successfully
        assert hasattr(result, "doppler_shift_hz")
        assert hasattr(result, "radial_velocity_mps")
        assert isinstance(result["doppler_shift_hz"], float)
        assert isinstance(result["radial_velocity_mps"], float)


# =============================================================================
# String Representation Tests
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation of transforms."""

    @pytest.mark.parametrize(
        ("transform", "expected_strings"),
        [
            (PathLoss(model="custom", loss_db=30.0), ["PathLoss", "custom", "30"]),
            (PathLoss(model="free_space"), ["PathLoss", "free_space"]),
            (PathDelay(), ["PathDelay"]),
            (PathDelay(propagation_constant=0.5), ["PathDelay", "0.500"]),
            (DopplerShift(), ["DopplerShift"]),
            (DopplerShift(propagation_constant=0.5), ["DopplerShift", "0.500"]),
        ],
    )
    def test_repr(self, transform, expected_strings):
        """Verify string representation contains expected substrings."""
        repr_str = repr(transform)
        for expected in expected_strings:
            assert expected in repr_str


class TestDopplerShift:
    """Tests for Doppler calculations and metadata updates."""

    def test_inherited_carrier_is_not_duplicated_after_shift(self) -> None:
        """A Doppler shift must add an offset, not copy the inherited carrier."""
        center_freq = 2.4e9
        sample_rate = 1.0e6

        path_signal = Signal(
            data=np.ones(64, dtype=np.complex64),
            center_freq=center_freq,
            sample_rate=sample_rate,
            tx_lat=0.0,
            tx_lon=0.0,
            tx_alt=100.0,
            rx_lat=0.0,
            rx_lon=0.01,
            rx_alt=100.0,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
        )
        leaf = Signal(
            data=np.ones(64, dtype=np.complex64),
        )
        leaf.add_parent(path_signal, register=False)

        DopplerShift()(leaf)

        radial_velocity = leaf["radial_velocity_mps"]
        expected_shift = radial_velocity / SPEED_OF_LIGHT_M_PER_S * center_freq

        assert "center_freq" in leaf.keys()

        # The local contribution should contain only the Doppler offset.
        assert leaf["center_freq"] == pytest.approx(expected_shift)

        # The complete frequency is inherited carrier plus local offset.
        assert get_absolute_center_freq(leaf) == pytest.approx(center_freq + expected_shift)

    @pytest.mark.parametrize(
        "sample_rate",
        [0.0, -1.0, np.nan, np.inf, -np.inf],
    )
    def test_rejects_invalid_sample_rate(
        self,
        sample_rate: float,
    ) -> None:
        """Doppler processing requires a positive finite sample rate."""
        signal = Signal(
            data=np.ones(64, dtype=np.complex64),
            center_freq=2.4e9,
            sample_rate=sample_rate,
            tx_lat=0.0,
            tx_lon=0.0,
            tx_alt=100.0,
            rx_lat=0.0,
            rx_lon=0.01,
            rx_alt=100.0,
            tx_vel_east=100.0,
            tx_vel_north=0.0,
            tx_vel_up=0.0,
            rx_vel_east=0.0,
            rx_vel_north=0.0,
            rx_vel_up=0.0,
        )

        with pytest.raises(
            ValueError,
            match=r"sample_rate.*positive.*finite",
        ):
            DopplerShift()(signal)


class TestPathLoss:
    """Tests for free-space and custom path loss."""

    @pytest.mark.parametrize(
        "loss_db",
        [np.nan, np.inf, -np.inf, "invalid"],
    )
    def test_custom_model_rejects_invalid_loss(
        self,
        loss_db,
    ) -> None:
        """Custom loss must be a finite numeric value."""
        with pytest.raises(
            ValueError,
            match=r"loss_db.*finite",
        ):
            PathLoss(
                model="custom",
                loss_db=loss_db,
            )


class TestPathDelay:
    """Tests for propagation-delay calculations."""

    @pytest.mark.parametrize(
        "distance",
        [-1.0, np.nan, np.inf, -np.inf],
    )
    def test_rejects_invalid_path_distance(
        self,
        distance: float,
    ) -> None:
        """Path distance must be nonnegative and finite."""
        signal = Signal(
            data=np.ones(64, dtype=np.complex64),
            path_distance=distance,
            sample_rate=1.0e6,
        )

        with pytest.raises(
            ValueError,
            match=r"path_distance.*nonnegative.*finite",
        ):
            PathDelay()(signal)

    @pytest.mark.parametrize(
        "sample_rate",
        [0.0, -1.0, np.nan, np.inf, -np.inf],
    )
    def test_rejects_invalid_sample_rate(
        self,
        sample_rate: float,
    ) -> None:
        """Path delay requires a positive finite sample rate."""
        signal = Signal(
            data=np.ones(64, dtype=np.complex64),
            path_distance=1000.0,
            sample_rate=sample_rate,
        )

        with pytest.raises(
            ValueError,
            match=r"sample_rate.*positive.*finite",
        ):
            PathDelay()(signal)
