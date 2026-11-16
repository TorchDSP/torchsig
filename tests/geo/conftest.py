"""Shared fixtures and constants for geo module tests.

This module provides common test infrastructure to:
- Eliminate magic numbers through named constants
- Consolidate duplicate fixtures
- Provide reusable helper functions
- Improve test maintainability
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.geo.datasets import Receiver, TorchSigGeoDataset, Transmitter
from torchsig.geo.transforms import PathLoss
from torchsig.geo.types import GeoPoint
from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import TorchSigDefaults

# =============================================================================
# Constants - Single Source of Truth for Magic Numbers
# =============================================================================

# Coordinates (San Francisco Bay Area)
SF_LAT = 37.7749
SF_LON = -122.4194
SF_ALT = 10.0
NEAR_SF_LAT = 37.7759
NEAR_SF_LON = -122.4194
NEAR_SF_ALT = 10.0

# Signal parameters
SAMPLE_RATE = 10_000_000.0  # 10 MHz
SIGNAL_LENGTH = 256
MIN_SIGNAL_CENTER_FREQ = 1e3  # Only physically realizable signals (>0 Hz, while avoiding float imprecision)
MIN_SIGNAL_DURATION = 128
MAX_SIGNAL_DURATION = SIGNAL_LENGTH  # Match signal length to avoid warnings
CENTER_FREQ = 2.4e9  # 2.4 GHz
PATH_DISTANCE = 1000.0  # 1 km

# Common test values
SMALL_SIGNAL_SIZE = 100
DEFAULT_ALT = 10.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for file I/O tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def minimal_metadata():
    """Create minimal metadata configuration for testing."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
    metadata["signal_center_freq_min"] = MIN_SIGNAL_CENTER_FREQ
    metadata["signal_duration_in_samples_min"] = MIN_SIGNAL_DURATION
    metadata["signal_duration_in_samples_max"] = MAX_SIGNAL_DURATION
    metadata["sample_rate"] = SAMPLE_RATE
    return metadata


@pytest.fixture
def source_dataset(minimal_metadata):
    """Create a source dataset for transmitters."""
    return TorchSigIterableDataset(metadata=minimal_metadata, signal_generators=["2fsk"])


@pytest.fixture
def tx_pos():
    """Default transmitter position (San Francisco)."""
    return GeoPoint(lat=SF_LAT, lon=SF_LON, alt=SF_ALT)


@pytest.fixture
def rx_pos():
    """Default receiver position (near San Francisco)."""
    return GeoPoint(lat=NEAR_SF_LAT, lon=NEAR_SF_LON, alt=NEAR_SF_ALT)


@pytest.fixture
def transmitter(source_dataset, tx_pos, receiver):
    """Create a transmitter attached to a geo dataset."""
    tx = Transmitter(source_dataset, tx_pos, identifier="tx_0")
    TorchSigGeoDataset(
        transmitters=[tx],
        receivers=[receiver],
    )
    return tx


@pytest.fixture
def receiver(rx_pos, minimal_metadata):
    """Create a receiver for testing."""
    return Receiver(rx_pos, sample_rate=minimal_metadata["sample_rate"], identifier="rx_0")


@pytest.fixture
def simple_geo_ds(transmitter):
    """Return the geo dataset containing the shared transmitter."""
    assert isinstance(transmitter.parent, TorchSigGeoDataset)
    return transmitter.parent


@pytest.fixture
def sample_signal():
    """Create a basic sample signal."""
    return Signal(
        data=np.ones(SMALL_SIGNAL_SIZE, dtype=np.complex64),
        center_freq=CENTER_FREQ,
        sample_rate=1e6,
    )


@pytest.fixture
def path_loss_free_space():
    """Create a free-space path loss transform."""
    return PathLoss(model="free_space")


@pytest.fixture
def path_loss_custom():
    """Create a custom path loss transform with 30 dB loss."""
    return PathLoss(model="custom", loss_db=30.0)


# =============================================================================
# Helper Functions
# =============================================================================


def make_tx(
    source_ds=None,
    lat=SF_LAT,
    lon=SF_LON,
    alt=SF_ALT,
    identifier=None,
    vel=None,
):
    """Create a transmitter with optional customization.

    Args:
        source_ds: Source dataset (creates default if None)
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        identifier: Unique identifier string (required)
        vel: Velocity as GeoVelocity or tuple (east, north, up)

    Returns:
        Transmitter instance
    """
    if source_ds is None:
        metadata = TorchSigDefaults().default_dataset_metadata.copy()
        metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
        metadata["signal_duration_in_samples_min"] = MIN_SIGNAL_DURATION
        metadata["signal_duration_in_samples_max"] = MAX_SIGNAL_DURATION
        metadata["sample_rate"] = SAMPLE_RATE
        source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators=["2fsk"])

    pos = GeoPoint(lat=lat, lon=lon, alt=alt)
    return Transmitter(source_ds, pos, identifier=identifier or f"tx_{lat}_{lon}", velocity=vel)


def make_rx(
    lat=NEAR_SF_LAT,
    lon=NEAR_SF_LON,
    alt=NEAR_SF_ALT,
    identifier=None,
    sample_rate=SAMPLE_RATE,
):
    """Create a receiver with optional customization.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        identifier: Unique identifier string (required)
        sample_rate: Sample rate in Hz

    Returns:
        Receiver instance
    """
    pos = GeoPoint(lat=lat, lon=lon, alt=alt)
    return Receiver(pos, sample_rate=sample_rate, identifier=identifier or f"rx_{lat}_{lon}")


def make_geo_ds(
    tx_count=1,
    rx_count=1,
    tx_offset=0.0,
    rx_offset=0.0,
    channel_transforms=None,
    **kwargs,
):
    """Create a geo dataset with configurable transmitters and receivers.

    Args:
        tx_count: Number of transmitters
        rx_count: Number of receivers
        tx_offset: Latitude offset for transmitters
        rx_offset: Latitude offset for receivers
        channel_transforms: Optional channel transforms
        **kwargs: Additional arguments for TorchSigGeoDataset

    Returns:
        TorchSigGeoDataset instance
    """
    from torchsig.utils.defaults import TorchSigDefaults

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["num_iq_samples_dataset"] = SIGNAL_LENGTH
    metadata["signal_duration_in_samples_min"] = MIN_SIGNAL_DURATION
    metadata["signal_duration_in_samples_max"] = MAX_SIGNAL_DURATION
    metadata["sample_rate"] = SAMPLE_RATE
    source_ds = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=["2fsk"],
    )

    txs = [make_tx(source_ds, lat=SF_LAT + i * tx_offset, identifier=f"tx_{i}") for i in range(tx_count)]
    rxs = [make_rx(lat=NEAR_SF_LAT + i * rx_offset, identifier=f"rx_{i}") for i in range(rx_count)]

    kwargs["channel_transforms"] = channel_transforms or []
    return TorchSigGeoDataset(transmitters=txs, receivers=rxs, **kwargs)


def compute_fspl(distance, frequency, propagation_constant=1.0):
    """Compute free-space path loss in dB.

    Args:
        distance: Path distance in meters
        frequency: Frequency in Hz
        propagation_constant: Speed scaling factor (default 1.0 for vacuum)

    Returns:
        Path loss in dB
    """
    from torchsig.geo.utils.propagation import SPEED_OF_LIGHT_M_PER_S

    effective_speed = SPEED_OF_LIGHT_M_PER_S * propagation_constant
    wavelength = effective_speed / frequency
    return 20 * np.log10(4 * np.pi * distance / wavelength)
