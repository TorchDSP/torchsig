# Copyright 2024 TorchSig contributors
# SPDX-License-Identifier: MIT
"""Geolocation-aware dataset for RF signal propagation simulation.

This module provides classes for creating datasets that simulate RF propagation
from multiple transmitters to multiple receivers with realistic channel effects.

The main classes are:
    - GeoPoint: Represents a physical location (imported from torchsig.geo)
    - Transmitter: Wraps a dataset with a geographic position
    - Receiver: Represents a receiver position with transforms
    - TorchSigGeoDataset: Main dataset class that orchestrates everything
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union
import hashlib

import numpy as np
from torch.utils.data import IterableDataset

from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigIterableDataset, apply_transforms_and_labels_to_signal
from torchsig.geo.transforms import GeoSignalTransform, align_signal_length, map_signal_leaves, map_signal_tree, rebuild_signal_from_leaves
from torchsig.geo.types import GeoPoint, GeoVelocity
from torchsig.geo.utils.file_handler import GeoDatasetWriter
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import SignalTransform
from torchsig.utils.abstractions import HierarchicalMetadataObject, MetadataAttributeError
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.dsp import multistage_polyphase_resampler
from torchsig.utils.file_handlers.hdf5 import HDF5Writer
from torchsig.utils.random import Seedable
from torchsig.utils.writer import DatasetCreator

__all__ = ["PositionType", "Receiver", "StaticTorchSigGeoDataset", "TorchSigGeoDataset", "Transmitter", "VelocityType", "apply_transforms_to_signal"]

if TYPE_CHECKING:
    from pathlib import Path

    from torchsig.transforms.base_transforms import Transform


# Constants for signal tree traversal
MAX_SIGNAL_TREE_DEPTH = 100


def apply_transforms_to_signal(signal: Signal, transforms: list[Transform]) -> Signal:
    """Apply a list of transforms to a signal.

    GeoSignalTransforms operate on the entire signal tree with metadata.
    SignalTransforms operate on the raw component signals.

    Args:
        signal: The signal to transform
        transforms: The list of transforms to apply

    Returns:
        The transformed signal
    """
    for transform in transforms:
        if isinstance(transform, GeoSignalTransform):
            # GeoSignalTransforms handle their own tree traversal via map_signal_leaves + rebuild
            signal = transform(signal)
        else:
            # SignalTransforms operate on the raw component signals
            # Apply to all leaf component signals and rebuild wrapper data
            # Note: SignalTransform.__call__ modifies in-place and returns the signal,
            # but map_signal_leaves only needs the in-place modification side effect
            map_signal_leaves(signal, transform, max_depth=MAX_SIGNAL_TREE_DEPTH)
            rebuild_signal_from_leaves(signal)
    return signal


def _validate_callable_or_instance(
    value: Any,
    entity_type: str,
    attribute_name: str,
    expected_type: type,
    allow_none: bool = False,
) -> None:
    """Validate a static value or callable without invoking the callable."""
    if allow_none and value is None:
        return

    if callable(value):
        return

    if not isinstance(value, expected_type):
        raise TypeError(f"{entity_type} {attribute_name} must be a {expected_type.__name__} or callable returning {expected_type.__name__}, got {type(value).__name__}")


def _resolve_callable_or_instance(
    value: Any,
    frame_index: int,
    entity_type: str,
    attribute_name: str,
    expected_type: type,
) -> Any:
    """Resolve and validate a dynamic or static geo value."""
    resolved_value = value(frame_index) if callable(value) else value

    if not isinstance(resolved_value, expected_type):
        raise TypeError(f"{entity_type} {attribute_name} must resolve to a {expected_type.__name__}, got {type(resolved_value).__name__}")

    return resolved_value


# Type alias for position: can be a static GeoPoint or a callable that returns GeoPoint
PositionType = Union["GeoPoint", Callable[[int], "GeoPoint"]]
# Type alias for velocity: can be a static GeoVelocity or a callable that returns GeoVelocity
VelocityType = Union["GeoVelocity", Callable[[int], "GeoVelocity"]]


class Transmitter(Seedable):
    """A RF transmitter at a geographic location.

    Wraps an external dataset (TorchSigIterableDataset) with geographic position.
    The transmitter generates signals that can be propagated to receivers.

    Supports both static positions (GeoPoint) and moving positions (callable that
    returns GeoPoint based on frame index). This enables simulation of moving
    transmitters such as aircraft, vehicles, or satellites.

    Attributes:
        dataset: The source dataset generating signals
        identifier: Unique identifier for this transmitter

    Example:
        >>> from torchsig.datasets import TorchSigIterableDataset, Transmitter
        >>> from torchsig.geo.types import GeoPoint
        >>> from torchsig.utils import TorchSigDefaults
        >>>
        >>> # Create source dataset
        >>> metadata = TorchSigDefaults().default_dataset_metadata
        >>> source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators="bpsk")
        >>>
        >>> # Create transmitter at a static location
        >>> position = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        >>> transmitter = Transmitter(source_ds, position)
        >>>
        >>> # Create transmitter with moving position (e.g., aircraft moving east)
        >>> def moving_position(frame_index):
        ...     base_lat = 37.7749
        ...     base_lon = -122.4194
        ...     # Move 100m east per frame (simplified)
        ...     lon_offset = 100 * frame_index / 111320  # meters to degrees
        ...     return GeoPoint(lat=base_lat, lon=base_lon + lon_offset, alt=1000)
        >>> moving_transmitter = Transmitter(source_ds, moving_position)
    """

    def __init__(
        self,
        dataset: TorchSigIterableDataset,
        position: PositionType,
        identifier: str,
        velocity: VelocityType | None = None,
        **kwargs,
    ):
        """Initialize a Transmitter.

        Args:
            dataset: The source dataset that generates signals. Must have 'sample_rate'
                in its metadata.
            position: Geographic position of the transmitter. Can be:
                - GeoPoint: Static position
                - Callable[[int], GeoPoint]: Function that returns position for a given sample index
            identifier: Unique identifier (default: auto-generated)
            velocity: Velocity of the transmitter in ENU (East-North-Up) frame. Can be:
                - GeoVelocity: Constant velocity (east, north, up) in m/s
                - Callable[[int], GeoVelocity]: Function that returns velocity for a given frame index
                - None: Static transmitter (default: GeoVelocity(0, 0, 0))
            **kwargs: Additional arguments for Seedable base class

        Raises:
            TypeError: If dataset is not a TorchSigIterableDataset
            ValueError: If dataset is missing 'sample_rate' metadata
        """
        super().__init__(**kwargs)  # Initialize Seedable for RNG infrastructure

        # Validate dataset type
        if not isinstance(dataset, TorchSigIterableDataset):
            raise TypeError(f"Transmitter dataset must be a TorchSigIterableDataset, got {type(dataset).__name__}")

        # Validate position and velocity
        _validate_callable_or_instance(position, "Transmitter", "position", GeoPoint, allow_none=False)
        _validate_callable_or_instance(velocity, "Transmitter", "velocity", GeoVelocity, allow_none=True)

        self.dataset = dataset
        self._position = position
        self._velocity = velocity
        self.identifier = identifier

        # Validate dataset has sample_rate declared
        if not hasattr(dataset, "sample_rate"):
            raise ValueError(f"Transmitter dataset must have 'sample_rate' in its metadata. Available keys: {sorted(dataset.keys())}") from None
        self.sample_rate = float(dataset["sample_rate"])

        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError(f"Transmitter dataset sample_rate must be positive and finite, got {self.sample_rate}")

        # Set up seeding relationship - use direct seed propagation instead of parent chain
        # This avoids issues with HMO metadata lookup trying to access non-HMO Transmitter
        # We'll manually handle seeding in generate_signal() method

    def get_position(self, frame_index: int = 0) -> GeoPoint:
        """Get and validate the position at a frame index."""
        return _resolve_callable_or_instance(
            self._position,
            frame_index,
            self.__class__.__name__,
            "position",
            GeoPoint,
        )

    def get_velocity(self, frame_index: int = 0) -> GeoVelocity:
        """Get and validate the velocity at a frame index."""
        if self._velocity is None:
            return GeoVelocity(east=0.0, north=0.0, up=0.0)

        return _resolve_callable_or_instance(
            self._velocity,
            frame_index,
            self.__class__.__name__,
            "velocity",
            GeoVelocity,
        )

    def generate_signal(self, frame_index: int) -> Signal:
        """Generate a signal from the source dataset.

        Args:
            frame_index: The frame index for deterministic signal generation.
                All receivers in the same frame will get the same transmitter signal.

        Returns:
            Signal: A signal generated by this transmitter's dataset, with
                   transmitter identifier added. Position metadata is added
                   by TorchSigGeoDataset based on the current sample index.
        """
        # Seed the dataset based on frame_index to ensure all receivers
        # in the same frame get the same transmitter signal.
        # Include the TorchSigGeoDataset's seed for single-point control of the entire simulation.
        # The parent of a Transmitter is always the TorchSigGeoDataset that owns it.
        if self.parent is None or not isinstance(self.parent, TorchSigGeoDataset):
            raise RuntimeError(f"Transmitter '{self.identifier}' must have a TorchSigGeoDataset as its parent. Got parent: {type(self.parent).__name__ if self.parent else 'None'}")
        seed_input = (f"{self.parent.rng_seed}:{self.identifier}:{frame_index}").encode()

        tx_seed = int.from_bytes(
            hashlib.blake2b(seed_input, digest_size=8).digest(),
            byteorder="little",
        ) % (2**31)
        self.dataset.seed(tx_seed)

        # Get next signal from dataset
        sample = next(self.dataset)

        # Handle different return types from dataset
        if isinstance(sample, tuple):
            # Dataset returns (data, labels) or (signal, labels)
            signal = sample[0] if isinstance(sample[0], Signal) else Signal(data=sample[0])
        elif isinstance(sample, Signal):
            signal = sample
        elif isinstance(sample, np.ndarray):
            signal = Signal(data=sample)
        else:
            signal = Signal(data=np.array(sample))

        # Add transmitter identifier and current position/velocity metadata
        # These are transmitter-specific and belong on the transmitter's signal
        tx_pos = self.get_position(frame_index)
        tx_vel = self.get_velocity(frame_index)
        signal["tx_id"] = self.identifier
        signal["tx_lat"] = float(tx_pos.lat)
        signal["tx_lon"] = float(tx_pos.lon)
        signal["tx_alt"] = float(tx_pos.alt)
        signal["tx_vel_east"] = float(tx_vel.east)
        signal["tx_vel_north"] = float(tx_vel.north)
        signal["tx_vel_up"] = float(tx_vel.up)

        return signal

    def __repr__(self) -> str:
        """Return string representation of the transmitter.

        Returns:
            str: Formatted string with transmitter info
        """
        return f"{self.__class__.__name__}(id={self.identifier}, position={self._position})"


class Receiver(Seedable):
    """A RF receiver at a geographic location.

    Represents a receiver position with its own set of transforms that are applied
    to signals received at this location. These transforms model receiver-specific
    effects such as hardware noise, AGC, etc.

    Supports both static positions (GeoPoint) and moving positions (callable that
    returns GeoPoint based on sample index). This enables simulation of moving
    receivers such as mobile ground stations, handheld devices, or aircraft.

    Attributes:
        receiver_transforms: Transforms applied to received signals
        identifier: Unique identifier for this receiver

    Example:
        >>> from torchsig.datasets import Receiver
        >>> from torchsig.geo.types import GeoPoint
        >>> from torchsig.transforms import AWGN
        >>>
        >>> position = GeoPoint(lat=37.7759, lon=-122.4194, alt=10)
        >>> receiver_transforms = [AWGN(noise_power_db=-100.0)]
        >>> receiver = Receiver(position, sample_rate=1e6, receiver_transforms=receiver_transforms)
        >>>
        >>> # Create receiver with moving position (e.g., mobile ground station)
        >>> def moving_position(frame_index):
        ...     base_lat = 37.7759
        ...     base_lon = -122.4194
        ...     # Move 50m north per frame (simplified)
        ...     lat_offset = 50 * frame_index / 111320  # meters to degrees
        ...     return GeoPoint(lat=base_lat + lat_offset, lon=base_lon, alt=10)
        >>> moving_receiver = Receiver(moving_position, sample_rate=1e6, receiver_transforms=receiver_transforms)
    """

    def __init__(
        self,
        position: PositionType,
        sample_rate: float,
        identifier: str,
        receiver_transforms: list[Transform] | None = None,
        velocity: VelocityType | None = None,
        **kwargs,
    ):
        """Initialize a Receiver.

        Args:
            position: Geographic position of the receiver. Can be:
                - GeoPoint: Static position
                - Callable[[int], GeoPoint]: Function that returns position for a given sample index
            sample_rate: Sample rate in Hz (required).
            identifier: Unique identifier (required for reproducibility)
            receiver_transforms: List of transforms to apply to received signals
            velocity: Velocity of the receiver in ENU (East-North-Up) frame. Can be:
                - GeoVelocity: Constant velocity (east, north, up) in m/s
                - Callable[[int], GeoVelocity]: Function that returns velocity for a given frame index
                - None: Static receiver (default: GeoVelocity(0, 0, 0))
            **kwargs: Additional arguments for Seedable base class
        """
        super().__init__(**kwargs)  # Initialize Seedable for RNG infrastructure

        # Validate position and velocity
        _validate_callable_or_instance(position, "Receiver", "position", GeoPoint, allow_none=False)
        _validate_callable_or_instance(velocity, "Receiver", "velocity", GeoVelocity, allow_none=True)

        self._position = position
        self._velocity = velocity
        self.identifier = identifier
        self.sample_rate = float(sample_rate)

        receiver_transforms = receiver_transforms or []

        for index, transform in enumerate(receiver_transforms):
            if not isinstance(transform, SignalTransform):
                raise TypeError(f"receiver_transforms[{index}] must be a SignalTransform, got {type(transform).__name__}")

        self.receiver_transforms = receiver_transforms

        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError(f"Receiver.sample_rate must be positive and finite, got {sample_rate}")

        # Set up parent-child relationships for seeding
        # Note: Transforms are Seedable but we handle seeding at TorchSigGeoDataset level
        for transform in self.receiver_transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)

    def get_position(self, frame_index: int = 0) -> GeoPoint:
        """Get and validate the position at a frame index."""
        return _resolve_callable_or_instance(
            self._position,
            frame_index,
            self.__class__.__name__,
            "position",
            GeoPoint,
        )

    def get_velocity(self, frame_index: int = 0) -> GeoVelocity:
        """Get and validate the velocity at a frame index."""
        if self._velocity is None:
            return GeoVelocity(east=0.0, north=0.0, up=0.0)

        return _resolve_callable_or_instance(
            self._velocity,
            frame_index,
            self.__class__.__name__,
            "velocity",
            GeoVelocity,
        )

    def __repr__(self) -> str:
        """Return string representation of the receiver.

        Returns:
            str: Formatted string with receiver info
        """
        return f"{self.__class__.__name__}(id={self.identifier}, position={self._position})"


class TorchSigGeoDataset(HierarchicalMetadataObject, IterableDataset):
    """Geolocation-aware dataset with multiple transmitters and receivers.

    Simulates RF propagation from transmitters to receivers, applying channel
    effects and receiver-specific transforms.

    Each sample represents one receiver's complete signal, which is the
    combination of signals from all visible transmitters after applying
    path-specific channel effects.

    Iteration is round-robin through receivers (deterministic, no random
    selection as per user requirement).

    Inherits from HierarchicalMetadataObject and IterableDataset to align with
    TorchSig's standard dataset patterns and enable reuse of seeding, metadata
    handling, and transform infrastructure.

    Note: PathLoss is NOT automatically applied. To apply path loss, users must
    explicitly add PathLoss transform to channel_transforms. For example::

        channel_transforms = [PathLoss(model="free_space")]

    Signals from TorchSigIterableDataset have center_freq set automatically,
    which is used by PathLoss via parent-aware metadata lookup.

    Attributes:
        transmitters: List of Transmitter objects
        receivers: List of Receiver objects
        channel_transforms: Global channel transforms applied to each transmitter->receiver path
            (e.g., PathLoss, DopplerShift, LineOfSight). These are applied during signal propagation.
        per_path_transforms: Optional per-path transform configuration. Dict mapping path keys
            (tx_id, rx_id) to lists of transforms for specific transmitter->receiver pairs.
        topology: Mapping of transmitter->receiver connections.
        transforms: Dataset-level transforms applied to the final combined signal at each receiver
            (e.g., AWGN, normalization). Applied after all geo-specific processing.

    Example:
        >>> from torchsig.geo.datasets import TorchSigGeoDataset, Transmitter, Receiver
        >>> from torchsig.geo.types import GeoPoint
        >>> from torchsig.datasets import TorchSigIterableDataset
        >>> from torchsig.geo.transforms import PathLoss
        >>> from torchsig.utils import TorchSigDefaults
        >>>
        >>> # Create positions
        >>> tx_pos = GeoPoint(lat=37.7749, lon=-122.4194)
        >>> rx_pos = GeoPoint(lat=37.7759, lon=-122.4194)
        >>>
        >>> # Create transmitter dataset
        >>> metadata = TorchSigDefaults().default_dataset_metadata
        >>> tx_ds = TorchSigIterableDataset(metadata=metadata, signal_generators="bpsk")
        >>> transmitter = Transmitter(tx_ds, tx_pos)
        >>>
        >>> # Create receiver
        >>> receiver = Receiver(rx_pos, sample_rate=1e6)
        >>>
        >>> # Create geo dataset with explicit path loss
        >>> geo_ds = TorchSigGeoDataset(
        ...     transmitters=[transmitter],
        ...     receivers=[receiver],
        ...     channel_transforms=[PathLoss(model="free_space", center_freq=2.4e9)],
        ... )
        >>>
        >>> # Example with custom topology
        >>> tx0 = Transmitter(TorchSigIterableDataset(metadata=metadata, signal_generators="bpsk"), GeoPoint(lat=37.7749, lon=-122.4194), identifier="tx0")
        >>> tx1 = Transmitter(TorchSigIterableDataset(metadata=metadata, signal_generators="qpsk"), GeoPoint(lat=37.7759, lon=-122.4194), identifier="tx1")
        >>> rx0 = Receiver(GeoPoint(lat=37.7769, lon=-122.4194), sample_rate=1e6, identifier="rx0")
        >>> rx1 = Receiver(GeoPoint(lat=37.7779, lon=-122.4194), sample_rate=1e6, identifier="rx1")
        >>> geo_ds = TorchSigGeoDataset(
        ...     transmitters=[tx0, tx1],
        ...     receivers=[rx0, rx1],
        ...     topology={"tx0": ["rx0", "rx1"], "tx1": ["rx0"]},  # tx0 -> rx0, rx1; tx1 -> rx0
        ...     channel_transforms=[PathLoss(model="free_space", center_freq=2.4e9)],
        ... )
        >>>
        >>> # Iterate
        >>> for signal in geo_ds:
        ...     print(signal["rx_id"])
    """

    def __init__(
        self,
        transmitters: list[Transmitter],
        receivers: list[Receiver],
        *,
        channel_transforms: list[SignalTransform] | None = None,
        per_path_transforms: dict[str, list[SignalTransform]] | None = None,
        topology: dict[str, list[str]] | None = None,
        transforms: list[Transform | callable] | None = None,
        target_labels: list | None = None,
        sample_rate: float | None = None,
        **kwargs,
    ):
        """Initialize a TorchSigGeoDataset.

        Args:
            transmitters: List of Transmitter objects
            receivers: List of Receiver objects
            channel_transforms: List of transforms applied to all paths
            per_path_transforms: Dict mapping (tx_id, rx_id) to list of transforms
            topology: Custom transmitter->receiver connection mapping.
                Dict mapping transmitter identifiers to lists of receiver identifiers.
                For example: {"tx0": ["rx_0", "rx_1", "rx_2"]}.
                Distances are computed automatically from transmitter/receiver positions.
                If None, a full mesh topology is created (all transmitters -> all receivers).
            transforms: List of dataset-level transforms to apply to signals
                (applied after geo-specific processing)
            target_labels: Labels to extract from the signal
            sample_rate: Target sample rate for output signals. If provided, all output
                signals will be resampled to this rate after receiver processing. This
                ensures consistent sample rates across all samples regardless of individual
                transmitter/receiver hardware rates. If None (default), output signals
                retain each receiver's sample_rate. Note: Receiver objects still require
                their own sample_rate which models hardware properties; this parameter
                provides an optional normalization step for ML pipeline convenience.
            **kwargs: Additional metadata passed to HierarchicalMetadataObject

        Raises:
            ValueError: If transmitters or receivers list is empty
            ValueError: If topology references unknown transmitter or receiver identifiers
        """
        # Call parent init
        HierarchicalMetadataObject.__init__(self, **kwargs)

        # Store sample_rate in metadata for library convention consistency
        if sample_rate is not None:
            sample_rate = float(sample_rate)

            if not np.isfinite(sample_rate) or sample_rate <= 0:
                raise ValueError(f"TorchSigGeoDataset sample_rate must be positive and finite, got {sample_rate}")

            self["sample_rate"] = sample_rate

        # Store transforms and target_labels like TorchSigIterableDataset
        self.transforms = transforms or []
        self.component_transforms = []  # Not used in TorchSigGeoDataset (we use channel_transforms)
        self.target_labels = target_labels

        # Set up parent-child relationships for seeding
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)

        if not transmitters:
            raise ValueError("TorchSigGeoDataset requires at least one transmitter")
        if not receivers:
            raise ValueError("TorchSigGeoDataset requires at least one receiver")

        # Validate identifier uniqueness
        tx_ids = [tx.identifier for tx in transmitters]
        if len(tx_ids) != len(set(tx_ids)):
            dup_tx_ids = [tid for tid, cnt in Counter(tx_ids).items() if cnt > 1]
            raise ValueError(f"TorchSigGeoDataset requires unique transmitter identifiers. Duplicates: {sorted(dup_tx_ids)}")

        rx_ids = [rx.identifier for rx in receivers]
        if len(rx_ids) != len(set(rx_ids)):
            dup_rx_ids = [rid for rid, cnt in Counter(rx_ids).items() if cnt > 1]
            raise ValueError(f"TorchSigGeoDataset requires unique receiver identifiers. Duplicates: {sorted(dup_rx_ids)}")

        # Track if we've warned about resampling to avoid repeated warnings
        self._resampling_warned = False

        # Validate and store channel_transforms
        self.channel_transforms: list[SignalTransform] = []
        if channel_transforms:
            for i, transform in enumerate(channel_transforms):
                if not isinstance(transform, SignalTransform):
                    raise TypeError(f"channel_transforms[{i}] must be a SignalTransform, got {type(transform).__name__}")
                transform.add_parent(self)
                self.channel_transforms.append(transform)

        # Validate and store per_path_transforms
        self.per_path_transforms: dict[str, list[SignalTransform]] = {}
        if per_path_transforms:
            for path_key, transforms_list in per_path_transforms.items():
                validated_transforms = []
                for i, transform in enumerate(transforms_list):
                    if not isinstance(transform, SignalTransform):
                        raise TypeError(f"per_path_transforms['{path_key}'][{i}] must be a SignalTransform, got {type(transform).__name__}")
                    transform.add_parent(self)
                    validated_transforms.append(transform)
                self.per_path_transforms[path_key] = validated_transforms

        # Initialize internal state
        self["_receiver_counter"] = 0

        # Store components
        self.transmitters = list(transmitters)
        self.receivers = list(receivers)

        # Build topology (transmitter -> receiver connections)
        self.topology = self._create_topology(connections=topology)

        # Set up parent-child relationships for metadata inheritance
        for tx in self.transmitters:
            tx.add_parent(self)
        for rx in self.receivers:
            rx.add_parent(self)

    def _create_topology(self, connections: dict[str, list[str]] | None = None) -> dict:
        """Create topology for transmitter->receiver connections.

        For moving transmitters/receivers, the distance is NOT stored in the topology.
        Instead, it is computed on-the-fly in _generate_receiver_signal() based on
        the current sample index.

        Args:
            connections: Dict mapping transmitter identifiers to lists of receiver
                identifiers. For example: {"tx0": ["rx_0", "rx_1", "rx_2"]}.
                If None, creates a full mesh topology (all transmitters -> all receivers).

        Returns:
            dict: Mapping of (tx_id, rx_id) -> {transmitter, receiver}

        Raises:
            ValueError: If connections is provided and a transmitter or receiver identifier is not found
            ValueError: If connections contains a self-loop (transmitter -> itself)
        """
        topology = {}

        # Build lookup maps for quick access
        tx_map = {tx.identifier: tx for tx in self.transmitters}
        rx_map = {rx.identifier: rx for rx in self.receivers}
        tx_ids = set(tx_map.keys())
        rx_ids = set(rx_map.keys())

        if connections is None:
            # Full mesh: connect all transmitters to all receivers
            for tx in self.transmitters:
                for rx in self.receivers:
                    key = (tx.identifier, rx.identifier)
                    topology[key] = {
                        "transmitter": tx,
                        "receiver": rx,
                    }
        else:
            # Custom connections: connect specific transmitter-receiver pairs
            for tx_id, rx_id_list in connections.items():
                # Validate transmitter exists
                if tx_id not in tx_ids:
                    raise ValueError(f"Topology references unknown transmitter: '{tx_id}'.\nAvailable transmitters: {sorted(tx_ids)}\nAvailable receivers: {sorted(rx_ids)}")
                tx = tx_map[tx_id]

                for rx_id in rx_id_list:
                    # Validate receiver exists
                    if rx_id not in rx_ids:
                        raise ValueError(
                            f"Topology references unknown receiver: '{rx_id}' (for transmitter '{tx_id}').\nAvailable transmitters: {sorted(tx_ids)}\nAvailable receivers: {sorted(rx_ids)}"
                        )
                    # Prevent self-loop: transmitter -> itself
                    if tx_id == rx_id:
                        raise ValueError(f"Topology contains self-loop: transmitter '{tx_id}' cannot connect to receiver '{rx_id}'. Transmitter and receiver identifiers must be distinct.")
                    rx = rx_map[rx_id]

                    key = (tx_id, rx_id)
                    topology[key] = {
                        "transmitter": tx,
                        "receiver": rx,
                    }

        return topology

    def _get_path_transforms(
        self,
        transmitter: Transmitter,
        receiver: Receiver,
    ) -> list[Transform]:
        """Get channel transforms for a specific transmitter->receiver path.

        Combines:
            1. Per-path transforms from per_path_transforms (if configured)
            2. Global channel transforms

        Note: PathLoss is NOT automatically applied. Users must explicitly add
        PathLoss to channel_transforms if they want path loss effects.
        Signals from TorchSigIterableDataset have center_freq set automatically,

        Args:
            transmitter: The source transmitter
            receiver: The destination receiver

        Returns:
            list[Transform]: Transforms to apply to signals on this path
        """
        path_key = (transmitter.identifier, receiver.identifier)
        path_info = self.topology.get(path_key)
        if path_info is None:
            # No path in topology, return empty list
            return []

        transforms = []

        # Add per-path transforms if configured
        if path_key in self.per_path_transforms:
            transforms.extend(self.per_path_transforms[path_key])

        # Add global channel transforms
        transforms.extend(self.channel_transforms)

        return transforms

    def _resample_signal(
        self,
        signal: Signal,
        to_rate: float,
    ) -> Signal:
        """Resample a signal to a target sample rate.

        Applies polyphase resampling to all leaf component signals and updates
        the sample_rate metadata on all nodes in the tree. Each leaf uses its
        own sample_rate metadata to compute the correct resampling ratio.

        Args:
            signal: The signal to resample (leaves must have sample_rate in metadata)
            to_rate: Target sample rate in Hz

        Returns:
            Signal: The resampled signal with updated sample_rate metadata
        """

        # Apply resampling to leaf signals, computing ratio per-leaf from own sample_rate
        def _resample_leaf(s: Signal, target_rate: float = to_rate) -> None:
            if not hasattr(s, "sample_rate"):
                raise ValueError("Leaf signal missing 'sample_rate' metadata; cannot resample")
            leaf_rate = s["sample_rate"]
            if leaf_rate == target_rate:
                return
            ratio = target_rate / leaf_rate
            s.data = multistage_polyphase_resampler(s.data, ratio)

        map_signal_leaves(signal, _resample_leaf, max_depth=MAX_SIGNAL_TREE_DEPTH)
        rebuild_signal_from_leaves(signal)

        # Update sample_rate on all nodes in the tree using map_signal_tree
        def _update_sample_rate_node(s: Signal) -> None:
            s["sample_rate"] = to_rate

        map_signal_tree(signal, _update_sample_rate_node, max_depth=MAX_SIGNAL_TREE_DEPTH)
        return signal

    def _generate_receiver_signal(self, receiver: Receiver, frame_index: int) -> Signal:
        """Generate the complete signal for a receiver.

        This combines signals from all transmitters that have a path to this
        receiver, applying appropriate channel effects for each path.
        For moving transmitters/receivers, positions and distances are computed
        based on the provided frame_index.

        Args:
            receiver: The receiver to generate a signal for
            frame_index: The frame index for position calculations (all receivers
                in one full cycle share the same frame_index)

        Returns:
            Signal: Combined signal at the receiver with all transmitter
                   signals and receiver effects applied.
        """
        # Collect component signals from all transmitters
        component_signals = []
        # First pass: generate all signals, resample to receiver rate, and find max duration
        tx_signals_with_info = []
        max_duration = 0.0

        # Get receiver position once (always needed for metadata, even when no transmitters are connected)
        rx_pos = receiver.get_position(frame_index)

        for tx in self.transmitters:
            path_key = (tx.identifier, receiver.identifier)

            # Skip if no path exists in topology
            if path_key not in self.topology:
                continue

            # Get current positions for transmitter
            tx_pos = tx.get_position(frame_index)
            distance = tx_pos.distance_to(rx_pos)

            # Generate signal from transmitter
            tx_signal = tx.generate_signal(frame_index)

            # Resample if transmitter and receiver have different sample rates
            if tx.sample_rate != receiver.sample_rate:
                if not self._resampling_warned:
                    warnings.warn(
                        f"Transmitter sample_rate ({tx.sample_rate:.2e} Hz) differs from "
                        f"receiver sample_rate ({receiver.sample_rate:.2e} Hz). "
                        f"Signal will be resampled to receiver rate using polyphase resampling. "
                        f"This adds computational overhead. Consider using matching sample rates "
                        f"for all transmitters and receivers to avoid resampling.",
                        UserWarning,
                        stacklevel=3,
                    )
                    self._resampling_warned = True
                tx_signal = self._resample_signal(tx_signal, receiver.sample_rate)

            # Validate signal duration is positive
            duration = len(tx_signal.data) / receiver.sample_rate
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Signal duration must be positive and finite, got {duration} seconds. Signal length: {len(tx_signal.data)}, sample_rate: {receiver.sample_rate}")

            tx_signals_with_info.append(
                {
                    "signal": tx_signal,
                    "tx": tx,
                    "tx_pos": tx_pos,
                    "rx_pos": rx_pos,
                    "distance": distance,
                    "duration": duration,
                }
            )
            max_duration = max(max_duration, duration)

        # Calculate target length based on max duration at receiver's sample rate
        target_length = round(max_duration * receiver.sample_rate)

        # Issue warning if signals have different lengths (before transforms)
        # This indicates signals will be padded/truncated for combination
        signal_lengths = [len(info["signal"].data) for info in tx_signals_with_info]
        unique_lengths = set(signal_lengths)
        if len(unique_lengths) > 1:
            warnings.warn(
                f"Signals from different transmitters have different lengths: {sorted(unique_lengths)}. "
                f"Signals will be padded or truncated at the END to a common length ({target_length}) "
                f"for element-wise addition. For physically accurate time alignment, "
                f"use PathDelay transform in channel_transforms to account for propagation delays first.",
                UserWarning,
                stacklevel=3,
            )

        # Second pass: add metadata, apply path transforms, then align lengths
        for info in tx_signals_with_info:
            tx_signal = info["signal"]
            tx = info["tx"]
            tx_pos = info["tx_pos"]
            rx_pos = info["rx_pos"]
            distance = info["distance"]

            # Add path-specific metadata to transmitter signal.
            # Path-specific metadata (distance, positions, velocities) are stored at the
            # transmitter component level since they can vary per path.
            # Receiver-level metadata (rx_id, rx_lat/lon/alt) and global metadata (frame_index)
            # are stored at the root level only, leveraging hierarchical inheritance.
            tx_signal["tx_id"] = tx.identifier
            tx_signal["path_distance"] = float(distance)
            tx_signal["tx_lat"] = float(tx_pos.lat)
            tx_signal["tx_lon"] = float(tx_pos.lon)
            tx_signal["tx_alt"] = float(tx_pos.alt)

            # Add velocity metadata for DopplerShift transform (path-specific)
            tx_vel = tx.get_velocity(frame_index)
            tx_signal["tx_vel_east"] = float(tx_vel.east)
            tx_signal["tx_vel_north"] = float(tx_vel.north)
            tx_signal["tx_vel_up"] = float(tx_vel.up)

            # Update parent pointers on all component signals to point to tx_signal
            # so they can inherit path-specific metadata for channel transform validation
            for comp in tx_signal.component_signals:
                comp.add_parent(tx_signal, register=False)

            component_signals.append(tx_signal)

        # Create the combined signal as a wrapper - do NOT flatten/pre-sum the data.
        # The combined_signal's data should be computed on-demand from component_signals
        # using rebuild_signal_from_leaves, preserving the signal tree structure.
        # This allows parent-aware metadata access to work correctly and enables
        # Initialize with None -> empty array; will be rebuilt from component_signals after transforms
        combined_signal = Signal(data=None, component_signals=component_signals)

        # Set up parent relationship: combined_signal -> TorchSigGeoDataset
        # This allows hierarchical metadata lookup and parent chain traversal
        combined_signal.add_parent(self, register=False)

        # Add dataset-level and receiver-level metadata at the ROOT level only.
        # Transforms and child signals can access these via hierarchical inheritance
        # through the parent chain (combined_signal -> TorchSigGeoDataset).
        # This conforms to the principle of storing metadata at the highest level
        # possible to reduce redundancy.
        combined_signal["rx_id"] = receiver.identifier
        combined_signal["rx_lat"] = float(rx_pos.lat)
        combined_signal["rx_lon"] = float(rx_pos.lon)
        combined_signal["rx_alt"] = float(rx_pos.alt)
        connected_tx_ids = tuple(component_signal["tx_id"] for component_signal in component_signals)
        combined_signal["num_transmitters"] = len(connected_tx_ids)
        combined_signal["tx_ids"] = connected_tx_ids
        combined_signal["sample_rate"] = receiver.sample_rate
        combined_signal["frame_index"] = frame_index

        # Add receiver velocity metadata to combined_signal (shared across all transmitters)
        rx_vel = receiver.get_velocity(frame_index)
        combined_signal["rx_vel_east"] = float(rx_vel.east)
        combined_signal["rx_vel_north"] = float(rx_vel.north)
        combined_signal["rx_vel_up"] = float(rx_vel.up)

        # Set up parent relationships for hierarchical metadata access.
        # Component signals (transmitter signals) need explicit parent assignment
        # to inherit receiver-level metadata from combined_signal.
        # Note: Signal constructor does not auto-set parent on component_signals.
        for component_signal in component_signals:
            component_signal.add_parent(combined_signal, register=False)

        # Apply path-specific transforms to each component signal.
        # Transforms access path metadata (tx_*, path_distance, etc.) from the
        # component_signal itself and receiver metadata (rx_*, sample_rate, etc.)
        # from combined_signal via the parent chain. Transforms do NOT access
        # combined_signal.data, so the empty data array is safe.
        for i, component_signal in enumerate(component_signals):
            tx_id = component_signal["tx_id"]
            tx = next(t for t in self.transmitters if t.identifier == tx_id)
            path_transforms = self._get_path_transforms(tx, receiver)

            component_signal = apply_transforms_to_signal(component_signal, path_transforms)

            # Align length after transforms (PathDelay may have changed it)
            map_signal_leaves(
                component_signal,
                lambda leaf: align_signal_length(leaf, target_length),
                max_depth=MAX_SIGNAL_TREE_DEPTH,
            )
            rebuild_signal_from_leaves(
                component_signal,
                max_depth=MAX_SIGNAL_TREE_DEPTH,
            )

            component_signals[i] = component_signal

        # Rebuild combined signal data from aligned component signals
        if component_signals:
            combined_signal.data = component_signals[0].data.copy()
            for comp_signal in component_signals[1:]:
                combined_signal.data += comp_signal.data
        else:
            combined_signal.data = np.zeros(
                target_length,
                dtype=np.complex64,
            )

        # Apply receiver-specific transforms
        combined_signal = apply_transforms_to_signal(combined_signal, receiver.receiver_transforms)

        # Apply dataset-level sample_rate normalization if configured
        # This allows consistent output sample rates across all samples for ML pipelines,
        # while still modeling individual receiver hardware rates during processing
        if "sample_rate" in self.keys():
            target_rate = self["sample_rate"]
            if combined_signal["sample_rate"] != target_rate:
                combined_signal = self._resample_signal(combined_signal, target_rate)

        return combined_signal

    def __iter__(self):
        """Returns an iterator object for the dataset.

        Returns:
            An iterator object that yields samples from the dataset.
        """
        return self

    def __next__(self) -> Signal | np.ndarray | tuple:
        """Returns a dataset sample and (optionally) corresponding targets.

        Returns:
            The sample data and the target values.
        """
        sample = self.__generate_new_signal__()
        return apply_transforms_and_labels_to_signal(sample, self.transforms, self.target_labels)

    def __generate_new_signal__(self) -> Signal:
        """Generate a signal for the next receiver in round-robin order.

        This method is called by the parent class's __next__() method.
        It implements round-robin iteration through receivers.

        Returns:
            Signal: The signal for the next receiver in round-robin order
        """
        # Get current counter value
        current_counter = self["_receiver_counter"]

        # Compute frame_index: all receivers in one full cycle share the same frame_index
        frame_index = current_counter // len(self.receivers)

        # Get next receiver using round-robin
        idx = current_counter % len(self.receivers)
        receiver = self.receivers[idx]

        # Increment counter
        self["_receiver_counter"] += 1

        # Generate the receiver signal with the computed frame_index
        return self._generate_receiver_signal(receiver, frame_index)

    def __len__(self) -> int:
        """Return the number of receivers.

        Returns:
            int: Number of receivers in the dataset
        """
        return len(self.receivers)

    def __getitem__(self, idx: int | str) -> Signal | Any:
        """Get metadata by key.

        Note: This class is an IterableDataset and does not support random access
        to signals by integer index. To get signals, iterate via next() or for loop.
        For metadata access, use string keys.

        Args:
            idx: Metadata key (string) to retrieve

        Returns:
            The metadata value

        Raises:
            MetadataAttributeError: If idx is not found in metadata
        """
        # Only string keys are supported (for metadata access)
        if isinstance(idx, str):
            try:
                return super().__getitem__(idx)
            except MetadataAttributeError as e:
                full_metadata = self.get_full_metadata()
                available_keys = sorted(full_metadata.keys())
                raise MetadataAttributeError(
                    f"Metadata key '{idx}' not found in TorchSigGeoDataset.\nAvailable keys ({len(available_keys)}): {available_keys}\nUse iteration (next() or for loop) to get signals by index."
                ) from e

        # Integer indices are not supported for signal access
        # Use iteration (next() or for loop) to get signals
        raise TypeError(
            f"{self.__class__.__name__} does not support integer indexing for signals.\n"
            f"Use iteration (next() or for loop) to get signals.\n"
            f"To access metadata, use string keys.\n"
            f"Got idx={idx} of type {type(idx).__name__}."
        )

    def to_file(
        self,
        root: str,
        dataset_length: int,
        file_handler_class: type | None = None,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        """Save the dataset to disk.

        Uses TorchSig's DatasetCreator and HDF5Writer for file output.

        Args:
            root: Root directory for the dataset
            dataset_length: Number of samples to generate and save
            file_handler_class: File handler class (defaults to HDF5Writer)
            overwrite: Whether to overwrite existing dataset (default: True)
            **kwargs: Additional arguments for DatasetCreator

        Example:
            >>> geo_dataset.to_file("./output", dataset_length=1000)
        """
        if file_handler_class is None:
            file_handler_class = HDF5Writer

        # Create dataloader
        dataloader = WorkerSeedingDataLoader(self, batch_size=1)

        # Create and run dataset creator
        creator = DatasetCreator(
            dataloader=dataloader,
            dataset_length=dataset_length,
            root=root,
            overwrite=overwrite,
            file_handler=file_handler_class,
            **kwargs,
        )
        creator.create()

    def to_yaml_dat_pairs(
        self,
        root: str,
        dataset_length: int,
        *,
        data_type: str = "float32",
        field_mapping: dict[str, str] | None = None,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        """Save the dataset to .yaml and .dat file pairs.

        Each sample is saved as two files:
        - {index}.yaml: YAML metadata file with receiver information
        - {index}.dat: Binary file with interleaved I/Q samples

        For integer data types (int16, int32), IQ samples are scaled to the
        full integer range: float values in [-1, 1] are mapped to [INT_MIN, INT_MAX].

        Args:
            root: Root directory for the dataset
            dataset_length: Number of samples to generate and save
            data_type: Data type for .dat files ('float32', 'short', 'int16', etc.)
            field_mapping: Dict mapping internal metadata keys to output YAML keys
                (e.g., {"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"})
            allowlist: If provided, only metadata keys in this list will be written
            blocklist: If provided, metadata keys in this list will be excluded
            overwrite: Whether to overwrite existing dataset (default: True)
            **kwargs: Additional arguments for DatasetCreator

        Example:
            >>> geo_dataset.to_yaml_dat_pairs("./output", dataset_length=1000, data_type="float32", field_mapping={"rx_lat": "lat", "rx_lon": "lon", "rx_alt": "alt"})
        """
        # Create dataloader
        dataloader = WorkerSeedingDataLoader(self, batch_size=1)

        # Create and run dataset creator
        creator = DatasetCreator(
            dataloader=dataloader,
            dataset_length=dataset_length,
            root=root,
            overwrite=overwrite,
            file_handler_overwrite=overwrite,
            file_handler=GeoDatasetWriter,
            data_type=data_type,
            field_mapping=field_mapping,
            allowlist=allowlist,
            blocklist=blocklist,
            **kwargs,
        )
        creator.create()

    def get_topology_summary(self, frame_index: int = 0) -> dict[str, Any]:
        """Get a summary of the network topology at a specific frame index.

        For moving transmitters/receivers, this returns positions and distances
        at the specified frame index.

        Args:
            frame_index: The frame index at which to compute positions and distances
                (default: 0)

        Returns:
            dict: Information about all transmitter-receiver pairs including
                  positions and distances. The topology can be set at construction
                  using a user-friendly format like {"tx0": ["rx0", "rx1"]}.

        Example:
            >>> summary = geo_dataset.get_topology_summary()
            >>> print(f"Number of paths: {len(summary['paths'])}")
            >>> # For moving objects, get summary at a different frame index
            >>> summary_at_10 = geo_dataset.get_topology_summary(10)
        """
        summary: dict[str, Any] = {
            "transmitters": {},
            "receivers": {},
            "paths": {},
        }

        # Transmitter info
        for tx in self.transmitters:
            tx_pos = tx.get_position(frame_index)
            summary["transmitters"][tx.identifier] = {
                "position": {
                    "lat": tx_pos.lat,
                    "lon": tx_pos.lon,
                    "alt": tx_pos.alt,
                },
            }

        # Receiver info
        for rx in self.receivers:
            rx_pos = rx.get_position(frame_index)
            summary["receivers"][rx.identifier] = {
                "position": {
                    "lat": rx_pos.lat,
                    "lon": rx_pos.lon,
                    "alt": rx_pos.alt,
                },
            }

        # Path info
        for (tx_id, rx_id), info in self.topology.items():
            tx = info["transmitter"]
            rx = info["receiver"]
            tx_pos = tx.get_position(frame_index)
            rx_pos = rx.get_position(frame_index)
            distance = tx_pos.distance_to(rx_pos)
            summary["paths"][(tx_id, rx_id)] = {"distance_m": distance}

        return summary

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}(transmitters={len(self.transmitters)}, receivers={len(self.receivers)}, paths={len(self.topology)})"


class StaticTorchSigGeoDataset(StaticTorchSigDataset):
    """Static dataset for loading pre-generated TorchSigGeoDataset files.

    This class provides a convenient way to load TorchSigGeoDataset data that has been
    saved to disk using the to_file() method. It inherits from StaticTorchSigDataset
    and provides the same interface for loading geo-located signal data.

    Example:
        >>> # Save a TorchSigGeoDataset
        >>> geo_ds = TorchSigGeoDataset(transmitters=[...], receivers=[...])
        >>> geo_ds.to_file("./output", dataset_length=1000)
        >>>
        >>> # Load it back
        >>> static_geo_ds = StaticTorchSigGeoDataset(root="./output")
        >>> signal = static_geo_ds[0]
        >>> print(signal["rx_id"])
    """

    def __init__(self, root: str | Path, **kwargs) -> None:
        """Initialize a StaticTorchSigGeoDataset.

        Args:
            root: Path to the directory containing the saved dataset.
            **kwargs: Additional arguments passed to StaticTorchSigDataset.
        """
        super().__init__(root=root, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the dataset.

        Returns:
            str: Formatted string with dataset info
        """
        return f"{self.__class__.__name__}(root={self.root})"
