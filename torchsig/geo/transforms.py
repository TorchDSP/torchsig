"""Channel transforms for RF propagation modeling.

This module provides transforms that model RF channel effects such as path loss,
path delay, fading, and other propagation phenomena between transmitter and receiver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from torchsig.geo.utils.coordinate_system import (
    WGS84_A,
    WGS84_B,
    enu_to_ecef,
    lla_to_ecef,
)
from torchsig.geo.utils.propagation import SPEED_OF_LIGHT_M_PER_S, free_space_path_loss_db
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import SignalTransform
from torchsig.utils.dsp import frequency_shift

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "DopplerShift",
    "GeoSignalTransform",
    "LineOfSight",
    "PathDelay",
    "PathLoss",
    "align_signal_length",
    "get_absolute_center_freq",
    "map_signal_leaves",
    "map_signal_tree",
    "rebuild_signal_from_leaves",
]

# Constants for signal tree traversal
MAX_SIGNAL_TREE_DEPTH = 100


def get_absolute_center_freq(signal: Signal) -> float:
    """Calculate a signal's absolute center frequency from its metadata hierarchy.

    Walks from ``signal`` through its parent chain and sums only the
    ``center_freq`` values stored locally on each object. This avoids counting
    inherited metadata more than once and permits signed relative-frequency
    offsets on child signals.

    Traversal stops at the end of the parent chain or when an already-visited
    object is encountered.

    Args:
        signal: Signal at which to begin the metadata traversal.

    Returns:
        Positive, finite absolute center frequency in Hz.

    Raises:
        ValueError: If a locally stored center frequency is nonfinite.
        ValueError: If no nonzero center frequency is found.
        ValueError: If the summed absolute center frequency is negative or
            nonfinite.
    """
    current = signal
    visited: set[int] = set()
    total_freq = 0.0

    while current is not None:
        signal_id = id(current)
        if signal_id in visited:
            break
        visited.add(signal_id)

        if "center_freq" in current.keys():
            center_freq = float(current["center_freq"])
            if not np.isfinite(center_freq):
                raise ValueError(f"get_absolute_center_freq found non-finite center_freq: {center_freq}")
            total_freq += center_freq

        current = current.parent

    if total_freq == 0:
        raise ValueError(f"No non-zero center_freq found for signal. Searched: {len(visited)} objects in parent chain.")

    if not np.isfinite(total_freq) or total_freq < 0:
        raise ValueError(f"Absolute center frequency must be positive and finite, got {total_freq}")

    return total_freq


def map_signal_leaves(
    signal: Signal,
    func: Callable[[Signal], None],
    *,
    max_depth: int | None = None,
) -> Signal:
    """Apply a function only to leaf signals (those with no component_signals).

    This helper function traverses the signal tree and applies the given function
    only to leaf nodes - signals that have empty component_signals lists.
    This is useful for operations where only the raw signal data at the leaves
    needs to be modified, and parent signals will be rebuilt via superposition
    of their children.

    This is more efficient than map_signal_tree when the function should only
    be applied to leaves, as it avoids unnecessary function calls on intermediate
    wrapper signals.

    Args:
        signal: The signal to process.
        func: A function that takes a Signal and modifies it in-place.
            Only applied to leaf nodes (signals with no component_signals).
        max_depth: Maximum recursion depth. If None, no limit is enforced.
            Use this to prevent infinite recursion on circular references.
            Defaults to None (no limit).

    Returns:
        The same signal object (modified in-place for leaf signals).

    Raises:
        RecursionError: If max_depth is exceeded, indicating potential circular
            references or an unexpectedly deep tree.

    Example:
        >>> from torchsig.signals import Signal
        >>> import numpy as np
        >>> def pad_signal(s, length):
        ...     pad_len = length - len(s.data)
        ...     s.data = np.pad(s.data, (0, pad_len), mode="constant")
        >>> wrapper = Signal(
        ...     data=np.zeros(200),
        ...     component_signals=[
        ...         Signal(data=np.ones(100)),  # leaf
        ...         Signal(data=np.ones(80)),  # leaf
        ...     ],
        ... )
        >>> # Pad only the leaf signals to length 100
        >>> map_signal_leaves(wrapper, lambda s: pad_signal(s, 100))
        >>> # wrapper.data is now stale and should be rebuilt
    """
    _visited: set[int] = set()

    def _traverse(s: Signal | None, current_depth: int) -> None:
        if s is None:
            return

        # Check recursion limit
        if max_depth is not None and current_depth > max_depth:
            raise RecursionError(f"Maximum recursion depth {max_depth} exceeded while traversing signal tree. This may indicate circular references in component_signals.")

        # Detect circular references using object id
        signal_id = id(s)
        if signal_id in _visited:
            return  # Already processed, skip to avoid infinite loops
        _visited.add(signal_id)

        # For leaf nodes (no component_signals), apply the function
        if not s.component_signals:
            func(s)
        else:
            # Non-leaf node: recursively process children but don't apply func
            for comp in s.component_signals:
                _traverse(comp, current_depth + 1)

    _traverse(signal, 0)
    return signal


def rebuild_signal_from_leaves(signal: Signal, *, max_depth: int | None = None) -> Signal:
    """Rebuild signal data in a tree by summing component signals from leaves.

    This function rebuilds the data of all non-leaf signals by summing the data
    of their component_signals. This is typically used after modifying leaf signals
    to ensure that parent/wrapper signals have consistent data that reflects the
    changes made to their children.

    The function operates recursively in a bottom-up manner: leaf signals are left
    unchanged, while each parent signal's data is rebuilt as the sum of its
    component_signals' data.

    Note: This function also updates the duration_in_samples metadata to match
    the length of the rebuilt data.

    Args:
        signal: The signal tree to rebuild.
        max_depth: Maximum recursion depth. If None, no limit is enforced.
            Use this to prevent infinite recursion on circular references.
            Defaults to None (no limit).

    Returns:
        The same signal object with data rebuilt from component_signals.
        The root signal's data will be the sum of all leaf signals' data.

    Raises:
        RecursionError: If max_depth is exceeded, indicating potential circular
            references or an unexpectedly deep tree.

    Example:
        >>> from torchsig.signals import Signal
        >>> import numpy as np
        >>> leaf1 = Signal(data=np.ones(100))
        >>> leaf2 = Signal(data=np.ones(100) * 2)
        >>> wrapper = Signal(data=np.zeros(200), component_signals=[leaf1, leaf2])
        >>> rebuild_signal_from_leaves(wrapper)
        >>> np.allclose(wrapper.data, np.ones(100) + np.ones(100) * 2)  # True
    """
    _visited: set[int] = set()

    def _rebuild(s: Signal | None, current_depth: int) -> None:
        if s is None:
            return

        # Check recursion limit
        if max_depth is not None and current_depth > max_depth:
            raise RecursionError(f"Maximum recursion depth {max_depth} exceeded while rebuilding signal tree. This may indicate circular references in component_signals.")

        signal_id = id(s)
        if signal_id in _visited:
            return
        _visited.add(signal_id)

        # First, recursively rebuild all component signals
        for comp in s.component_signals:
            _rebuild(comp, current_depth + 1)

        # Then rebuild this signal from its components
        if s.component_signals:
            s.data = s.component_signals[0].data.copy()
            for comp in s.component_signals[1:]:
                s.data += comp.data
            s["duration_in_samples"] = len(s.data)

    _rebuild(signal, 0)
    return signal


def align_signal_length(signal: Signal, target_length: int) -> Signal:
    """Align signal data length to target with zero-padding, updating duration metadata.

    Args:
        signal: Signal to align
        target_length: Desired length in samples

    Returns:
        Signal with aligned data and updated duration_in_samples metadata
    """
    current_length = len(signal.data)
    if current_length < target_length:
        signal.data = np.pad(signal.data, (0, target_length - current_length), mode="constant")
        signal["duration_in_samples"] = target_length
    elif current_length > target_length:
        signal.data = signal.data[:target_length]
        signal["duration_in_samples"] = target_length
    return signal


def map_signal_tree(
    signal: Signal,
    func: Callable[[Signal], None],
    *,
    max_depth: int | None = None,
) -> Signal:
    """Recursively apply a function to a signal and all its component signals.

    This helper function traverses the signal tree (a signal and its nested
    component_signals) and applies the given function to each signal in a depth-first
    manner. The function is applied to the root signal first, then to all component
    signals recursively.

    This is useful for transforms that need to apply operations consistently across
    an entire signal hierarchy, such as length normalization, metadata updates, or
    applying the same transformation to all components.

    Args:
        signal: The signal to process.
        func: A function that takes a Signal and modifies it in-place.
        max_depth: Maximum recursion depth. If None, no limit is enforced.
            Use this to prevent infinite recursion on circular references
            (though the Signal class should not have circular references by design).
            Defaults to None (no limit).

    Returns:
        The same signal object (modified in-place by func).

    Raises:
        RecursionError: If max_depth is exceeded, indicating potential circular
            references or an unexpectedly deep tree.

    Example:
        >>> from torchsig.signals import Signal
        >>> import numpy as np
        >>> def truncate_signal(s, length):
        ...     s.data = s.data[:length]
        ...     s["duration_in_samples"] = length
        >>> signal = Signal(data=np.ones(100), component_signals=[Signal(data=np.ones(100)), Signal(data=np.ones(100))])
        >>> # Truncate all signals in the tree to length 50
        >>> map_signal_tree(signal, lambda s: truncate_signal(s, 50))
    """
    # Track traversed signals to detect circular references
    # This is defensive - Signal trees should never be circular, but it's good practice
    _visited: set[int] = set()

    def _traverse(s: Signal | None, current_depth: int) -> None:
        if s is None:
            return

        # Check recursion limit
        if max_depth is not None and current_depth > max_depth:
            raise RecursionError(f"Maximum recursion depth {max_depth} exceeded while traversing signal tree. This may indicate circular references in component_signals.")

        # Detect circular references using object id
        signal_id = id(s)
        if signal_id in _visited:
            return  # Already processed, skip to avoid infinite loops
        _visited.add(signal_id)

        # Apply the function to this signal
        func(s)

        # Recursively process all component signals
        for comp in s.component_signals:
            _traverse(comp, current_depth + 1)

    _traverse(signal, 0)
    return signal


class GeoSignalTransform(SignalTransform):
    """Marker base class for transforms that operate on wrapped geo signals with component_signals.

    This class serves as a marker to distinguish geo transforms that handle wrapped
    signals (with component_signals) from standard SignalTransforms that expect raw signals.

    Callers can use isinstance(transform, GeoSignalTransform) to detect which transforms
    handle wrapped signals. The contract is that GeoSignalTransform subclasses can do
    whatever they need with the signal - operate on signal.data, component_signals, or
    walk the parent chain as needed.

    Transforms that apply the same operation to all signals in a hierarchy should use
    map_signal_tree() to ensure consistency across the entire signal tree.

    Example:
        >>> # In a GeoSignalTransform subclass
        >>> def __apply__(self, signal: Signal) -> Signal:
        ...     signal = super().__apply__(signal)
        ...     # Apply to all component signals recursively
        ...     map_signal_tree(signal, self, max_depth=MAX_SIGNAL_TREE_DEPTH)
        ...     return signal
    """

    def __validate__(self, signal: Signal) -> Signal:
        """Validates signal and all leaf signals have required metadata.

        Uses hasattr which performs hierarchical metadata lookup through parent chain.
        Checks both the signal itself and all leaves in component_signals.

        Args:
            signal: Signal to be validated.

        Raises:
            TypeError: If signal is not a Signal object.
            ValueError: If signal or any leaf is missing required metadata.

        Returns:
            Valid signal.
        """
        if not isinstance(signal, Signal):
            raise TypeError(f"Must be Signal class for transform {self.__class__.__name__}, signal is {type(signal)}.")

        for rm in self.required_metadata:

            def check_leaf(s: Signal, key: str = rm) -> None:
                if not hasattr(s, key):
                    raise ValueError(f"{self.__class__.__name__} requires metadata '{key}' on signal, but it was missing. Signal keys: {list(s.keys())}")

            map_signal_leaves(signal, check_leaf)

        return signal


class PathLoss(GeoSignalTransform):
    """Apply path loss attenuation to a signal based on propagation distance.

    This transform attenuates a signal according to the path loss model, which
    describes how signal strength decreases with distance from the transmitter.

    Supports multiple path loss models:
        - "free_space": Free Space Path Loss (FSPL) - default. Requires:
          - signal metadata 'path_distance' and 'center_freq'
        - "custom": Use a directly specified loss value in dB via loss_db parameter.

    The free space path loss formula is::

        L = 20 * log10(4 * pi * d / lambda)

    where d is distance, lambda is wavelength (c/f), c is speed of light.

    Attributes:
        model: Path loss model to use ("free_space" or "custom")
        loss_db: Direct path loss value in dB (required for "custom" model)
        propagation_constant: Scaling factor for speed of light. Defaults to 1.0
            (vacuum). Set to 1/n where n is refractive index for other media.

    Example:
        >>> # Apply free-space path loss (requires signal['path_distance'] and signal['center_freq'])
        >>> transform = PathLoss(model="free_space")
        >>> signal = transform(signal)
        >>>
        >>> # Apply custom 30 dB path loss
        >>> transform = PathLoss(model="custom", loss_db=30.0)
        >>> signal = transform(signal)
        >>>
        >>> # Apply path loss in fiber (n=1.5, propagation_constant = 2/3)
        >>> transform = PathLoss(model="free_space", propagation_constant=2 / 3)
        >>> signal = transform(signal)
    """

    def __init__(
        self,
        model: str = "free_space",
        loss_db: float | None = None,
        propagation_constant: float = 1.0,
        **kwargs,
    ):
        """Initialize the PathLoss transform.

        Args:
            model: Path loss model. Options: "free_space", "custom".
                Defaults to "free_space".
            loss_db: Direct path loss in dB. Required for "custom" model.
                Defaults to None.
            propagation_constant: Scaling factor for speed of light. Defaults to 1.0
                (vacuum). Set to 1/n where n is refractive index for other media.
                Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            ValueError: If model is not one of the supported options.
            ValueError: If model is "custom" but loss_db is not specified.
            ValueError: If propagation_constant is not finite.
            ValueError: If propagation_constant is not positive.
        """
        if model not in ("free_space", "custom"):
            raise ValueError(f"Unknown path loss model: {model}. Supported models: 'free_space', 'custom'")

        if model == "custom":
            if loss_db is None:
                raise ValueError(f"{self.__class__.__name__} custom model requires loss_db parameter to be specified.")

            if not isinstance(loss_db, (int, float)) or not np.isfinite(loss_db):
                raise ValueError(f"loss_db must be finite, got {loss_db}")

            loss_db = float(loss_db)

        if not isinstance(propagation_constant, (int, float)) or not np.isfinite(propagation_constant):
            raise ValueError(f"propagation_constant must be finite, got {propagation_constant}")
        if propagation_constant <= 0:
            raise ValueError(f"propagation_constant must be positive, got {propagation_constant}")

        # Set required metadata based on model
        # Note: center_freq will be obtained via get_absolute_center_freq which walks the parent chain
        # path_distance and center_freq may be stored at transmitter component level or inherited
        required_metadata = []
        if model == "free_space":
            required_metadata.extend(["path_distance", "center_freq"])

        super().__init__(required_metadata=required_metadata, **kwargs)

        self.model = model
        self.loss_db = loss_db
        self.propagation_constant = propagation_constant

    def __apply__(self, signal: Signal) -> Signal:
        """Apply path loss attenuation to the signal.

        Args:
            signal: Signal to be transformed (already validated by __validate__)

        Returns:
            Signal: Transformed signal with path loss applied

        Raises:
            ValueError: If center_freq is not available from signal metadata
        """
        if self.model == "custom":
            # Custom model: use explicitly provided loss_db (already validated in __init__)
            loss_db = self.loss_db
            attenuation = 10 ** (-loss_db / 20)

            # Apply same attenuation to all leaf signals
            def apply_attenuation_to_leaf(s: Signal) -> None:
                s.data = s.data * attenuation
                if hasattr(s, "snr_db"):
                    s["snr_db"] = s["snr_db"] - loss_db
                s["path_loss_db"] = loss_db

            map_signal_leaves(signal, apply_attenuation_to_leaf, max_depth=MAX_SIGNAL_TREE_DEPTH)
            rebuild_signal_from_leaves(signal)
        else:
            # Free space model: compute loss per leaf using each leaf's center_freq
            distance = signal["path_distance"]

            def apply_free_space_to_leaf(s: Signal) -> None:
                frequency = get_absolute_center_freq(s)
                leaf_loss_db = free_space_path_loss_db(distance, frequency, self.propagation_constant)
                attenuation = 10 ** (-leaf_loss_db / 20)
                s.data = s.data * attenuation
                if hasattr(s, "snr_db"):
                    s["snr_db"] = s["snr_db"] - leaf_loss_db
                s["path_loss_db"] = leaf_loss_db

            map_signal_leaves(signal, apply_free_space_to_leaf, max_depth=MAX_SIGNAL_TREE_DEPTH)
            rebuild_signal_from_leaves(signal)

        return signal

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: Formatted string with transform parameters
        """
        params = [f"model={self.model}"]
        if self.model == "custom" and self.loss_db is not None:
            params.append(f"loss_db={self.loss_db:.1f}dB")
        elif self.model == "free_space" and self.propagation_constant != 1.0:
            params.append(f"propagation_constant={self.propagation_constant:.3f}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class LineOfSight(GeoSignalTransform):
    """Apply line-of-sight blocking to a signal based on WGS84 Earth model geometry.

    This transform calculates whether a direct line-of-sight path exists between
    transmitter and receiver by checking if the line segment connecting them intersects
    the WGS84 ellipsoid. When line-of-sight is blocked (no direct path), the signal
    data is zeroed out to simulate complete signal obstruction.

    The algorithm works by:
    1. Converting both points to ECEF Cartesian coordinates
    2. Parameterizing the line segment between them: P(t) = tx + t * (rx - tx), t in [0, 1]
    3. Substituting into the WGS84 ellipsoid equation: x^2/a^2 + y^2/a^2 + z^2/b^2 = 1
    4. Solving the resulting quadratic equation for t
    5. Checking if any solution exists in the interval [0, 1]

    When line-of-sight is blocked, the signal data is set to zero. When line-of-sight
    exists, the signal passes through unchanged.

    Metadata Added:
        los: Boolean indicating line-of-sight existence (True if LOS, False if blocked).

    Example:
        >>> # Apply LOS blocking between transmitter and receiver
        >>> transform = LineOfSight()
        >>> signal = Signal(data=..., metadata={"tx_lat": 37.7749, "tx_lon": -122.4194, "tx_alt": 100, "rx_lat": 37.7750, "rx_lon": -122.4195, "rx_alt": 10})
        >>> signal = transform(signal)
        >>> if not signal["los"]:
        ...     # Signal data will be zeroed
        ...     print("Signal blocked by Earth!")

    Note:
        Uses the full WGS84 ellipsoid model for accurate LOS determination.
        Points at or above the Earth's surface (including altitude) are considered
        to have LOS if the direct line between them doesn't pass through the ellipsoid.
    """

    def __init__(self, **kwargs):
        """Initialize the LineOfSight transform.

        Args:
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(
            required_metadata=[
                "tx_lat",
                "tx_lon",
                "tx_alt",
                "rx_lat",
                "rx_lon",
                "rx_alt",
            ],
            **kwargs,
        )

    def __apply__(self, signal: Signal) -> Signal:
        """Apply line-of-sight blocking using the WGS84 Earth ellipsoid.

        This transform determines whether the direct path between a transmitter and
        receiver passes through the interior of the Earth. If the path is blocked, the
        signal data is replaced with zeros. Otherwise, the signal passes through
        unchanged.

        The calculation:

        1. Converts the transmitter and receiver coordinates from latitude,
        longitude, and altitude to Earth-Centered, Earth-Fixed (ECEF) coordinates.
        2. Parameterizes the connecting segment as
        ``P(t) = transmitter + t * (receiver - transmitter)``, where
        ``0 <= t <= 1``.
        3. Substitutes the segment into the WGS84 ellipsoid equation.
        4. Finds the minimum ellipsoid value over the open segment ``0 < t < 1``.
        5. Marks the path as blocked when either endpoint is inside the ellipsoid or
        the segment passes through the ellipsoid interior.

        Contact with the ellipsoid only at an endpoint does not block line of sight.
        A path tangent to the ellipsoid is also considered to have line of sight.

        Metadata Added:
            los: ``True`` when a direct line-of-sight path exists; otherwise ``False``.

        Example:
            >>> transform = LineOfSight()
            >>> signal = Signal(
            ...     data=np.ones(128, dtype=np.complex64),
            ...     metadata={
            ...         "tx_lat": 37.7749,
            ...         "tx_lon": -122.4194,
            ...         "tx_alt": 100.0,
            ...         "rx_lat": 37.7750,
            ...         "rx_lon": -122.4195,
            ...         "rx_alt": 10.0,
            ...     },
            ... )
            >>> signal = transform(signal)
            >>> signal["los"]
            True

        Note:
            The calculation uses the WGS84 oblate ellipsoid. It models obstruction by
            the Earth only; terrain, buildings, vegetation, atmospheric refraction,
            diffraction, and multipath propagation are not included.
        """
        # Use parent-aware metadata access to get coordinates
        # These may be stored at the transmitter component level or inherited from parent
        tx_lat = signal["tx_lat"]
        tx_lon = signal["tx_lon"]
        tx_alt = signal["tx_alt"]
        # rx_* metadata is stored at root level and inherited by child signals
        rx_lat = signal["rx_lat"]
        rx_lon = signal["rx_lon"]
        rx_alt = signal["rx_alt"]

        # Convert to ECEF coordinates
        tx_x, tx_y, tx_z = lla_to_ecef(tx_lat, tx_lon, tx_alt)
        rx_x, rx_y, rx_z = lla_to_ecef(rx_lat, rx_lon, rx_alt)

        # Vector from transmitter to receiver
        dx = rx_x - tx_x
        dy = rx_y - tx_y
        dz = rx_z - tx_z

        # WGS84 ellipsoid equation: x^2/a^2 + y^2/a^2 + z^2/b^2 = 1
        # For a point on the line: P(t) = (tx_x + t*dx, tx_y + t*dy, tx_z + t*dz)
        # Substitute into ellipsoid equation and collect terms to form quadratic:
        # t^2 * A + t * B + C = 0

        a_sq = WGS84_A**2
        b_sq = WGS84_B**2

        # Coefficient a (t^2 term)
        a = (dx**2 + dy**2) / a_sq + dz**2 / b_sq

        # Coefficient b (t term)
        b = 2 * (tx_x * dx + tx_y * dy) / a_sq + 2 * tx_z * dz / b_sq

        # Coefficient c (constant term)
        c = (tx_x**2 + tx_y**2) / a_sq + tx_z**2 / b_sq - 1.0

        # The quadratic gives the ellipsoid value minus one along the segment:
        #
        #     q(t) = a*t**2 + b*t + c
        #
        # Earth blocks the path only when an endpoint is inside the ellipsoid or
        # q(t) is negative somewhere in the open segment 0 < t < 1. Contact at an
        # endpoint or a tangential contact does not block line of sight.
        tolerance = 1e-12

        tx_value = c
        rx_value = a + b + c

        tx_inside = tx_value < -tolerance
        rx_inside = rx_value < -tolerance

        passes_through_earth = False
        if a > 0.0:
            t_min = -b / (2.0 * a)

            if 0.0 < t_min < 1.0:
                minimum_value = a * t_min**2 + b * t_min + c
                passes_through_earth = minimum_value < -tolerance

        los = bool(not tx_inside and not rx_inside and not passes_through_earth)

        # Store result at root level only - los is a property of the path, not individual components
        signal["los"] = los

        # Zero the signal data in-place if LOS is blocked - apply to leaves only
        if not los:

            def zero_signal_data(s: Signal) -> None:
                s.data[:] = 0

            map_signal_leaves(signal, zero_signal_data, max_depth=MAX_SIGNAL_TREE_DEPTH)
            rebuild_signal_from_leaves(signal)

        return signal

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: Formatted string with transform parameters.
        """
        return f"{self.__class__.__name__}()"


class DopplerShift(GeoSignalTransform):
    """Apply Doppler frequency shift based on transmitter and receiver velocities.

    This transform models the constant frequency shift caused by relative motion
    between transmitter and receiver when moving at constant velocity (Tier 2).
    It applies a constant Doppler shift to the entire signal.

    The Doppler shift formula is:
        f_doppler = (v_radial / c) * f_center

    where v_radial is the relative radial velocity (positive when objects are
    moving toward each other), c is the speed of light, and f_center is the
    carrier frequency.

    This transform reads velocity from the Transmitter and Receiver objects
    associated with the signal (via tx_id and rx_id metadata) and computes
    the relative radial velocity geometrically.

    Velocities are specified in ENU (East-North-Up) frame as (east, north, up)
    tuples in m/s.

    Attributes:
        propagation_constant: Scaling factor for speed of light. Defaults to 1.0
            (vacuum). Set to 1/n where n is refractive index for other media.

    Note:
        This transform requires the signal to have a TorchSigGeoDataset as an
        ancestor in the metadata hierarchy, from which it retrieves the
        Transmitter and Receiver objects.

        The center_freq is obtained from signal metadata via parent-aware lookup.

    Example:
        >>> from torchsig.geo.datasets import Transmitter, Receiver, TorchSigGeoDataset
        >>> from torchsig.geo.transforms import DopplerShift, PathLoss
        >>> from torchsig.geo.types import GeoPoint
        >>> from torchsig.datasets import TorchSigIterableDataset
        >>> from torchsig.utils import TorchSigDefaults
        >>>
        >>> # Create transmitter moving east at 100 m/s
        >>> metadata = TorchSigDefaults().default_dataset_metadata
        >>> source_ds = TorchSigIterableDataset(metadata=metadata, signal_generators="bpsk")
        >>> tx = Transmitter(source_ds, GeoPoint(37.7749, -122.4194, 100), velocity=(100, 0, 0))
        >>> rx = Receiver(GeoPoint(37.7759, -122.4194, 10))
        >>>
        >>> geo_ds = TorchSigGeoDataset(
        ...     transmitters=[tx],
        ...     receivers=[rx],
        ...     channel_transforms=[DopplerShift()],
        ... )
        >>> signal = next(geo_ds)
        >>> print(f"Doppler shift: {signal['doppler_shift_hz']:.1f} Hz")
        >>>
        >>> # In fiber (n=1.5, propagation_constant = 2/3)
        >>> geo_ds = TorchSigGeoDataset(
        ...     transmitters=[tx],
        ...     receivers=[rx],
        ...     channel_transforms=[DopplerShift(propagation_constant=2 / 3)],
        ... )
    """

    def __init__(self, propagation_constant: float = 1.0, **kwargs):
        """Initialize the DopplerShift transform.

        Args:
            propagation_constant: Scaling factor for speed of light. Defaults to 1.0
                (vacuum). Set to 1/n where n is refractive index for other media.
                Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            ValueError: If propagation_constant is not positive.
        """
        if not isinstance(propagation_constant, (int, float)) or not np.isfinite(propagation_constant):
            raise ValueError(f"propagation_constant must be finite, got {propagation_constant}")
        if propagation_constant <= 0:
            raise ValueError(f"propagation_constant must be positive, got {propagation_constant}")

        self.propagation_constant = propagation_constant
        super().__init__(
            required_metadata=[
                "tx_lat",
                "tx_lon",
                "tx_alt",
                "rx_lat",
                "rx_lon",
                "rx_alt",
                "tx_vel_east",
                "tx_vel_north",
                "tx_vel_up",
                "rx_vel_east",
                "rx_vel_north",
                "rx_vel_up",
                "sample_rate",
                "center_freq",
            ],
            **kwargs,
        )

    def __apply__(self, signal: Signal) -> Signal:
        """Apply Doppler frequency shift to the signal.

        Args:
            signal: Signal to be transformed (already validated by __validate__).
                Must have tx_id, rx_id, and position metadata.

        Returns:
            Signal: Transformed signal with Doppler shift applied. Adds
                doppler_shift_hz and radial_velocity_mps metadata.

        Raises:
            ValueError: If center_freq is not available from signal metadata.
        """
        sr = float(signal["sample_rate"])

        if not np.isfinite(sr) or sr <= 0:
            raise ValueError(f"DopplerShift sample_rate must be positive and finite, got {sr}")

        # Compute radial velocity from metadata (accessed via hierarchical lookup)
        v_radial = self._compute_radial_velocity(signal)
        signal["radial_velocity_mps"] = float(v_radial)

        # Doppler shift: f_d = (v_radial / c) * f_c
        # Get center_freq from signal metadata via parent traversal
        effective_speed = SPEED_OF_LIGHT_M_PER_S * self.propagation_constant

        # Apply frequency shift to leaf signals only (more efficient)
        def apply_doppler_to_leaf(s: Signal) -> None:
            f_center = get_absolute_center_freq(s)
            f_doppler = (v_radial / effective_speed) * f_center

            s.data = frequency_shift(s.data, f_doppler, sr)
            if "center_freq" in s.keys():
                s["center_freq"] = float(s["center_freq"]) + f_doppler
            else:
                # The carrier is inherited; store only the new local offset.
                s["center_freq"] = float(f_doppler)
            s["doppler_shift_hz"] = float(f_doppler)

        map_signal_leaves(signal, apply_doppler_to_leaf, max_depth=MAX_SIGNAL_TREE_DEPTH)
        rebuild_signal_from_leaves(signal)

        return signal

    def _compute_radial_velocity(self, signal: Signal) -> float:
        """Compute relative radial velocity between transmitter and receiver.

        Radial velocity is the component of relative velocity along the line of
        sight from transmitter to receiver. Positive when objects are moving
        toward each other.

        Args:
            signal: Signal with position and velocity metadata (accessed via hierarchical lookup)

        Returns:
            Radial velocity in m/s (positive = objects moving toward each other)
        """
        # Use parent-aware metadata access to get positions
        # tx_* may be at signal level, rx_* inherited from root
        tx_lat = signal["tx_lat"]
        tx_lon = signal["tx_lon"]
        tx_alt = signal["tx_alt"]
        rx_lat = signal["rx_lat"]
        rx_lon = signal["rx_lon"]
        rx_alt = signal["rx_alt"]

        # Convert positions to ECEF
        tx_ecef = np.array(lla_to_ecef(tx_lat, tx_lon, tx_alt))
        rx_ecef = np.array(lla_to_ecef(rx_lat, rx_lon, rx_alt))

        # Vector from transmitter to receiver
        r_vec = rx_ecef - tx_ecef
        r_norm = np.linalg.norm(r_vec)
        # Zero separation: no defined direction, radial velocity is zero
        r_hat = np.zeros(3) if r_norm == 0 else r_vec / r_norm  # Unit vector from tx to rx

        # Get velocities from metadata (accessed via hierarchical lookup)
        tx_vel_enu = (signal["tx_vel_east"], signal["tx_vel_north"], signal["tx_vel_up"])
        rx_vel_enu = (signal["rx_vel_east"], signal["rx_vel_north"], signal["rx_vel_up"])

        # Unpack velocities
        tx_e, tx_n, tx_u = tx_vel_enu
        rx_e, rx_n, rx_u = rx_vel_enu

        # Convert ENU velocities to ECEF
        tx_vel_ecef = np.array(enu_to_ecef(tx_lat, tx_lon, tx_e, tx_n, tx_u))
        rx_vel_ecef = np.array(enu_to_ecef(rx_lat, rx_lon, rx_e, rx_n, rx_u))

        # Relative velocity: v_rx - v_tx
        v_rel = rx_vel_ecef - tx_vel_ecef

        # Radial component of relative velocity.
        # Note: We negate because Doppler convention uses closing velocity (positive = decreasing range),
        # but r_hat points from tx to rx, so (v_rx - v_tx) · r_hat gives range rate (positive = increasing range).
        # Closing velocity = -range rate.
        return -np.dot(v_rel, r_hat)

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: Formatted string with transform parameters.
        """
        params = []
        if self.propagation_constant != 1.0:
            params.append(f"propagation_constant={self.propagation_constant:.3f}")
        if not params:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({', '.join(params)})"


class PathDelay(GeoSignalTransform):
    """Apply time delay to a signal based on propagation distance.

    This transform delays a signal according to the time it takes for the signal
    to propagate over the path distance at the given propagation speed.

    The delay formula is:
        delay_seconds = path_distance / (speed_of_light * propagation_constant)
        delay_samples = delay_seconds * sample_rate

    The delay is applied by prepending zeros to the signal data and truncating
    the end to maintain the original signal length. This simulates the time
    delay of signal propagation in a discrete-time system.

    Attributes:
        propagation_constant: Scaling factor for speed of light. Defaults to 1.0
            (vacuum). Set to 1/n where n is refractive index for other media.
            For example, 0.667 for fiber (n=1.5).

    Example:
        >>> # Default: uses speed of light, requires signal['path_distance'] and signal['sample_rate']
        >>> transform = PathDelay()
        >>> signal = transform(signal)
        >>>
        >>> # In fiber (n=1.5, propagation_constant = 2/3)
        >>> transform = PathDelay(propagation_constant=2 / 3)
        >>> signal = transform(signal)
    """

    def __init__(
        self,
        propagation_constant: float = 1.0,
        **kwargs,
    ):
        """Initialize the PathDelay transform.

        Args:
            propagation_constant: Scaling factor for speed of light. Defaults to 1.0
                (vacuum). Set to 1/n where n is refractive index for other media.
                Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            ValueError: If propagation_constant is not positive
        """
        if not isinstance(propagation_constant, (int, float)) or not np.isfinite(propagation_constant):
            raise ValueError(f"propagation_constant must be finite, got {propagation_constant}")
        if propagation_constant <= 0:
            raise ValueError(f"PathDelay.propagation_constant must be positive, got {propagation_constant}")
        self.propagation_constant = propagation_constant
        required_metadata = ["path_distance", "sample_rate"]

        super().__init__(required_metadata=required_metadata, **kwargs)

    def __apply__(self, signal: Signal) -> Signal:
        """Apply time delay to the signal.

        Args:
            signal: Signal to be transformed (already validated by __validate__).
                Must have 'path_distance' metadata and 'sample_rate' metadata.

        Returns:
            Signal: Transformed signal with time delay applied. The signal data
                is shifted right (zeros prepended, end truncated) and metadata
                is updated with path_delay_seconds and path_delay_samples.
        """
        distance = float(signal["path_distance"])
        sr = float(signal["sample_rate"])

        if not np.isfinite(distance) or distance < 0:
            raise ValueError(f"PathDelay path_distance must be nonnegative and finite, got {distance}")

        if not np.isfinite(sr) or sr <= 0:
            raise ValueError(f"PathDelay sample_rate must be positive and finite, got {sr}")

        # Calculate delay
        effective_speed = SPEED_OF_LIGHT_M_PER_S * self.propagation_constant
        delay_seconds = distance / effective_speed
        delay_samples = delay_seconds * sr
        delay_samples_int = round(delay_samples)

        # Apply delay to leaf signals only (more efficient)
        def apply_delay_to_leaf(s: Signal) -> None:
            s.data = np.pad(s.data, (delay_samples_int, 0), mode="constant")[: len(s.data)]

        map_signal_leaves(signal, apply_delay_to_leaf, max_depth=MAX_SIGNAL_TREE_DEPTH)
        rebuild_signal_from_leaves(signal)

        # Apply metadata at root level only
        # path_delay_seconds and path_delay_samples are properties of the propagation path,
        # not of individual signals, so they are stored at the root level only
        signal["path_delay_seconds"] = delay_seconds
        signal["path_delay_samples"] = delay_samples_int

        return signal

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            str: Formatted string with transform parameters.
        """
        params = []
        if self.propagation_constant != 1.0:
            params.append(f"propagation_constant={self.propagation_constant:.3f}")
        if not params:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({', '.join(params)})"
