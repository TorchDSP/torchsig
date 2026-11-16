"""Utility functions for TorchSigGeoDataset examples.

This module provides helper functions for visualization and geolocation
algorithms that are not directly related to using the TorchSig GeoDataset API.

Typical usage:
    from geo_example_utils import (
        plot_geo_network_at_frame,
        estimate_position_from_ranges,
        calculate_tdoa_fix,
        collect_multi_frame_data,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torchsig.geo.datasets import Receiver, Transmitter


def plot_geo_network_at_frame(
    transmitters: list[Transmitter],
    receivers: list[Receiver],
    frame_index: int = 0,
    connections: bool = True,
    ax: Any | None = None,
    title: str | None = None,
    topology: dict | None = None,
) -> Any:
    """Plot transmitters and receivers on a 2D map at a specific frame.

    Args:
        transmitters: List of Transmitter objects.
        receivers: List of Receiver objects.
        frame_index: Frame index for position lookup.
        connections: Whether to draw lines between tx and rx.
        ax: Matplotlib axes object (if None, creates new figure).
        title: Optional title for the plot.
        topology: Optional dictionary mapping (tx_id, rx_id) to connection info.
            If provided and connections=True, only draws the connections defined
            in the topology. If None, draws a full mesh (all TX->RX pairs).

    Returns:
        The matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_ax = True

    # Get positions at this frame using get_position()
    tx_lats = [tx.get_position(frame_index).lat for tx in transmitters]
    tx_lons = [tx.get_position(frame_index).lon for tx in transmitters]
    rx_lats = [rx.get_position(frame_index).lat for rx in receivers]
    rx_lons = [rx.get_position(frame_index).lon for rx in receivers]

    # Plot transmitters (red triangles)
    ax.scatter(tx_lons, tx_lats, c="red", marker="^", s=200, label="Transmitters", zorder=5)
    for tx, lat, lon in zip(transmitters, tx_lats, tx_lons):
        ax.text(lon + 0.0004, lat + 0.0004, tx.identifier, fontsize=10, color="red")

    # Plot receivers (blue circles)
    ax.scatter(rx_lons, rx_lats, c="blue", marker="o", s=200, label="Receivers", zorder=5)
    for rx, lat, lon in zip(receivers, rx_lats, rx_lons):
        ax.text(lon + 0.0004, lat - 0.0004, rx.identifier, fontsize=10, color="blue")

    # Get connection pairs based on topology or full mesh
    connection_pairs = []
    if topology is not None:
        for (tx_id, rx_id), info in topology.items():
            tx = info["transmitter"]
            rx = info["receiver"]
            connection_pairs.append((tx, rx))
    else:
        for tx in transmitters:
            for rx in receivers:
                connection_pairs.append((tx, rx))

    # Draw connections if requested
    if connections:
        for tx, rx in connection_pairs:
            tx_pos = tx.get_position(frame_index)
            rx_pos = rx.get_position(frame_index)
            ax.plot(
                [tx_pos.lon, rx_pos.lon],
                [tx_pos.lat, rx_pos.lat],
                "k--",
                alpha=0.3,
                linewidth=0.5,
            )

    # Calculate and display distances in meters for this configuration
    for tx, rx in connection_pairs:
        tx_pos = tx.get_position(frame_index)
        rx_pos = rx.get_position(frame_index)
        distance = tx_pos.distance_to(rx_pos)
        mid_lat = (tx_pos.lat + rx_pos.lat) / 2
        mid_lon = (tx_pos.lon + rx_pos.lon) / 2
        ax.text(
            mid_lon,
            mid_lat,
            f"{distance / 1000:.1f}km",
            fontsize=8,
            ha="center",
            va="center",
            alpha=0.7,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Frame {frame_index}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created_ax:
        plt.tight_layout()
        plt.show()

    return ax


def collect_multi_frame_data(
    geo_dataset: Any,
    num_frames: int,
    num_receivers: int | None = None,
) -> list[list[Any]]:
    """Collect samples from multiple frames of a geo dataset.

    Each frame contains one sample per receiver.

    Args:
        geo_dataset: A TorchSigGeoDataset instance.
        num_frames: Number of frames to collect.
        num_receivers: Number of receivers (if None, uses len(geo_dataset.receivers)).

    Returns:
        A list of lists: frames[frame_idx][rx_idx] contains the sample for
        receiver rx_idx at frame frame_idx.
    """
    if num_receivers is None:
        num_receivers = len(geo_dataset.receivers)

    frames = []
    for _ in range(num_frames):
        frame_samples = []
        for rx_idx in range(num_receivers):
            sample = next(geo_dataset)
            sample["receiver_index"] = rx_idx
            frame_samples.append(sample)
        frames.append(frame_samples)

    return frames


def lla_to_local_cartesian(
    reference_lat: float,
    reference_lon: float,
    lats: list[float],
    lons: list[float],
    alts: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert lat/lon coordinates to local Cartesian (East, North) coordinates.

    Args:
        reference_lat: Reference latitude in degrees.
        reference_lon: Reference longitude in degrees.
        lats: List of latitudes in degrees.
        lons: List of longitudes in degrees.
        alts: Optional list of altitudes in meters (defaults to 0).

    Returns:
        Tuple of (eastings, northings, east_vec, north_vec).
    """
    from torchsig.geo.utils.coordinate_system import lla_to_ecef

    if alts is None:
        alts = np.zeros_like(lats)

    ref_x, ref_y, ref_z = lla_to_ecef(reference_lat, reference_lon, 0)
    points_ecef = np.array([lla_to_ecef(lat, lon, alt) for lat, lon, alt in zip(lats, lons, alts)])

    ref_lat_rad = np.radians(reference_lat)
    ref_lon_rad = np.radians(reference_lon)

    east_vec = np.array([-np.sin(ref_lon_rad), np.cos(ref_lon_rad), 0])
    north_vec = np.array(
        [
            -np.sin(ref_lat_rad) * np.cos(ref_lon_rad),
            -np.sin(ref_lat_rad) * np.sin(ref_lon_rad),
            np.cos(ref_lat_rad),
        ]
    )

    ecef_offsets = points_ecef - np.array([ref_x, ref_y, ref_z])
    eastings = np.dot(ecef_offsets, east_vec)
    northings = np.dot(ecef_offsets, north_vec)

    return eastings, northings, east_vec, north_vec


def estimate_position_from_ranges(
    receiver_positions: list[tuple[float, float, float]],
    delays: list[float],
    speed_of_light: float = 2.9979e8,
) -> tuple[float, float, float]:
    """Estimate transmitter position from absolute propagation delays using least squares (2D).

    This function performs **range-based multilateration**, NOT true TDOA.

    Range-based multilateration converts absolute propagation delays directly to ranges
    (distance = delay * speed_of_light) and solves a range-based localization problem
    using least squares.

    **Important distinction from true TDOA:**
    - True TDOA uses differences in arrival times BETWEEN receiver pairs, which cancels
      out the unknown transmit time. This does NOT require clock synchronization.
    - Range-based multilateration uses ABSOLUTE delays from transmitter to each receiver,
      which requires either synchronized clocks or knowledge of the absolute transmit time.

    Uses proper ECEF to LLA conversion to avoid flat-Earth approximation bias.

    Args:
        receiver_positions: List of (lat, lon, alt) tuples for each receiver.
        delays: List of propagation delays (in seconds) from transmitter to each receiver.
            These are absolute delays, not delay differences.
        speed_of_light: Speed of light in m/s (default: 2.9979e8).

    Returns:
        Tuple of (estimated_lat, estimated_lon, estimated_alt) in degrees and meters.

    Raises:
        ValueError: If fewer than 3 receivers are provided.
    """
    from torchsig.geo.utils.coordinate_system import ecef_to_lla, lla_to_ecef

    if len(receiver_positions) != len(delays):
        raise ValueError(f"receiver_positions and delays must have the same length, got {len(receiver_positions)} and {len(delays)}")

    if len(receiver_positions) < 3:
        raise ValueError("Need at least 3 receivers for range-based multilateration")

    ref_lat = receiver_positions[0][0]
    ref_lon = receiver_positions[0][1]
    ref_alt = receiver_positions[0][2]

    lats = [p[0] for p in receiver_positions]
    lons = [p[1] for p in receiver_positions]
    alts = [p[2] for p in receiver_positions]

    eastings, northings, east_vec, north_vec = lla_to_local_cartesian(ref_lat, ref_lon, lats, lons, alts)
    distances = [d * speed_of_light for d in delays]

    d0 = distances[0]
    e0, n0 = eastings[0], northings[0]

    # Build least squares problem for 2D localization (multilateration)
    A = []
    b = []

    for i in range(1, len(receiver_positions)):
        ei, ni = eastings[i], northings[i]
        di = distances[i]
        A.append([2 * (ei - e0), 2 * (ni - n0)])
        b.append((d0**2 - di**2) + (ei**2 + ni**2) - (e0**2 + n0**2))

    A = np.array(A)
    b = np.array(b)

    x_sol, y_sol = np.linalg.lstsq(A, b, rcond=None)[0]

    # Convert local Cartesian solution back to ECEF
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    # Add the offset in ECEF space
    est_ecef_x = ref_x + x_sol * east_vec[0] + y_sol * north_vec[0]
    est_ecef_y = ref_y + x_sol * east_vec[1] + y_sol * north_vec[1]
    est_ecef_z = ref_z + x_sol * east_vec[2] + y_sol * north_vec[2]

    # Convert back to LLA
    est_lat, est_lon, est_alt = ecef_to_lla(est_ecef_x, est_ecef_y, est_ecef_z)

    # For 2D TDOA with 3 receivers, altitude is unobservable
    # Use mean receiver altitude as best estimate
    est_alt = float(np.mean(alts))

    return est_lat, est_lon, est_alt


def calculate_tdoa_fix(
    all_frames: list[list[Any]],
    transmitter_idx: int,
    transmitters: list[Transmitter],
    speed_of_light: float = 2.9979e8,
) -> tuple[float | None, float | None, float | None, list[tuple[float, float, float]]]:
    """Calculate range-based position fix averaged over multiple frames for a specific transmitter.

    **IMPORTANT: Despite the name, this function performs RANGE-BASED MULTILATERATION,
    NOT true TDOA.**

    True TDOA (Time Difference of Arrival) uses differences in arrival times between
    receiver pairs: tdoa_ij = delay_i - delay_j. The unknown transmit time cancels out
    when taking differences, so no clock synchronization is required.

    This function instead uses ABSOLUTE propagation delays (delay = distance / c) from
    the transmitter to each receiver, which is range-based multilateration. This requires
    either synchronized clocks or knowledge of the absolute transmit time.

    For true TDOA, you would need a different solver that operates on delay differences
    between receiver pairs and solves the resulting hyperbolic multilateration problem
    (e.g., using Chan's algorithm or a Taylor series expansion).

    Args:
        all_frames: List of frames from collect_multi_frame_data().
        transmitter_idx: Index of the transmitter to locate.
        transmitters: List of Transmitter objects (for identifier lookup).
        speed_of_light: Speed of light in m/s.

    Returns:
        Tuple of (avg_lat, avg_lon, avg_alt, frame_estimates).
        Returns (None, None, None, []) if estimation fails for all frames.
    """
    frame_estimates: list[tuple[float, float, float]] = []
    tx_id = transmitters[transmitter_idx].identifier

    for frame_samples in all_frames:
        receiver_positions: list[tuple[float, float, float]] = []
        delays: list[float] = []

        for sample in frame_samples:
            rx_lat = sample["rx_lat"]
            rx_lon = sample["rx_lon"]
            rx_alt = sample["rx_alt"]

            # Find the component signal for the specific transmitter we're locating
            # Only append position and delay together when we find a matching component
            # This prevents misalignment when topology is partial (not all TX->RX connected)
            for comp in sample.component_signals:
                if comp["tx_id"] == tx_id:
                    receiver_positions.append((rx_lat, rx_lon, rx_alt))
                    delays.append(comp["path_delay_seconds"])
                    break

        if len(delays) >= 3:
            try:
                est_lat, est_lon, est_alt = estimate_position_from_ranges(receiver_positions, delays, speed_of_light)
                frame_estimates.append((est_lat, est_lon, est_alt))
            except (np.linalg.LinAlgError, ValueError) as e:
                # Log numerical failures but continue with remaining frames
                # This allows partial results when some frames fail
                import warnings

                warnings.warn(
                    f"Failed to estimate position for frame: {e}. Continuing with remaining frames.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    if not frame_estimates:
        return None, None, None, []

    # Average all frame estimates
    avg_lat = float(np.mean([e[0] for e in frame_estimates]))
    avg_lon = float(np.mean([e[1] for e in frame_estimates]))
    avg_alt = float(np.mean([e[2] for e in frame_estimates]))

    return avg_lat, avg_lon, avg_alt, frame_estimates
