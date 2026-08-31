"""Geo types for representing physical locations and velocities.

This module provides the GeoPoint and GeoVelocity classes for working with
geographic coordinates (lat, lon, alt) and velocity vectors (east, north, up).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from torchsig.geo.utils.coordinate_system import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ecef_distance,
)

__all__ = ["VELOCITY_COMPONENTS", "GeoPoint", "GeoVelocity"]

# Number of components in a velocity vector (east, north, up)
VELOCITY_COMPONENTS = 3


class GeoPoint:  # noqa: PLW1641
    """A point in geographic space with lat, lon, and alt coordinates.

    This class represents a physical location on Earth using geographic coordinates
    and provides utilities for distance calculations between points.

    Attributes:
        lat: Lat in degrees, ranging from -90 (South Pole) to 90 (North Pole)
        lon: Lon in degrees, ranging from -180 to 180
        alt: Alt in meters above sea level (default: 0.0)

    Example:
        >>> # Create a point for San Francisco
        >>> sf = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        >>> # Create a point for New York
        >>> ny = GeoPoint(lat=40.7128, lon=-74.0060, alt=20)
        >>> # Calculate distance
        >>> distance = sf.distance_to(ny)
        >>> print(f"Distance: {distance:.1f} meters")
    """

    def __init__(self, lat: float, lon: float, alt: float = 0.0):
        """Initialize a GeoPoint.

        Args:
            lat: Latitude in degrees (LAT_MIN to LAT_MAX)
            lon: Longitude in degrees (LON_MIN to LON_MAX)
            alt: Altitude in meters (default: 0.0)

        Raises:
            ValueError: If lat is outside [LAT_MIN, LAT_MAX] range
            ValueError: If lon is outside [LON_MIN, LON_MAX] range
            ValueError: If alt is not finite
        """
        if not LAT_MIN <= lat <= LAT_MAX:
            raise ValueError(f"GeoPoint.lat must be between -90 and 90, got {lat}")
        if not LON_MIN <= lon <= LON_MAX:
            raise ValueError(f"GeoPoint.lon must be between -180 and 180, got {lon}")
        if not np.isfinite(alt):
            raise ValueError(f"GeoPoint.alt must be finite, got {alt}")

        self.lat = float(lat)
        self.lon = float(lon)
        self.alt = float(alt)

    def distance_to(self, other: GeoPoint) -> float:
        """Calculate the straight-line (3D) distance to another point in meters.

        Uses ECEF coordinate conversion for accurate 3D Euclidean distance.
        This is the true line-of-sight distance appropriate for free-space path
        loss calculations.

        Args:
            other: Another GeoPoint to calculate distance to

        Returns:
            float: Distance in meters (always non-negative and finite)

        Raises:
            TypeError: If other is not a GeoPoint
            ValueError: If the computed distance is not finite
        """
        if not isinstance(other, GeoPoint):
            raise TypeError(f"GeoPoint.distance_to requires a GeoPoint argument, got {type(other).__name__}")
        result = ecef_distance(self.lat, self.lon, self.alt, other.lat, other.lon, other.alt)
        if not np.isfinite(result):
            raise ValueError(f"GeoPoint.distance_to computed non-finite distance: {result}")
        return result

    def to_dict(self) -> dict[str, float]:
        """Convert the point to a dictionary representation.

        Returns:
            dict: Dictionary with keys 'lat', 'lon', 'alt'

        Example:
            >>> point = GeoPoint(37.7749, -122.4194, 100)
            >>> point.to_dict()
            {'lat': 37.7749, 'lon': -122.4194, 'alt': 100.0}
        """
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeoPoint:
        """Create a GeoPoint from a dictionary.

        Args:
            data: Dictionary with keys 'lat', 'lon', and optionally 'alt'

        Returns:
            GeoPoint: A new point instance

        Example:
            >>> data = {"lat": 37.7749, "lon": -122.4194, "alt": 100}
            >>> point = GeoPoint.from_dict(data)
        """
        return cls(
            lat=data["lat"],
            lon=data["lon"],
            alt=data.get("alt", 0.0),
        )

    def __repr__(self) -> str:
        """Return a string representation of the point.

        Returns:
            str: Formatted string with lat, lon, and alt
        """
        return f"GeoPoint(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt:.1f}m)"

    def __eq__(self, other: object) -> bool:
        """Check equality with another object.

        Args:
            other: Object to compare with

        Returns:
            bool: True if other is a GeoPoint with same coordinates
        """
        if not isinstance(other, GeoPoint):
            return False
        return self.lat == other.lat and self.lon == other.lon and self.alt == other.alt


class GeoVelocity:  # noqa: PLW1641
    """A velocity vector in the local East-North-Up (ENU) coordinate frame.

    This class provides a structured representation of velocity with east, north,
    and up components in meters per second. It mirrors the GeoPoint class structure
    for consistency and eliminates the asymmetry where position uses GeoPoint but
    velocity uses bare tuples.

    The ENU frame is a local Cartesian coordinate system where:
    - East: Positive in the direction of increasing longitude (east)
    - North: Positive in the direction of increasing latitude (north)
    - Up: Positive upward from the Earth's surface

    Attributes:
        east: Velocity component in the east direction (m/s)
        north: Velocity component in the north direction (m/s)
        up: Velocity component in the upward direction (m/s)

    Example:
        >>> # Create a velocity vector (10 m/s east, 5 m/s north, 0 m/s up)
        >>> vel = GeoVelocity(east=10.0, north=5.0, up=0.0)
        >>> print(vel)
        >>> # Convert to tuple for compatibility
        >>> vel.to_tuple()
        (10.0, 5.0, 0.0)
    """

    def __init__(self, east: float, north: float, up: float) -> None:
        """Initialize a GeoVelocity with east, north, and up components.

        Args:
            east: Velocity component in the east direction (m/s)
            north: Velocity component in the north direction (m/s)
            up: Velocity component in the upward direction (m/s)

        Raises:
            ValueError: If any component is not finite
        """
        self.east = float(east)
        self.north = float(north)
        self.up = float(up)

        if not (np.isfinite(self.east) and np.isfinite(self.north) and np.isfinite(self.up)):
            raise ValueError(f"GeoVelocity components must be finite, got east={self.east}, north={self.north}, up={self.up}")

    def __repr__(self) -> str:
        """Return a string representation of the GeoVelocity."""
        return f"GeoVelocity(east={self.east}, north={self.north}, up={self.up})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another object.

        Args:
            other: Object to compare with

        Returns:
            bool: True if other is a GeoVelocity with same components
        """
        if not isinstance(other, GeoVelocity):
            return False
        return self.east == other.east and self.north == other.north and self.up == other.up

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert GeoVelocity to a tuple of (east, north, up) components.

        Returns:
            tuple[float, float, float]: The velocity components as a tuple.
        """
        return (self.east, self.north, self.up)

    @classmethod
    def from_tuple(cls, velocity: tuple[float, float, float]) -> GeoVelocity:
        """Create a GeoVelocity from a tuple of (east, north, up) components.

        Args:
            velocity: The velocity components as a tuple.

        Returns:
            GeoVelocity: A new GeoVelocity instance.

        Raises:
            ValueError: If the tuple does not have exactly VELOCITY_COMPONENTS elements.
        """
        if not isinstance(velocity, tuple) or len(velocity) != VELOCITY_COMPONENTS:
            raise ValueError(
                f"Velocity tuple must have exactly 3 elements (east, north, up), got {type(velocity).__name__} with {len(velocity) if hasattr(velocity, '__len__') else 'unknown'} elements"
            )
        return cls(east=velocity[0], north=velocity[1], up=velocity[2])
