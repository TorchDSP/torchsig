"""Coordinate system conversion utilities for geolocation.

This module provides functions for converting between geographic coordinates
(latitude, longitude, altitude) and ECEF (Earth-Centered, Earth-Fixed) Cartesian
coordinates using the WGS84 ellipsoid model.

ECEF (Earth-Centered, Earth-Fixed) is a Cartesian coordinate system where:
- Origin is at Earth's center
- Z-axis points to North Pole
- X-axis points to (0 deg lat, 0 deg lon)
- Y-axis points to (0 deg lat, 90 deg lon)
"""

from __future__ import annotations

import numpy as np

# Geographic coordinate bounds
LAT_MIN = -90.0
LAT_MAX = 90.0
LON_MIN = -180.0
LON_MAX = 180.0

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_B = 6356752.314245  # Semi-minor axis (m)
WGS84_F = 1.0 / 298.257223563  # Flattening (exact value for WGS84)

# Derived parameters
WGS84_E_SQ = 2.0 * WGS84_F - WGS84_F**2  # Square of first eccentricity

# Numerical precision thresholds
POLE_EPSILON = 1e-10
CONVERGENCE_EPSILON = 1e-12

__all__ = [
    "CONVERGENCE_EPSILON",
    "LAT_MAX",
    "LAT_MIN",
    "LON_MAX",
    "LON_MIN",
    "POLE_EPSILON",
    "WGS84_A",
    "WGS84_B",
    "WGS84_E_SQ",
    "WGS84_F",
    "ecef_distance",
    "ecef_to_lla",
    "enu_to_ecef",
    "lla_to_ecef",
]


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
    """Convert geodetic coordinates (lat, lon, alt) to ECEF (x, y, z).

    Uses WGS84 ellipsoid model. Based on the standard geodetic to ECEF conversion
    algorithm (e.g., as described in "Astronomical Algorithms" by Jean Meeus).

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)
        alt: Altitude in meters above WGS84 ellipsoid (default: 0.0)

    Returns:
        tuple: (x, y, z) in meters

    Raises:
        ValueError: If lat is outside [-90, 90] range
        ValueError: If lon is outside [-180, 180] range
        ValueError: If any coordinate is not finite

    Example:
        >>> # A point at the equator, prime meridian
        >>> lla_to_ecef(0.0, 0.0, 0.0)
        (6378137.0, 0.0, 0.0)
        >>> # North Pole
        >>> lla_to_ecef(90.0, 0.0, 0.0)
        (0.0, 0.0, 6356752.314245)
    """
    # Validate finiteness first (before range checks, since NaN comparisons fail)
    if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
        raise ValueError(f"lla_to_ecef requires finite coordinates, got lat={lat}, lon={lon}, alt={alt}")
    # Validate input ranges
    if not LAT_MIN <= lat <= LAT_MAX:
        raise ValueError(f"lat must be between -90 and 90 degrees, got {lat}")
    if not LON_MIN <= lon <= LON_MAX:
        raise ValueError(f"lon must be between -180 and 180 degrees, got {lon}")

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate sin/cos
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Prime vertical radius of curvature (radius of curvature in the prime vertical)
    # This is the distance from center to surface along the normal at latitude lat
    n = WGS84_A / np.sqrt(1.0 - WGS84_E_SQ * sin_lat**2)

    # ECEF coordinates
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E_SQ) + alt) * sin_lat

    return (float(x), float(y), float(z))


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF (x, y, z) coordinates to geodetic (lat, lon, alt).

    Uses WGS84 ellipsoid model. Implements Bowring's method for accurate
    conversion, including handling of edge cases.

    Args:
        x: X-coordinate in meters
        y: Y-coordinate in meters
        z: Z-coordinate in meters

    Returns:
        tuple: (lat, lon, alt) in degrees, degrees, meters

    Raises:
        ValueError: If any coordinate is not finite

    Example:
        >>> ecef_to_lla(6378137.0, 0.0, 0.0)
        (0.0, 0.0, 0.0)
    """
    # Validate finiteness first (NaN/Inf comparisons fail, so check before any math)
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        raise ValueError(f"ecef_to_lla requires finite coordinates, got x={x}, y={y}, z={z}")
    # Calculate longitude (simple, no iteration needed)
    lon_rad = np.arctan2(y, x)
    lon = float(np.degrees(lon_rad))

    # Calculate radial distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Latitude and longitude are undefined at Earth's center
    if p == 0.0 and z == 0.0:
        raise ValueError("Geodetic coordinates are undefined at the ECEF origin")

    # Handle edge case: point at pole (p = 0)
    if p < POLE_EPSILON:
        if z >= 0:
            return (90.0, lon, z - WGS84_B)
        return (-90.0, lon, -z - WGS84_B)

    # Initial estimates for latitude and altitude
    # Use atan2 to get initial lat estimate
    lat_rad = np.arctan2(z, p * (1.0 - WGS84_E_SQ))

    # Iterate to refine latitude and altitude (Bowring's method)
    # Typically converges in 2-3 iterations
    for _ in range(5):
        sin_lat = np.sin(lat_rad)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E_SQ * sin_lat**2)
        alt = p / np.cos(lat_rad) - n
        lat_rad_new = np.arctan2(z, p * (1.0 - WGS84_E_SQ * n / (n + alt)))

        diff = np.abs(lat_rad_new - lat_rad)
        lat_rad = lat_rad_new
        if diff < CONVERGENCE_EPSILON:
            lat_rad = lat_rad_new
            break
    else:
        lat_rad = lat_rad_new
        raise RuntimeError(f"Bowring's method failed to converge after 5 iterations. Final difference: {diff:.2e} rad")

    lat = float(np.degrees(lat_rad))
    alt = float(alt)

    return (lat, lon, alt)


def ecef_distance(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> float:
    """Calculate 3D Euclidean distance between two geodetic points using ECEF conversion.

    This computes the true straight-line distance through 3D space by:
    1. Converting both points to ECEF Cartesian coordinates
    2. Computing the Euclidean distance between them

    This is the correct distance to use for free-space path loss calculations,
    as FSPL assumes a direct line-of-sight path between transmitter and receiver.

    Args:
        lat1: Latitude of first point in degrees (-90 to 90)
        lon1: Longitude of first point in degrees (-180 to 180)
        alt1: Altitude of first point in meters
        lat2: Latitude of second point in degrees (-90 to 90)
        lon2: Longitude of second point in degrees (-180 to 180)
        alt2: Altitude of second point in meters

    Returns:
        float: 3D Euclidean distance in meters

    Example:
        >>> # Distance between two points 1km apart at same altitude
        >>> ecef_distance(37.7749, -122.4194, 0, 37.7749, -122.4194, 1000)
        1000.0
    """
    x1, y1, z1 = lla_to_ecef(lat1, lon1, alt1)
    x2, y2, z2 = lla_to_ecef(lat2, lon2, alt2)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def enu_to_ecef(ref_lat: float, ref_lon: float, east: float, north: float, up: float) -> tuple[float, float, float]:
    """Convert ENU (East, North, Up) vector to ECEF vector.

    ENU is a local Cartesian coordinate system where:
    - East: Tangent to the parallel, pointing east
    - North: Tangent to the meridian, pointing north
    - Up: Normal to the ellipsoid surface, pointing up

    The rotation from ENU to ECEF is the transpose of the ECEF to ENU rotation.
    ECEF to ENU matrix (R_e2n)::

        Row 0 (East):  [-sin_lon, cos_lon, 0]
        Row 1 (North): [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
        Row 2 (Up):    [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]

    ENU to ECEF matrix (R_n2e = R_e2n^T)::

        Col 0 (East):  [-sin_lon, cos_lon, 0]
        Col 1 (North): [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
        Col 2 (Up):    [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]

    Args:
        ref_lat: Reference latitude in degrees (-90 to 90)
        ref_lon: Reference longitude in degrees (-180 to 180)
        east: East component of vector in meters or m/s
        north: North component of vector in meters or m/s
        up: Up component of vector in meters or m/s

    Returns:
        tuple: (x, y, z) vector in ECEF frame in the same units as input

    Raises:
        ValueError: If ref_lat is outside [-90, 90] range
        ValueError: If ref_lon is outside [-180, 180] range
        ValueError: If any coordinate is not finite

    Example:
        >>> # At equator, prime meridian: 100 m/s east -> (0, 100, 0) in ECEF
        >>> enu_to_ecef(0.0, 0.0, 100.0, 0.0, 0.0)
        (0.0, 100.0, 0.0)
        >>> # At equator, prime meridian: 100 m/s north -> (0, 0, 100) in ECEF
        >>> enu_to_ecef(0.0, 0.0, 0.0, 100.0, 0.0)
        (0.0, 0.0, 100.0)
    """
    # Validate finiteness first (before range checks, since NaN comparisons fail)
    if not (np.isfinite(ref_lat) and np.isfinite(ref_lon) and np.isfinite(east) and np.isfinite(north) and np.isfinite(up)):
        raise ValueError(f"enu_to_ecef requires finite coordinates, got ref_lat={ref_lat}, ref_lon={ref_lon}, east={east}, north={north}, up={up}")
    # Validate input ranges for geographic coordinates
    if not LAT_MIN <= ref_lat <= LAT_MAX:
        raise ValueError(f"ref_lat must be between -90 and 90 degrees, got {ref_lat}")
    if not LON_MIN <= ref_lon <= LON_MAX:
        raise ValueError(f"ref_lon must be between -180 and 180 degrees, got {ref_lon}")

    # Convert to radians
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # ENU to ECEF rotation matrix (transpose of ECEF to ENU)
    # Column 0: ENU East basis in ECEF
    r00 = -sin_lon
    r10 = cos_lon
    r20 = 0.0

    # Column 1: ENU North basis in ECEF
    r01 = -sin_lat * cos_lon
    r11 = -sin_lat * sin_lon
    r21 = cos_lat

    # Column 2: ENU Up basis in ECEF
    r02 = cos_lat * cos_lon
    r12 = cos_lat * sin_lon
    r22 = sin_lat

    # Apply rotation: ecef = R * enu
    x = r00 * east + r01 * north + r02 * up
    y = r10 * east + r11 * north + r12 * up
    z = r20 * east + r21 * north + r22 * up

    return (float(x), float(y), float(z))
