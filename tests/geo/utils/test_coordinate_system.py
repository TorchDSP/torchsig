"""Tests for ECEF coordinate system conversions.

This module tests:
- LLA to ECEF conversion
- ECEF to LLA conversion
- ECEF distance calculations
- Round-trip conversions
- Error handling for invalid coordinates
- Edge cases (poles, equator, date line)
- Numerical precision and accuracy
"""

import numpy as np
import pytest

from torchsig.geo.utils.coordinate_system import (
    WGS84_A,
    WGS84_B,
    WGS84_E_SQ,
    WGS84_F,
    ecef_distance,
    ecef_to_lla,
    enu_to_ecef,
    lla_to_ecef,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def origin_lla():
    """Origin point (0, 0, 0) in LLA."""
    return (0.0, 0.0, 0.0)


@pytest.fixture
def origin_ecef():
    """Origin point in ECEF (WGS84_A, 0, 0)."""
    return (WGS84_A, 0.0, 0.0)


@pytest.fixture
def north_pole_lla():
    """North Pole in LLA."""
    return (90.0, 0.0, 0.0)


@pytest.fixture
def north_pole_ecef():
    """North Pole in ECEF."""
    return (0.0, 0.0, WGS84_B)


@pytest.fixture
def south_pole_lla():
    """South Pole in LLA."""
    return (-90.0, 0.0, 0.0)


@pytest.fixture
def south_pole_ecef():
    """South Pole in ECEF."""
    return (0.0, 0.0, -WGS84_B)


@pytest.fixture
def sf_lla():
    """San Francisco in LLA."""
    return (37.7749, -122.4194, 10.0)


@pytest.fixture
def equator_prime_meridian():
    """Point on equator at prime meridian."""
    return (0.0, 0.0, 0.0)


@pytest.fixture
def international_date_line():
    """Point on equator at International Date Line."""
    return (0.0, 180.0, 0.0)


# =============================================================================
# Helper Functions
# =============================================================================


def assert_ecef_approx(actual, expected, rel=1e-6, abs_tol=1e-6):
    """Helper to assert ECEF coordinates match expected values."""
    assert actual[0] == pytest.approx(expected[0], rel=rel, abs=abs_tol)
    assert actual[1] == pytest.approx(expected[1], rel=rel, abs=abs_tol)
    assert actual[2] == pytest.approx(expected[2], rel=rel, abs=abs_tol)


def assert_lla_approx(actual, expected, lat_abs=1e-6, lon_abs=1e-6, alt_abs=1e-3):
    """Helper to assert LLA coordinates match expected values.

    Altitude has lower precision than lat/lon.
    """
    assert actual[0] == pytest.approx(expected[0], abs=lat_abs)
    assert actual[1] == pytest.approx(expected[1], abs=lon_abs)
    assert actual[2] == pytest.approx(expected[2], abs=alt_abs)


# =============================================================================
# Constants Tests
# =============================================================================


class TestWGS84Constants:
    """Tests for WGS84 ellipsoid constants."""

    def test_wgs84_a_is_positive(self):
        """Test that semi-major axis is positive."""
        assert WGS84_A > 0

    def test_wgs84_b_is_positive(self):
        """Test that semi-minor axis is positive."""
        assert WGS84_B > 0

    def test_wgs84_a_greater_than_b(self):
        """Test that semi-major axis > semi-minor axis (Earth is oblate)."""
        assert WGS84_A > WGS84_B

    def test_wgs84_f_is_positive(self):
        """Test that flattening is positive."""
        assert WGS84_F > 0

    def test_wgs84_f_is_small(self):
        """Test that flattening is small (Earth is nearly spherical)."""
        assert WGS84_F < 0.01

    def test_wgs84_e_sq_derived_correctly(self):
        """Test that WGS84_E_SQ is correctly derived from WGS84_F."""
        expected_e_sq = 2.0 * WGS84_F - WGS84_F**2
        assert pytest.approx(expected_e_sq) == WGS84_E_SQ


# =============================================================================
# LLA to ECEF Conversion Tests
# =============================================================================


class TestLlaToEcef:
    """Tests for LLA to ECEF conversion."""

    def test_origin(self, origin_lla, origin_ecef):
        """Test ECEF conversion at (0, 0, 0)."""
        lat, lon, alt = origin_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        assert_ecef_approx((x, y, z), origin_ecef)

    def test_north_pole(self, north_pole_lla, north_pole_ecef):
        """Test ECEF conversion at North Pole."""
        lat, lon, alt = north_pole_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        assert_ecef_approx((x, y, z), north_pole_ecef, abs_tol=1e-6)

    def test_south_pole(self, south_pole_lla, south_pole_ecef):
        """Test ECEF conversion at South Pole."""
        lat, lon, alt = south_pole_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        assert_ecef_approx((x, y, z), south_pole_ecef, abs_tol=1e-6)

    def test_equator_prime_meridian(self, equator_prime_meridian, origin_ecef):
        """Test ECEF conversion at equator, prime meridian."""
        lat, lon, alt = equator_prime_meridian
        x, y, z = lla_to_ecef(lat, lon, alt)
        assert_ecef_approx((x, y, z), origin_ecef)

    def test_international_date_line(self, international_date_line):
        """Test ECEF conversion at International Date Line."""
        lat, lon, alt = international_date_line
        x, y, z = lla_to_ecef(lat, lon, alt)
        # At 180 degrees, x should be negative, y should be near zero
        assert x == pytest.approx(-WGS84_A, rel=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)
        assert z == pytest.approx(0.0, abs=1e-6)

    def test_with_altitude(self):
        """Test ECEF conversion with non-zero altitude."""
        x1, y1, z1 = lla_to_ecef(0.0, 0.0, 0.0)
        x2, y2, z2 = lla_to_ecef(0.0, 0.0, 1000.0)

        # With altitude, point should be further from center
        assert x2 > x1
        assert y2 == y1
        assert z2 == z1

        # Distance should be exactly 1000m in x-direction
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        assert distance == pytest.approx(1000.0, rel=1e-6)

    @pytest.mark.parametrize("alt", [0.0, 100.0, 1000.0])
    def test_altitude_scaling(self, alt):
        """Test that altitude scales correctly in ECEF."""
        x1, y1, z1 = lla_to_ecef(45.0, 45.0, 0.0)
        x2, y2, z2 = lla_to_ecef(45.0, 45.0, alt)

        # The vector from (lat,lon,0) to (lat,lon,alt) should have length alt
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        assert distance == pytest.approx(alt, rel=1e-6)

    @pytest.mark.parametrize(
        ("lon", "expected_x", "expected_y"),
        [
            (90.0, 0.0, WGS84_A),
            (-90.0, 0.0, -WGS84_A),
            (180.0, -WGS84_A, 0.0),
        ],
    )
    def test_equator_various_longitudes(self, lon, expected_x, expected_y):
        """Test ECEF conversion at equator with various longitudes."""
        x, y, z = lla_to_ecef(0.0, lon, 0.0)
        assert x == pytest.approx(expected_x, abs=1e-6)
        assert y == pytest.approx(expected_y, abs=1e-6)
        assert z == pytest.approx(0.0, abs=1e-6)

    def test_san_francisco(self, sf_lla):
        """Test ECEF conversion for San Francisco coordinates."""
        lat, lon, alt = sf_lla
        x, y, z = lla_to_ecef(lat, lon, alt)

        # San Francisco is in the northern and western hemisphere
        # x and y should be negative (west of prime meridian)
        # z should be positive (north of equator)
        # All should be within Earth's radius
        assert abs(x) < WGS84_A * 1.1
        assert abs(y) < WGS84_A * 1.1
        assert abs(z) < WGS84_A * 1.1

        # Magnitude should be approximately Earth's radius + altitude
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        assert magnitude == pytest.approx(np.sqrt(WGS84_A**2 + alt**2), rel=0.01)

    def test_returns_tuple_of_floats(self):
        """Test that lla_to_ecef returns a tuple of floats."""
        result = lla_to_ecef(45.0, -45.0, 100.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.parametrize("lat", [90.0001, -90.0001, 91, -91, 180, -180])
    def test_invalid_lat_raises(self, lat):
        """Test that invalid latitude raises ValueError."""
        with pytest.raises(ValueError, match="lat must be between"):
            lla_to_ecef(lat, 0.0, 0.0)

    @pytest.mark.parametrize("lon", [180.0001, -180.0001, 181, -181, 360, -360])
    def test_invalid_lon_raises(self, lon):
        """Test that invalid longitude raises ValueError."""
        with pytest.raises(ValueError, match="lon must be between"):
            lla_to_ecef(0.0, lon, 0.0)

    @pytest.mark.parametrize(("lat", "expected_x", "expected_y"), [(90.0, 0.0, 0.0), (-90.0, 0.0, 0.0)])
    def test_lat_boundary_exactly_poles(self, lat, expected_x, expected_y):
        """Test that latitude of exactly +/-90 is valid."""
        # Should not raise
        x, y, _ = lla_to_ecef(lat, 0.0, 0.0)
        assert x == pytest.approx(expected_x, abs=1e-6)
        assert y == pytest.approx(expected_y, abs=1e-6)

    @pytest.mark.parametrize(("lon", "expected_x"), [(-180.0, -WGS84_A), (180.0, -WGS84_A)])
    def test_lon_boundary_exactly_180(self, lon, expected_x):
        """Test that longitude of exactly +/-180 is valid."""
        # Should not raise
        x, _, _ = lla_to_ecef(0.0, lon, 0.0)
        assert x == pytest.approx(expected_x, rel=1e-6)


# =============================================================================
# ECEF to LLA Conversion Tests
# =============================================================================


class TestEcefToLla:
    """Tests for ECEF to LLA conversion."""

    def test_origin(self, origin_ecef, origin_lla):
        """Test reverse conversion at origin."""
        x, y, z = origin_ecef
        lat, lon, alt = ecef_to_lla(x, y, z)
        assert_lla_approx((lat, lon, alt), origin_lla)

    def test_north_pole(self, north_pole_ecef):
        """Test reverse conversion at North Pole."""
        x, y, z = north_pole_ecef
        lat, _, alt = ecef_to_lla(x, y, z)
        assert lat == pytest.approx(90.0, abs=1e-6)
        assert alt == pytest.approx(0.0, abs=1e-3)

    def test_south_pole(self, south_pole_ecef):
        """Test reverse conversion at South Pole."""
        x, y, z = south_pole_ecef
        lat, _, alt = ecef_to_lla(x, y, z)
        assert lat == pytest.approx(-90.0, abs=1e-6)
        assert alt == pytest.approx(0.0, abs=1e-3)

    def test_equator_prime_meridian(self, origin_ecef):
        """Test reverse conversion at equator, prime meridian."""
        x, y, z = origin_ecef
        lat, lon, _ = ecef_to_lla(x, y, z)
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon == pytest.approx(0.0, abs=1e-6)

    def test_returns_tuple_of_floats(self):
        """Test that ecef_to_lla returns a tuple of floats."""
        result = ecef_to_lla(WGS84_A, 0.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.parametrize(
        ("x", "y", "z"),
        [
            (float("nan"), 0.0, 0.0),
            (0.0, float("nan"), 0.0),
            (0.0, 0.0, float("nan")),
            (float("inf"), 0.0, 0.0),
            (0.0, float("inf"), 0.0),
            (0.0, 0.0, float("inf")),
            (float("-inf"), 0.0, 0.0),
            (0.0, float("-inf"), 0.0),
            (0.0, 0.0, float("-inf")),
        ],
    )
    def test_non_finite_raises(self, x, y, z):
        """Test that non-finite ECEF coordinates raise ValueError."""
        with pytest.raises(ValueError, match="ecef_to_lla requires finite coordinates"):
            ecef_to_lla(x, y, z)

    def test_bowring_convergence_error(self, monkeypatch):
        """Test that RuntimeError is raised when Bowring's method fails to converge.

        We patch CONVERGENCE_EPSILON to 0 to force non-convergence within the 5
        iteration limit, since diff < 0 will never be true.
        """
        import torchsig.geo.utils.coordinate_system as cs_module

        # Patch the convergence threshold to 0 so iteration never converges
        monkeypatch.setattr(cs_module, "CONVERGENCE_EPSILON", 0.0)

        # Use a normal coordinate - with epsilon=0, it won't converge
        with pytest.raises(RuntimeError, match="Bowring's method failed to converge"):
            ecef_to_lla(WGS84_A, 0.0, 0.0)


# =============================================================================
# Round-Trip Conversion Tests
# =============================================================================


class TestRoundTripConversions:
    """Tests for round-trip LLA <-> ECEF conversions."""

    def test_round_trip_origin(self, origin_lla):
        """Test round-trip at origin."""
        lat, lon, alt = origin_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)
        assert_lla_approx((lat2, lon2, alt2), (lat, lon, alt))

    def test_round_trip_north_pole(self, north_pole_lla):
        """Test round-trip at North Pole."""
        lat, lon, alt = north_pole_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, _, alt2 = ecef_to_lla(x, y, z)

        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert alt2 == pytest.approx(alt, abs=1e-3)

    def test_round_trip_south_pole(self, south_pole_lla):
        """Test round-trip at South Pole."""
        lat, lon, alt = south_pole_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, _, alt2 = ecef_to_lla(x, y, z)

        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert alt2 == pytest.approx(alt, abs=1e-3)

    @pytest.mark.parametrize("alt", [0.0, 100.0, 200.0])
    def test_round_trip_poles_with_altitude(self, alt):
        """Test round-trip conversion at poles with non-zero altitude."""
        for pole_lat in [90.0, -90.0]:
            x, y, z = lla_to_ecef(pole_lat, 0.0, alt)
            lat, _, retrieved_alt = ecef_to_lla(x, y, z)
            assert lat == pytest.approx(pole_lat, abs=1e-6)
            assert retrieved_alt == pytest.approx(alt, abs=1e-3)

    @pytest.mark.parametrize("lon", [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0])
    def test_round_trip_equator_various_longitudes(self, lon):
        """Test round-trip conversion at equator with various longitudes."""
        x, y, z = lla_to_ecef(0.0, lon, 50.0)
        lat, lon2, alt = ecef_to_lla(x, y, z)
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon2 == pytest.approx(lon, abs=1e-6)
        assert alt == pytest.approx(50.0, abs=1e-3)

    @pytest.mark.parametrize(
        "lat",
        [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, -15.0, -30.0, -45.0, -60.0, -75.0],
    )
    def test_round_trip_various_latitudes(self, lat):
        """Test round-trip conversion at various latitudes."""
        x, y, z = lla_to_ecef(lat, 45.0, 100.0)
        lat2, retrieved_lon, alt = ecef_to_lla(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert retrieved_lon == pytest.approx(45.0, abs=1e-6)
        assert alt == pytest.approx(100.0, abs=1e-3)

    def test_round_trip_san_francisco(self, sf_lla):
        """Test round-trip conversion for San Francisco."""
        lat, lon, alt = sf_lla
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)
        assert_lla_approx((lat2, lon2, alt2), (lat, lon, alt))

    def test_round_trip_random_points(self):
        """Test round-trip conversion for multiple random points."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            lat = rng.uniform(-90, 90)
            lon = rng.uniform(-180, 180)
            alt = rng.uniform(-1000, 10000)

            x, y, z = lla_to_ecef(lat, lon, alt)
            lat2, lon2, alt2 = ecef_to_lla(x, y, z)

            assert lat2 == pytest.approx(lat, abs=1e-6), f"Failed for lat={lat}"
            assert lon2 == pytest.approx(lon, abs=1e-6), f"Failed for lon={lon}"
            assert alt2 == pytest.approx(alt, abs=1e-3), f"Failed for alt={alt}"


# =============================================================================
# ECEF Distance Tests
# =============================================================================


class TestEcefDistance:
    """Tests for ECEF distance calculations."""

    def test_same_point(self, sf_lla):
        """Test that distance to same point is zero."""
        lat, lon, alt = sf_lla
        dist = ecef_distance(lat, lon, alt, lat, lon, alt)
        assert dist == 0.0

    def test_origin_to_origin(self, origin_lla):
        """Test distance from origin to itself."""
        lat, lon, alt = origin_lla
        dist = ecef_distance(lat, lon, alt, lat, lon, alt)
        assert dist == 0.0

    @pytest.mark.parametrize("alt_diff", [1.0, 10.0, 100.0, 1000.0])
    def test_vertical_only(self, alt_diff):
        """Test distance for vertical separation only."""
        dist = ecef_distance(37.7749, -122.4194, 0, 37.7749, -122.4194, alt_diff)
        assert np.isclose(dist, alt_diff, rtol=1e-6)

    @pytest.mark.parametrize("lat", [0.0, 45.0, 90.0, -45.0, -90.0])
    @pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, -90.0])
    def test_vertical_distance_various(self, lat, lon):
        """Test vertical distance at various locations."""
        dist = ecef_distance(lat, lon, 0, lat, lon, 1000)
        assert dist == pytest.approx(1000.0, rel=1e-6)

    def test_antipodal_points(self):
        """Test distance for antipodal points."""
        dist = ecef_distance(0.0, 0.0, 0, 0.0, 180.0, 0)
        expected = 2.0 * WGS84_A
        assert dist == pytest.approx(expected, rel=1e-4)

    def test_north_pole_to_south_pole(self):
        """Test distance from North Pole to South Pole."""
        dist = ecef_distance(90.0, 0.0, 0, -90.0, 0.0, 0)
        # Distance should be approximately the major axis length
        # (through the Earth, not along the surface)
        assert dist == pytest.approx(2.0 * WGS84_B, rel=1e-4)

    def test_symmetry(self):
        """Test that distance is symmetric: d(a,b) == d(b,a)."""
        dist_ab = ecef_distance(37.7749, -122.4194, 10, 40.7128, -74.0060, 20)
        dist_ba = ecef_distance(40.7128, -74.0060, 20, 37.7749, -122.4194, 10)
        assert dist_ab == pytest.approx(dist_ba)

    def test_positive_distance(self):
        """Test that distance is always non-negative."""
        dist = ecef_distance(37.7749, -122.4194, 10, 40.7128, -74.0060, 20)
        assert dist >= 0.0

    def test_distance_zero_altitude_difference(self):
        """Test distance with zero altitude difference."""
        dist = ecef_distance(37.7749, -122.4194, 100, 37.7749, -122.4194, 100)
        assert dist == 0.0

    @pytest.mark.parametrize(("alt1", "alt2", "expected"), [(0, 1000, 1000.0), (1000, 2000, 1000.0)])
    def test_distance_with_altitude_both_points(self, alt1, alt2, expected):
        """Test distance with both points having non-zero altitude."""
        dist = ecef_distance(0.0, 0.0, alt1, 0.0, 0.0, alt2)
        assert dist == pytest.approx(expected, rel=1e-6)

    def test_distance_equals_euclidean_in_ecef(self):
        """Test that ecef_distance equals Euclidean distance in ECEF space."""
        lat1, lon1, alt1 = 45.0, -45.0, 100.0
        lat2, lon2, alt2 = 46.0, -46.0, 200.0

        # Calculate using ecef_distance
        dist_direct = ecef_distance(lat1, lon1, alt1, lat2, lon2, alt2)

        # Calculate manually via ECEF conversion
        x1, y1, z1 = lla_to_ecef(lat1, lon1, alt1)
        x2, y2, z2 = lla_to_ecef(lat2, lon2, alt2)
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        dist_manual = np.sqrt(dx**2 + dy**2 + dz**2)

        assert dist_direct == pytest.approx(dist_manual, rel=1e-10)

    def test_distance_sf_to_ny_approximate(self):
        """Test that SF to NY distance is approximately 4124 km (straight-line)."""
        dist = ecef_distance(37.7749, -122.4194, 0, 40.7128, -74.0060, 0)
        # Straight-line distance through Earth is ~4.1 million meters
        assert 4_000_000 < dist < 4_300_000


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestNumericalPrecision:
    """Tests for numerical precision and stability."""

    def test_origin_magnitude(self):
        """Test that origin ECEF has correct magnitude."""
        x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        assert magnitude == pytest.approx(WGS84_A, rel=1e-6)

    def test_north_pole_magnitude(self):
        """Test that North Pole ECEF has correct magnitude."""
        x, y, z = lla_to_ecef(90.0, 0.0, 0.0)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        assert magnitude == pytest.approx(WGS84_B, rel=1e-6)

    def test_ecef_to_lla_origin_alt_precision(self):
        """Test altitude precision at origin."""
        x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
        _, _, alt = ecef_to_lla(x, y, z)
        # Altitude precision is lower than lat/lon
        assert abs(alt) < 1e-3  # Should be very close to zero

    @pytest.mark.parametrize("alt", [100_000, 35_786_000])
    def test_high_altitude_precision(self, alt):
        """Test precision at high altitudes."""
        x, y, z = lla_to_ecef(45.0, 45.0, alt)
        lat, lon, alt2 = ecef_to_lla(x, y, z)

        assert lat == pytest.approx(45.0, abs=1e-6)
        assert lon == pytest.approx(45.0, abs=1e-6)
        # ~10cm precision at 100km, ~1m precision at GEO
        assert alt2 == pytest.approx(alt, abs=max(0.1, alt * 1e-8))

    def test_small_coordinate_values(self):
        """Test conversion with very small coordinate values."""
        lat, lon, alt = 1e-10, 1e-10, 1e-10
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)

        # Small values should round-trip reasonably
        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert lon2 == pytest.approx(lon, abs=1e-6)
        assert alt2 == pytest.approx(alt, abs=1e-6)


# =============================================================================
# Edge Cases and Special Values
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special values."""

    @pytest.mark.parametrize("alt", [-100.0, -10000.0])
    def test_negative_altitude(self, alt):
        """Test conversion with negative altitude (below WGS84 ellipsoid)."""
        x, y, z = lla_to_ecef(45.0, 45.0, alt)
        lat, lon, retrieved_alt = ecef_to_lla(x, y, z)

        assert lat == pytest.approx(45.0, abs=1e-6)
        assert lon == pytest.approx(45.0, abs=1e-6)
        assert retrieved_alt == pytest.approx(alt, abs=0.1)

    @pytest.mark.parametrize("lat", [89.999, 89.99, 89.9, 89.0, -89.999, -89.99, -89.9, -89.0])
    def test_near_poles(self, lat):
        """Test conversion near the poles."""
        x, y, z = lla_to_ecef(lat, 45.0, 100.0)
        lat2, _, _ = ecef_to_lla(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-5)

    @pytest.mark.parametrize("lon", [179.999, 179.99, 179.9, -179.999, -179.99, -179.9])
    def test_near_date_line(self, lon):
        """Test conversion near International Date Line."""
        x, y, z = lla_to_ecef(45.0, lon, 100.0)
        _, lon2, _ = ecef_to_lla(x, y, z)
        # Longitude may wrap, so check modulo 360
        assert lon2 == pytest.approx(lon, abs=1e-5) or abs(lon2 - lon) % 360 < 1e-5

    def test_exactly_zero_values(self):
        """Test conversion with exactly zero values."""
        x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
        assert x == pytest.approx(WGS84_A, rel=1e-6)
        assert y == 0.0
        assert z == 0.0

    @pytest.mark.parametrize("lat", [-90.0, 0.0, 90.0])
    def test_multiple_of_90_latitudes(self, lat):
        """Test conversion at latitudes that are multiples of 90."""
        x, y, z = lla_to_ecef(lat, 45.0, 0.0)
        lat2, _, _ = ecef_to_lla(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-6)

    @pytest.mark.parametrize("lon", [-180.0, -90.0, 0.0, 90.0, 180.0])
    def test_multiple_of_90_longitudes(self, lon):
        """Test conversion at longitudes that are multiples of 90."""
        x, y, z = lla_to_ecef(45.0, lon, 0.0)
        _, lon2, _ = ecef_to_lla(x, y, z)
        # Handle longitude wrapping
        if lon in {-180.0, 180.0}:
            assert abs(lon2) == pytest.approx(180.0, abs=1e-6)
        else:
            assert lon2 == pytest.approx(lon, abs=1e-6)


# =============================================================================
# ENU to ECEF Conversion Tests
# =============================================================================


class TestEnuToEcef:
    """Tests for ENU (East, North, Up) to ECEF vector conversion."""

    def test_origin_enu_to_ecef(self):
        """Test ENU to ECEF at origin (0, 0)."""
        ref_lat, ref_lon = 0.0, 0.0

        # At equator, prime meridian: East -> Y, North -> Z, Up -> X
        x, y, z = enu_to_ecef(ref_lat, ref_lon, 1.0, 0.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

        x, y, z = enu_to_ecef(ref_lat, ref_lon, 0.0, 1.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(1.0, abs=1e-10)

        x, y, z = enu_to_ecef(ref_lat, ref_lon, 0.0, 0.0, 1.0)
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_north_pole_enu_to_ecef(self):
        """Test ENU to ECEF at North Pole."""
        ref_lat, ref_lon = 90.0, 0.0

        # At North Pole, Up should map to positive Z (ECEF)
        x, y, z = enu_to_ecef(ref_lat, ref_lon, 0.0, 0.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)
        assert z == pytest.approx(1.0, abs=1e-6)

    def test_equator_90e_longitude(self):
        """Test ENU to ECEF at equator, 90 degrees east."""
        ref_lat, ref_lon = 0.0, 90.0

        # At equator, 90E: East -> -X, North -> Z, Up -> Y
        x, y, z = enu_to_ecef(ref_lat, ref_lon, 1.0, 0.0, 0.0)
        assert x == pytest.approx(-1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

        x, y, z = enu_to_ecef(ref_lat, ref_lon, 0.0, 1.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(1.0, abs=1e-10)

        x, y, z = enu_to_ecef(ref_lat, ref_lon, 0.0, 0.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_san_francisco_enu_to_ecef(self):
        """Test ENU to ECEF at San Francisco coordinates."""
        ref_lat, ref_lon = 37.7749, -122.4194

        # Test that ENU vectors are orthogonal
        east = np.array(enu_to_ecef(ref_lat, ref_lon, 1.0, 0.0, 0.0))
        north = np.array(enu_to_ecef(ref_lat, ref_lon, 0.0, 1.0, 0.0))
        up = np.array(enu_to_ecef(ref_lat, ref_lon, 0.0, 0.0, 1.0))

        # ENU basis vectors should be approximately orthogonal (unit vectors)
        assert np.abs(np.dot(east, north)) < 1e-10
        assert np.abs(np.dot(east, up)) < 1e-10
        assert np.abs(np.dot(north, up)) < 1e-10

        # Each should have unit length
        assert np.linalg.norm(east) == pytest.approx(1.0, rel=1e-10)
        assert np.linalg.norm(north) == pytest.approx(1.0, rel=1e-10)
        assert np.linalg.norm(up) == pytest.approx(1.0, rel=1e-10)

    def test_returns_tuple_of_floats(self):
        """Test that enu_to_ecef returns a tuple of floats."""
        result = enu_to_ecef(45.0, -45.0, 100.0, 50.0, 25.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_zero_vector(self):
        """Test that zero ENU vector returns zero ECEF vector."""
        x, y, z = enu_to_ecef(45.0, -45.0, 0.0, 0.0, 0.0)
        assert x == 0.0
        assert y == 0.0
        assert z == 0.0

    def test_linearity(self):
        """Test that enu_to_ecef is linear."""
        ref_lat, ref_lon = 30.0, -60.0

        # Test scaling
        x1, y1, z1 = enu_to_ecef(ref_lat, ref_lon, 1.0, 2.0, 3.0)
        x2, y2, z2 = enu_to_ecef(ref_lat, ref_lon, 2.0, 4.0, 6.0)
        assert x2 == pytest.approx(2.0 * x1, rel=1e-10)
        assert y2 == pytest.approx(2.0 * y1, rel=1e-10)
        assert z2 == pytest.approx(2.0 * z1, rel=1e-10)

        # Test additivity
        x_a, y_a, z_a = enu_to_ecef(ref_lat, ref_lon, 1.0, 0.0, 0.0)
        x_b, y_b, z_b = enu_to_ecef(ref_lat, ref_lon, 0.0, 1.0, 0.0)
        x_c, y_c, z_c = enu_to_ecef(ref_lat, ref_lon, 1.0, 1.0, 0.0)
        assert x_c == pytest.approx(x_a + x_b, rel=1e-10)
        assert y_c == pytest.approx(y_a + y_b, rel=1e-10)
        assert z_c == pytest.approx(z_a + z_b, rel=1e-10)

    def test_velocity_conversion_sf(self):
        """Test ENU velocity to ECEF velocity conversion for San Francisco."""
        ref_lat, ref_lon = 37.7749, -122.4194

        # Test 100 m/s east velocity
        v_east = 100.0
        x, y, z = enu_to_ecef(ref_lat, ref_lon, v_east, 0.0, 0.0)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        assert magnitude == pytest.approx(v_east, rel=1e-10)

        # Test combined velocity
        v_e, v_n, v_u = 100.0, 50.0, 25.0
        x, y, z = enu_to_ecef(ref_lat, ref_lon, v_e, v_n, v_u)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        expected_magnitude = np.sqrt(v_e**2 + v_n**2 + v_u**2)
        assert magnitude == pytest.approx(expected_magnitude, rel=1e-10)

    @pytest.mark.parametrize("ref_lat", [90.0001, -90.0001, 91, -91, 180, -180])
    def test_invalid_ref_lat_raises(self, ref_lat):
        """Test that invalid reference latitude raises ValueError."""
        with pytest.raises(ValueError, match="ref_lat must be between"):
            enu_to_ecef(ref_lat, 0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize("ref_lon", [180.0001, -180.0001, 181, -181, 360, -360])
    def test_invalid_ref_lon_raises(self, ref_lon):
        """Test that invalid reference longitude raises ValueError."""
        with pytest.raises(ValueError, match="ref_lon must be between"):
            enu_to_ecef(0.0, ref_lon, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        ("ref_lat", "ref_lon", "east", "north", "up"),
        [
            (float("nan"), 0.0, 0.0, 0.0, 0.0),
            (0.0, float("nan"), 0.0, 0.0, 0.0),
            (0.0, 0.0, float("nan"), 0.0, 0.0),
            (0.0, 0.0, 0.0, float("nan"), 0.0),
            (0.0, 0.0, 0.0, 0.0, float("nan")),
            (float("inf"), 0.0, 0.0, 0.0, 0.0),
            (0.0, float("inf"), 0.0, 0.0, 0.0),
            (0.0, 0.0, float("inf"), 0.0, 0.0),
            (0.0, 0.0, 0.0, float("inf"), 0.0),
            (0.0, 0.0, 0.0, 0.0, float("inf")),
            (float("-inf"), 0.0, 0.0, 0.0, 0.0),
            (0.0, float("-inf"), 0.0, 0.0, 0.0),
            (0.0, 0.0, float("-inf"), 0.0, 0.0),
            (0.0, 0.0, 0.0, float("-inf"), 0.0),
            (0.0, 0.0, 0.0, 0.0, float("-inf")),
        ],
    )
    def test_non_finite_raises(self, ref_lat, ref_lon, east, north, up):
        """Test that non-finite coordinates raise ValueError."""
        with pytest.raises(ValueError, match="enu_to_ecef requires finite coordinates"):
            enu_to_ecef(ref_lat, ref_lon, east, north, up)


# =============================================================================
# Edge Cases and Special Values
# =============================================================================
