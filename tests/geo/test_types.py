"""Tests for GeoPoint and GeoVelocity classes.

This module tests:
- GeoPoint creation and validation
- Distance calculations
- Equality and hashing
- Serialization
- Edge cases and boundary conditions
- GeoVelocity creation and validation
- GeoVelocity equality and hashing
- GeoVelocity tuple conversion
"""

import numpy as np
import pytest

from torchsig.geo.types import GeoPoint, GeoVelocity
from torchsig.geo.utils.coordinate_system import WGS84_A, WGS84_B

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sf_point():
    """GeoPoint for San Francisco."""
    return GeoPoint(lat=37.7749, lon=-122.4194, alt=10)


@pytest.fixture
def ny_point():
    """GeoPoint for New York City."""
    return GeoPoint(lat=40.7128, lon=-74.0060, alt=20)


@pytest.fixture
def origin_point():
    """GeoPoint at (0, 0, 0)."""
    return GeoPoint(lat=0.0, lon=0.0, alt=0.0)


@pytest.fixture
def north_pole():
    """GeoPoint at North Pole."""
    return GeoPoint(lat=90.0, lon=0.0, alt=0.0)


@pytest.fixture
def south_pole():
    """GeoPoint at South Pole."""
    return GeoPoint(lat=-90.0, lon=0.0, alt=0.0)


# =============================================================================
# Helper Functions
# =============================================================================


def assert_point_equals(point, lat, lon, alt):
    """Helper to assert GeoPoint coordinates match expected values."""
    assert point.lat == lat
    assert point.lon == lon
    assert point.alt == alt


def assert_distance_symmetry(p1, p2):
    """Helper to assert distance calculation is symmetric."""
    assert p1.distance_to(p2) == pytest.approx(p2.distance_to(p1))


def assert_round_trip_serialization(original):
    """Helper to assert serialization round-trip preserves the point."""
    data = original.to_dict()
    restored = GeoPoint.from_dict(data)
    assert restored == original


# =============================================================================
# Creation and Validation Tests
# =============================================================================


class TestGeoPointCreation:
    """Tests for GeoPoint creation and validation."""

    def test_create_point_with_all_params(self, sf_point):
        """Test creating a GeoPoint with lat, lon, and alt."""
        assert_point_equals(sf_point, 37.7749, -122.4194, 10.0)

    def test_create_point_default_alt(self):
        """Test creating a GeoPoint with default altitude (0.0)."""
        point = GeoPoint(lat=37.7749, lon=-122.4194)
        assert point.alt == 0.0

    def test_create_point_explicit_zero_alt(self):
        """Test creating a GeoPoint with explicit zero altitude."""
        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=0.0)
        assert point.alt == 0.0

    def test_create_point_negative_alt(self):
        """Test creating a GeoPoint with negative altitude (below sea level)."""
        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=-100.0)
        assert point.alt == -100.0

    def test_create_point_float_coercion(self):
        """Test that integer coordinates are converted to float."""
        point = GeoPoint(lat=37, lon=-122, alt=10)
        assert isinstance(point.lat, float)
        assert isinstance(point.lon, float)
        assert isinstance(point.alt, float)

    @pytest.mark.parametrize("lat", [90.0, -90.0])
    def test_valid_lat_boundaries(self, lat):
        """Test that boundary latitudes (+/-90) are valid."""
        point = GeoPoint(lat=lat, lon=0.0)
        assert point.lat == lat

    @pytest.mark.parametrize("lat", [90.0001, -90.0001, 91, -91, 180, -180])
    def test_invalid_lat_too_high_or_low(self, lat):
        """Test that latitude outside [-90, 90] raises ValueError."""
        with pytest.raises(ValueError, match="GeoPoint.lat must be between -90 and 90"):
            GeoPoint(lat=lat, lon=0)

    @pytest.mark.parametrize("lon", [180.0, -180.0])
    def test_valid_lon_boundaries(self, lon):
        """Test that boundary longitudes (+/-180) are valid."""
        point = GeoPoint(lat=0.0, lon=lon)
        assert point.lon == lon

    @pytest.mark.parametrize("lon", [180.0001, -180.0001, 181, -181, 360, -360])
    def test_invalid_lon_too_high_or_low(self, lon):
        """Test that longitude outside [-180, 180] raises ValueError."""
        with pytest.raises(ValueError, match="GeoPoint.lon must be between -180 and 180"):
            GeoPoint(lat=0, lon=lon)

    def test_invalid_both_lat_and_lon(self):
        """Test that both invalid lat and lon raises ValueError for lat first."""
        with pytest.raises(ValueError, match="GeoPoint.lat must be between -90 and 90"):
            GeoPoint(lat=100, lon=200)


# =============================================================================
# Distance Calculation Tests
# =============================================================================


class TestDistanceCalculation:
    """Tests for distance calculation between GeoPoints."""

    def test_distance_to_same_point(self):
        """Test that distance to the same point is exactly zero."""
        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        assert point.distance_to(point) == 0.0

    def test_distance_to_self(self, sf_point):
        """Test that distance to self is zero."""
        assert sf_point.distance_to(sf_point) == 0.0

    def test_distance_to_invalid_type_raises(self):
        """Test that distance_to with non-GeoPoint raises TypeError."""
        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        with pytest.raises(TypeError, match="GeoPoint.distance_to requires a GeoPoint argument"):
            point.distance_to("invalid")

    def test_distance_to_with_none_raises(self):
        """Test that distance_to with None raises TypeError."""
        point = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        with pytest.raises(TypeError, match="GeoPoint.distance_to requires a GeoPoint argument"):
            point.distance_to(None)

    def test_distance_symmetry(self, sf_point, ny_point):
        """Test that distance calculation is symmetric: d(a,b) == d(b,a)."""
        assert_distance_symmetry(sf_point, ny_point)

    def test_distance_positive(self, sf_point, ny_point):
        """Test that distance between distinct points is positive."""
        assert sf_point.distance_to(ny_point) > 0.0

    def test_distance_non_negative(self, sf_point, ny_point):
        """Test that distance is always non-negative."""
        assert sf_point.distance_to(ny_point) >= 0.0

    def test_distance_to_vertical_only(self):
        """Test distance calculation for vertical separation only (same lat/lon)."""
        p1 = GeoPoint(lat=37.7749, lon=-122.4194, alt=0)
        p2 = GeoPoint(lat=37.7749, lon=-122.4194, alt=1000)
        distance = p1.distance_to(p2)
        assert np.isclose(distance, 1000.0, rtol=1e-6)

    @pytest.mark.parametrize(
        ("alt1", "alt2", "expected"),
        [(0, 100, 100.0), (0, 200, 200.0), (100, 200, 100.0)],
    )
    def test_distance_altitude_difference(self, alt1, alt2, expected):
        """Test that altitude difference contributes correctly to distance."""
        p1 = GeoPoint(lat=0, lon=0, alt=alt1)
        p2 = GeoPoint(lat=0, lon=0, alt=alt2)
        assert p1.distance_to(p2) == pytest.approx(expected, rel=1e-6)

    def test_distance_sf_to_ny_approximate(self, sf_point, ny_point):
        """Test that SF to NY distance is approximately 4124 km.

        Note: This is a straight-line (3D Euclidean) distance through the Earth,
        not the great-circle distance along the surface. For FSPL calculations,
        the straight-line distance is appropriate.
        """
        distance = sf_point.distance_to(ny_point)
        # SF to NY straight-line distance is approximately 4.1 million meters
        assert 4_000_000 < distance < 4_300_000

    def test_distance_equator_to_pole(self, origin_point, north_pole):
        """Test distance from equator to North Pole."""
        distance = origin_point.distance_to(north_pole)
        # Expected result: sqrt((WGS84_A - 0)^2 + (0 - 0)^2 + (0 - WGS84_B)^2)
        # But actually both are at surface, so it's the chord length
        expected = np.sqrt(WGS84_A**2 + WGS84_B**2)
        assert distance == pytest.approx(expected, rel=1e-4)

    def test_distance_antipodal_points(self):
        """Test distance between antipodal points (opposite sides of Earth)."""
        p1 = GeoPoint(lat=0.0, lon=0.0, alt=0)
        p2 = GeoPoint(lat=0.0, lon=180.0, alt=0)
        distance = p1.distance_to(p2)

        # Antipodal points at equator: distance should be ~2 * WGS84_A
        # (diameter of Earth at equator)
        assert distance == pytest.approx(2.0 * WGS84_A, rel=1e-4)

    def test_distance_with_different_altitudes(self):
        """Test distance calculation with different altitudes at same lat/lon."""
        p1 = GeoPoint(lat=45.0, lon=45.0, alt=0)
        p2 = GeoPoint(lat=45.0, lon=45.0, alt=10000)  # 10 km up
        distance = p1.distance_to(p2)
        # Should be exactly the altitude difference
        assert distance == pytest.approx(10000.0, rel=1e-6)

    @pytest.mark.parametrize("non_finite_value", [float("nan"), float("inf"), float("-inf")])
    def test_distance_to_non_finite_ecef_result_raises(self, monkeypatch, non_finite_value):
        """Test that non-finite ecef_distance result raises ValueError.

        This tests the defensive error handling in distance_to when ecef_distance
        returns a non-finite value (nan or inf). This is unreachable through normal
        operation since GeoPoint validates coordinates, but tested for completeness.
        """
        from unittest.mock import patch

        p1 = GeoPoint(lat=0.0, lon=0.0, alt=0.0)
        p2 = GeoPoint(lat=0.0, lon=0.0, alt=0.0)

        with patch("torchsig.geo.types.ecef_distance", return_value=non_finite_value), pytest.raises(ValueError, match="computed non-finite distance"):
            p1.distance_to(p2)


# =============================================================================
# Equality and Hashing Tests
# =============================================================================


class TestEquality:
    """Tests for equality comparison between GeoPoints."""

    @pytest.mark.parametrize(
        ("lat", "lon", "alt"),
        [(37.7749, -122.4194, 10), (0.0, 0.0, 0.0), (90.0, 0.0, 0.0), (-90.0, 180.0, 100)],
    )
    def test_equality_same_coordinates(self, lat, lon, alt):
        """Test that points with same coordinates are equal."""
        point1 = GeoPoint(lat=lat, lon=lon, alt=alt)
        point2 = GeoPoint(lat=lat, lon=lon, alt=alt)
        assert point1 == point2

    @pytest.mark.parametrize("alt_diff", [1, 0.1, -5])
    def test_equality_different_altitude(self, alt_diff):
        """Test that points with different altitudes are not equal."""
        point1 = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        point2 = GeoPoint(lat=37.7749, lon=-122.4194, alt=10 + alt_diff)
        assert point1 != point2

    @pytest.mark.parametrize("lat_diff", [0.0001, 0.1, 1.0])
    def test_equality_different_latitude(self, lat_diff):
        """Test that points with different latitudes are not equal."""
        point1 = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        point2 = GeoPoint(lat=37.7749 + lat_diff, lon=-122.4194, alt=10)
        assert point1 != point2

    @pytest.mark.parametrize("lon_diff", [0.0001, 0.1, 1.0])
    def test_equality_different_longitude(self, lon_diff):
        """Test that points with different longitudes are not equal."""
        point1 = GeoPoint(lat=37.7749, lon=-122.4194, alt=10)
        point2 = GeoPoint(lat=37.7749, lon=-122.4194 + lon_diff, alt=10)
        assert point1 != point2

    @pytest.mark.parametrize(
        "other",
        [
            {"lat": 37.7749, "lon": -122.4194},
            (37.7749, -122.4194),
            "37.7749,-122.4194",
            None,
        ],
    )
    def test_equality_different_type(self, other):
        """Test that comparison with non-GeoPoint returns False."""
        point = GeoPoint(lat=37.7749, lon=-122.4194)
        assert point != other
        assert point is not None

    def test_equality_with_float_precision(self):
        """Test equality with floating-point precision considerations."""
        point1 = GeoPoint(lat=37.7749, lon=-122.4194, alt=10.0)
        point2 = GeoPoint(lat=37.7749000001, lon=-122.4194000001, alt=10.0000001)
        # These should NOT be equal due to floating-point differences
        # Python's == for floats is exact
        assert point1 != point2


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for GeoPoint serialization and deserialization."""

    def test_to_dict_contains_all_fields(self, sf_point):
        """Test that to_dict returns all coordinate fields."""
        data = sf_point.to_dict()
        assert "lat" in data
        assert "lon" in data
        assert "alt" in data

    def test_to_dict_values(self, sf_point):
        """Test that to_dict returns correct values."""
        data = sf_point.to_dict()
        assert data["lat"] == 37.7749
        assert data["lon"] == -122.4194
        assert data["alt"] == 10.0

    def test_to_dict_preserves_type(self, sf_point):
        """Test that to_dict values are floats."""
        data = sf_point.to_dict()
        assert isinstance(data["lat"], float)
        assert isinstance(data["lon"], float)
        assert isinstance(data["alt"], float)

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields present."""
        data = {"lat": 37.7749, "lon": -122.4194, "alt": 10.0}
        point = GeoPoint.from_dict(data)
        assert_point_equals(point, 37.7749, -122.4194, 10.0)

    def test_from_dict_with_missing_alt(self):
        """Test from_dict with missing altitude (defaults to 0.0)."""
        data = {"lat": 37.7749, "lon": -122.4194}
        point = GeoPoint.from_dict(data)
        assert point.lat == 37.7749
        assert point.lon == -122.4194
        assert point.alt == 0.0

    def test_from_dict_with_missing_fields(self):
        """Test from_dict raises KeyError with missing lat/lon fields."""
        data = {"alt": 10.0}
        with pytest.raises(KeyError, match="lat"):
            GeoPoint.from_dict(data)

    def test_from_dict_with_empty_dict(self):
        """Test from_dict raises KeyError with empty dictionary."""
        with pytest.raises(KeyError, match="lat"):
            GeoPoint.from_dict({})

    def test_round_trip_serialization(self, sf_point):
        """Test that to_dict -> from_dict preserves the point."""
        assert_round_trip_serialization(sf_point)

    @pytest.mark.parametrize(
        ("lat", "lon", "alt"),
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (-90.0, 0.0, 0.0),
            (45.5, -123.456, 1234.567),
            (-45.5, 123.456, -100.0),
        ],
    )
    def test_round_trip_with_various_points(self, lat, lon, alt):
        """Test round-trip serialization with various points."""
        original = GeoPoint(lat=lat, lon=lon, alt=alt)
        assert_round_trip_serialization(original)


# =============================================================================
# String Representation Tests
# =============================================================================


class TestStringRepresentation:
    """Tests for GeoPoint string representation."""

    def test_repr_contains_class_name(self, sf_point):
        """Test that repr contains the class name."""
        repr_str = repr(sf_point)
        assert "GeoPoint" in repr_str

    def test_repr_contains_all_coordinates(self, sf_point):
        """Test that repr contains all coordinate values."""
        repr_str = repr(sf_point)
        assert "37.7749" in repr_str
        assert "-122.4194" in repr_str
        assert "10" in repr_str or "10.0" in repr_str

    def test_repr_format(self, sf_point):
        """Test that repr has expected format."""
        repr_str = repr(sf_point)
        # Should have lat, lon, alt in order
        assert "lat=" in repr_str
        assert "lon=" in repr_str
        assert "alt=" in repr_str

    def test_repr_precision(self):
        """Test that repr shows coordinates with 6 decimal places."""
        point = GeoPoint(lat=37.123456789, lon=-122.987654321, alt=100.123456789)
        repr_str = repr(point)
        # Check that we get reasonable precision (not truncated too early)
        assert "37.123457" in repr_str or "37.123456" in repr_str

    def test_repr_negative_values(self):
        """Test repr with negative coordinates."""
        point = GeoPoint(lat=-45.5, lon=-123.456, alt=-100.0)
        repr_str = repr(point)
        assert "-45.5" in repr_str
        assert "-123.456" in repr_str
        assert "-100" in repr_str


# =============================================================================
# Edge Cases and Special Values
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_zero_coordinates(self, origin_point):
        """Test point at (0, 0, 0)."""
        assert origin_point.lat == 0.0
        assert origin_point.lon == 0.0
        assert origin_point.alt == 0.0

    def test_north_pole(self, north_pole):
        """Test point at North Pole."""
        assert north_pole.lat == 90.0
        assert north_pole.lon == 0.0

    def test_south_pole(self, south_pole):
        """Test point at South Pole."""
        assert south_pole.lat == -90.0
        assert south_pole.lon == 0.0

    def test_international_date_line(self):
        """Test point at International Date Line."""
        point = GeoPoint(lat=0.0, lon=180.0, alt=0.0)
        assert point.lat == 0.0
        assert point.lon == 180.0

    def test_prime_meridian(self):
        """Test point at Prime Meridian."""
        point = GeoPoint(lat=51.4779, lon=0.0, alt=0.0)  # ~Greenwich
        assert point.lat == pytest.approx(51.4779)
        assert point.lon == 0.0

    def test_very_high_altitude(self):
        """Test point at very high altitude (e.g., satellite)."""
        point = GeoPoint(lat=0.0, lon=0.0, alt=400_000_000)  # 400 km
        assert point.alt == 400_000_000.0

    def test_alt_edge_cases(self):
        """Test altitude at edge cases."""
        # Very large positive altitude
        point1 = GeoPoint(lat=0.0, lon=0.0, alt=1e9)
        assert point1.alt == 1e9
        # Very large negative altitude (deep underground)
        point2 = GeoPoint(lat=0.0, lon=0.0, alt=-1e9)
        assert point2.alt == -1e9
        # Exactly zero
        point3 = GeoPoint(lat=0.0, lon=0.0, alt=0.0)
        assert point3.alt == 0.0

    def test_very_precise_coordinates(self):
        """Test point with very precise coordinates."""
        point = GeoPoint(lat=37.774929374829374, lon=-122.419415498123456, alt=10.123456789012345)
        assert point.lat == pytest.approx(37.774929374829374)
        assert point.lon == pytest.approx(-122.419415498123456)
        assert point.alt == pytest.approx(10.123456789012345)

    def test_distance_at_poles(self):
        """Test distance calculation involving poles."""
        north_pole = GeoPoint(lat=90.0, lon=0.0, alt=0)
        south_pole = GeoPoint(lat=-90.0, lon=0.0, alt=0)

        # Distance between poles should be approximately the Earth's diameter
        distance = north_pole.distance_to(south_pole)
        # Earth's diameter through poles is ~2 * WGS84_B (semi-minor axis)
        # But actual distance depends on the ellipsoid
        assert distance > 12_000_000  # More than 12,000 km
        assert distance < 13_000_000  # Less than 13,000 km

    def test_distance_with_zero_altitude_difference(self):
        """Test that same lat/lon with same alt gives zero distance."""
        p1 = GeoPoint(lat=45.0, lon=45.0, alt=100)
        p2 = GeoPoint(lat=45.0, lon=45.0, alt=100)
        assert p1.distance_to(p2) == 0.0


# =============================================================================
# GeoVelocity Tests
# =============================================================================


class TestGeoVelocityCreation:
    """Tests for GeoVelocity creation and validation."""

    def test_create_velocity_with_all_params(self):
        """Test creating a GeoVelocity with east, north, and up components."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        assert vel.east == 10.0
        assert vel.north == 5.0
        assert vel.up == 2.0

    def test_create_velocity_zero(self):
        """Test creating a zero GeoVelocity."""
        vel = GeoVelocity(east=0.0, north=0.0, up=0.0)
        assert vel.east == 0.0
        assert vel.north == 0.0
        assert vel.up == 0.0

    def test_create_velocity_negative_components(self):
        """Test creating a GeoVelocity with negative components."""
        vel = GeoVelocity(east=-10.0, north=-5.0, up=-2.0)
        assert vel.east == -10.0
        assert vel.north == -5.0
        assert vel.up == -2.0

    def test_create_velocity_float_coercion(self):
        """Test that integer components are converted to float."""
        vel = GeoVelocity(east=10, north=5, up=2)
        assert isinstance(vel.east, float)
        assert isinstance(vel.north, float)
        assert isinstance(vel.up, float)

    def test_to_tuple(self):
        """Test converting GeoVelocity to tuple."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        result = vel.to_tuple()
        assert result == (10.0, 5.0, 2.0)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_from_tuple_valid(self):
        """Test creating GeoVelocity from a valid tuple."""
        velocity_tuple = (10.0, 5.0, 2.0)
        vel = GeoVelocity.from_tuple(velocity_tuple)
        assert vel.east == 10.0
        assert vel.north == 5.0
        assert vel.up == 2.0

    def test_from_tuple_invalid_length(self):
        """Test that from_tuple raises ValueError for invalid tuple length."""
        with pytest.raises(ValueError, match="Velocity tuple must have exactly 3 elements"):
            GeoVelocity.from_tuple((10.0, 5.0))  # Only 2 elements

    def test_from_tuple_invalid_length_too_long(self):
        """Test that from_tuple raises ValueError for tuple with too many elements."""
        with pytest.raises(ValueError, match="Velocity tuple must have exactly 3 elements"):
            GeoVelocity.from_tuple((10.0, 5.0, 2.0, 1.0))  # 4 elements

    def test_from_tuple_invalid_type(self):
        """Test that from_tuple raises ValueError for non-tuple."""
        with pytest.raises(ValueError, match="Velocity tuple must have exactly 3 elements"):
            GeoVelocity.from_tuple([10.0, 5.0, 2.0])  # List, not tuple

    def test_nan_components_raise(self):
        """Test that NaN components raise ValueError."""
        with pytest.raises(ValueError, match="components must be finite"):
            GeoVelocity(east=float("nan"), north=0.0, up=0.0)
        with pytest.raises(ValueError, match="components must be finite"):
            GeoVelocity(east=0.0, north=float("nan"), up=0.0)
        with pytest.raises(ValueError, match="components must be finite"):
            GeoVelocity(east=0.0, north=0.0, up=float("nan"))

    def test_inf_components_raise(self):
        """Test that Inf components raise ValueError."""
        with pytest.raises(ValueError, match="components must be finite"):
            GeoVelocity(east=float("inf"), north=0.0, up=0.0)
        with pytest.raises(ValueError, match="components must be finite"):
            GeoVelocity(east=float("-inf"), north=0.0, up=0.0)

    def test_nan_alt_raises(self):
        """Test that NaN altitude raises ValueError."""
        with pytest.raises(ValueError, match="GeoPoint.alt must be finite"):
            GeoPoint(lat=0.0, lon=0.0, alt=float("nan"))

    def test_inf_alt_raises(self):
        """Test that Inf altitude raises ValueError."""
        with pytest.raises(ValueError, match="GeoPoint.alt must be finite"):
            GeoPoint(lat=0.0, lon=0.0, alt=float("inf"))
        with pytest.raises(ValueError, match="GeoPoint.alt must be finite"):
            GeoPoint(lat=0.0, lon=0.0, alt=float("-inf"))


class TestGeoVelocityEquality:
    """Tests for GeoVelocity equality comparison."""

    def test_equality_same_components(self):
        """Test that velocities with same components are equal."""
        vel1 = GeoVelocity(east=10.0, north=5.0, up=2.0)
        vel2 = GeoVelocity(east=10.0, north=5.0, up=2.0)
        assert vel1 == vel2

    def test_equality_different_east(self):
        """Test that velocities with different east components are not equal."""
        vel1 = GeoVelocity(east=10.0, north=5.0, up=2.0)
        vel2 = GeoVelocity(east=11.0, north=5.0, up=2.0)
        assert vel1 != vel2

    def test_equality_different_north(self):
        """Test that velocities with different north components are not equal."""
        vel1 = GeoVelocity(east=10.0, north=5.0, up=2.0)
        vel2 = GeoVelocity(east=10.0, north=6.0, up=2.0)
        assert vel1 != vel2

    def test_equality_different_up(self):
        """Test that velocities with different up components are not equal."""
        vel1 = GeoVelocity(east=10.0, north=5.0, up=2.0)
        vel2 = GeoVelocity(east=10.0, north=5.0, up=3.0)
        assert vel1 != vel2

    def test_equality_different_type(self):
        """Test that comparison with non-GeoVelocity returns False."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        assert vel != (10.0, 5.0, 2.0)
        assert vel != [10.0, 5.0, 2.0]
        assert vel != "velocity"
        assert vel is not None


class TestCoordinateSystemValidation:
    """Tests for coordinate system validation."""

    def test_lla_to_ecef_nan_raises(self):
        """Test that NaN coordinates raise ValueError in lla_to_ecef."""
        from torchsig.geo.utils.coordinate_system import lla_to_ecef

        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(float("nan"), 0.0, 0.0)
        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(0.0, float("nan"), 0.0)
        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(0.0, 0.0, float("nan"))

    def test_lla_to_ecef_inf_raises(self):
        """Test that Inf coordinates raise ValueError in lla_to_ecef."""
        from torchsig.geo.utils.coordinate_system import lla_to_ecef

        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(float("inf"), 0.0, 0.0)
        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(0.0, float("inf"), 0.0)
        with pytest.raises(ValueError, match="finite coordinates"):
            lla_to_ecef(0.0, 0.0, float("-inf"))


class TestGeoVelocityStringRepresentation:
    """Tests for GeoVelocity string representation."""

    def test_repr_contains_class_name(self):
        """Test that repr contains the class name."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        repr_str = repr(vel)
        assert "GeoVelocity" in repr_str

    def test_repr_contains_all_components(self):
        """Test that repr contains all component values."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        repr_str = repr(vel)
        assert "10.0" in repr_str
        assert "5.0" in repr_str
        assert "2.0" in repr_str

    def test_repr_format(self):
        """Test that repr has expected format."""
        vel = GeoVelocity(east=10.0, north=5.0, up=2.0)
        repr_str = repr(vel)
        # Should have east, north, up in order
        assert "east=" in repr_str
        assert "north=" in repr_str
        assert "up=" in repr_str

    def test_repr_negative_values(self):
        """Test repr with negative components."""
        vel = GeoVelocity(east=-10.0, north=-5.0, up=-2.0)
        repr_str = repr(vel)
        assert "-10.0" in repr_str
        assert "-5.0" in repr_str
        assert "-2.0" in repr_str
