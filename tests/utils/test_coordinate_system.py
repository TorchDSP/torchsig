"""Unit tests for spectrogram rectangle overlap utilities."""

import pytest

from torchsig.utils.coordinate_system import (
    Coordinate,
    Rectangle,
    counter_clock_wise,
    is_corner_in_rectangle,
    is_rectangle_inside_rectangle,
    is_rectangle_overlap,
    is_within_range,
    line_intersection,
)


@pytest.fixture
def unit_rectangle() -> Rectangle:
    """Return a rectangle spanning (0, 0) through (1, 1)."""
    return Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))


class TestCoordinate:
    """Tests for Coordinate."""

    def test_stores_coordinates(self):
        coord = Coordinate(1.5, -2.0)

        assert coord.x == 1.5
        assert coord.y == -2.0

    def test_string_representation(self):
        coord = Coordinate(1.5, -2.0)

        assert str(coord) == "x = 1.5, y = -2.0"


class TestRectangle:
    """Tests for Rectangle."""

    def test_constructs_all_four_corners(self):
        rectangle = Rectangle(Coordinate(1.0, 2.0), Coordinate(4.0, 6.0))

        assert (
            rectangle.coord_lower_left.x,
            rectangle.coord_lower_left.y,
        ) == (1.0, 2.0)
        assert (
            rectangle.coord_upper_right.x,
            rectangle.coord_upper_right.y,
        ) == (4.0, 6.0)
        assert (
            rectangle.coord_upper_left.x,
            rectangle.coord_upper_left.y,
        ) == (1.0, 6.0)
        assert (
            rectangle.coord_lower_right.x,
            rectangle.coord_lower_right.y,
        ) == (4.0, 2.0)

    def test_preserves_supplied_corner_objects(self):
        lower = Coordinate(1.0, 2.0)
        upper = Coordinate(4.0, 6.0)

        rectangle = Rectangle(lower, upper)

        assert rectangle.coord_lower_left is lower
        assert rectangle.coord_upper_right is upper


class TestCounterClockWise:
    """Tests for the counter-clockwise orientation test."""

    @pytest.mark.parametrize(
        ("a", "b", "c", "expected"),
        [
            (
                Coordinate(0.0, 0.0),
                Coordinate(1.0, 0.0),
                Coordinate(1.0, 1.0),
                True,
            ),
            (
                Coordinate(0.0, 0.0),
                Coordinate(1.0, 1.0),
                Coordinate(1.0, 0.0),
                False,
            ),
            (
                Coordinate(0.0, 0.0),
                Coordinate(1.0, 1.0),
                Coordinate(2.0, 2.0),
                False,
            ),
        ],
        ids=["counter-clockwise", "clockwise", "collinear"],
    )
    def test_orientation(self, a, b, c, expected):
        assert counter_clock_wise(a, b, c) is expected


class TestLineIntersection:
    """Tests for line segment intersection."""

    def test_crossing_segments_intersect(self):
        a = Coordinate(0.0, 0.0)
        b = Coordinate(2.0, 2.0)
        c = Coordinate(0.0, 2.0)
        d = Coordinate(2.0, 0.0)

        assert line_intersection(a, b, c, d)

    def test_separated_segments_do_not_intersect(self):
        a = Coordinate(0.0, 0.0)
        b = Coordinate(1.0, 0.0)
        c = Coordinate(0.0, 1.0)
        d = Coordinate(1.0, 1.0)

        assert not line_intersection(a, b, c, d)

    def test_intersection_is_independent_of_segment_order(self):
        a = Coordinate(0.0, 0.0)
        b = Coordinate(2.0, 2.0)
        c = Coordinate(0.0, 2.0)
        d = Coordinate(2.0, 0.0)

        assert line_intersection(a, b, c, d)
        assert line_intersection(c, d, a, b)

    def test_collinear_overlapping_segments_are_not_detected(self):
        """Document the current strict-orientation behavior."""
        a = Coordinate(0.0, 0.0)
        b = Coordinate(2.0, 0.0)
        c = Coordinate(1.0, 0.0)
        d = Coordinate(3.0, 0.0)

        assert not line_intersection(a, b, c, d)

    def test_segments_touching_at_endpoint_are_not_detected(self):
        """Document the current strict-orientation behavior."""
        a = Coordinate(0.0, 0.0)
        b = Coordinate(1.0, 1.0)
        c = Coordinate(1.0, 1.0)
        d = Coordinate(2.0, 0.0)

        assert not line_intersection(a, b, c, d)


class TestIsWithinRange:
    """Tests for closed interval membership."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (-0.1, False),
            (0.0, True),
            (0.5, True),
            (1.0, True),
            (1.1, False),
        ],
    )
    def test_closed_interval(self, value, expected):
        assert is_within_range(value, 0.0, 1.0) is expected


class TestIsCornerInRectangle:
    """Tests for point containment."""

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            (Coordinate(0.5, 0.5), True),
            (Coordinate(0.0, 0.5), True),
            (Coordinate(1.0, 1.0), True),
            (Coordinate(-0.1, 0.5), False),
            (Coordinate(0.5, 1.1), False),
        ],
        ids=[
            "interior",
            "edge",
            "corner",
            "outside-x",
            "outside-y",
        ],
    )
    def test_point_containment(self, unit_rectangle, point, expected):
        assert is_corner_in_rectangle(point, unit_rectangle) is expected


class TestIsRectangleInsideRectangle:
    """Tests for complete rectangle containment."""

    def test_rectangle_strictly_inside(self):
        outer = Rectangle(Coordinate(0.0, 0.0), Coordinate(4.0, 4.0))
        inner = Rectangle(Coordinate(1.0, 1.0), Coordinate(3.0, 3.0))

        assert is_rectangle_inside_rectangle(inner, outer)

    def test_equal_rectangles_are_inside_each_other(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))
        rectangle_b = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))

        assert is_rectangle_inside_rectangle(rectangle_a, rectangle_b)
        assert is_rectangle_inside_rectangle(rectangle_b, rectangle_a)

    def test_rectangle_sharing_outer_boundary_is_inside(self):
        outer = Rectangle(Coordinate(0.0, 0.0), Coordinate(4.0, 4.0))
        inner = Rectangle(Coordinate(0.0, 1.0), Coordinate(3.0, 4.0))

        assert is_rectangle_inside_rectangle(inner, outer)

    def test_partial_overlap_is_not_complete_containment(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))
        rectangle_b = Rectangle(Coordinate(1.0, 1.0), Coordinate(3.0, 3.0))

        assert not is_rectangle_inside_rectangle(rectangle_a, rectangle_b)
        assert not is_rectangle_inside_rectangle(rectangle_b, rectangle_a)

    def test_separate_rectangle_is_not_inside(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
        rectangle_b = Rectangle(Coordinate(2.0, 2.0), Coordinate(3.0, 3.0))

        assert not is_rectangle_inside_rectangle(rectangle_a, rectangle_b)


class TestIsRectangleOverlap:
    """Tests for overall rectangle overlap detection."""

    def test_partially_overlapping_rectangles(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))
        rectangle_b = Rectangle(Coordinate(1.0, 1.0), Coordinate(3.0, 3.0))

        assert is_rectangle_overlap(rectangle_a, rectangle_b)

    def test_contained_rectangle(self):
        outer = Rectangle(Coordinate(0.0, 0.0), Coordinate(4.0, 4.0))
        inner = Rectangle(Coordinate(1.0, 1.0), Coordinate(2.0, 2.0))

        assert is_rectangle_overlap(inner, outer)
        assert is_rectangle_overlap(outer, inner)

    def test_identical_rectangles(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))
        rectangle_b = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))

        assert is_rectangle_overlap(rectangle_a, rectangle_b)

    def test_separated_horizontally(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
        rectangle_b = Rectangle(Coordinate(2.0, 0.0), Coordinate(3.0, 1.0))

        assert not is_rectangle_overlap(rectangle_a, rectangle_b)

    def test_separated_vertically(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
        rectangle_b = Rectangle(Coordinate(0.0, 2.0), Coordinate(1.0, 3.0))

        assert not is_rectangle_overlap(rectangle_a, rectangle_b)

    def test_cross_overlap_without_corner_containment(self):
        """Detect a plus-shaped overlap where no corner lies in the other box."""
        horizontal = Rectangle(
            Coordinate(0.0, 1.0),
            Coordinate(4.0, 2.0),
        )
        vertical = Rectangle(
            Coordinate(1.5, 0.0),
            Coordinate(2.5, 3.0),
        )

        assert not is_corner_in_rectangle(
            horizontal.coord_lower_left,
            vertical,
        )
        assert not is_corner_in_rectangle(
            vertical.coord_lower_left,
            horizontal,
        )
        assert is_rectangle_overlap(horizontal, vertical)

    def test_shared_edge_counts_as_overlap(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
        rectangle_b = Rectangle(Coordinate(1.0, 0.0), Coordinate(2.0, 1.0))

        assert is_rectangle_overlap(rectangle_a, rectangle_b)

    def test_shared_corner_counts_as_overlap(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
        rectangle_b = Rectangle(Coordinate(1.0, 1.0), Coordinate(2.0, 2.0))

        assert is_rectangle_overlap(rectangle_a, rectangle_b)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.5, 0.5), (0.5, 0.5)),
        ],
        ids=["zero-width", "zero-height", "point"],
    )
    def test_degenerate_rectangle_inside_regular_rectangle(
        self,
        unit_rectangle,
        lower,
        upper,
    ):
        degenerate = Rectangle(
            Coordinate(*lower),
            Coordinate(*upper),
        )

        assert is_rectangle_overlap(degenerate, unit_rectangle)

    def test_overlap_is_symmetric(self):
        rectangle_a = Rectangle(Coordinate(0.0, 0.0), Coordinate(2.0, 2.0))
        rectangle_b = Rectangle(Coordinate(1.0, -1.0), Coordinate(3.0, 1.0))

        assert is_rectangle_overlap(
            rectangle_a,
            rectangle_b,
        ) == is_rectangle_overlap(
            rectangle_b,
            rectangle_a,
        )
