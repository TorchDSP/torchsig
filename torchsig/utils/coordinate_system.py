"""Library for overlap detection in spectrograms to control co-channel interference.

This module provides classes and functions to define 2D coordinates and axis-aligned
rectangles, and to detect overlaps between rectangles using line-segment intersection
and containment tests.
"""

__all__ = ["Coordinate", "Rectangle", "counter_clock_wise", "is_corner_in_rectangle", "is_rectangle_inside_rectangle", "is_rectangle_overlap", "is_within_range", "line_intersection"]


# class object to contain (x, y) coordinates
class Coordinate:
    """Represents a point in 2D space with x and y coordinates.

    Attributes:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
    """

    def __init__(self, x: float, y: float):
        """Initialize a Coordinate.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
        """
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Return a human-readable string representation of the coordinate.

        Returns:
            str: Formatted as 'x = {x}, y = {y}'.
        """
        return f"x = {self.x}, y = {self.y}"


# represents a rectangle shape with four vertices, each a Coordinate
class Rectangle:
    """Represents an axis-aligned rectangle defined by two opposite corners.

    The rectangle is built from a lower-left and an upper-right corner,
    from which the other two corners are inferred.

    Attributes:
        coord_lower_left (Coordinate): Lower-left corner.
        coord_upper_right (Coordinate): Upper-right corner.
        coord_upper_left (Coordinate): Upper-left corner.
        coord_lower_right (Coordinate): Lower-right corner.
    """

    def __init__(self, lower_coord: Coordinate, upper_coord: Coordinate):
        """Initialize a Rectangle from two corner coordinates.

        Args:
            lower_coord (Coordinate): Lower-left corner of the rectangle.
            upper_coord (Coordinate): Upper-right corner of the rectangle.
        """
        self.coord_lower_left = lower_coord
        self.coord_upper_right = upper_coord

        self.coord_upper_left = Coordinate(self.coord_lower_left.x, self.coord_upper_right.y)
        self.coord_lower_right = Coordinate(self.coord_upper_right.x, self.coord_lower_left.y)


# function used in determining if lines intersect
# based on the counter-clockwise test algorithm
def counter_clock_wise(a: Coordinate, b: Coordinate, c: Coordinate) -> bool:
    """Determine if three points a, b, c are in counter-clockwise order.

    Args:
        a (Coordinate): First point.
        b (Coordinate): Second point.
        c (Coordinate): Third point.

    Returns:
        bool: True if the sequence (a → b → c) is counter-clockwise.
    """
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


# determine if two line segments (AB and CD) intersect
def line_intersection(a: Coordinate, b: Coordinate, c: Coordinate, d: Coordinate) -> bool:
    """Check if the line segments AB and CD intersect.

    Uses the counter-clockwise orientation test.

    Args:
        a (Coordinate): First endpoint of segment AB.
        b (Coordinate): Second endpoint of segment AB.
        c (Coordinate): First endpoint of segment CD.
        d (Coordinate): Second endpoint of segment CD.

    Returns:
        bool: True if segments AB and CD intersect.
    """
    return counter_clock_wise(a, c, d) != counter_clock_wise(b, c, d) and counter_clock_wise(a, b, c) != counter_clock_wise(a, b, d)


# determine if a point lies within the 1D interval [left, right]
def is_within_range(test_coord_x: float, rectangle_left_x: float, rectangle_right_x: float) -> bool:
    """Check if a coordinate lies within a closed interval on the x-axis.

    Args:
        test_coord_x (float): The x-value to test.
        rectangle_left_x (float): Lower bound of the interval.
        rectangle_right_x (float): Upper bound of the interval.

    Returns:
        bool: True if rectangle_left_x <= test_coord_x <= rectangle_right_x.
    """
    return rectangle_left_x <= test_coord_x <= rectangle_right_x


# determine if a rectangle corner lies inside another rectangle
def is_corner_in_rectangle(corner_coord: Coordinate, reference_box: Rectangle) -> bool:
    """Check if a corner point is within the bounds of a reference rectangle.

    Args:
        corner_coord (Coordinate): The corner to test.
        reference_box (Rectangle): The rectangle in which to test containment.

    Returns:
        bool: True if the corner is inside reference_box (including edges).
    """
    x_inside = is_within_range(
        corner_coord.x,
        reference_box.coord_lower_left.x,
        reference_box.coord_lower_right.x,
    )
    y_inside = is_within_range(
        corner_coord.y,
        reference_box.coord_lower_left.y,
        reference_box.coord_upper_left.y,
    )
    return x_inside and y_inside


# determine if one rectangle is entirely within another
def is_rectangle_inside_rectangle(rectangle_1: Rectangle, rectangle_2: Rectangle) -> bool:
    """Check if rectangle_1 is completely inside rectangle_2.

    Tests whether all four corners of rectangle_1 lie within rectangle_2.

    Args:
        rectangle_1 (Rectangle): The inner rectangle to test.
        rectangle_2 (Rectangle): The outer rectangle to test against.

    Returns:
        bool: True if rectangle_1 is fully contained in rectangle_2.
    """
    corners = [
        rectangle_1.coord_lower_left,
        rectangle_1.coord_upper_left,
        rectangle_1.coord_upper_right,
        rectangle_1.coord_lower_right,
    ]
    return all(is_corner_in_rectangle(c, rectangle_2) for c in corners)


# determine if two rectangles have any overlap
def is_rectangle_overlap(rectangle_a: Rectangle, rectangle_b: Rectangle) -> bool:
    """Check whether two axis-aligned rectangles overlap or touch.

    Comparing the projections on both axes handles partial overlap,
    containment, and collinear or touching edges uniformly.

    Args:
        rectangle_a (Rectangle): First rectangle.
        rectangle_b (Rectangle): Second rectangle.

    Returns:
        bool: True if the rectangles overlap.
    """
    horizontal_overlap = max(rectangle_a.coord_lower_left.x, rectangle_b.coord_lower_left.x) <= min(
        rectangle_a.coord_upper_right.x,
        rectangle_b.coord_upper_right.x,
    )
    vertical_overlap = max(rectangle_a.coord_lower_left.y, rectangle_b.coord_lower_left.y) <= min(
        rectangle_a.coord_upper_right.y,
        rectangle_b.coord_upper_right.y,
    )
    return bool(horizontal_overlap and vertical_overlap)
