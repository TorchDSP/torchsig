
# library for doing overlap detection in spectrograms so the amount of
# co-channel interference can be controlled

# class object to contain (x,y) coordinates
class Coordinate:
    def __init__( self, x, y):
        self.x = x
        self.y = y

    def __str__( self ):
        return f'x = {self.x}, y = {self.y}'

# represents a rectangle shape with four vertices, with each vertex represented
# using the Coordinate() class
class Rectangle:
    def __init__ ( self, lower_coord:Coordinate, upper_coord:Coordinate):
        # build four verticies as coordinates
        self.coord_lower_left = lower_coord
        self.coord_upper_right = upper_coord

        self.coord_upper_left = Coordinate(self.coord_lower_left.x,self.coord_upper_right.y)
        self.coord_lower_right = Coordinate(self.coord_upper_right.x,self.coord_lower_left.y)


# function used in determining if lines intersect
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def counter_clock_wise(A:Coordinate,B:Coordinate,C:Coordinate):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

# determine if two lines (AB and CD) intersect
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def line_intersection(A,B,C,D):
    return counter_clock_wise(A,C,D) != counter_clock_wise(B,C,D) and counter_clock_wise(A,B,C) != counter_clock_wise(A,B,D)

# determine the input point is within the range of two points on a 1D line
def is_within_range ( test_coord_x, rectangle_left_x, rectangle_right_x ):
    linear_overlap_bool = rectangle_left_x <= test_coord_x and test_coord_x <= rectangle_right_x
    return linear_overlap_bool

# determine if the corner point is within the boundary of the box
def is_corner_in_rectangle ( corner_coord:Coordinate, reference_box:Rectangle ):

    # check if x position is within boundary of box
    corner_overlap_x = is_within_range( corner_coord.x, reference_box.coord_lower_left.x, reference_box.coord_lower_right.x )
    # check if y position is within boundary of box
    corner_overlap_y = is_within_range( corner_coord.y, reference_box.coord_lower_left.y, reference_box.coord_upper_left.y  )

    corner_overlap_bool = corner_overlap_x and corner_overlap_y

    return corner_overlap_bool

# determine if a box is fully within the bounds of another box
def is_rectangle_inside_rectangle( rectangle_a:Rectangle, rectangle_b:Rectangle ):

    # check if the four verices are within boundary
    corner_overlap_bool_0 = is_corner_in_rectangle( rectangle_a.coord_lower_left,  rectangle_b )
    corner_overlap_bool_1 = is_corner_in_rectangle( rectangle_a.coord_upper_left,  rectangle_b )
    corner_overlap_bool_2 = is_corner_in_rectangle( rectangle_a.coord_upper_right, rectangle_b )
    corner_overlap_bool_3 = is_corner_in_rectangle( rectangle_a.coord_lower_right, rectangle_b )

    rectangle_overlap_bool = corner_overlap_bool_0 and corner_overlap_bool_1 and corner_overlap_bool_2 and corner_overlap_bool_3

    return rectangle_overlap_bool

# determine if two boxes have any overlap. the conditions include:
#  1. one (or more) of the sides intersect with one another
#  2. one box is totally enclosed by another box
def is_rectangle_overlap ( rectangle_a:Rectangle, rectangle_b:Rectangle ):

    # check all combinations for the overlap of all sides
    line_intersection_0  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_lower_right, rectangle_b.coord_lower_left,  rectangle_b.coord_lower_right)
    line_intersection_1  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_lower_right, rectangle_b.coord_lower_left,  rectangle_b.coord_upper_left)
    line_intersection_2  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_lower_right, rectangle_b.coord_upper_left,  rectangle_b.coord_upper_right)
    line_intersection_3  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_lower_right, rectangle_b.coord_upper_right, rectangle_b.coord_lower_right)

    line_intersection_4  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_upper_left,  rectangle_b.coord_lower_left,  rectangle_b.coord_lower_right)
    line_intersection_5  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_upper_left,  rectangle_b.coord_lower_left,  rectangle_b.coord_upper_left)
    line_intersection_6  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_upper_left,  rectangle_b.coord_upper_left,  rectangle_b.coord_upper_right)
    line_intersection_7  = line_intersection(rectangle_a.coord_lower_left,  rectangle_a.coord_upper_left,  rectangle_b.coord_upper_right, rectangle_b.coord_lower_right)

    line_intersection_8  = line_intersection(rectangle_a.coord_upper_left,  rectangle_a.coord_upper_right, rectangle_b.coord_lower_left,  rectangle_b.coord_lower_right)
    line_intersection_9  = line_intersection(rectangle_a.coord_upper_left,  rectangle_a.coord_upper_right, rectangle_b.coord_lower_left,  rectangle_b.coord_upper_left)
    line_intersection_10 = line_intersection(rectangle_a.coord_upper_left,  rectangle_a.coord_upper_right, rectangle_b.coord_upper_left,  rectangle_b.coord_upper_right)
    line_intersection_11 = line_intersection(rectangle_a.coord_upper_left,  rectangle_a.coord_upper_right, rectangle_b.coord_upper_right, rectangle_b.coord_lower_right)

    line_intersection_12 = line_intersection(rectangle_a.coord_upper_right, rectangle_a.coord_lower_right, rectangle_b.coord_lower_left,  rectangle_b.coord_lower_right)
    line_intersection_13 = line_intersection(rectangle_a.coord_upper_right, rectangle_a.coord_lower_right, rectangle_b.coord_lower_left,  rectangle_b.coord_upper_left)
    line_intersection_14 = line_intersection(rectangle_a.coord_upper_right, rectangle_a.coord_lower_right, rectangle_b.coord_upper_left,  rectangle_b.coord_upper_right)
    line_intersection_15 = line_intersection(rectangle_a.coord_upper_right, rectangle_a.coord_lower_right, rectangle_b.coord_upper_right, rectangle_b.coord_lower_right)

    # determine if one box is within another box
    rectangle_overlap_0 = is_rectangle_inside_rectangle( rectangle_a, rectangle_b )
    rectangle_overlap_1 = is_rectangle_inside_rectangle( rectangle_b, rectangle_a )

    return line_intersection_0 or line_intersection_1 or line_intersection_2 or line_intersection_3 or line_intersection_4 or line_intersection_5 or line_intersection_6 or line_intersection_7 or line_intersection_8 or line_intersection_9 or line_intersection_10 or line_intersection_11 or line_intersection_12 or line_intersection_13 or line_intersection_14 or line_intersection_15 or rectangle_overlap_0 or rectangle_overlap_1

