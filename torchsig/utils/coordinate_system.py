
# contains (x,y) coordinates
class Coordinate:
    def __init__( self, x, y):
        self.x = x
        self.y = y

    def __str__( self ):
        return f'x = {self.x}, y = {self.y}'
# represents a rectangle shape with four vertices, each a Coordinate()
class Rectangle:
    def __init__ ( self, lower_coord:Coordinate, upper_coord:Coordinate):
        # build four verticies as coordinates
        self.coord_lower_left = lower_coord
        self.coord_upper_right = upper_coord

        self.coord_upper_left = Coordinate(self.coord_lower_left.x,self.coord_upper_right.y)
        self.coord_lower_right = Coordinate(self.coord_upper_right.x,self.coord_lower_left.y)


# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def counter_clock_wise(A:Coordinate,B:Coordinate,C:Coordinate):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def line_intersection(A,B,C,D):
    return counter_clock_wise(A,C,D) != counter_clock_wise(B,C,D) and counter_clock_wise(A,B,C) != counter_clock_wise(A,B,D)


def linear_overlap ( test_coord_x, box_left_x, box_right_x ):
    linear_overlap_bool = box_left_x <= test_coord_x and test_coord_x <= box_right_x
    return linear_overlap_bool

def corner_overlap ( corner_coord:Coordinate, reference_box:Rectangle ):

    #print('corner overlap:')
    #print(f'corner x = {corner_coord.x}')
    #print(f'box x = {reference_box.coord_lower_left.x}, {reference_box.coord_lower_right.x}')
    #print(f'corner y = {corner_coord.y}')
    #print(f'box y = {reference_box.coord_lower_left.y}, {reference_box.coord_upper_left.y}')

    corner_overlap_x = linear_overlap( corner_coord.x, reference_box.coord_lower_left.x, reference_box.coord_lower_right.x )
    corner_overlap_y = linear_overlap( corner_coord.y, reference_box.coord_lower_left.y, reference_box.coord_upper_left.y  )

    corner_overlap_bool = corner_overlap_x and corner_overlap_y

    #print(f'overlap = {corner_overlap_bool}')
    #print()

    return corner_overlap_bool


def is_box_inside_box( box_a:Rectangle, box_b:Rectangle ):

    # check if one the four verices if they are within boundary. since
    # we are checking line intersections, we only need to validate the
    # case in which one vertex is within the boundary.
    #print('corner lower left')
    corner_overlap_bool_0 = corner_overlap( box_a.coord_lower_left, box_b )
    #print('corner upper left')
    #corner_overlap_bool_1 = corner_overlap( box_a.coord_upper_left, box_b )
    #print('corner upper right')
    #corner_overlap_bool_2 = corner_overlap( box_a.coord_upper_right, box_b )
    #print('corner lower right')
    #corner_overlap_bool_3 = corner_overlap( box_a.coord_lower_right, box_b )

    box_overlap_bool = corner_overlap_bool_0 #or corner_overlap_bool_1 or corner_overlap_bool_2 or corner_overlap_bool_3

    return box_overlap_bool


def is_box_overlap ( box_a:Rectangle, box_b:Rectangle ):

    # check all combinations for overlap of all sides

    line_intersection_0  = line_intersection(box_a.coord_lower_left,  box_a.coord_lower_right, box_b.coord_lower_left,  box_b.coord_lower_right)
    line_intersection_1  = line_intersection(box_a.coord_lower_left,  box_a.coord_lower_right, box_b.coord_lower_left,  box_b.coord_upper_left)
    line_intersection_2  = line_intersection(box_a.coord_lower_left,  box_a.coord_lower_right, box_b.coord_upper_left,  box_b.coord_upper_right)
    line_intersection_3  = line_intersection(box_a.coord_lower_left,  box_a.coord_lower_right, box_b.coord_upper_right, box_b.coord_lower_right)

    line_intersection_4  = line_intersection(box_a.coord_lower_left,  box_a.coord_upper_left,  box_b.coord_lower_left,  box_b.coord_lower_right)
    line_intersection_5  = line_intersection(box_a.coord_lower_left,  box_a.coord_upper_left,  box_b.coord_lower_left,  box_b.coord_upper_left)
    line_intersection_6  = line_intersection(box_a.coord_lower_left,  box_a.coord_upper_left,  box_b.coord_upper_left,  box_b.coord_upper_right)
    line_intersection_7  = line_intersection(box_a.coord_lower_left,  box_a.coord_upper_left,  box_b.coord_upper_right, box_b.coord_lower_right)

    line_intersection_8  = line_intersection(box_a.coord_upper_left,  box_a.coord_upper_right, box_b.coord_lower_left,  box_b.coord_lower_right)
    line_intersection_9  = line_intersection(box_a.coord_upper_left,  box_a.coord_upper_right, box_b.coord_lower_left,  box_b.coord_upper_left)
    line_intersection_10 = line_intersection(box_a.coord_upper_left,  box_a.coord_upper_right, box_b.coord_upper_left,  box_b.coord_upper_right)
    line_intersection_11 = line_intersection(box_a.coord_upper_left,  box_a.coord_upper_right, box_b.coord_upper_right, box_b.coord_lower_right)

    line_intersection_12 = line_intersection(box_a.coord_upper_right, box_a.coord_lower_right, box_b.coord_lower_left,  box_b.coord_lower_right)
    line_intersection_13 = line_intersection(box_a.coord_upper_right, box_a.coord_lower_right, box_b.coord_lower_left,  box_b.coord_upper_left)
    line_intersection_14 = line_intersection(box_a.coord_upper_right, box_a.coord_lower_right, box_b.coord_upper_left,  box_b.coord_upper_right)
    line_intersection_15 = line_intersection(box_a.coord_upper_right, box_a.coord_lower_right, box_b.coord_upper_right, box_b.coord_lower_right)

    box_overlap_0 = is_box_inside_box( box_a, box_b )
    box_overlap_1 = is_box_inside_box( box_b, box_a )

    return line_intersection_0 or line_intersection_1 or line_intersection_2 or line_intersection_3 or line_intersection_4 or line_intersection_5 or line_intersection_6 or line_intersection_7 or line_intersection_8 or line_intersection_9 or line_intersection_10 or line_intersection_11 or line_intersection_12 or line_intersection_13 or line_intersection_14 or line_intersection_15 or box_overlap_0 or box_overlap_1

