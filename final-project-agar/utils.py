import numpy as np
import math
import config as conf
from actions import Action
import time


def mass_to_radius(mass):
    """
    Determine radius from mass of blob

    Parameters:

        mass - number

    Returns number
    """
    return int(4 + np.sqrt(mass) * 4)


def mass_to_velocity(mass):
    """
    Determine velocity from mass of blob

    Parameters:

        mass - number

    Returns number
    """
    return int(2.2 * np.power(mass / 1000, -0.439))


def gen_non_overlap_position(agents, radius):
    # generate 10 candidate positions, and find the first one that isn't overlapping
    # with any agents
    candidates = [gen_random_position(radius) for i in range(10)]
    for cand in candidates:
        overlaps = False
        for agent in agents:
            for cell in agent.cells:
                if is_point_in_circle(cand, cell.get_pos(), cell.radius):
                    overlaps = True
                    break
            if overlaps:
                break

        if not overlaps:
            return cand

    # if none of them work, give up and return a random position
    print('[DEBUG] couldn\'t find non-overlapping position for item :(')
    return gen_random_position(radius)


def gen_random_position(radius):
    """
    Generate a random position within the field of play. NOTE the `radius` is
    used to position within bounds of the board.

    Parameters

        radius : number

    Returns

        tuple (x y)
    """
    return (
        gen_rand_int_in_range(radius, conf.BOARD_WIDTH - radius),
        gen_rand_int_in_range(radius, conf.BOARD_HEIGHT - radius)
    )


def gen_rand_int_in_range(lo, hi):
    """
    Generate random int in the range lo to hi

    Parameters

        lo : number
        hi : number

    Returns

        number
    """
    return int(np.floor(np.random.random() * (hi - lo)) + lo)


def get_euclidean_dist(p1, p2):
    """
    Calculate euclidean distance between two points

    Parameters

        p1 : tuple (x, y)
        p2 : tuple (x, y)

    Returns

        number
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_object_dist(a, b):
    return get_euclidean_dist(a.get_pos(), b.get_pos())


def is_point_in_circle(point, circle, radius):
    """
    Return if the provided point is within the circle

    Parameters

        point  : tuple (x, y)
        circle : tuple (x, y)
        radius : number

    Returns

        boolean
    """
    return radius >= get_euclidean_dist(point, circle)


def get_circle_overlap(c1, r1, c2, r2):
    """
    Measure overlap between two circles

    NOTE this is more positive the more cirlces overlap, this is negative if the
    circles do not overlap

    Parameters

        c1 - tuple (x, y) position of circle
        r1 - radius
        c2 - tuple (x, y) position of circle
        r2 - radius

    Returns

        number
    """
    dist_btwn_centers = np.linalg.norm(np.array(c1) - np.array(c2))
    sum_radii = r1 + r2
    return sum_radii - dist_btwn_centers


def get_object_overlap(a, b):
    return get_circle_overlap(a.get_pos(), a.radius, b.get_pos(), b.radius)


def are_circles_colliding(c1, r1, c2, r2):
    """
    Parameters

        c1 : tuple (x, y) position of circle
        r1 : radius
        c2 : tuple (x, y) position of circle
        r2 : radius

    Returns boolean if the circles overlap
    """
    overlap = get_circle_overlap(c1, r1, c2, r2)
    return overlap > 0


def get_angle_between_points(p1, p2):
    """
    Parameters

        p1 : tuple (x, y)
        p2 : tuple (x, y)

    Returns angle in degrees of line drawn between points and positive x dir

    Note this angle is in the range [0, 360)
    """
    (x1, y1) = p1
    (x2, y2) = p2
    dx = x2 - x1
    dy = y1 - y2  # Since 0 is in the top left corner
    
    angle = math.atan2(dy, dx) * 180 / math.pi
    if angle < 0:
        return angle + 360
    return angle


def get_angle_between_objects(a, b):
    return get_angle_between_points(a.get_pos(), b.get_pos())


def get_action_closest_to_angle(angle):
    half_angle = 45 / 2
    if angle <= (45 - half_angle):
        return Action.MOVE_RIGHT
    elif angle <= (90 - half_angle):
        return Action.MOVE_UP_RIGHT
    elif angle <= (135 - half_angle):
        return Action.MOVE_UP
    elif angle <= (180 - half_angle):
        return Action.MOVE_UP_LEFT
    elif angle <= (225 - half_angle):
        return Action.MOVE_LEFT
    elif angle <= (270 - half_angle):
        return Action.MOVE_DOWN_LEFT
    elif angle <= (315 - half_angle):
        return Action.MOVE_DOWN
    elif angle <= (360 - half_angle):
        return Action.MOVE_DOWN_RIGHT
    else:
        return Action.MOVE_RIGHT


def get_action_farthest_from_angle(angle):
    half_angle = 45 / 2
    if angle <= (45 - half_angle):
        return Action.MOVE_LEFT
    elif angle <= (90 - half_angle):
        return Action.MOVE_DOWN_LEFT
    elif angle <= (135 - half_angle):
        return Action.MOVE_DOWN
    elif angle <= (180 - half_angle):
        return Action.MOVE_DOWN_RIGHT
    elif angle <= (225 - half_angle):
        return Action.MOVE_RIGHT
    elif angle <= (270 - half_angle):
        return Action.MOVE_UP_RIGHT
    elif angle <= (315 - half_angle):
        return Action.MOVE_UP
    elif angle <= (360 - half_angle):
        return Action.MOVE_UP_LEFT
    else:
        return Action.MOVE_LEFT


def get_random_action():
    return Action(np.random.randint(len(Action)))


def is_action_feasible(action, pos, radius):
    """
    Check if the given action will actually move the object with given position
    and radius
    """
    (x_pos, y_pos) = pos
    if action == Action.MOVE_RIGHT:
        return x_pos < conf.BOARD_WIDTH - radius
    elif action == Action.MOVE_LEFT:
        return x_pos > radius
    elif action == Action.MOVE_DOWN:
        return y_pos < conf.BOARD_HEIGHT - radius
    elif action == Action.MOVE_UP:
        return y_pos > radius
    elif action == Action.MOVE_UP_RIGHT:
        return x_pos < conf.BOARD_WIDTH - radius or y_pos > radius
    elif action == Action.MOVE_UP_LEFT:
        return x_pos > radius or y_pos > radius
    elif action == Action.MOVE_DOWN_RIGHT:
        return x_pos < conf.BOARD_WIDTH - radius or y_pos < conf.BOARD_HEIGHT - radius
    elif action == Action.MOVE_DOWN_LEFT:
        return x_pos > radius or y_pos < conf.BOARD_HEIGHT - radius
    else:
        raise ValueError('invalid action given to is_action_feasible()')


def move_in_bounds(obj):
    """move the given object such that it is entirely in bounds"""
    if obj.x_pos < obj.radius:
        obj.x_pos = obj.radius
    if obj.x_pos > conf.BOARD_WIDTH - obj.radius:
        obj.x_pos = conf.BOARD_WIDTH - obj.radius
    if obj.y_pos < obj.radius:
        obj.y_pos = obj.radius
    if obj.y_pos > conf.BOARD_HEIGHT - obj.radius:
        obj.y_pos = conf.BOARD_HEIGHT - obj.radius


def move_object_left(obj, vel):
    obj.x_pos = max(obj.x_pos - vel, obj.radius)


def move_object_right(obj, vel):
    obj.x_pos = min(obj.x_pos + vel, conf.BOARD_WIDTH - obj.radius)


def move_object_up(obj, vel):
    obj.y_pos = max(obj.y_pos - vel, obj.radius)


def move_object_down(obj, vel):
    obj.y_pos = min(obj.y_pos + vel, conf.BOARD_HEIGHT - obj.radius)


def move_object(obj, angle, vel):
    if angle is None:
        return

    radians = angle / 180 * math.pi
    dx = math.cos(radians) * vel
    dy = math.sin(radians) * vel
    if dx > 0:
        move_object_right(obj, dx)
    elif dx < 0:
        move_object_left(obj, dx * -1)

    if dy > 0:
        move_object_up(obj, dy)
    elif dy < 0:
        move_object_down(obj, dy * -1)


def current_milli_time():
    return int(round(time.time() * 1000))