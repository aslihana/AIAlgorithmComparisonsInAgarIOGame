import torch
import math

from models.DeepRLModel import *
from gamestate import GameState
from agent import Agent
from food import Food
import config as conf


"""
constants
"""


agent_x = 200
agent_y = 200
offset = 50


"""
testing helper functions
"""


def get_food(offset_x, offset_y):
    food_radius = utils.mass_to_radius(conf.FOOD_MASS)
    return Food(agent_x + offset_x, agent_y + offset_y, food_radius, conf.RED_COLOR)


def assert_shape(t, d0, d1=None):
    assert t.shape[0] == d0
    if d1 is not None:
        assert t.shape[1] == d1


def assert_tensor_eq(t, arr):
    arr_tensor = torch.Tensor(arr)
    arr_tensor = arr_tensor.to(t.dtype)

    if t.dtype == torch.float:
        return torch.allclose(t, arr_tensor)

    assert torch.all(torch.eq(t, arr_tensor)).item()


def assert_all_zero_except(arr, exceptIdx):
    """
    Assert that all elements in the provided array are zero except the element at
    the specified exceptIdx
    """
    for (idx, elt) in enumerate(arr):
        if idx != exceptIdx:
            assert elt == 0


"""
get_avg_angles
"""


angles = conf.ANGLES
avg_angles = get_avg_angles(angles)

assert len(angles) == len(avg_angles)
assert avg_angles == [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

angles = [0, 90, 180, 270]
avg_angles = get_avg_angles(angles)

assert len(angles) == len(avg_angles)
assert avg_angles == [45, 135, 225, 315]


"""
get_direction_score
"""


game = GameState()
model = DeepRLModel()
agent = Agent(game, model, agent_x, agent_y, 20)

assert game
assert model
assert agent

obj_angles = torch.Tensor([]).to(torch.float)
obj_dists = torch.Tensor([]).to(torch.float)

score = get_direction_score(agent, obj_angles, obj_dists, 0, 90)
assert score == 0

score = get_direction_score(agent, obj_angles, obj_dists, 22.5, 67.5)
assert score == 0

obj_angles = torch.Tensor([45, 45]).to(torch.float)
obj_dists = torch.Tensor([100, 200]).to(torch.float)

score = get_direction_score(agent, obj_angles, obj_dists, 0, 22.5)
assert score == 0

score = get_direction_score(agent, obj_angles, obj_dists, 22.5, 67.5)
assert score > 0

# Moving objects closer increases the score
obj_angles = torch.Tensor([45, 45]).to(torch.float)
obj_dists = torch.Tensor([50, 100]).to(torch.float)

new_score = get_direction_score(agent, obj_angles, obj_dists, 22.5, 67.5)
assert new_score > score


"""
get_obj_poses_tensor
"""


food_right = get_food(offset, 0)
food_up_right = get_food(offset, -offset)
food_up = get_food(0, -offset)
food_up_left = get_food(-offset, -offset)
food_left = get_food(-offset, 0)
food_down_left = get_food(-offset, offset)
food_down = get_food(0, offset)
food_down_right = get_food(offset, offset)

food_list = [
    food_right,
    food_up_right,
    food_up,
    food_up_left,
    food_left,
    food_down_left,
    food_down,
    food_down_right,
]

# Assert that dimensions returned are correct
t = get_obj_poses_tensor([food_right])
assert_shape(t, 1, 2)
t = get_obj_poses_tensor([food_right, food_left])
assert_shape(t, 2, 2)
t = get_obj_poses_tensor([food_right, food_left, food_up, food_down])
assert_shape(t, 4, 2)

# Assert values are correct
t = get_obj_poses_tensor([food_right])
assert_tensor_eq(t, [[food_right.x_pos, food_right.y_pos]])
t = get_obj_poses_tensor(food_list)
assert_tensor_eq(t, [
    [food_right.x_pos, food_right.y_pos],
    [food_up_right.x_pos, food_up_right.y_pos],
    [food_up.x_pos, food_up.y_pos],
    [food_up_left.x_pos, food_up_left.y_pos],
    [food_left.x_pos, food_left.y_pos],
    [food_down_left.x_pos, food_down_left.y_pos],
    [food_down.x_pos, food_down.y_pos],
    [food_down_right.x_pos, food_down_right.y_pos],
])


"""
get_diff_tensor
"""


t = get_diff_tensor(agent, [food_right])
assert_shape(t, 1, 2)
assert_tensor_eq(t, [[offset, 0]])

t = get_diff_tensor(agent, food_list)
assert_shape(t, 8, 2)
assert_tensor_eq(t, [
    [offset, 0],
    [offset, -offset],
    [0, -offset],
    [-offset, -offset],
    [-offset, 0],
    [-offset, offset],
    [0, offset],
    [offset, offset],
])


"""
get_dists_tensor
"""


t = get_diff_tensor(agent, [food_right])
t = get_dists_tensor(t)
assert_shape(t, 1)
assert_tensor_eq(t, [offset])

t = get_diff_tensor(agent, [food_right, food_up, food_left, food_down])
t = get_dists_tensor(t)
assert_shape(t, 4)
assert_tensor_eq(t, [offset, offset, offset, offset])

t = get_diff_tensor(agent, food_list)
t = get_dists_tensor(t)
assert_shape(t, 8)
offset_diag = math.sqrt(2 * (offset ** 2))
assert_tensor_eq(t, [offset, offset_diag, offset, offset_diag,
                     offset, offset_diag, offset, offset_diag])


"""
get_filtered_angles_tensor
"""


t = get_diff_tensor(agent, [food_right])
t = get_filtered_angles_tensor(t)
assert_shape(t, 1)
assert_tensor_eq(t, [0.0])

t = get_diff_tensor(agent, [food_right, food_up])
t = get_filtered_angles_tensor(t)
assert_shape(t, 2)
assert_tensor_eq(t, [0.0, 90.0])

t = get_diff_tensor(agent, [food_right, food_up, food_left, food_down])
t = get_filtered_angles_tensor(t)
assert_shape(t, 4)
assert_tensor_eq(t, [0, 90, 180, 270])

t = get_diff_tensor(
    agent, [food_up_right, food_up_left, food_down_left, food_down_right])
t = get_filtered_angles_tensor(t)
assert_shape(t, 4)
assert_tensor_eq(t, [45, 135, 225, 315])


"""
get_direction_score
"""


# No objs -> no score
objs = []
scores = get_direction_scores(agent, objs)
for score in scores:
    assert score == 0


scores = get_direction_scores(agent, [food_right])
assert scores[0] > 0
assert_all_zero_except(scores, 0)

scores = get_direction_scores(agent, [food_up_right])
assert scores[1] > 0
assert_all_zero_except(scores, 1)

scores = get_direction_scores(agent, [food_up])
assert scores[2] > 0
assert_all_zero_except(scores, 2)

scores = get_direction_scores(agent, [food_up_left])
assert scores[3] > 0
assert_all_zero_except(scores, 3)

scores = get_direction_scores(agent, [food_left])
assert scores[4] > 0
assert_all_zero_except(scores, 4)

scores = get_direction_scores(agent, [food_down_left])
assert scores[5] > 0
assert_all_zero_except(scores, 5)

scores = get_direction_scores(agent, [food_down])
assert scores[6] > 0
assert_all_zero_except(scores, 6)

scores = get_direction_scores(agent, [food_down_right])
assert scores[7] > 0
assert_all_zero_except(scores, 7)


# Two objs in two directions -> two scores
objs = [food_right, food_up_right]
scores = get_direction_scores(agent, objs)
assert len(scores) == len(conf.ANGLES)
assert scores[0] > 0
assert scores[1] > 0
assert scores[0] > scores[1]  # This food is closer -> score higher
for i in range(2, len(conf.ANGLES)):
    assert scores[i] == 0

# Two foods which are equidistant in different directions -> same score
scores = get_direction_scores(agent, [food_up_right, food_down_right])
score1 = scores[1]
score2 = scores[-1]
assert score1 == score2

# Two objects -> twice the score
scores = get_direction_scores(
    agent, [food_up_right, food_up_right, food_down_right])
score1 = scores[1]
score2 = scores[-1]
assert score1 == 2 * score2
