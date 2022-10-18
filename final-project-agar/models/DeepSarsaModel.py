from collections import deque
import torch
import numpy as np
import random
import torch
import torch.nn as nn
import math

import xlsxwriter

from models.ModelInterface import ModelInterface
from model_utils.ReplayBuffer import ReplayBuffer
from actions import Action
import config as conf
import utils

STATE_ENCODING_LENGTH = 9
# INERTIA_PROB = 0.05
# NOISE = 0
# STUCK_BUFFER_SIZE = 20

# 0.2 is too high
ANGLE_PENALTY_FACTOR = 0.05

# Anything further than this will (likely, unless very large) be outside
# of the agent's field of view
MAX_DIST = math.sqrt(conf.SCREEN_WIDTH ** 2 + conf.SCREEN_HEIGHT ** 2) / 2


# -------------------------------
# Other helpers
# -------------------------------


def get_avg_angles(angles):
    """
    For example, this would go from conf.ANGLES of [0, 90, 180, 270] to
    [45, 135, 225, 315]

    NOTE this is effectively a closure that should only be run once
    """
    angles = angles + [360]
    avg_angles = []
    for idx in range(0, len(angles) - 1):
        angle = angles[idx]
        next_angle = angles[idx + 1]
        avg_angle = (angle + next_angle) / 2
        avg_angles.append(avg_angle)
    return avg_angles


avg_angles = get_avg_angles(conf.ANGLES)


# def get_all_angles(angles):
#     """
#     Get list of all angles and average angles:
#     [0, 22.5, 45, ..., 360]
#     """
#     angles = angles + [360]
#     all_angles = []
#     for idx in range(0, len(angles) - 1):
#         angle = angles[idx]
#         next_angle = angles[idx + 1]
#         avg_angle = (angle + next_angle) / 2
#         all_angles.append(angle)
#         all_angles.append(avg_angle)
#     all_angles = all_angles + [360]
#     return all_angles


# all_angles = get_all_angles(conf.ANGLES)


def get_direction_score(agent, obj_angles, obj_dists, min_angle, max_angle):
    """
    Returns score for all objs that are between min_angle and max_angle relative
    to the agent. Gives a higher score to objects which are closer. Returns 0 if
    there are no objects between the provided angles.

    Parameters

        agent      (Agent)        : player to compute state relative to
        obj_angles (torch.Tensor) : angles between agent and each object
        obj_dists  (torch.Tensor) : distance between agent and each object
        min_angle  (number)       :
        max_angle  (number)       : greater than min_angle

    Returns

        score (number)
    """
    if min_angle is None or max_angle is None or min_angle < 0 or max_angle < 0:
        raise Exception('max_angle and min_angle must be positive numbers')
    elif min_angle >= max_angle:
        raise Exception('max_angle must be larger than min_angle')

    filter_mask_tensor = (obj_angles < max_angle) & (obj_angles >= min_angle)
    filtered_obj_dists = obj_dists[filter_mask_tensor]

    if filtered_obj_dists.shape[0] == 0:
        return 0

    # If just number of food, use this:
    # return filtered_obj_dists.shape[0]

    obj_dists_inv = 1 / torch.sqrt(filtered_obj_dists)
    # obj_dists_inv = torch.sqrt(MAX_DIST - filtered_obj_dists) / MAX_DIST
    return torch.sum(obj_dists_inv).item()


def get_obj_poses_tensor(objs):
    """
    Get positions of all objects in a Tensor of shape (n, 2) where the 2 is
    each object's [x, y] position tuple

    Parameters

        objs (list) : objects with get_pos() method implemented

    Returns

        positions (torch.Tensor)
    """
    obj_poses = []
    for obj in objs:
        (x, y) = obj.get_pos()
        obj_poses.append([x, y])
    obj_poses_tensor = torch.Tensor(obj_poses)
    return obj_poses_tensor


def get_diff_tensor(agent, objs):
    """
    Get dx and dy distance between the agent and each object

    Parameters:

        agent (object) : object with get_pos() method implemented
        objs  (list)   : n objects with get_pos() method implemented

    Returns:

        diff tensor (torch.Tensor) of size n by 2
    """
    obj_poses_tensor = get_obj_poses_tensor(objs)
    (agent_x, agent_y) = agent.get_pos()
    agent_pos_tensor = torch.Tensor([agent_x, agent_y])
    diff_tensor = obj_poses_tensor - agent_pos_tensor
    return diff_tensor


def get_dists_tensor(diff_tensor):
    diff_sq_tensor = diff_tensor ** 2
    sum_sq_tensor = torch.sum(diff_sq_tensor, 1)  # sum all x's and y's
    dists_tensor = torch.sqrt(sum_sq_tensor)
    return dists_tensor


def get_filtered_angles_tensor(filtered_diff_tensor):
    diff_invert_y_tensor = filtered_diff_tensor * torch.Tensor([1, -1])
    dx = diff_invert_y_tensor[:, 0]
    dy = diff_invert_y_tensor[:, 1]
    radians_tensor = torch.atan2(dy, dx)
    filtered_angles_tensor = radians_tensor * 180 / math.pi

    # Convert negative angles to positive ones
    filtered_angles_tensor = filtered_angles_tensor + \
                             ((filtered_angles_tensor < 0) * 360.0)
    filtered_angles_tensor = filtered_angles_tensor.to(torch.float)

    return filtered_angles_tensor


def get_direction_scores(agent, objs):
    """
    For each direction (from right around the circle to down-right), compute a
    score quantifying how many and how close the proided objects are in each
    direction.

    Parameters

        agent : Agent
        objs  : list of objects with get_pos() methods

    Returns

        list of numbers of length the number of directions agent can move in
    """
    if len(objs) == 0:
        return np.zeros(len(conf.ANGLES))

    # Build an array to put into a tensor
    diff_tensor = get_diff_tensor(agent, objs)
    dists_tensor = get_dists_tensor(diff_tensor)

    filter_mask_tensor = (dists_tensor <= MAX_DIST) & (dists_tensor > 0)
    filter_mask_tensor = filter_mask_tensor.to(
        torch.bool)  # Ensure type is correct
    fitlered_dists_tensor = dists_tensor[filter_mask_tensor]
    filtered_diff_tensor = diff_tensor[filter_mask_tensor]

    # Invert y dimension since y increases as we go down
    filtered_angles_tensor = get_filtered_angles_tensor(filtered_diff_tensor)

    """
    Calculate score for the conic section immediately in the positive x
    direction of the agent (this is from -22.5 degrees to 22.5 degrees if
    there are 8 allowed directions)

    This calculation is unique from the others because it requires summing the
    state across two edges based on how angles are stored
    """
    zero_to_first_angle = get_direction_score(
        agent,
        filtered_angles_tensor,
        fitlered_dists_tensor,
        avg_angles[-1],
        360)
    last_angle_to_360 = get_direction_score(
        agent,
        filtered_angles_tensor,
        fitlered_dists_tensor,
        0,
        avg_angles[0])
    first_direction_state = zero_to_first_angle + last_angle_to_360

    # Compute score for each conic section
    direction_states = [first_direction_state]

    for i in range(0, len(avg_angles) - 1):
        min_angle = avg_angles[i]
        max_angle = avg_angles[i + 1]
        state = get_direction_score(
            agent,
            filtered_angles_tensor,
            fitlered_dists_tensor,
            min_angle,
            max_angle)
        direction_states.append(state)

    # Return list of scores (one for each direction)
    return direction_states


def get_angle_penalties(angle):
    """
    Penalty of 0 if moving in same direction, negative penalty if moving in a
    different direction, with a more negative value the more different the
    direction is
    """
    if angle is None:
        return torch.zeros(len(conf.ANGLES))
    angle_penalties = torch.Tensor(conf.ANGLES)
    angle_penalties = ((angle_penalties - angle) % 360) * -1 / 360
    return angle_penalties * ANGLE_PENALTY_FACTOR


def get_agent_score(model, state):
    (agents, foods, _viruses, _masses, _time) = state

    # If the agent is dead
    if model.id not in agents:
        return np.zeros((STATE_ENCODING_LENGTH,))

    agent = agents[model.id]
    agent_mass = agent.get_mass()
    score = agent.max_mass - agent.starting_mass

    return score


def encode_agent_state(model, state):
    (agents, foods, _viruses, _masses, _time) = state

    # If the agent is dead
    if model.id not in agents:
        return np.zeros((STATE_ENCODING_LENGTH,))

    agent = agents[model.id]
    agent_mass = agent.get_mass()
    # angle = agent.get_angle()
    # angle_penalites = get_angle_penalties(angle)
    # angle_weights = (1 + angle_penalites).numpy()

    # Compute a list of all cells in the game not belonging to this model's agent
    all_agent_cells = []
    for other_agent in agents.values():
        if other_agent == agent:
            continue
        all_agent_cells.extend(other_agent.cells)

    # Partition all other cells into sets of those larger and smaller than
    # the current agent in aggregate
    # all_larger_agent_cells = []
    # all_smaller_agent_cells = []
    # for cell in all_agent_cells:
    #     if cell.mass >= agent_mass:
    #         all_larger_agent_cells.append(cell)
    #     else:
    #         all_smaller_agent_cells.append(cell)

    # # Compute scores for cells for each direction
    # larger_agent_state = get_direction_scores(agent, all_larger_agent_cells)
    # smaller_agent_state = get_direction_scores(agent, all_smaller_agent_cells)

    # other_agent_state = np.concatenate(
    #     (larger_agent_state, smaller_agent_state))
    food_state = torch.Tensor(get_direction_scores(agent, foods))
    # food_state = food_state * torch.Tensor(angle_weights)

    # noise = np.random.normal(1, NOISE, len(conf.ANGLES))
    # food_state = food_state * torch.Tensor(noise)

    # Normalize
    # food_state = food_state - torch.min(food_state)
    # food_state = food_state / torch.max(food_state)
    # food_state = torch.eq(food_state, torch.max(food_state)).to(torch.float)

    food_state = food_state.numpy()

    # virus_state = get_direction_scores(agent, viruses)
    # mass_state = get_direction_scores(agent, masses)

    # Encode important attributes about this agent
    this_agent_state = [
        agent_mass,
        # len(agent.cells),
        # agent.get_avg_x_pos() / conf.BOARD_WIDTH,
        # agent.get_avg_y_pos() / conf.BOARD_HEIGHT,
        # agent.get_angle() / 360,
        # agent.get_stdev_mass(),
    ]

    encoded_state = np.concatenate((
        this_agent_state,
        food_state,
        # other_agent_state,
        # virus_state,
        # mass_state,
    ))

    return encoded_state


class DQN(nn.Module):
    """
    Neural network model for the deep learning agent with fully connected layers
    """

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_size = 32

        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DeepSarsaModel(ModelInterface):
    """
    Model agent class that contains the deep learning network, as well as performs actions,
    Model agent class that contains the deep learning network, as well as performs actions,
    remembers state for the model, and executes the training.

    Params:
        epsilon: starting epsilon for epsilon greedy action (with probability epsilon does random action)
        min_epsilon: smallest epsilon for the agent
        epsilon_decay: the decay factor for epsilon
        buffer_capacity: how large the replay memory is
        gamma: discount factor
        batch_size: batch size for training
        replay_buffer_learn_thresh: threshold for how full the replay buffer should be before starting training
        lr: learning rate for optimizer
        model: if inserting pretrained model
    """

    def __init__(
            self,
            epsilon=1,
            min_epsilon=0.01,
            epsilon_decay=0.999,
            buffer_capacity=10000,
            gamma=0.99,
            batch_size=64,
            replay_buffer_learn_thresh=0.5,
            lr=1e-3,
            model=None,
    ):
        super().__init__()

        # init replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.replay_buffer_learn_thresh = replay_buffer_learn_thresh

        # Before we start learning, we populate the replay buffer with states
        # derived from taking random actions
        self.learning_start = False

        # init model
        if model:
            self.model = model
        else:
            self.model = DQN(STATE_ENCODING_LENGTH, len(Action))

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

        # run on a GPU if we have access to one in this env
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)

        # target net
        self.target_net = DQN(STATE_ENCODING_LENGTH,
                              len(Action)).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        # other parameters
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        self.prev_action = None

        # self.positions_buffer = deque(maxlen=STUCK_BUFFER_SIZE)
        # self.stuck_count = 0

        # self.state_buffer = deque(maxlen=self.tau)
        # self.next_state_buffer = deque(maxlen=self.tau)

        # Fill the buffers
        # for _ in range(self.tau):
        #     self.state_buffer.append(np.zeros(STATE_ENCODING_LENGTH))
        #     self.next_state_buffer.append(np.zeros(STATE_ENCODING_LENGTH))

    def is_replay_buffer_ready(self):
        """Wait until the replay buffer has reached a certain thresh"""
        return len(self.replay_buffer) >= self.replay_buffer_learn_thresh * self.replay_buffer.capacity

    # def record_position(self, state):
    #     (agents, _foods, _viruses, _masses, _time) = state

    #     # If the agent is dead
    #     if self.id not in agents:
    #         return

    #     agent = agents[self.id]
    #     self.positions_buffer.append(agent.get_pos())

    # def get_radius(self, state):
    #     (agents, _foods, _viruses, _masses, _time) = state

    #     # If the agent is dead
    #     if self.id not in agents:
    #         return 0

    #     agent = agents[self.id]
    #     return agent.get_avg_radius()

    # def is_stuck(self):
    #     if len(self.positions_buffer) < STUCK_BUFFER_SIZE:
    #         return False
    #     positions = torch.Tensor(self.positions_buffer)
    #     maxpos, _ = torch.max(positions, 0)
    #     minpos, _ = torch.min(positions, 0)
    #     diff = maxpos - minpos

    #     if diff[0].item() > 48:
    #         return False
    #     if diff[1].item() > 48:
    #         return False

    #     prod = torch.prod(diff)
    #     return prod.item() < 1200

    def get_policy_action(self, state):
        """ returns an action based on policy from state input"""
        with torch.no_grad():
            state = encode_agent_state(self, state)
            state = torch.Tensor(state).to(self.device)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            action = Action(action)
            return action

    def get_random_action(self):
        """ returns a random action"""
        action = Action(np.random.randint(len(Action)))
        return action

    # def get_prev_action(self, state):
    #     action = self.prev_action

    #     if not action or not utils.is_action_feasible(
    #         action,
    #         self.positions_buffer[-1],
    #         self.get_radius(state)
    #     ):
    #         action = self.get_random_action()

    #     return action

    def get_action(self, state):
        # self.record_position(state)

        if not self.is_replay_buffer_ready() and not self.eval:
            action = self.get_random_action()
        # elif self.is_stuck():
        #     self.stuck_count = 20
        #     action = self.get_prev_action(state)
        # elif self.stuck_count > 0:
        #     action = self.get_prev_action(state)
        #     self.stuck_count = self.stuck_count - 1
        elif self.eval:
            action = self.get_policy_action(state)

            # if random.random() <= INERTIA_PROB:
            #     action = self.get_prev_action(state)
            # else:
            #     action = self.get_policy_action(state)
        elif self.done:
            return None
        elif random.random() > self.epsilon:
            # take the action which maximizes expected reward
            action = self.get_policy_action(state)
        else:
            action = self.get_random_action()

        self.prev_action = action
        return action

    def remember(self, state, action, next_state, reward, done):
        """Update the replay buffer with this example if we are not done yet"""
        if self.done:
            return

        self.replay_buffer.push(
            (encode_agent_state(self, state), action.value, encode_agent_state(self, next_state), reward, done))
        self.done = done

    def optimize(self):
        """Training method for agent"""
        if self.done:
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        if not self.is_replay_buffer_ready():
            return

        if not self.learning_start:
            # Stop taking only random actions
            self.learning_start = True
            print("----LEARNING BEGINS----")

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.Tensor(states).to(self.device)
        actions = torch.LongTensor(list(actions)).to(self.device)
        rewards = torch.Tensor(list(rewards)).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(list(dones)).to(self.device)

        # do Q computation
        currQ = self.model(states).gather(
            1, actions.unsqueeze(1))
        nextQ = self.target_net(next_states)
        # max_nextQ = torch.max(nextQ, 1)[0].detach()
        mean_nextQ = torch.mean(nextQ, 1)[0].detach()
        mask = 1 - dones
        expectedQ = rewards + mask * self.gamma * mean_nextQ

        loss = self.loss(currQ, expectedQ.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon without dipping below min_epsilon"""
        if self.epsilon != self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        return self.epsilon

    def sync_target_net(self):
        """Syncs target and policy net"""
        self.target_net.load_state_dict(self.model.state_dict())
