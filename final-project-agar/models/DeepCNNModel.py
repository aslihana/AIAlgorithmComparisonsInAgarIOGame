from models.ModelInterface import ModelInterface
from model_utils.ReplayBuffer import ReplayBuffer
from actions import Action
from utils import get_random_action

from collections import deque
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from skimage import transform

import matplotlib.pyplot as plt

# configurable hyperparameters
BATCH_SIZE = 32


# CNN which takes in the game state as stack of frames and returns Q-values for each possible action
class CNN(nn.Module):
    def __init__(self, tau, downsample_size):
        super(CNN, self).__init__()
        self.tau = tau
        self.input_dim = downsample_size
        self.output_dim = len(Action)

        self.convnet = nn.Sequential(
            nn.Conv2d(self.tau, 32, kernel_size=8, stride=1),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc_input_dim = self.calc_cnn_out_dim()

        self.fcnet = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.reshape(-1, self.fc_input_dim)
        return self.fcnet(x)

    def calc_cnn_out_dim(self):
        return self.convnet(torch.zeros(1, self.tau, *self.input_dim)).flatten().shape[0]

# tau is the number of frames to stack to create one "state"
class DeepCNNModel(ModelInterface):
    def __init__(self, tau=4, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay_factor=0.99,
                 replay_buf_capacity=10000, replay_buf_prefill_amt=1000, lr=0.001,
                 downsample_size=(112, 112), batch_size=32, camera_follow=True):
        super(DeepCNNModel, self).__init__()
        self.camera_follow = camera_follow
        self.downsample_size = downsample_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.epsilon = eps_start
        self.end_epsilon = eps_end
        self.epsilon_decay_fac = eps_decay_factor
        self.replay_buffer = ReplayBuffer(replay_buf_capacity, replay_buf_prefill_amt)

        self.net = CNN(tau, downsample_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.net.to(self.device)

        self.step_count = 0
        self.net_update_count = 0

        # initialize tau-sized frame buffers with zeros
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        for i in range(self.tau):
            self.state_buffer.append(np.zeros(downsample_size))
            self.next_state_buffer.append(np.zeros(downsample_size))
        
        # target network for more stable error
        self.target_net = deepcopy(self.net)


    def get_action(self, state):
        """Used when playing the actual game"""
        if random.random() < self.end_epsilon:
            return get_random_action()
        else:
            pp_state = self.preprocess_state(state)
            self.state_buffer.append(pp_state)
            stacked_state = np.stack([self.state_buffer])
            q_vals = self.net(torch.FloatTensor(stacked_state).to(self.device)).to(self.device)
            action_idx = torch.argmax(q_vals).item()
            return Action(action_idx)

    def get_stacked_action(self, stacked_state):
        """Given the current (stacked) game state, determine what action the model will output"""
        # take a random action epsilon fraction of the time
        if random.random() < self.epsilon:
            return get_random_action()
        # otherwise, take the action which maximizes expected reward
        else:
            q_values = self.net(torch.FloatTensor(stacked_state).to(self.device))
            action_idx = torch.argmax(q_values).item()
            return Action(action_idx)

    def optimize(self):
        """Given reward received, optimize the model"""
        # wait for a full training batch before doing any optimizing
        if len(self.replay_buffer) < self.batch_size:
            return

        self.optimizer.zero_grad()
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def remember(self, state, action, next_state, reward, done):
        """Update replay buffer with what model chose to do"""
        # stack last tau states buffers into single array
        stacked_state = np.stack(self.state_buffer)
        stacked_next_state = np.stack(self.next_state_buffer)

        # push to memory
        self.replay_buffer.push((deepcopy(stacked_state), action,
                                 deepcopy(stacked_next_state), reward, done))

    def calculate_loss(self, batch):
        # set net back to training mode for optimizing
        self.net.train()

        states, actions, next_states, rewards, dones = zip(*batch)

        # convert actions from enum to ints
        actions = np.array([action.value for action in actions])

        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        qvals = torch.gather(self.net(states_t), 1, actions_t.unsqueeze(1))
        qvals_next = torch.max(self.target_net(next_states_t), dim=-1)[0].detach()
        qvals_next[dones_t] = 0         # zero out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = self.loss_fn(qvals, expected_qvals.reshape(-1, 1))
        return loss

    def sync_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def preprocess_state(self, state):
        # convert RGB to grayscale via relative luminance
        gray_state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        # size down the image to speed up training
        resized_state = transform.resize(gray_state, self.downsample_size, mode='constant')
        return resized_state