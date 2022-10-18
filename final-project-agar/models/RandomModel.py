import numpy as np
import utils
from models.ModelInterface import ModelInterface
from actions import Action


class RandomModel(ModelInterface):
    def __init__(self, min_steps, max_steps):
        super().__init__()

        if min_steps is None or max_steps is None or max_steps < min_steps or min_steps < 1:
            raise ValueError(
                'min_steps must be positive number less than max_steps')

        self.min_steps = min_steps
        self.max_steps = max_steps
        self.steps_remaining = 0
        self.curr_action = None

    def get_action(self, state):
        """
        RandomModel always moves between min_steps and max_steps (inclusive) steps
        in randomly selected direction
        """
        if self.steps_remaining <= 0:
            self.steps_remaining = np.random.randint(
                self.min_steps, self.max_steps)
            self.curr_action = utils.get_random_action()

        self.steps_remaining -= 1
        return self.curr_action

    def optimize(self):
        """no optimization occurs for RandomModel"""
        return

    def remember(self, state, action, next_state, reward, done):
        """no remembering occurs for RandomModel"""
        self.done = done
        return
