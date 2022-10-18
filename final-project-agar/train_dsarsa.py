from gamestate import GameState, start_ai_only_game
from models.DeepSarsaModel import DeepSarsaModel
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepRLModel import DeepRLModel
from model_utils.train_utils import train_models, test_models, get_epsilon_decay_factor
import random
import model_utils.fs_utils as fs
import sys
import xlsxwriter
import xlrd
"""
Constants
"""

PRINT_EVERY = 2000
NUM_CHECKPOINTS = 5

"""
Hyperparameters
"""

START_EPSILON = 1.0
MIN_EPSILON = 0.02
DECAY_EPISODE_WINDOW = 2

GAMMA = 0.8
BATCH_SIZE = 35

REPLAY_BUFFER_LEARN_THRESH = 0.05
REPLAY_BUFFER_CAPACITY = 100000

EPISODES = 5
STEPS_PER_EPISODE = 1000
LEARNING_RATE = 0.001


def train():
    """
    Training loop for training the DeepRLModel
    """
    print("Running Train | Episodes: {} | Steps: {}".format(
        EPISODES, STEPS_PER_EPISODE))

    # Define environment
    env = GameState(with_masses=False, with_viruses=False,
                    with_random_mass_init=False)

    # Define and pass in model parameters
    epsilon_decay = get_epsilon_decay_factor(
        START_EPSILON, MIN_EPSILON, DECAY_EPISODE_WINDOW)
    deep_sarsa_model = DeepSarsaModel(
        epsilon=START_EPSILON,
        min_epsilon=MIN_EPSILON,
        epsilon_decay=epsilon_decay,
        buffer_capacity=REPLAY_BUFFER_CAPACITY,
        replay_buffer_learn_thresh=REPLAY_BUFFER_LEARN_THRESH,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    # define enemy players
    # heuristic_model = HeuristicModel()
    # rand_model_1 = RandomModel(min_steps=5, max_steps=10)
    # rand_model_2 = RandomModel(min_steps=5, max_steps=10)
    # enemies = [heuristic_model, rand_model_1, rand_model_2]
    enemies = []
    model_name = "train_drl_with_others_{}".format(random.randint(0, 2 ** 16))
    train_models(
        env,
        deep_sarsa_model,
        enemies,
        episodes=EPISODES,
        steps=STEPS_PER_EPISODE,
        print_every=PRINT_EVERY,
        model_name=model_name,
        num_checkpoints=NUM_CHECKPOINTS)

    # save the model
    fs.save_net_to_disk(deep_sarsa_model.model, "dsarsa_model")


if __name__ == "__main__":
    train()
