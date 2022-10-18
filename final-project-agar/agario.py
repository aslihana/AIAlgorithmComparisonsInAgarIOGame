import pygame
import config as conf
from gamestate import start_game
from models.DeepSarsaModel import DeepSarsaModel
from models.HeuristicModel import HeuristicModel
from models.RandomModel import RandomModel
from models.DeepCNNModel import DeepCNNModel
from models.DeepRLModel import DeepRLModel

ai_models = [
    ('Random', RandomModel(5, 10)),
    ('CNN', DeepCNNModel()),
    ('RL', DeepRLModel()),
    ('DeepSarsa', DeepSarsaModel()),
    ('Heuristic', HeuristicModel()),
    ('Brain', HeuristicModel()),
]
start_game(ai_models)
