from gamestate import GameState, start_ai_only_game
import model_utils.fs_utils as fs
import sys

from models.DeepRLModel import DeepRLModel
from models.RandomModel import RandomModel
from models.HeuristicModel import HeuristicModel
from models.DeepCNNModel import DeepCNNModel
from model_utils.train_utils import get_epsilon_decay_factor

# CNN hyperparams
TAU = 4
GAMMA = 0.95
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY_WINDOW = 50
REPLAY_BUF_CAPACITY = 10000
REPLAY_BUF_PREFILL_AMT = 5000
LR = 0.001
DOWNSAMPLE_SIZE = (112, 112)
BATCH_SIZE = 32


DRL = 'drl'
CNN = 'cnn'


def test(model_type, model_name):
    if model_type == DRL:
        agarai_model = DeepRLModel()
        fs.load_net_from_disk(agarai_model.model, model_name)
    elif model_type == CNN:
        agarai_model = DeepCNNModel(
            tau=TAU, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
            eps_decay_factor=get_epsilon_decay_factor(
                EPS_START, EPS_END, EPS_DECAY_WINDOW),
            replay_buf_capacity=REPLAY_BUF_CAPACITY, replay_buf_prefill_amt=REPLAY_BUF_PREFILL_AMT,
            lr=LR, downsample_size=DOWNSAMPLE_SIZE, batch_size=BATCH_SIZE)
        agarai_model.net = fs.load_net_from_disk(agarai_model.net, model_name)
        agarai_model.net.eval()
    else:
        raise ValueError(
            'Invalid model type, please use one of \'cnn\' or \'drl\'')

    agarai_model.eval = True
    main_model = ('AgarAI', agarai_model)

    rand_model_1 = RandomModel(min_steps=5, max_steps=10)
    rand_model_2 = RandomModel(min_steps=5, max_steps=10)
    rand_model_3 = RandomModel(min_steps=5, max_steps=10)
    heur_model_1 = HeuristicModel()
    heur_model_2 = HeuristicModel()
    other_models = [
        ('Random1', rand_model_1),
        ('Random2', rand_model_2),
        ('Random3', rand_model_3),
        # ('Heur1', heur_model_1),
        # ('Heur2', heur_model_2),
    ]
    start_ai_only_game(main_model, other_models)


if __name__ == "__main__":
    num_args = len(sys.argv)

    if num_args == 3:
        model_type = sys.argv[1]
        model_name = sys.argv[2]

        if not (model_type in [DRL, CNN]) or not model_name:
            raise ValueError('Usage: test.py {drl/cnn} {model_name}')

        test(model_type, model_name)
    else:
        raise ValueError('Usage: test.py {drl/cnn} {model_name}')
