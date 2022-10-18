from models.DeepCNNModel import DeepCNNModel
from model_utils.train_utils import train_deepcnn_model, get_epsilon_decay_factor
import xlsxwriter
import xlrd

# model hyperparameters
MODEL_NAME = 'my_cnn_model'
TAU = 4
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_WINDOW = 50
REPLAY_BUF_CAPACITY = 10000
REPLAY_BUF_PREFILL_AMT = 1000
LR = 0.001
DOWNSAMPLE_SIZE = (112, 112)
BATCH_SIZE = 32

# training hyperparameters
ADVERSARY_MODELS = []
FRAME_SKIP = 4
UPDATE_FREQ = 4
TARGET_NET_SYNC_FREQ = 1000
MAX_EPS = 1
MAX_STEPS_PER_EP = 500
WINDOW_SIZE = 10
ENABLE_PREFILL_BUFFER = True

def CreateFile(hyperparameters):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('cnn_trainingInfo.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    for item, value in (hyperparameters):
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, value)
        row += 1

    merge_format = workbook.add_format({
        'align': 'center',
        'valign': 'vcenter'
      })
    worksheet.merge_range('A7:C7', '',merge_format)
    workbook.close()
if __name__ == '__main__':
    hyperparameters = (
        ['Tau: ', '{:.1f}'.format(TAU)],
        ['Gamma: ', '{:.3f}'.format(GAMMA)],
        ['Learning Rate:', '{:.3f}'.format(LR)],
        ['Batch Size:', '{:.1f}'.format(BATCH_SIZE)],
        ['Episode:', '{:.1f}'.format(MAX_EPS)],
        ['Step per Episode:', '{:.1f}'.format(MAX_STEPS_PER_EP)],
    )
    CreateFile(hyperparameters)

cnn_model = DeepCNNModel(tau=TAU, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
                         eps_decay_factor=get_epsilon_decay_factor(EPS_START, EPS_END, EPS_DECAY_WINDOW),
                         replay_buf_capacity=REPLAY_BUF_CAPACITY, replay_buf_prefill_amt=REPLAY_BUF_PREFILL_AMT,
                         lr=LR, downsample_size=DOWNSAMPLE_SIZE, batch_size=BATCH_SIZE)

train_deepcnn_model(cnn_model, MODEL_NAME, ADVERSARY_MODELS, frame_skip=FRAME_SKIP,
                    update_freq=UPDATE_FREQ, target_net_sync_freq=TARGET_NET_SYNC_FREQ,
                    max_eps=MAX_EPS, max_steps_per_ep=MAX_STEPS_PER_EP,
                    mean_window=WINDOW_SIZE, prefill_buffer=ENABLE_PREFILL_BUFFER)