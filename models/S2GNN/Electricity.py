import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner, MyTimeSeriesForecastingRunner
from basicts.losses import masked_mae
from .arch import S2GNN
from .arch import s2gnn_loss
from basicts.utils.serialization import load_pkl

import pdb

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "S2GNN model configuration"
CFG.RUNNER = MyTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Electricity"
CFG.DATASET_TYPE = "Electricity"
CFG.DATASET_INPUT_LEN = 96
CFG.DATASET_OUTPUT_LEN = 96
CFG.NUM_NODES = 321
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0
CFG.RESCALE = False # default True
CFG.CHANNEL_NORM = True # default True
CFG.SAVE_RESULT = True # default False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = int(sys.argv[-1])
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "S2GNN"
CFG.MODEL.ARCH = S2GNN
CFG.MODEL.PARAM = {
    "num_nodes": CFG.NUM_NODES,
    "input_len": CFG.DATASET_INPUT_LEN,
    "input_dim": 1,
    "embed_dim": 32,
    "embed_sparsity": 0.5,
    "embed_std": 0.01,
    "constant": 10,
    "output_len": CFG.DATASET_OUTPUT_LEN,
    "num_layer": 5,
    "num_gnn_layer": 10,
    "init": "RWR", # KI Random  
    "stop_grad": 100,
    "if_rp": True,
    'gnn_type': 'chebnetii',
    # -------node&time-----------
    "if_node": True,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    # -------kernel-------
    "if_feat": False,
    "n_kernel": 20, 
    "topk": 1,
    # -------Bernstein appro-------
    "use_bern": False
}

CFG.MODEL.FORWARD_FEATURES = [0, 1, 2] # traffic flow, time in day
CFG.MODEL.TARGET_FEATURES = [0] # traffic flow

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.005,
    "weight_decay": 0.01,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 30, 50, 80],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.DATASET_NAME), str(CFG.TRAIN.NUM_EPOCHS),  str(CFG.DATASET_INPUT_LEN), str(CFG.DATASET_OUTPUT_LEN)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = False
# CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]
# CFG.EVAL.HORIZONS = [3, 6, 12]
