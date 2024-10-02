import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

from generate_adj_mx import generate_adj_wp
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform
from basicts.data.embedding import time_in_day, day_in_week, day_in_month, day_in_year
from basicts.data.utils import save_data

import pdb

def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """
    target_dict = {0:'KM', 1:'TN', 2:'TP'}
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel

    # read data
    data = np.load(data_file_path)["data"]
    stations = np.load(data_file_path, allow_pickle=True)["stations"]

    time_index = pd.to_datetime(np.load(data_file_path)["time_index"])
    rain = data[..., [-1]]
    data = data[..., [target_channel]]
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape 
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    dirs = {}
    dirs['output_dir'] = output_dir.format(target_dict[target_channel])
    dirs['index'] = "/index_in_{0}_out_{1}_norm_each_channel_{2}.pkl"
    dirs['data'] = "/data_in_{0}_out_{1}_norm_each_channel_{2}.pkl"
    dirs['scaler'] = "/scaler_in_{0}_out_{1}_norm_each_channel_{2}.pkl"

    if not os.path.exists(dirs['output_dir']):
        os.makedirs(dirs['output_dir'])

    # normalize data
    scaler = standard_transform
    # Following related works (e.g. informer and autoformer), we normalize each channel separately.
    data_norm = scaler(data, dirs, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm, rain]

    if add_time_of_day:
        # numerical time_of_day
        tod = (
            time_index.values - time_index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = time_index.dayofweek / 7

        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (time_index.day - 1 ) / 31 # time_index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (time_index.dayofyear - 1) / 366 # time_index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index

    data = {}
    data["processed_data"] = processed_data

    save_data(index, data, dirs, history_seq_len, future_seq_len, norm_each_channel)

    generate_adj_wp(stations, directed=False)
    shutil.copyfile(graph_file_path, output_dir.format(target_dict[target_channel]) + "/adj_mx.pkl")

if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 186
    FUTURE_SEQ_LEN = 18 # 3天 7天 31天

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = 0       # target channel(s) KM TN TP
    STEPS_PER_DAY = 6          # every 4 hour

    DATASET_NAME = "WaterQuality"      # sampling frequency: every 4 hour
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True                  # if add day_of_month feature
    DOY = True                  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME + "{0}" # 跟TARGET_CHANNEL保持一致
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=int,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
