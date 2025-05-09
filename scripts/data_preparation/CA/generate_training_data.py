import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

import torch

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform
from basicts.data.utils import save_data

import pdb
# Dataset Description: 
#   LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

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
    norm_each_channel = args.norm_each_channel

    # read data
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
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
    dirs['output_dir'] = output_dir
    dirs['index'] = "/index_in_{0}_out_{1}_norm_each_channel_{2}.pkl"
    dirs['data'] = "/data_in_{0}_out_{1}_norm_each_channel_{2}.pkl"
    dirs['scaler'] = "/scaler_in_{0}_out_{1}_norm_each_channel_{2}.pkl"

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, dirs, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]

    if add_time_of_day:
        # numerical time_of_day
        tod = (
            df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek / 7

        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df.index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
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

   # # copy adj
   #  adj_mx = np.load(graph_file_path)
   #  with open(output_dir + "/adj_mx.pkl", "wb") as f:
   #      pickle.dump(adj_mx, f)
   #  # copy adj meta data
   #  shutil.copyfile(graph_file_path, output_dir + "/adj_meta.csv")


    # adj 
    adj = np.load(graph_file_path)
    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(adj, f)

   # # copy adj
   #  shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")
   #  # copy adj meta data
   #  shutil.copyfile(graph_file_path, output_dir + "/adj_meta.pkl")



if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 96
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]                   # target channel(s)

    DATASET_NAME = "CA"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = False                  # if add day_of_month feature
    DOY = False                  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.npy".format(DATASET_NAME)
    GRAPH_METE_PATH = "datasets/raw_data/{0}/meta_{0}.csv".format(DATASET_NAME)

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
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
