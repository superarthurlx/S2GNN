import os
import sys
import shutil
import pickle
import argparse
import pandas as pd 
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform

import pdb

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
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.

    # read geo data
    geofile = pd.read_csv(graph_file_path)
    geo_ids = list(geofile['geo_id'])
    num_nodes = len(geo_ids)
    geo_to_ind = {}
    geo_to_rc = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index
    for i in range(geofile.shape[0]):
        geo_to_rc[geofile['geo_id'][i]] = [geofile['row_id'][i], geofile['column_id'][i]]
    len_row = max(list(geofile['row_id'])) + 1
    len_column = max(list(geofile['column_id'])) + 1

    # read data
    gridfile = pd.read_csv(data_file_path)
    data_col = ['pickup', 'dropoff']
    data_col.insert(0, 'time')
    data_col.insert(1, 'row_id')
    data_col.insert(2, 'column_id')
    gridfile = gridfile[data_col]
    timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(geo_ids))])
    idx_of_timesolts = dict()
    if not gridfile['time'].isna().any():  # 时间没有空值
        timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        timesolts = np.array(timesolts, dtype='datetime64[ns]')
        for idx, _ts in enumerate(timesolts):
            idx_of_timesolts[_ts] = idx
    # 转3-d数组
    feature_dim = len(gridfile.columns) - 3
    df = gridfile[gridfile.columns[-feature_dim:]]
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
    data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
    print("raw time series shape: {0}".format(data.shape))
    pdb.set_trace()

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

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)
    # copy adj
    if os.path.exists(args.graph_file_path):
        # copy
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        # generate and copy
        generate_adj_pems03()
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 96
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 288

    DATASET_NAME = "NYCTaxi"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.grid".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/{0}.geo".format(DATASET_NAME)

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
