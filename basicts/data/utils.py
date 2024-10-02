import pickle
import os
import pdb
# import joblib

def save_data(index, data, dirs, history_seq_len, future_seq_len, norm_each_channel):
    with open(dirs["output_dir"] + dirs["index"].format(history_seq_len, future_seq_len, norm_each_channel), "wb") as f:
        pickle.dump(index, f)

    with open(dirs["output_dir"] + dirs["data"].format(history_seq_len, future_seq_len, norm_each_channel), "wb") as f:
        pickle.dump(data, f)
    # joblib.dump(data, dirs["output_dir"] + dirs["data"].format(history_seq_len, future_seq_len, norm_each_channel))

def mkdir(path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))