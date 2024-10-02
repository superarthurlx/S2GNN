import os
import sys
import time
sys.path.append(os.path.abspath(__file__ + '/../..'))
from argparse import ArgumentParser

from basicts import launch_runner, BaseRunner
import pdb

def inference(cfg: dict, runner: BaseRunner, ckpt: str = None, batch_size: int = 1):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    # init model
    cfg.TEST.DATA.BATCH_SIZE = batch_size
    runner.model.eval()
    runner.setup_graph(cfg=cfg, train=False)
    # load model checkpoint
    runner.load_model(ckpt_path=ckpt)
    # inference & speed
    t0 = time.perf_counter()
    runner.test_process(cfg)
    elapsed = time.perf_counter() - t0

    print('##############################')
    runner.logger.info('%s: %0.8fs' % ('Speed', elapsed))
    runner.logger.info('# Param: {0}'.format(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)))

if __name__ == '__main__':
    MODEL_NAME = 'CrossGNN'
    DATASET_NAME = 'PEMS08'
    BATCH_SIZE = 64
    GPUS = '0'

    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-m', '--model', default=MODEL_NAME, help='model name')
    parser.add_argument('-d', '--dataset', default=DATASET_NAME, help='dataset name')
    parser.add_argument('-g', '--gpus', default=GPUS, help='visible gpus')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('-s', '--seed', default=0, help='visible gpus')
    args = parser.parse_args()

    cfg_path = 'baselines\\CrossGNN\\PEMS08.py'
    ckpt_path = 'checkpoints\\CrossGNN_PEMS08_100_96_12\\a5ab9289d7b76f45eca4cca2e6116845\\CrossGNN_best_val_MAE.pt'
    # pdb.set_trace()
    launch_runner(cfg_path, inference, (ckpt_path, args.batch_size), devices=args.gpus)

    # python experiments/inference.py -c D:\\实验结果\\ICLR\\模型预测结果\\CrossGNN_PEMS\\CrossGNN_PEMS08_100_96_12\\3c6f6b04639c6756d3eb5c38ba7bef2e\\PEMS08.py --gpus 0 --seed 0
