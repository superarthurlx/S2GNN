#!/bin/bash
# spatial-temporal forecasting
# python scripts/data_preparation/METR-LA/generate_training_data.py  --history_seq_len 12 --future_seq_len 12
# python scripts/data_preparation/PEMS-BAY/generate_training_data.py --history_seq_len 12 --future_seq_len 12
# python scripts/data_preparation/PEMS03/generate_training_data.py   --history_seq_len 12 --future_seq_len 12
# python scripts/data_preparation/PEMS04/generate_training_data.py   --history_seq_len 12 --future_seq_len 12
# python scripts/data_preparation/PEMS07/generate_training_data.py   --history_seq_len 12 --future_seq_len 12
# python scripts/data_preparation/PEMS08/generate_training_data.py   --history_seq_len 12 --future_seq_len 12

python scripts/data_preparation/SD/generate_training_data.py    --history_seq_len 12 --future_seq_len 12
python scripts/data_preparation/GLA/generate_training_data.py   --history_seq_len 12 --future_seq_len 12
python scripts/data_preparation/GBA/generate_training_data.py   --history_seq_len 12 --future_seq_len 12
python scripts/data_preparation/SD/generate_training_data.py    --history_seq_len 12 --future_seq_len 12

# # long-term time series forecasting
# python scripts/data_preparation/ETTh1/generate_training_data.py        --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/ETTh2/generate_training_data.py        --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/ETTm1/generate_training_data.py        --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/ETTm2/generate_training_data.py        --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/Electricity/generate_training_data.py  --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/Weather/generate_training_data.py      --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/ExchangeRate/generate_training_data.py --history_seq_len 96 --future_seq_len 96
# # python scripts/data_preparation/Illness/generate_training_data.py      --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/Traffic/generate_training_data.py      --history_seq_len 96 --future_seq_len 96

# python scripts/data_preparation/METR-LA/generate_training_data.py      --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/PEMS-BAY/generate_training_data.py     --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/PEMS04/generate_training_data.py       --history_seq_len 96 --future_seq_len 96
# python scripts/data_preparation/PEMS08/generate_training_data.py       --history_seq_len 96 --future_seq_len 96

python scripts/data_preparation/BeijingAirQuality/generate_training_data.py
python scripts/data_preparation/ETTh1/generate_training_data.py --history_seq_len 96 --future_seq_len 12

# python scripts/data_preparation/WaterQuality/prepare_training_data.py

