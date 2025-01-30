# S2GNN

First, extract the Electricity data in the `/datasets/raw_data/` directory. Run `run_datasets.sh` to obtain the processed dataset, then execute `run.sh`. To predict different lengths, manually modify the configuration file.

# S2GNN
S2GNN: Improving Air Quality Forecasting by Leveraging Meteorological Information

## Dependencies
```bash
# Install Python
conda create -n S2GNN python=3.11
conda activate S2GNN
# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```
## Datasets
Processed datasets and raw data can be downloaded at: https://mega.nz/folder/AeVknA4C#MuQITYW9YPcaRX6w9uk_Hg

## Implementation
```bash
python experiments/train.py -c models/S2GNN/PEMS04.py -g 0
python experiments/train.py -c models/S2GNN/Electricity.py -g 0
```
### To run other baselines, for example: 
```bash
python experiments/train.py -c baselines/iTransformer/BJAQ-NCDC.py -g 0
```