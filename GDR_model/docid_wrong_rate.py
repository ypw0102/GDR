import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import pickle

import nltk
import pandas as pd
import time
import torch
import pytorch_lightning as pl

from main_metrics import recall, MRR100
from main_models_old import T5FineTuner, l1_query, decode_token
from main_utils_old import set_seed, get_ckpt, dec_2d, numerical_decoder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm
from transformers import T5Tokenizer

#print(1)
#nltk.download('punkt')
#print(2)
print(torch.__version__)  # 1.10.0+cu113
print(pl.__version__)  # 1.4.9

logger = None
YOUR_API_KEY = 'your api key' # wandb token, please get yours from wandb portal

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

if __name__ == "__main__":
    print(1)