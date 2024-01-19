import os
import pickle
import random
from time import time
import copy
import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.serialization import default_restore_location
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim.lr_scheduler as lr_scheduler
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder
from dataclasses import dataclass
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput
from main_helper_loss import loss_zoo
from typing import Dict, Optional
from main_utils import assert_all_frozen, load_data_infer, \
    load_data, numerical_decoder, dec_2d
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertModel,
    BertTokenizer
)
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput
from torch import nn, Tensor
from tqdm import tqdm
import math
from gensim.summarization.bm25 import BM25
def simple_tok(sent: str):
    return sent.split()

if __name__ == "__main__":

    filename = 'GDR_main/Data_process/NQ_dataset/Self_NQ_ar2_334314/nq_qa_fulldoc_334314.csv'
    out_filename = 'GDR_main/Data_process/NQ_dataset/Self_NQ_ar2_334314/neg_random.pkl'
    df1 = pd.read_csv(
        filename,
        names=["docid", "query", "doc"],header=0, sep='\t', dtype={'docid': str, 'query': str, 'doc': str})
    df1.dropna(axis=0, inplace=True)
    corpus = df1["doc"].tolist()
    query = df1["query"].tolist()
    id = df1["docid"].tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # corpus = ["The little fox ran home",
    #           "dogs are the best ",
    #           "Yet another doc ",
    #           "I see a little fox with another small fox",
    #           "last doc without animals"]

    tok_corpus = [tokenizer.tokenize(s) for s in tqdm(corpus)]
    bm25 = BM25(tok_corpus)
    query = [tokenizer.tokenize(s) for s in tqdm(query)]

    out_dict = {}
    for i in range(len(query)):
        scores = bm25.get_scores(query[i])

        best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:30]
        # for index, b in enumerate(best_docs):
            # print(f"rank {index + 1}: {corpus[b]}")
        out_dict[int(id[i])] = best_docs.copy()

    with open(out_filename, 'wb') as file:
        pickle.dump(out_dict, file)
        file.close()

    with open(out_filename, 'rb') as file:
        in_dict = pickle.load(file)
        file.close()

    print("here")
