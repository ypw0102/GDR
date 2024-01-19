import pandas as pd
import pickle
import torch
import os
import re
import random
import csv
import jsonlines
import numpy as np
import pickle
import time
import gzip
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from typing import Any, List, Sequence, Callable
from itertools import islice, zip_longest
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.cluster import MiniBatchKMeans

## Mapping tool

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def lower(x):
    text = tokenizer.tokenize(x)
    id_ = tokenizer.convert_tokens_to_ids(text)
    return tokenizer.decode(id_)
## doc_tac denotes the concatenation of title, abstract and content

nq_dev = pd.read_csv('nq_dev.tsv', \
                     names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'doc_tac', 'language'],\
                     header=None, sep='\t')

nq_train = pd.read_csv('nq_train.tsv', \
                       names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'doc_tac', 'language'],\
                       header=None, sep='\t')

nq_dev['title'] = nq_dev['title'].map(lower)
nq_train['title'] = nq_train['title'].map(lower)
## Concat train doc and validation doc to obtain full document collection

nq_all_doc = nq_train.append(nq_dev)
nq_all_doc.reset_index(inplace = True)
## Remove duplicated documents based on titles

nq_all_doc.drop_duplicates('title', inplace = True)
nq_all_doc.reset_index(inplace = True)

## Construct mapping relation

title_doc = {}
title_doc_id = {}
id_doc = {}
ran_id_old_id = {}
idx = 0
for i in range(len(nq_all_doc)):
    title_doc[nq_all_doc['title'][i]] =  nq_all_doc['doc_tac'][i]
    title_doc_id[nq_all_doc['title'][i]] = idx
    id_doc[idx] = nq_all_doc['doc_tac'][i]
    ran_id_old_id[idx] = nq_all_doc['id'][i]
    idx += 1



with open('kmeans/IDMapping_NQ_bert_512_k30_c30_seed_7.pkl', 'rb') as f:
    kmeans_nq_doc_dict = pickle.load(f)
## random id : newid
new_kmeans_nq_doc_dict_512 = {}
for old_docid in kmeans_nq_doc_dict.keys():
    new_kmeans_nq_doc_dict_512[str(old_docid)] = '-'.join(str(elem) for elem in kmeans_nq_doc_dict[old_docid])

new_kmeans_nq_doc_dict_512_int_key = {}
for key in new_kmeans_nq_doc_dict_512:
    new_kmeans_nq_doc_dict_512_int_key[int(key)] = new_kmeans_nq_doc_dict_512[key]



## merge parallel results
output_bert_base_tensor_nq_qg = []
output_bert_base_id_tensor_nq_qg = []
for num in trange(4):
    with open(f'qg/pkl/NQ_output_tensor_512_content_64_15_{num}.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()
    output_bert_base_tensor_nq_qg.extend(data)

    with open(f'qg/pkl/NQ_output_tensor_512_content_64_15_{num}_id.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()
    output_bert_base_id_tensor_nq_qg.extend(data)

qg_dict = {}
for i in trange(len(output_bert_base_tensor_nq_qg)):
    if(output_bert_base_id_tensor_nq_qg[i] not in qg_dict):
        qg_dict[output_bert_base_id_tensor_nq_qg[i]] = [output_bert_base_tensor_nq_qg[i]]
    else:
        qg_dict[output_bert_base_id_tensor_nq_qg[i]].append(output_bert_base_tensor_nq_qg[i])

## nq_512_qg20.tsv
QG_NUM = 15

qg_file = open("NQ_512_qg.tsv", 'w')

for queryid in tqdm(qg_dict):
    for query in qg_dict[queryid][:QG_NUM]:
        qg_file.write('\t'.join([query, str(ran_id_old_id[int(queryid)]), queryid, new_kmeans_nq_doc_dict_512[queryid]]) + '\n')
        qg_file.flush()

new_kmeans_nq_doc_dict_512_int_key = {}
for key in new_kmeans_nq_doc_dict_512:
    new_kmeans_nq_doc_dict_512_int_key[int(key)] = new_kmeans_nq_doc_dict_512[key]

nq_train['randomid'] = nq_train['title'].map(title_doc_id)
nq_train['id_512'] = nq_train['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)

nq_train_ = nq_train.loc[:, ['query', 'id', 'randomid', 'id_512']]
nq_train_.to_csv('nq_train_doc_newid.tsv', sep='\t', header=None, index=False, encoding='utf-8')

nq_dev['randomid'] = nq_dev['title'].map(title_doc_id)
nq_dev['id_512'] = nq_dev['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)


nq_dev_ = nq_dev.loc[:, ['query', 'id', 'randomid', 'id_512']]
nq_dev_.to_csv('nq_dev_doc_newid.tsv', sep='\t', header=None, index=False, encoding='utf-8')


nq_all_doc_non_duplicate = nq_train.append(nq_dev)
nq_all_doc_non_duplicate.reset_index(inplace = True)

nq_all_doc_non_duplicate['id_512'] = nq_all_doc_non_duplicate['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)

nq_all_doc_non_duplicate['ta'] = nq_all_doc_non_duplicate['title'] + ' ' + nq_all_doc_non_duplicate['abstract']

nq_all_doc_non_duplicate = nq_all_doc_non_duplicate.loc[:, ['ta', 'id', 'randomid','id_512']]
nq_all_doc_non_duplicate.to_csv('nq_title_abs.tsv', sep='\t', header=None, index=False, encoding='utf-8')



queryid_oldid_dict = {}
bertid_oldid_dict = {}
map_file = "./nq_title_abs.tsv"
with open(map_file, 'r') as f:
    for line in f.readlines():
        query, queryid, oldid, bert_k30_c30 = line.split("\t")
        queryid_oldid_dict[oldid] = queryid
        bertid_oldid_dict[oldid] = bert_k30_c30

train_file = "./NQ_doc_content.tsv"
doc_aug_file = open(f"./NQ_doc_aug.tsv", 'w')
with open(train_file, 'r') as f:
    for line in f.readlines():
        docid, _, _, content, _, _, _ = line.split("\t")
        content = content.split(' ')
        add_num = max(0, len(content)-3000) / 3000
        for i in range(10+int(add_num)):
            begin = random.randrange(0, len(content))
            # if begin >= (len(content)-64):
            #     begin = max(0, len(content)-64)
            end = begin + 64 if len(content) > begin + 64 else len(content)
            doc_aug = content[begin:end]
            doc_aug = ' '.join(doc_aug)
            queryid = queryid_oldid_dict[docid]
            bert_k30_c30 = bertid_oldid_dict[docid]
            # doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]) + '\n')
            doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]))
            doc_aug_file.flush()



