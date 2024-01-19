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

docnum = '334314'
aug_len = 16

def process_ori_data():
    if not os.path.exists('nq_qa_fulldoc1.csv'):

        with open(f'nq-train_num58622-corpus_num{docnum}/nq-{docnum}-corpus.pickle', 'rb') as f:
            data1 = pickle.load(f)
        f.close()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def lower(x):
            text = tokenizer.tokenize(x)
            id_ = tokenizer.convert_tokens_to_ids(text)
            return tokenizer.decode(id_)

        ## Construct mapping relation

        title_doc = {}
        title_doc_id = {}
        id_doc = {}
        ran_id_old_id = {}
        idx = 0
        doc_id = []
        title = []
        content = []
        for i in tqdm(range(len(data1))):
            doc_id.append(idx)
            title.append(data1[i]['title'])
            content.append(lower(data1[i]['text']))
            title_doc[data1[i]['title']] = lower(data1[i]['text'])
            title_doc_id[data1[i]['title']] = idx
            id_doc[idx] = lower(data1[i]['text'])
            idx += 1

        all_dict = {"docid": doc_id.copy(), "title": title, "content": content}
        df_all = pd.DataFrame(data=all_dict)
        df_all.to_csv('nq_qa_fulldoc.csv', sep='\t', index=False, encoding='utf-8')
    else:
        df_all = pd.read_csv('nq_qa_fulldoc.csv',
                              encoding='utf-8', sep='\t')

    title_doc_id = {}
    for i in trange(len(df_all)):
        title_doc_id[df_all['title'][i]] = i

    origin_new_id = {}
    for i in trange(len(df_all)):
        origin_new_id[i] = i

    if not os.path.exists("NQ_doc_content_{}.tsv"):
        file_pool = open("NQ_doc_content_{}.tsv".format(docnum), 'w')

        for i in trange(len(df_all)):
            file_pool.write('\t'.join([str(df_all['docid'][i]), str(origin_new_id[df_all['docid'][i]]),
                                       str(df_all['title'][i]), str(df_all['content'][i]),
                                       str(df_all['title'][i]) + str(df_all['content'][i])]) + '\n')
            file_pool.flush()

    return df_all, title_doc_id, origin_new_id

def load_kmeans(dir):
    with open(dir, 'rb') as f:
        kmeans_nq_doc_dict = pickle.load(f)
    ## random id : newid
    new_kmeans_nq_doc_dict_100 = {}
    for old_docid in kmeans_nq_doc_dict.keys():
        new_kmeans_nq_doc_dict_100[str(old_docid)] = '-'.join(str(elem) for elem in kmeans_nq_doc_dict[old_docid])

    return new_kmeans_nq_doc_dict_100


def load_qg(num_cuda):
    ## merge parallel results
    output_bert_base_tensor_nq_qg = []
    output_bert_base_id_tensor_nq_qg = []
    for num in trange(num_cuda):
        with open(f'qg/pkl/NQ_output_tensor_512_content_64_5_{num}.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_tensor_nq_qg.extend(data)

        with open(f'qg/pkl/NQ_output_tensor_512_content_64_5_{num}_id.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_id_tensor_nq_qg.extend(data)
    qg_dict = {}
    for i in trange(len(output_bert_base_tensor_nq_qg)):
        if (output_bert_base_id_tensor_nq_qg[i] not in qg_dict):
            qg_dict[output_bert_base_id_tensor_nq_qg[i]] = [output_bert_base_tensor_nq_qg[i]]
        else:
            qg_dict[output_bert_base_id_tensor_nq_qg[i]].append(output_bert_base_tensor_nq_qg[i])
    return qg_dict

if __name__ == "__main__":
    df_all, title_doc_id, origin_new_id = process_ori_data()
    new_kmeans_nq_doc_dict_100 = load_kmeans(dir='kmeans/IDMapping_NQ_real-{}_bert_100_k30_c30_seed_7.pkl'.format(docnum))
    new_kmeans_nq_doc_dict_100_int_key = {}
    for key in new_kmeans_nq_doc_dict_100:
        new_kmeans_nq_doc_dict_100_int_key[int(key)] = new_kmeans_nq_doc_dict_100[key]

    ## nq_100_qg20.tsv
    QG_NUM = 5
    qg_file = open("NQ_100_qg.tsv", 'w')

    qg_dict = load_qg(num_cuda=12)
    for queryid in tqdm(qg_dict):
        for query in qg_dict[queryid][:QG_NUM]:
            qg_file.write('\t'.join(
                [query, queryid, queryid, new_kmeans_nq_doc_dict_100[queryid]]) + '\n')
            qg_file.flush()

    ##############################################################################
    with open('nq-train_num58622-corpus_num{}/nq-58622-data.pickle'.format(docnum), 'rb') as f:
        data2 = pickle.load(f)
    f.close()

    ## Mapping tool

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def lower(x):
        text = tokenizer.tokenize(x)
        id_ = tokenizer.convert_tokens_to_ids(text)
        return tokenizer.decode(id_)

    title_doc = {}
    title_doc_id = {}
    id_doc = {}
    ran_id_old_id = {}
    idx = 0
    train_data = data2["train"]
    dev_data = data2["dev"]

    title = []
    text = []
    docid = []
    query = []
    queryid = []
    # train_data.extend(dev_data)
    # train_dev_data = train_data
    for i in tqdm(range(len(train_data))):
    #for i in tqdm(range(10)):
        queryid.append(train_data[i]['query_id'])
        query.append(train_data[i]['query'])
        docid_new = []
        for j in range(len(train_data[i]['positive_passages'])):
            title_doc[train_data[i]['positive_passages'][j]['title']] = lower(train_data[i]['positive_passages'][j]['text'])
            docid_new.append(str(train_data[i]['positive_passages'][j]['docid']))
            title_doc_id[train_data[i]['positive_passages'][j]['title']] = train_data[i]['positive_passages'][j]['docid']
            id_doc[train_data[i]['positive_passages'][j]['docid']] = title_doc[train_data[i]['positive_passages'][j]['title']]
            idx += 1
        docid.append(",".join(docid_new))
    train_dict = {"query":query.copy(),"docid":docid.copy()}
    df_train = pd.DataFrame(data=train_dict)

    title = []
    text = []
    docid = []
    queryid = []
    query = []
    random_id = []
    # train_data.extend(dev_data)
    # train_dev_data = train_data
    for i in tqdm(range(len(dev_data))):
    #for i in tqdm(range(10)):
        queryid.append(dev_data[i]['query_id'])
        query.append(dev_data[i]['query'])
        docid_new = []
        for j in range(len(dev_data[i]['positive_passages'])):
            title_doc[dev_data[i]['positive_passages'][j]['title']] = lower(
                dev_data[i]['positive_passages'][j]['text'])
            docid_new.append(str(dev_data[i]['positive_passages'][j]['docid']))
            title_doc_id[dev_data[i]['positive_passages'][j]['title']] = dev_data[i]['positive_passages'][j][
                'docid']
            id_doc[dev_data[i]['positive_passages'][j]['docid']] = title_doc[
                dev_data[i]['positive_passages'][j]['title']]
            idx += 1
        docid.append(",".join(docid_new))
    dev_dict = {"query": query.copy(), "docid": docid.copy()}
    df_dev = pd.DataFrame(data=dev_dict)
    # train_dev_file = open("NQ_doc_content_{}.tsv".format(docnum), 'w')
    # for docid in id_doc.keys():
    #     train_dev_file.write('\t'.join([str(docid), '', '', id_doc[docid], '', '', 'en']) + '\n')
    #     train_dev_file.flush()
    ##############################################################################


    train_query_docid = {}
    for i in trange(len(df_train)):
        if (len(df_train['query'][i].split('\n')) == 1):
            train_query_docid[df_train['query'][i]] = [int(elem) for elem in df_train['docid'][i].split(',')]

    file_train = open("train.tsv", 'w')

    count = 0
    for query in tqdm(train_query_docid.keys()):
        for i in range(len(train_query_docid[query])):
            id_ori = train_query_docid[query][i]
            new_id = origin_new_id[id_ori]
            file_train.write('\t'.join(
                [query, str(id_ori), str(new_id), new_kmeans_nq_doc_dict_100_int_key[int(new_id)]]) + '\n')
            file_train.flush()

    val_query_docid = {}
    for i in trange(len(df_dev)):
        if (len(df_dev['query'][i].split('\n')) == 1):
            val_query_docid[df_dev['query'][i]] = [int(elem) for elem in df_dev['docid'][i].split(',')]

    file_val = open("dev.tsv", 'w')

    count = 0
    for query in tqdm(val_query_docid.keys()):
        id_ori_ = []
        new_id_ = []
        kmeans_ = []
        for i in range(len(val_query_docid[query])):
            id_ori = str(val_query_docid[query][i])
            new_id = str(origin_new_id[int(id_ori)])
            id_ori_.append(id_ori)
            new_id_.append(new_id)
            kmeans_.append(new_kmeans_nq_doc_dict_100_int_key[int(new_id)])

        id_ori_ = ','.join(id_ori_)
        new_id_ = ','.join(new_id_)
        kmeans_ = ','.join(kmeans_)

        file_val.write('\t'.join([query, str(id_ori_), str(new_id_), kmeans_]) + '\n')
        file_val.flush()

    # QG_NUM = 5
    # qg_file = open("NQ_100_qg.tsv", 'w')
    #
    # for queryid in tqdm(qg_dict):
    #     for query in qg_dict[queryid][:QG_NUM]:
    #         qg_file.write('\t'.join([query, queryid, new_kmeans_nq_doc_dict_100_int_key[int(queryid)]]) + '\n')
    #         qg_file.flush()

    df_all['new_id'] = df_all['docid'].map(origin_new_id)

    df_all['kmeas_id'] = df_all['new_id'].map(new_kmeans_nq_doc_dict_100_int_key)

    df_all['tc'] = df_all['title'] + ' ' + df_all['content']

    df_all = df_all.loc[:, ['tc', 'docid', 'new_id', 'kmeas_id']]

    df_all.to_csv('NQ_title_cont_{}.tsv'.format(docnum), sep='\t', header=None, index=False, encoding='utf-8')

    queryid_oldid_dict = {}
    bertid_oldid_dict = {}
    map_file = "NQ_title_cont_{}.tsv".format(docnum)
    with open(map_file, 'r') as f:
        for line in f.readlines():
            query, queryid, oldid, bert_k30_c30 = line.split("\t")
            queryid_oldid_dict[oldid] = queryid
            bertid_oldid_dict[oldid] = bert_k30_c30

    train_file = "NQ_doc_content_{}.tsv".format(docnum)
    doc_aug_file = open("NQ_doc_aug_{}.tsv".format(docnum), 'w')
    with open(train_file, 'r') as f:
        for line in f.readlines():
            _, docid, _, _, content = line.split("\t")
            content = content.split(' ')
            add_num = max(0, len(content) - 3000) / 3000
            for i in range(5 + int(add_num)):
                begin = random.randrange(0, len(content))
                # if begin >= (len(content)-64):
                #     begin = max(0, len(content)-64)
                end = begin + aug_len if len(content) > begin + aug_len else len(content)
                doc_aug = content[begin:end]
                doc_aug = ' '.join(doc_aug).replace('\n', ' ')
                queryid = queryid_oldid_dict[docid]
                bert_k30_c30 = bertid_oldid_dict[docid]
                doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]) + '\n')
                # doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]))
                doc_aug_file.flush()
