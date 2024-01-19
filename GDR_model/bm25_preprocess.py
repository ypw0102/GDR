import os
import pickle
import pandas as pd
from transformers import (
    BertTokenizer
)

from tqdm import tqdm
import math
from gensim.summarization.bm25 import BM25
# setting thread num
import multiprocessing
from multiprocessing import cpu_count
# setting thread num as number of CPU core
num_processes = cpu_count()
import time
import threading
def simple_tok(sent: str):
    return sent.split()

if __name__ == "__main__":

    filename = 'GDR-main/Data_process/NQ_dataset/Self_NQ_ar2_334314/nq_qa_fulldoc_334314.csv'
    out_filename = 'GDR-main/Data_process/NQ_dataset/Self_NQ_ar2_334314/neg_bm25.pkl'
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
    #           "last doc
    #           without animals"]

    tok_corpus = [tokenizer.tokenize(s) for s in tqdm(corpus)]
    bm25 = BM25(tok_corpus)

    def bm25_search(query):
        scores = bm25.get_scores(query)
        best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:30]
        return best_docs
    query = [tokenizer.tokenize(s) for s in tqdm(query)]

    out_dict = {}
    results = []

    pool = multiprocessing.Pool(num_processes)
    results = []
    with tqdm(total=len(query)) as pbar:
        for query_result in pool.imap(bm25_search, query):
            results.append(query_result)
            pbar.update()

    pool.close()
    pool.join()
        # scores = bm25.get_scores(query[i])
        # best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:30]
        # for index, b in enumerate(best_docs):
            # print(f"rank {index + 1}: {corpus[b]}")

    for i in tqdm(range(len(results))):
        out_dict[int(id[i])] = results[i]
    with open(out_filename, 'wb') as file:
        pickle.dump(out_dict, file)
        file.close()

    with open(out_filename, 'rb') as file:
        in_dict = pickle.load(file)
        file.close()

    print("here")
