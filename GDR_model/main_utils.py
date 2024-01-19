import pickle
import random
from collections import defaultdict
from typing import List
import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_ckpt(args):
    certain_epoch, given_ckpt, ckpt_saved_folder = args.certain_epoch, args.given_ckpt, args.logs_dir
    ckpt_files = [f for f in listdir(ckpt_saved_folder) if isfile(join(ckpt_saved_folder, f))]
    assert len(ckpt_files) >= 1
    desired_ckpt_name = ''
    desired_ckpt_epoch = 0
    if given_ckpt is not None:
        desired_ckpt_epoch = int(given_ckpt.split('=')[1].split('-')[0])
        desired_ckpt_name = ckpt_saved_folder + given_ckpt
    else:
        for ckpt_name in ckpt_files:
            if ckpt_name[-4:] != 'ckpt':
                continue
            if ckpt_name.split('_epoch')[0] != args.tag_info:
                continue

            try:
                ckpt_epoch = int(ckpt_name.split('epoch=')[1].split('-')[0])
            except:
                continue
            if certain_epoch is not None:
                if certain_epoch == ckpt_epoch:
                    desired_ckpt_epoch, desired_ckpt_name = ckpt_epoch, ckpt_name
            else:
                if ckpt_epoch > desired_ckpt_epoch:
                    desired_ckpt_epoch, desired_ckpt_name = ckpt_epoch, ckpt_name
    print('=' * 20)
    print('Loading: ' + desired_ckpt_name)
    print('=' * 20)
    assert desired_ckpt_name in ckpt_files
    return ckpt_saved_folder + desired_ckpt_name, desired_ckpt_epoch


def grad_status(model):
    return (par.requires_grad for par in model.parameters())


def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"


def dec_2d(dec, size):
    res = []
    i = 0
    while i < len(dec):
        res.append(dec[i: i + size])
        i = i + size
    return res


###### decoder helper
def numerical_decoder(args, cuda_ids, output):
    np_ids = cuda_ids.cpu().numpy()
    begin_and_end_token = np.where(np_ids == 1)

    if output:
        if len(begin_and_end_token) != 1 or begin_and_end_token[0].size < 1:
            print("Invalid Case")
            return "0"
        if args.hierarchic_decode:
            np_ids = np_ids[1:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[1:begin_and_end_token[0][0]] - 2) % args.output_vocab_size
    else:
        if args.hierarchic_decode:
            np_ids = np_ids[:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[:begin_and_end_token[0][0]] - 2) % args.output_vocab_size

    bits = int(np.log10(args.output_vocab_size))
    num_list = list(map(str, np_ids))
    str_ids = ''.join([c.zfill(bits) if (i != len(num_list) - 1) else c for i, c in enumerate(num_list)])
    return str_ids


def random_shuffle(doc_id):
    new_doc_id = ""
    for index in range(0, len(doc_id)):
        while True:
            rand_digit = np.random.randint(0, 9)
            if not rand_digit == int(doc_id[index]):
                new_doc_id += str(rand_digit)
                break
    return new_doc_id


def augment(query):
    if len(query) < 20 * 10:
        start_pos = np.random.randint(0, int(len(query) + 1 / 2))
        end_pos = np.random.randint(start_pos, len(query))
        span_length = max(start_pos - end_pos, 10 * 10)
        new_query = str(query[start_pos:start_pos + span_length])
    else:
        start_pos = np.random.randint(0, len(query) - 10 * 10)
        end_pos = np.random.randint(start_pos + 5 * 10, len(query))
        span_length = min(start_pos - end_pos, 20 * 10)
        new_query = str(query[start_pos:start_pos + span_length])
    # print(new_query)
    return new_query


from tqdm import tqdm


def load_data(args):
    def process_func(doc_to_query_list, index, query, newid, oldid, rank=1):
        docid = newid
        softmax_index = None
        # if args.kary:
        #     docid = '-'.join([c for c in str(newid).split('-')])
        #     softmax_index_list = str(newid).split('-')[:3]
        #     softmax_index = 0
        #     for id, num in enumerate(softmax_index_list[::-1]):
        #         softmax_index += int(num) * (args.kary ** id)
        # else:
        #     docid = ''.join([c for c in str(newid)])
        #     softmax_index = int(str(newid)[:3])

        ##import hard negative here
        neg_docid_list = []

        if args.hard_negative:
            for i in range(1):
                neg_docid_list.append(random_shuffle(docid))

        aug_query_list = []
        if args.aug_query:
            if args.aug_query_type == 'aug_query':
                if newid in doc_to_query_list:
                    aug_query_list = doc_to_query_list[newid]
            else:
                for i in range(10):
                    aug_query_list.append(augment(query))
        return query, docid, oldid, rank, softmax_index, neg_docid_list, aug_query_list

    result = None
    q_emb, query_id_dict_train = None, None
    prefix_embedding = None
    prefix_mask = None
    prefix2idx_dict = None
    doc_to_query_list = None
    assert args.contrastive_variant == ''

    ## load pre-defined id_class in train.sh
    if 'gtq' in args.query_type:
        if args.nq:
            train_file = args.project_path + '/Data_process/NQ_dataset/{}/train.tsv'.format(args.dataset_name)

        df = pd.read_csv(
            train_file,
            encoding='utf-8',
            names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4",
                   "bert_k30_c30_5"],
            header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:,
             ["query", "oldid", args.id_class]]
        assert not df.isnull().values.any()
        doc_to_query_list = defaultdict(set)
        for [query, queryid, docid] in df.values.tolist():
            doc_to_query_list[docid].add(query)

        ## Query Generation Data
        if 'qg' in args.query_type:
            if args.nq:
                qg_file = args.project_path + '/Data_process/NQ_dataset/{}/qg.tsv'.format(args.dataset_name)
            gq_df1 = pd.read_csv(
                qg_file,
                names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3",
                       "bert_k30_c30_4", "bert_k30_c30_5"],
                encoding='utf-8', quoting=3, header=None, sep='\t',
                dtype={'query': str, 'queryid': str, "oldid": str, args.id_class: str}).loc[:,
                     ["query", "oldid", args.id_class]]
            gq_df1 = gq_df1.dropna(axis=0)
            print(len(gq_df1))
            for [query, oldid, docid] in gq_df1.values.tolist():
                doc_to_query_list[docid].add(query)
            temp = defaultdict(list)
            for k, v in doc_to_query_list.items():
                temp[k] = list(v)
            doc_to_query_list = temp

            result = tuple(
                process_func(doc_to_query_list, index, *row) for index, row in
                tqdm(enumerate(zip(df["query"], df[args.id_class], df["oldid"])))
            )
            result_add1 = tuple(
                process_func(doc_to_query_list, index, *row) for index, row in
                tqdm(enumerate(zip(gq_df1["query"], gq_df1[args.id_class], df["oldid"])))
            )
            result = result + result_add1

        else:
            result = tuple(
                process_func(doc_to_query_list, index, *row) for index, row in
                enumerate(zip(df["query"], df[args.id_class], df["oldid"]))
            )

    path_list = []
    if 'doc' in args.query_type:
        if args.nq:
            filename = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(args.dataset_name)
            path_list.append(filename)

    if 'aug' in args.query_type:
        if args.nq:
            filename = args.project_path + '/Data_process/NQ_dataset/{}/doc_aug.tsv'.format(args.dataset_name)
            path_list.append(filename)

    for file_path in path_list:
        print(file_path)
        df1 = pd.read_csv(
            file_path,
            names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4",
                   "bert_k30_c30_5"],
            header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:,
              ["query", "oldid", args.id_class]]
        df1.dropna(axis=0, inplace=True)
        assert not df1.isnull().values.any()
        result_add1 = tuple(
            filter(None,
                   (process_func(doc_to_query_list, index, *row) for index, row in
                    tqdm(enumerate(zip(df1["query"], df1[args.id_class], df1["oldid"]))))
                   )
        )
        result = result_add1 if result is None else result + result_add1

    if args.is_train_encoder:
        if args.nq:
            filename_doc = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(args.dataset_name)
            filename_queryid = args.project_path + "/Data_process/NQ_dataset/{}/indexmap.pkl".format(args.dataset_name)

        doc_frame = pd.read_csv(
            filename_doc,
            names=["doc", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4",
                   "bert_k30_c30_5"],
            header=None, sep='\t', dtype={'doc': str, 'queryid': str, 'oldid': str}).loc[:, ["doc"]]

        doc = doc_frame.values.tolist()

        with open(filename_queryid, 'rb') as f:
            id_mapping = pickle.load(f)
            f.close()
    else:
        doc = None
        id_mapping = None

    print('&' * 20)
    print(result[0])
    print('&' * 20)
    if args.train_num != -1:
        result = result[:args.train_num]
    return result, q_emb, query_id_dict_train, prefix_embedding, prefix_mask, prefix2idx_dict, doc, id_mapping


def load_data_infer(args):
    df = None
    index_to_id = {}
    if args.expand:
        filename_queryid_insert = args.project_path + "/Data_process/NQ_dataset/{}/indexmap_insert.pkl".format(
            args.dataset_name)
        with open(filename_queryid_insert, "rb") as f:
            id_mapping = pickle.load(f)
            f.close()
        for key in id_mapping.keys():
            for index in id_mapping[key]:
                index_to_id[index] = key
    if args.test_set == 'dev':
        if args.nq:
            dev_file = args.project_path + '/Data_process/NQ_dataset/{}/dev.tsv'.format(args.dataset_name)
        df = pd.read_csv(
            dev_file,
            encoding='utf-8',
            names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4",
                   "bert_k30_c30_5"],
            header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:,
             ["query", 'oldid', args.id_class]]
        if args.expand:
            dev_file = args.project_path + '/Data_process/NQ_dataset/NQ_ar2_334314_expand/nq-58622-data.pickle'
            with open(dev_file, "rb") as file:
                expand_dict = pickle.load(file)
                file.close()
            expand_dev = expand_dict["dev2"]
            query = []
            oldid =[]
            cluster_id = []
            for item in expand_dev:
                sub = []
                sub_cluster_id = []
                query.append(item['query'])
                for it in item['positive_passages']:
                    sub.append(str(it['docid']))
                    sub_cluster_id.append(str(index_to_id[it['docid']]))
                oldid.append(",".join(sub))
                cluster_id.append(",".join(sub_cluster_id))
            df = pd.DataFrame({'query': query, 'oldid': oldid, args.id_class:cluster_id})
    assert not df.isnull().values.any()

    result = []
    softmax_index = -1
    for index, row in df.iterrows():
        query = row["query"]
        oldid = row["oldid"]
        if True:
            if args.expand:
                if args.kary:
                    rank1 = row[args.id_class]
                    rank1 = rank1.split(",")
                    rank1 = rank1[0]
                    rank1 = '-'.join([c for c in str(rank1).split('-')])
                else:
                    rank1 = row[args.id_class]
                    rank1 = rank1.split(",")
                    rank1 = rank1[0]
            else:
                if args.kary:
                    rank1 = row[args.id_class]
                    rank1 = rank1.split(",")
                    rank1 = rank1[0]
                    rank1 = '-'.join([c for c in str(rank1).split('-')])
                else:
                    rank1 = row[args.id_class]
                    rank1 = rank1.split(",")
                    rank1 = rank1[0]

        list_sum = []
        if args.kary:
            docid = '-'.join([c for c in str(row[args.id_class]).split('-')])
            softmax_index_list = str(row[args.id_class]).split('-')[:3]
        else:
            docid = ''.join([c for c in str(row[args.id_class])])
            softmax_index = int(str(row[args.id_class])[:3])

        rank = 1

        list_sum.append((docid, rank))
        neg_docid_list = []
        if args.hard_negative:
            for i in range(1):
                neg_docid_list.append(random_shuffle(rank1))
        aug_query_list = []  # do not need for inference
        if args.aug_query:
            for i in range(20):
                aug_query_list.append(augment(query))
        result.append((query, rank1, oldid, list_sum, softmax_index, neg_docid_list, aug_query_list))

        if args.eval_num != -1:
            result = result[:args.eval_num]
    return tuple(result)
