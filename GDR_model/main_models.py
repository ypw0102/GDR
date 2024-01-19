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


# class DensePooler(nn.Module):
#     def __init__(self, input_dim: int = 768, output_dim: int = 768, normalize=False):
#         super(DensePooler, self).__init__()
#         self.normalize = normalize
#         self.linear_q = nn.Linear(input_dim, output_dim)
#         self.linear_p = nn.Linear(input_dim, output_dim)
#         self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'normalize': normalize}
#
#     def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
#         if q is not None:
#             if q.ndim < 3:
#                 q = q.unsqueeze(1)
#             rep = self.linear_q(q[:, 0])
#         elif p is not None:
#             rep = self.linear_p(p[:, 0])
#         else:
#             raise ValueError
#         if self.normalize:
#             rep = nn.functional.normalize(rep, dim=-1)
#         return rep


class EncoderModel(nn.Module):
    def __init__(self,args
                 ):
        super(EncoderModel, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(args.project_path + "/GDR_model/dpr-base-passage")
        if args.kmeans_model == "bert":
            self.model.ctx_encoder.bert_model = BertModel.from_pretrained(args.project_path + "/GDR_model/bert-base-uncased")
        elif args.kmeans_model == "ar2":
            with open(args.project_path + "/GDR_model/ar2_base.pkl", "rb")as f:
                state_dict = pickle.load(f)
                f.close()
            state_dict = {a.replace("ctx_model","ctx_encoder.bert_model"):state_dict[a] for a in state_dict.keys() if "question" not in a}
            state_dict['ctx_encoder.bert_model.embeddings.token_type_embeddings.weight'] = state_dict['ctx_encoder.bert_model.embeddings.token_type_embeddings.weight'][:2]
            self.model.load_state_dict(state_dict)
        # self.output = DensePooler(self.model.config.hidden_size, self.model.config.hidden_size)
        self.output = None
        self.args = args
    def forward(self, passage: Dict[str, Tensor] = None, query_enc=None):
        if passage is not None:
            for key in passage.keys():
                passage[key] = passage[key].view(-1, passage[key].size(-1))
            #p_reps = self.encode_passage(passage)
            #p_reps = self.model(**passage,return_dict=True)
            p_reps = self.model(**passage,return_dict=True).pooler_output
            return p_reps
        if query_enc is not None:
            q_reps = self.encode_query(query_enc)
            return q_reps

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.model(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.output is not None:
            p_reps = self.output(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry_hidden):
        if qry_hidden is None:
            return None
        if self.output is not None:
            q_reps = self.output(q=qry_hidden)
        else:
            q_reps = qry_hidden[:, 0]
        return q_reps


class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}
        self.embedding_index = []
        self.embedding = None
        self.all_leaf_num = 0

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq, embedding_index) -> None:
        '''
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        '''
        cur = self.root
        cur_top = None
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur_top = cur
            cur = cur.children[tok]
        cur_top.embedding_index.append(embedding_index)


def tree_embedding_calculate(root, embedding):
    def dfs(cur):
        if len(cur.embedding_index) > 0:
            cur.embedding = sum([embedding[item] for item in cur.embedding_index]) / len(cur.embedding_index)
            cur.all_leaf_num = len(cur.embedding_index)
            return cur.embedding, cur.all_leaf_num
        else:
            embeds = []
            all_leaf_num = []
            for key in cur.children.keys():
                embed, leaf_num = dfs(cur.children[key])
                embeds.append(embed)
                all_leaf_num.append(leaf_num)

            cur_embed = None
            for i in range(len(embeds)):
                if cur_embed is None:
                    cur_embed = embeds[i].clone() * all_leaf_num[i]
                else:
                    cur_embed += embeds[i].clone() * all_leaf_num[i]

            cur.embedding = cur_embed / sum(all_leaf_num)
            cur.all_leaf_num = sum(all_leaf_num)
            return cur.embedding, cur.all_leaf_num

    dfs(root)


def load_embedding(args):
    if args.nq:
        with open(args.project_path+'/Data_process/NQ_dataset/{}/doc_embedding.pkl'.format(args.dataset_name), "rb") as input_file:
            embed = pickle.load(input_file)
            input_file.close()
    return embed


def build_or_load_embedding_tree(tree_save_path, args):
    if os.path.isfile(tree_save_path):
        print('tree not true')
        with open(tree_save_path, "rb") as input_file:
            root = pickle.load(input_file)
            input_file.close()
        embedding = load_embedding(args)
        tree_embedding_calculate(root, embedding)
        print(1)
    else:
        embedding = load_embedding(args)

        print("Begin build tree")
        builder = TreeBuilder()
        if args.nq:
            all_file = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(args.dataset_name)
            df = pd.read_csv(
                all_file,
                encoding='utf-8', names=['tc', 'docid', 'new_id', 'kmeas_id', "a", "b", "c", "d"],
                header=None, sep='\t').loc[:, ["tc", 'kmeas_id']]
        for index, (_, newid) in tqdm(df.iterrows()):
            if args.label_length_cutoff:
                newid = newid[:args.max_output_length - 2]

            newid = newid.split(",")
            for i in range(len(newid)):
                toks = encode_single_newid(args, newid[i])
                builder.add(toks, index)
        if args.tree == 1:
            root = builder.build()
            print(1)
        else:
            print('No Tree')
            root = None

        with open(tree_save_path, "wb") as f:
            pickle.dump(root, f)
            f.close()
        tree_embedding_calculate(root, embedding)
    return root


def tree_match(cur, doc_embed):
    match_answer = [0]
    while True:
        if len(cur.children) == 1 and cur.children[list(cur.children.keys())[0]].embedding is None:
            break
        else:
            candidate_embeds = None
            keys = []
            for key in cur.children.keys():
                if candidate_embeds is None:
                    candidate_embeds = cur.children[key].embedding.unsqueeze(0)
                else:
                    candidate_embeds = torch.cat([candidate_embeds, cur.children[key].embedding.unsqueeze(0)], dim=0)
                keys.append(key)
            sim = torch.mul(doc_embed, candidate_embeds).sum(dim=-1)
            max_index = np.argmax(sim)
            target = keys[max_index]
            match_answer.append(target)
            cur = cur.children[target]
    match_answer.append(1)
    return np.array(match_answer)       # concate eos token


# def tree_embedding_insert(root, id_mapping, insert_doc, cluster, args):
#     cur = root
#     for index in tqdm(range(len(insert_doc))):
#         if index < args.docnum:
#             continue
#         else:
#             id = tree_match(cur, insert_doc[index])
#             id = decode_token(args, [id])[0]
#             id_mapping[id].append(index)
#             id_mapping[id] = list(set(id_mapping[id]))
#     print("done")
#     return id_mapping

def tree_embedding_insert(root, id_mapping, insert_doc, cluster, args):
    cur = root
    cluster_list = list(cluster)
    cluster_embedding = None
    for index in tqdm(range(len(cluster_list))):
        cluster_id = cluster_list[index]
        road = cluster_id.split("-")
        for sub in road:
            cur = cur.children[int(sub)]
        if cluster_embedding is None:
            cluster_embedding = cur.embedding.unsqueeze(0)
        else:
            cluster_embedding = torch.cat([cluster_embedding, cur.embedding.unsqueeze(0)], dim=0)
        cur = root
    for index in tqdm(range(len(insert_doc))):
        if index < args.docnum:
            continue
        else:
            sim = torch.mul(insert_doc[index], cluster_embedding).sum(dim=-1)
            max_index = np.argmax(sim)
            target = cluster_list[max_index]
            id = [0] + list(map(int, target.split("-"))) + [1]

            id = decode_token(args, [np.array(id)])[0]
            id_mapping[id].append(index)
            id_mapping[id] = list(set(id_mapping[id]))
    print("done")
    return id_mapping

def encode_single_newid(args, seq):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    if args.kary:
        for i, c in enumerate(seq.split('-')):
            if args.position:
                cur_token = i * args.kary + int(c) + 2
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    else:
        for i, c in enumerate(seq):
            if args.position:
                cur_token = i * 10 + int(c) + 2  # hardcoded vocab_size = 10
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    return target_id_int + [1]  # append eos_token


def decode_token(args, seqs):
    '''
    Param:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    '''
    result = []
    for seq in seqs:
        try:
            eos_idx = seq.tolist().index(1)
            seq = seq[1: eos_idx]
        except:
            print("no eos token found")
        if args.position:
            offset = np.arange(len(seq)) * args.output_vocab_size + 2
        else:
            offset = 2
        res = seq - offset
        # assert np.all(res >= 0)
        if args.kary:
            result.append('-'.join(str(c) for c in res))
        else:
            result.append(''.join(str(c) for c in res))
    return result


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class l1_query(Dataset):
    def __init__(self, args, tokenizer, encoder_tokenizer, num_samples, print_text=False, task='train'):
        assert task in ['train', 'test']
        #print("into l1_query")
        self.args = args
        input_length = args.max_input_length
        output_length = args.max_output_length * int(np.log10(args.output_vocab_size))
        inf_input_length = args.inf_max_input_length
        random_gen = args.random_gen
        softmax = args.softmax
        aug = args.aug
        print("into check task")
        if task == 'train':
            print("begin loading!")
            self.dataset, self.q_emb, self.query_dict, \
                self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict, self.doc, self.id_mapping = \
                load_data(args)
            print("loading over!")
        elif task == 'test':
            self.dataset = load_data_infer(args)
            if args.is_train_encoder:
                if args.nq:
                    filename_doc = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(
                        args.dataset_name)
                    filename_doc_emb = args.project_path + '/Data_process/NQ_dataset/{}/doc_embedding.pkl'.format(
                        args.dataset_name)
                    filename_queryid = args.project_path + "/Data_process/NQ_dataset/{}/indexmap.pkl".format(
                        args.dataset_name)

                doc_frame = pd.read_csv(
                    filename_doc,
                    names=["doc", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3",
                           "bert_k30_c30_4",
                           "bert_k30_c30_5"],
                    header=None, sep='\t', dtype={'doc': str, 'queryid': str, 'oldid': str}).loc[:, ["doc"]]

                self.doc = doc_frame.values.tolist()

                with open(filename_doc_emb, "rb") as f:
                    self.doc_embed = pickle.load(f)
                    f.close()
                with open(filename_queryid, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                    f.close()
            else:
                self.doc = None
                self.id_mapping = None
            self.q_emb, self.query_dict, \
                self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict \
                = None, None, None, None, None
        else:
            raise NotImplementedError("No Corresponding Task.")

        if num_samples:
            self.dataset = self.dataset[:num_samples]

        self.epoch = 0
        self.task = task
        self.input_length = input_length
        self.doc_length = self.args.doc_length
        self.inf_input_length = inf_input_length
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.softmax = softmax
        self.aug = aug
        self.random_gen = random_gen
        if random_gen:
            assert len(self.dataset[0]) >= 3
        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token,
                      self.tokenizer.pad_token, self.tokenizer.cls_token,
                      self.tokenizer.mask_token] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def convert_to_features(self, example_batch, length_constraint):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return output_

    def __getitem__(self, index):
        inputs = self.dataset[index]

        query_embedding = torch.tensor([0])
        prefix_embedding, prefix_mask = torch.tensor([0]), torch.tensor([0])
        # if self.args.old_data:
        if len(inputs) >= 7:
            query, target, oldid, rank, neg_target, aug_query = inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], \
            inputs[6]
        elif len(inputs) >= 6:
            query, target, oldid, rank, neg_target = inputs[0], inputs[1], inputs[2], inputs[3], inputs[5]
        else:
            query, target, oldid, rank = inputs[0], inputs[1], inputs[2], inputs[3]

        if hasattr(self, 'query_dict') and self.query_dict is not None:
            query_embedding = self.q_emb[self.query_dict[query]]
        neg_targets_list = []
        if self.args.hard_negative:
            neg_targets_list = np.random.choice(neg_target, self.args.sample_neg_num)
        if self.args.aug_query and len(aug_query) >= 1:
            aug_query = np.random.choice(aug_query, 1)[0]
        else:
            aug_query = ""
        if self.args.label_length_cutoff:
            target = target[:self.args.max_output_length - 2]

        source = self.convert_to_features(query, self.input_length if self.task == 'train' else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        if 'print_token' in self.args.query_type:
            print("Input Text: ", query, '\n', "Output Text: ", source_ids)
        src_mask = source["attention_mask"].squeeze()
        aug_source = self.convert_to_features(aug_query,
                                              self.input_length if self.task == 'train' else self.inf_input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()
        if self.args.multiple_decoder:
            target_ids, target_mask = [], []
            for i in range(self.args.decoder_num):
                targets = self.convert_to_features(target[i], self.output_length)
                target_ids.append(targets["input_ids"].squeeze())
                target_mask.append(targets["attention_mask"].squeeze())
        # else:
        #     targets = self.convert_to_features(target, self.output_length)
        #     target_ids = targets["input_ids"].squeeze()
        #     target_mask = targets["attention_mask"].squeeze()

        def target_to_prefix_emb(target, tgt_length):
            tgt_prefix_emb = []
            prefix_masks = []
            for i in range(tgt_length):
                if i < len(target):
                    ###### fake data
                    _prefix_emb = np.random.rand(10, 768)
                    ###### real data
                    # _prefix_emb = self.prefix_embedding[self.prefix2idx_dict[target[:i]]]
                    _prefix_emb = torch.tensor(_prefix_emb)
                    tgt_prefix_emb.append(_prefix_emb.unsqueeze(0))
                    ##############################
                    ###### fake data
                    _prefix_mask = np.random.rand(10, )
                    _prefix_mask[_prefix_mask < 0.5] = 0
                    _prefix_mask[_prefix_mask > 0.5] = 1
                    ###### real data
                    # _prefix_mask = self.prefix_mask[self.prefix2idx_dict[target[:i]]]
                    _prefix_mask = torch.LongTensor(_prefix_mask)
                    prefix_masks.append(_prefix_mask.unsqueeze(0))
                    ##############################
                else:
                    tgt_prefix_emb.append(torch.zeros((1, 10, 768)))
                    prefix_masks.append(torch.zeros((1, 10)))
            return torch.cat(tgt_prefix_emb, dim=0), torch.cat(prefix_masks, dim=0)

        if self.prefix_embedding is not None:
            prefix_embedding, prefix_mask = target_to_prefix_emb(target, self.output_length)

        neg_target_ids_list = []
        neg_target_mask_list = []
        neg_rank_list = []

        if self.args.hard_negative:
            for cur_target in neg_targets_list:
                cur_targets = self.convert_to_features(cur_target, self.output_length)
                cur_target_ids = cur_targets["input_ids"].squeeze()
                cur_target_mask = cur_targets["attention_mask"].squeeze()
                neg_target_ids_list.append(cur_target_ids)
                neg_target_mask_list.append(cur_target_mask)
                neg_rank_list.append(999)  # denote hard nagative

        lm_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)

        if self.args.decode_embedding:
            ## func target_id+target_id2, twice or k
            def decode_embedding_process(target_id):
                target_id_int = []
                if self.args.kary:
                    target_id = target_id.split('-')
                    for i in range(0, len(target_id)):
                        c = target_id[i]
                        if self.args.position:
                            temp = i * self.args.output_vocab_size + int(c) + 2 \
                                if not self.args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                else:
                    bits = int(np.log10(self.args.output_vocab_size))
                    idx = 0
                    for i in range(0, len(target_id), bits):
                        if i + bits >= len(target_id):
                            c = target_id[i:]
                        c = target_id[i:i + bits]
                        if self.args.position:
                            temp = idx * self.args.output_vocab_size + int(c) + 2 \
                                if not self.args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                        idx += 1
                lm_labels[:len(target_id_int)] = torch.LongTensor(target_id_int)
                lm_labels[len(target_id_int)] = 1
                decoder_attention_mask = lm_labels.clone()
                decoder_attention_mask[decoder_attention_mask != 0] = 1
                target_ids = lm_labels
                target_mask = decoder_attention_mask
                return target_ids, target_mask

            if self.args.multiple_decoder:
                target_mask = []
                for i in range(len(target_ids)):
                    target_ids[i], cur_target_mask = decode_embedding_process(target_ids[i])
                    target_mask.append(cur_target_mask)
            else:
                target_ids, target_mask = decode_embedding_process(target)

            if self.args.hard_negative:
                for i in range(len(neg_target_ids_list)):
                    cur_target_ids = neg_target_ids_list[i]
                    cur_target_ids, cur_target_mask = decode_embedding_process(cur_target_ids)
                    neg_target_ids_list[i] = cur_target_ids
                    neg_target_mask_list[i] = cur_target_mask

        if self.args.is_train_encoder:
            if self.task == "train":
                c_sample_candidate = self.id_mapping[target]
                if int(oldid) in c_sample_candidate:
                    c_sample_candidate.remove(int(oldid))
                num = min(len(c_sample_candidate), self.args.max_intraclass_num)
                random.shuffle(c_sample_candidate)
                candidates = random.sample(c_sample_candidate, num)
                candidates_doc = []
                for candidate in candidates:
                    candidates_doc.append(self.doc[candidate][0])
                candidates.append(int(oldid))
                positive_doc = [self.doc[int(oldid)][0]]
                candidates_doc.append(self.doc[int(oldid)][0])
                num += 1

                if self.epoch > self.args.train_encoder_epoch:
                    candidates_doc = self.encoder_tokenizer.batch_encode_plus(candidates_doc,
                                                                              max_length=self.args.encoder_max_len,
                                                                              padding='max_length', truncation=True,
                                                                              return_tensors="pt")
                    positive_doc = self.encoder_tokenizer.batch_encode_plus(positive_doc,
                                                                            max_length=self.args.encoder_max_len,
                                                                            padding='max_length', truncation=True,
                                                                            return_tensors="pt")

                    if num < self.args.max_intraclass_num + 1:
                        pad_num = self.args.max_intraclass_num + 1 - num
                        pad_tensor = torch.zeros(pad_num, self.args.encoder_max_len).long()
                        candidates_doc["input_ids"] = torch.cat([candidates_doc["input_ids"], pad_tensor], dim=0)
                        candidates_doc["attention_mask"] = torch.cat([candidates_doc["attention_mask"], pad_tensor], dim=0)
                        candidates_doc["token_type_ids"] = torch.cat([candidates_doc["token_type_ids"], pad_tensor], dim=0)

                    oldid = [oldid]

                else:
                    oldid = [oldid]
                    candidates_doc = [",".join(list(map(str, candidates)))]
                    positive_doc = [str(oldid[0])]
                    num = num
            else:
                oldid = [oldid]
                candidates_doc = []
                positive_doc = []
                num = 0
        else:
            oldid = []
            candidates_doc = []
            positive_doc = []
            num = 0

        # position = torch.sum(target_mask, dim=-1)
        # target_ids[position-1] = 0
        # target_mask[position-1] = 0
        return_dict = {"source_ids": source_ids,
                       "source_mask": src_mask,
                       "aug_source_ids": aug_source_ids,
                       "aug_source_mask": aug_source_mask,
                       "target_ids": target_ids,
                       "target_mask": target_mask,
                       "neg_target_ids": neg_target_ids_list,
                       "neg_rank": neg_rank_list,
                       "neg_target_mask": neg_target_mask_list,
                       "doc_ids": doc_ids if self.args.contrastive_variant != '' else torch.tensor([-1997],
                                                                                                   dtype=torch.int64),
                       "doc_mask": doc_mask if self.args.contrastive_variant != '' else torch.tensor([-1997],
                                                                                                     dtype=torch.int64),
                       "softmax_index": torch.tensor([inputs[-1]], dtype=torch.int64)
                       if self.softmax else torch.tensor([-1997], dtype=torch.int64),
                       "rank": rank,
                       "query_emb": query_embedding,
                       "prefix_emb": prefix_embedding,
                       "prefix_mask": prefix_mask,
                       "candidates_doc": candidates_doc,
                       "positive_doc": positive_doc,
                       "valid_num": num,
                       "oldid": oldid}

        # if self.task == "train":
        #     print(1)
        # for key_ in ["candidates_doc"]:
        #     if return_dict[key_] is not None and len(return_dict[key_])>6:
        #         print(1)

        return return_dict


class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train=True):
        super(T5FineTuner, self).__init__()
        self.cluster = set()
        ## Bulid tree
        if args.nq:
            datas = "NQ"
        if args.is_train_encoder:
            tree_save_path = args.output_dir + "data-" + datas + "_kmeans-model-" + args.kmeans_model + "_docnum-" + str(
                args.docnum) + 'cluster_id_tree_expand.pkl'
            # tree_save_path = args.output_dir + args.dataset_name + '_cluster_id_tree.pkl'
        else:
            tree_save_path = args.output_dir + "data-" + datas + "_kmeans-model-" + args.kmeans_model + "_docnum-" + str(
                args.docnum) + 'doc_id_tree.pkl'
        if os.path.isfile(tree_save_path):
            print('tree not true')
            with open(tree_save_path, "rb") as input_file:
                root = pickle.load(input_file)
            embedding = load_embedding(args)
            # tree_embedding_calculate(root, embedding)
            self.root = root
        else:
            embedding = load_embedding(args)
            print("Begin build tree")
            builder = TreeBuilder()
            if args.nq:
                all_file = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(args.dataset_name)
                df = pd.read_csv(
                    all_file,
                    encoding='utf-8', names=['tc', 'docid', 'new_id', 'kmeas_id', "a", "b", "c", "d"],
                    header=None, sep='\t').loc[:, ["tc", 'kmeas_id']]

            for index, (_, newid) in tqdm(df.iterrows()):
                if args.label_length_cutoff:
                    newid = newid[:args.max_output_length - 2]

                newid = newid.split(",")
                for i in range(len(newid)):
                    toks = encode_single_newid(args, newid[i])
                    builder.add(toks, index)
            if args.tree == 1:
                root = builder.build()
            else:
                print('No Tree')
                root = None
            with open(tree_save_path, "wb") as f:
                pickle.dump(root, f)
                f.close()

        if args.expand:
            tree_embedding_calculate(root, embedding)
            self.root = root
        ######

        self.args = args
        self.save_hyperparameters(args)
        # assert args.tie_word_embedding is not args.decode_embedding
        if args.decode_embedding:
            if self.args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                self.decode_vocab_size = 12
        else:
            self.decode_vocab_size = None

        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=0 if args.softmax else args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size,
            output_vocab_size=args.output_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            contrastive=args.contrastive,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num=args.adaptor_layer_num,
            embedding_distillation=args.embedding_distillation,
            weight_distillation=args.weight_distillation,
            input_dropout=args.input_dropout,
            denoising=args.denoising,
            multiple_decoder=args.multiple_decoder,
            decoder_num=args.decoder_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
        )

        print(t5_config)
        self.model_config = t5_config
        model = T5ForConditionalGeneration(t5_config)
        dual_encoder = EncoderModel(args)
        if args.pretrain_encoder:
            pretrain_model = T5ForConditionalGeneration.from_pretrained(
                args.project_path + "/GDR_model/t5-"+args.model_info)  # args.model_name_or_path)
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
        self.model = model
        self.tau = args.tau
        print(self.model)
        self.encoder = dual_encoder
        # self.fusion_layer = torch.nn.Linear(2*args.d_model, args.d_model)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(args.encoder_tokenizer_name_or_path)

        if args.is_train_encoder:
            if args.nq:
                filename_doc_emb = args.project_path + '/Data_process/NQ_dataset/{}/doc_embedding.pkl'.format(
                    args.dataset_name)
        if self.args.is_train_encoder:
            with open(filename_doc_emb, "rb") as f:
                self.doc_embed = pickle.load(f)
                f.close()
            if self.args.expand:
                filename_doc_emb_expand = args.project_path + '/Data_process/NQ_dataset/NQ_ar2_334314_expand/doc_embedding.pkl'
                with open(filename_doc_emb_expand, "rb") as f:
                    self.doc_embed = pickle.load(f)
                    f.close()
        # self.rouge_metric = load_metric('rouge')
        self.epoch = 0
        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        if self.args.softmax:
            self.fc = torch.nn.Linear(args.d_model, self.args.num_cls)  # [feature size, num cls]
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.5)
        if self.args.disc_loss:
            self.dfc = torch.nn.Linear(args.d_model, 1)

        n_observations_per_split = {
            "train": self.args.n_train,
            "validation": self.args.n_val,
            "test": self.args.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        if train:
            n_samples = self.n_obs['train']
            train_dataset = l1_query(self.args, self.tokenizer, self.encoder_tokenizer, n_samples)
            self.l1_query_train_dataset = train_dataset
            self.t_total = (
                    (len(train_dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                    // self.args.gradient_accumulation_steps
                    * float(self.args.num_train_epochs)
            )
        if args.is_train_encoder:
            if args.nq:
                filename_doc = args.project_path + '/Data_process/NQ_dataset/{}/title_content.tsv'.format(
                    args.dataset_name)
                filename_queryid = args.project_path + "/Data_process/NQ_dataset/{}/indexmap.pkl".format(
                    args.dataset_name)

            doc_frame = pd.read_csv(
                filename_doc,
                names=["doc", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3",
                       "bert_k30_c30_4",
                       "bert_k30_c30_5"],
                header=None, sep='\t', dtype={'doc': str, 'queryid': str, 'oldid': str}).loc[:, ["doc"]]

            self.doc = doc_frame.values.tolist()

            cluster_frame = pd.read_csv(
                filename_doc,
                names=["doc", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3",
                       "bert_k30_c30_4",
                       "bert_k30_c30_5"],
                header=None, sep='\t', dtype={'doc': str, 'queryid': str, 'oldid': str}).loc[:, ["bert_k30_c30_1"]]
            clusters = cluster_frame.values.tolist()
            for cluster in clusters:
                out = encode_single_newid(args, cluster[0])
                self.cluster.add("-".join(list(map(str, out[:-1]))))

            with open(filename_queryid, 'rb') as f:
                self.id_mapping = pickle.load(f)
                f.close()

            if self.args.expand:
                filename_queryid_insert = args.project_path + "/Data_process/NQ_dataset/{}/indexmap_insert.pkl".format(
                    args.dataset_name)
                if os.path.isfile(filename_queryid_insert):
                    with open(filename_queryid_insert, "rb") as f:
                        self.id_mapping = pickle.load(f)
                        f.close()
                else:
                    self.id_mapping = tree_embedding_insert(self.root, self.id_mapping, self.doc_embed, self.cluster,self.args)
                    with open(filename_queryid_insert, "wb") as f:
                        pickle.dump(self.id_mapping, f)
                        f.close()

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(self, input_ids, aug_input_ids=None, encoder_outputs=None, attention_mask=None, aug_attention_mask=None,
                logit_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None,
                query_embedding=None, prefix_emb=None, prefix_mask=None, only_encoder=False, decoder_index=-1,
                input_mask=None, candidates_doc=None, positive_doc=None, valid_num=None):
        input_mask = None
        loss_weight = None

        aug_input_ids=None
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder and self.training:
                if aug_input_ids is not None and self.training:
                    input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
                    attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
                elif self.training:
                    input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                    attention_mask = torch.cat([attention_mask, attention_mask.clone()], dim=0)
                if self.args.denoising:
                    if input_mask is None:
                        input_mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.9
                if self.args.input_dropout and np.random.rand() < 0.5:
                    if input_mask is None:
                        input_mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.9
                    input_ids = torch.where(input_mask == True, input_ids, torch.zeros_like(input_ids))
                if decoder_attention_mask is not None:
                    decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask], dim=0)
                if lm_labels is not None:
                    lm_labels = torch.cat([lm_labels, lm_labels], dim=0)
                if decoder_input_ids is not None:
                    decoder_input_ids = torch.cat([decoder_input_ids, decoder_input_ids], dim=0)


        out = self.model(
            input_ids,
            input_mask=input_mask,
            logit_mask=logit_mask,
            encoder_outputs=encoder_outputs,
            only_encoder=only_encoder,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            query_embedding=query_embedding,
            prefix_embedding=prefix_emb,
            prefix_mask=prefix_mask,
            return_dict=True,
            output_hidden_states=True,
            decoder_index=decoder_index,
            loss_weight=loss_weight,
        )
        if self.args.is_train_encoder:
            if self.epoch > self.args.train_encoder_epoch:
                candidates_doc_out = self.encoder(passage=candidates_doc)
                positive_doc_out = self.encoder(passage=positive_doc)
            else:
                candidates_doc_index = []
                candidates_doc_embed = None
                candidates_pos_index = []
                candidates_pos_embed = None
                for index_str in candidates_doc:
                    for i in range(len(index_str)):
                        tmp = list(map(int, index_str[i].split(",")))
                        candidates_doc_index.extend(tmp.copy())
                for index_str in positive_doc:
                    for i in range(len(index_str)):
                        tmp_pos = int(index_str[i])
                        candidates_pos_index.append(tmp_pos)

                for index in candidates_doc_index:
                    if candidates_doc_embed is None:
                        candidates_doc_embed = self.doc_embed[index].unsqueeze(0).cuda()
                    else:
                        candidates_doc_embed = torch.cat([candidates_doc_embed, self.doc_embed[index].unsqueeze(0).cuda()], dim=0)

                for index in candidates_pos_index:
                    if candidates_pos_embed is None:
                        candidates_pos_embed = self.doc_embed[index].unsqueeze(0).cuda()
                    else:
                        candidates_pos_embed = torch.cat([candidates_pos_embed, self.doc_embed[index].unsqueeze(0).cuda()], dim=0)

                candidates_doc_out = candidates_doc_embed
                positive_doc_out = candidates_pos_embed

            encoder_query_out = None
            if self.args.use_query_embed_encoder:
                if self.args.Rdrop > 0:
                    encoder_query_out = self.encoder(query_enc=out.encoder_last_hidden_state[:out.encoder_last_hidden_state.size(0)//2])
                else:
                    encoder_query_out = self.encoder(query_enc=out.encoder_last_hidden_state)
            # elif self.args.use_query_embed_decoder_special:
            #     print("under developing")
            if self.args.use_query_embed_decoder_avg:
                decoder_last_hidden = out.decoder_hidden_states[-1]
                masked_mean = torch.sum(decoder_last_hidden * decoder_attention_mask.unsqueeze(-1), dim=1) / torch.sum(decoder_attention_mask, dim=1, keepdim=True)
                encoder_query_out = self.encoder(query_enc=masked_mean)
            if self.args.use_query_embed_decoder_special:
                decoder_inputs = lm_labels.new_zeros(lm_labels.shape)
                decoder_inputs[..., 1:] = lm_labels[..., :-1].clone()
                decoder_inputs[..., 0] = self.model_config.decoder_start_token_id
                insert_place = torch.sum(~decoder_inputs.eq(-100), dim=-1) - 1
                # decoder_input_ids = decoder_inputs.scatter(-1, insert_place.unsqueeze(-1), self.model_config.eos_token_id)
                decoder_input_ids = decoder_inputs.masked_fill_(decoder_inputs == -100, self.model_config.pad_token_id)

                decoder_attention_mask_1 = torch.sum(decoder_attention_mask, dim=-1)
                # decoder_attention_mask_2 = decoder_attention_mask_1 + 1
                # decoder_attention_insert = torch.cat([decoder_attention_mask_1.unsqueeze(-1), decoder_attention_mask_2.unsqueeze(-1)], dim=-1)
                decoder_attention_mask = decoder_attention_mask.scatter(-1, decoder_attention_mask_1.unsqueeze(-1), 1)

                out_decoder_special = self.model(
                    input_ids,
                    input_mask=input_mask,
                    logit_mask=logit_mask,
                    encoder_outputs=encoder_outputs,
                    only_encoder=only_encoder,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    lm_labels=None,
                    query_embedding=query_embedding,
                    prefix_embedding=prefix_emb,
                    prefix_mask=prefix_mask,
                    return_dict=True,
                    output_hidden_states=True,
                    decoder_index=decoder_index,
                    loss_weight=loss_weight,
                )
                if encoder_query_out is not None and self.args.use_query_embed_encoder:
                    encoder_query_out_tmp = encoder_query_out.clone()
                    encoder_query_out = out_decoder_special.decoder_hidden_states[-1]
                    mask = torch.zeros((encoder_query_out.size(0), encoder_query_out.size(1))).cuda()
                    mask = mask.scatter(-1, insert_place.unsqueeze(-1), 1).bool()
                    encoder_query_out = encoder_query_out[mask]
                    if self.args.fusion_strategy == "concate":
                        encoder_query_out = torch.concat([encoder_query_out, encoder_query_out_tmp], dim=-1)
                        encoder_query_out = self.fusion_layer(encoder_query_out)
                    if self.args.fusion_strategy == "average":
                        encoder_query_out = (encoder_query_out + encoder_query_out_tmp)/2
                else:
                    encoder_query_out = out_decoder_special.decoder_hidden_states[-1]
                    mask = torch.zeros((encoder_query_out.size(0), encoder_query_out.size(1))).cuda()
                    mask = mask.scatter(-1, insert_place.unsqueeze(-1), 1).bool()
                    encoder_query_out = encoder_query_out[mask]

        else:
            candidates_doc_out = None
            positive_doc_out = None
            encoder_query_out = None
        return out, encoder_query_out, candidates_doc_out, positive_doc_out
        # return out, encoder_doc_out, encoder_query_out

    def _step(self, batch):
        loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation, contrast_loss = self._step_i(batch, -1)
        return loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation, contrast_loss

    def _step_i(self, batch, i, encoder_outputs=None, input_mask=None):
        if i < 0:
            lm_labels = batch["target_ids"]
            target_mask = batch['target_mask']
        else:
            lm_labels = batch["target_ids"][i]
            target_mask = batch['target_mask'][i]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, query_enc, candidates_doc_enc, positive_doc_enc = self.forward(input_ids=batch["source_ids"],
                                                                                aug_input_ids=batch["aug_source_ids"],
                                                                                attention_mask=batch["source_mask"],
                                                                                aug_attention_mask=batch[
                                                                                    "aug_source_mask"],
                                                                                lm_labels=lm_labels,
                                                                                decoder_attention_mask=target_mask,
                                                                                query_embedding=batch["query_emb"],
                                                                                decoder_index=i,
                                                                                encoder_outputs=encoder_outputs,
                                                                                prefix_emb=batch["prefix_emb"],
                                                                                prefix_mask=batch["prefix_mask"],
                                                                                input_mask=input_mask,
                                                                                candidates_doc=batch["candidates_doc"],
                                                                                positive_doc=batch["positive_doc"],
                                                                                valid_num=batch["valid_num"])

        neg_outputs = None
        # if self.args.hard_negative and self.args.sample_neg_num > 0:
        #     neg_lm_labels = torch.cat(batch['neg_target_ids'], dim=0)
        #     neg_decoder_attention_mask = torch.cat(batch['neg_target_mask'], dim=0)
        #     attention_mask = batch["source_mask"].repeat([self.args.sample_neg_num, 1])
        #     sources_ids = batch["source_ids"].repeat([self.args.sample_neg_num, 1])
        #     neg_lm_labels[neg_lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        #     neg_outputs = self.forward(input_ids=sources_ids, decoder_index=i, encoder_outputs=outputs.encoder_outputs,
        #                                attention_mask=attention_mask, lm_labels=neg_lm_labels,
        #                                decoder_attention_mask=neg_decoder_attention_mask,
        #                                query_embedding=batch['query_emb'])

        def select_lm_head_weight(cur_outputs):
            lm_head_weight = cur_outputs.lm_head_weight
            vocab_size = lm_head_weight.shape[-1]
            dim_size = lm_head_weight.shape[-2]
            lm_head_weight = lm_head_weight.view(-1, vocab_size)  # [batch_size, seq_length, dim_size, vocab_size]
            indices = cur_outputs.labels.unsqueeze(-1).repeat([1, 1, dim_size]).view(-1, 1)
            indices[indices[:, :] == -100] = self.tokenizer.pad_token_id
            lm_head_weight = torch.gather(lm_head_weight, -1, indices)  # [batch_size, seq_length, dim_size, 1]
            lm_head_weight = lm_head_weight.view(cur_outputs.decoder_hidden_states[-1].shape)
            return lm_head_weight

        def cal_contrastive(outputs, neg_outputs):
            vocab_size = outputs.lm_head_weight.shape[-1]
            dim_size = outputs.lm_head_weight.shape[-2]
            decoder_weight = select_lm_head_weight(outputs)

            if neg_outputs is not None:
                decoder_embed = torch.cat((outputs.decoder_hidden_states[-1], neg_outputs.decoder_hidden_states[-1]),
                                          dim=0).transpose(0, 1).transpose(1,
                                                                           2)  # [seq_length, embed_size, batch_size*2]
                neg_decoder_weight = select_lm_head_weight(neg_outputs)
                decoder_weight = torch.cat((decoder_weight, neg_decoder_weight), dim=0).transpose(0, 1).transpose(1, 2)
            else:
                decoder_embed = outputs.decoder_hidden_states[-1].transpose(0, 1).transpose(1, 2)
                decoder_weight = decoder_weight.transpose(0, 1).transpose(1, 2)
            seq_length = decoder_embed.shape[0]
            embed_size = decoder_embed.shape[1]
            bz = outputs.encoder_last_hidden_state.shape[0]
            # print("decoder_embed", decodfer_embed.shape)  #[seq_length, embed_size, batch_size + neg_bz]
            # print("decoder_weight", decoder_weight.shape) #[seq_length, embed_size, batch_size + neg_bz]
            query_embed = outputs.encoder_last_hidden_state[:, 0, :].unsqueeze(0).repeat(
                [seq_length, 1, 1])  # [seq_length, batch_size, embed_size]
            # query_tloss = self.triplet_loss(query_embed, decoder_embed[:,:,0:bz], decoder_embed[:,:,bz:])
            # query_tloss = self.triplet_loss(query_embed, decoder_weight[:,:,0:bz], decoder_weight[:,:,bz:])
            query_tloss = None
            weight_tloss = None
            disc_loss = None
            ranking_loss = None
            if self.args.query_tloss:
                all_doc_embed = decoder_embed  # [seq_length, embed_size, pos_bz+neg_bz]
                doc_logits = torch.bmm(query_embed, all_doc_embed)  # [sl, bz, bz+neg_bz]
                contrast_labels = torch.arange(0, bz).to(doc_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(seq_length, 1)
                # masks = outputs.labels.transpose(0, 1).repeat([1, 1+self.args.sample_neg_num])
                contrast_labels[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                query_tloss = self.ce(doc_logits.view(seq_length * bz, -1), contrast_labels.view(-1))
            if self.args.weight_tloss:
                query_embed = query_embed.transpose(1, 2)
                doc_embed = decoder_embed[:, :, 0:bz].transpose(1, 2)  # [seq_length, batch_size, embed_size]
                query_logits = torch.bmm(doc_embed, query_embed)  # [sl, bz, bz]
                contrast_labels = torch.arange(0, bz).to(query_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(seq_length, 1)  # [sl, bz]
                contrast_labels[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                weight_tloss = self.ce(query_logits.view(seq_length * bz, -1), contrast_labels.view(-1))
            if self.args.ranking_loss:
                rank_target = torch.ones(bz * seq_length).to(lm_labels.device)
                rank_indices = outputs.labels.detach().clone().reshape([-1, 1])
                rank_indices[rank_indices[:, :] == -100] = self.tokenizer.pad_token_id
                # pos_prob = torch.gather(self.softmax(outputs.lm_logits.detach().clone()).view(-1, vocab_size), -1, rank_indices).squeeze(-1)
                pos_prob = torch.gather(self.softmax(outputs.lm_logits).view(-1, vocab_size), -1, rank_indices)
                pos_prob[rank_indices[:, :] == self.tokenizer.pad_token_id] = 1.0
                pos_prob = pos_prob.squeeze(-1)
                # [bz, seq_length, vocab_size] -> [bz, seq_length]
                # pos_prob, _ = torch.max(self.softmax(outputs.lm_logits.detach()), -1)
                neg_prob, _ = torch.max(self.softmax(neg_outputs.lm_logits), -1)
                ranking_loss = self.ranking_loss(pos_prob.view(-1), neg_prob.view(-1), rank_target)
            if self.args.disc_loss:
                target = torch.zeros(seq_length, bz).to(lm_labels.device)
                target[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                all_logits = self.dfc(torch.reshape(decoder_embed.transpose(1, 2), (-1, embed_size))).view(seq_length,
                                                                                                           -1)  # [seq_length, bz+neg_bz]
                all_logits = all_logits.view(seq_length, self.args.sample_neg_num + 1, bz).transpose(1, 2)
                all_logits = torch.reshape(all_logits,
                                           (-1, self.args.sample_neg_num + 1))  # [seq_length*bz, pos+neg_num]
                disc_loss = self.ce(all_logits.view(-1, self.args.sample_neg_num + 1), target.view(-1).long())
            return query_tloss, weight_tloss, disc_loss, ranking_loss

        def encoder_cal(query, all_doc, valid_num):
            if self.args.intra_rate != 1:
                batch_size = len(valid_num)
                if self.args.loss_func == "sigmoid":
                    func = torch.sigmoid
                elif self.args.loss_func == "tanh":
                    func = torch.tanh
                dot_sim = func(torch.mul(query.unsqueeze(1), all_doc.unsqueeze(0)).sum(-1))

                loss = 0

                for i in range(batch_size):
                    nominator_i = torch.exp(dot_sim[i][i] / self.tau)
                    intra_denominator_i = torch.sum(torch.exp(
                        dot_sim[i][batch_size + sum(valid_num[:i]):batch_size + sum(valid_num[:i + 1])]/self.tau), dim=-1)
                    inter_denominator_i = torch.sum(torch.exp(torch.cat([dot_sim[i][batch_size:batch_size + sum(valid_num[:i])],
                                                               dot_sim[i][batch_size + sum(valid_num[:i + 1]):]],
                                                              dim=0)/self.tau), dim=-1)
                    loss_i = -torch.log(nominator_i) + torch.log(
                        self.args.intra_rate * intra_denominator_i + inter_denominator_i)
                    loss += loss_i
            else:
                batch_size = len(valid_num)
                if self.args.loss_func == "sigmoid":
                    func = torch.sigmoid
                elif self.args.loss_func == "tanh":
                    func = torch.tanh
                dot_sim = func(torch.mul(query.unsqueeze(1), all_doc.unsqueeze(0)).sum(-1))

                nominator_index = torch.arange(batch_size).cuda()
                nominator_index = nominator_index.unsqueeze(1)
                nominator = torch.exp(torch.gather(dot_sim, index=nominator_index, dim=1) / self.tau)

                denominator = torch.sum(torch.exp(dot_sim[:, batch_size:] / self.tau), dim=-1)

                loss = - torch.sum(torch.log(nominator), dim=0) + torch.sum(torch.log(denominator), dim=0)

            return loss/batch_size

        # if self.args.softmax:
        #     logits = self.fc(outputs.encoder_last_hidden_state)[:, 0, :].squeeze()
        #     loss = self.ce(logits, batch["softmax_index"].squeeze(dim=1))
        # else:
        #     if self.args.hard_negative:
        #         query_tloss, weight_tloss, disc_loss, ranking_loss = cal_contrastive(outputs, neg_outputs)
        #         loss = outputs.loss
        #         if self.args.ranking_loss:
        #             loss += ranking_loss
        #         if self.args.disc_loss:
        #             loss += disc_loss
        #             loss = outputs.loss
        #         if self.args.query_tloss:
        #             loss += query_tloss
        #         if self.args.weight_tloss:
        #             loss += weight_tloss
        #     else:
        loss = outputs.loss

        if self.args.Rdrop > 0:
            orig_loss = outputs.orig_loss
            dist_loss = outputs.dist_loss
        else:
            orig_loss, dist_loss = 0, 0

        # if self.args.embedding_distillation > 0:
        #     q_emb_distill_loss = outputs.emb_distill_loss
        # else:
        q_emb_distill_loss = 0

        # if self.args.weight_distillation > 0:
        #     weight_distillation = outputs.weight_distillation
        # else:
        weight_distillation = 0

        if self.args.is_train_encoder:
            valid_num = batch["valid_num"]
            batch_size = self.args.train_batch_size
            all_doc = None
            if self.epoch > self.args.train_encoder_epoch:
                candidates_doc_enc = candidates_doc_enc.view(batch_size, -1, candidates_doc_enc.size(-1))
                for i in range(batch_size):
                    if all_doc is None:
                        all_doc = candidates_doc_enc[i][:valid_num[i]]
                    else:
                        all_doc = torch.cat([all_doc, candidates_doc_enc[i][:valid_num[i]]], dim=0)

                all_doc = torch.cat([positive_doc_enc, all_doc], dim=0)

            else:
                all_doc = torch.cat([positive_doc_enc, candidates_doc_enc], dim=0)

            contrast_loss = encoder_cal(query_enc, all_doc, valid_num)
            loss += contrast_loss.squeeze(0)
        else:
            contrast_loss = 0

            # query_enc, candidates_doc_enc, positive_doc_enc = query_enc.view(batch_size)
            print("do contrastive loss calculate")

        return loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation, contrast_loss

    def _softmax_generative_step(self, batch):
        assert self.args.softmax
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        pred_index = torch.argmax(outputs[0], dim=1)
        return pred_index

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def training_step(self, batch, batch_idx):
        # set to train
        # if self.epoch == 1:
        return {"loss":torch.FloatTensor([0]).cuda().requires_grad_()}
        loss, orig_loss, kl_loss, q_emb_distill_loss, weight_distillation, contrast_loss = self._step(batch)
        self.log("nci_loss", loss - contrast_loss)
        self.log("contrast_loss", contrast_loss)
        self.log("train_loss", loss)
        return {"loss": loss, "nci_loss": loss, "orig_loss": orig_loss, "kl_loss": kl_loss,
                "Query_distill_loss": q_emb_distill_loss,
                "Weight_distillation": weight_distillation,
                "Contrast_loss": contrast_loss}

    def training_epoch_end(self, outputs):

        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        if self.epoch < self.args.begin_val_epoch and self.epoch>0:
            return None
        if self.args.multiple_decoder:
            result_list = []
            for i in range(self.args.decoder_num):
                result = self.validation_step_i(batch, i)
                result_list.append(result)
            return result_list
        else:
            result = self.validation_step_i(batch, -1)
            return result

    def validation_step_i(self, batch, i):

        inf_result_cache = []
        if self.args.decode_embedding:
            if self.args.position:
                expand_scale = self.args.max_output_length if not self.args.hierarchic_decode else 1
                decode_vocab_size = self.args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12

        else:
            decode_vocab_size = None

        assert not self.args.softmax and self.args.gen_method == "greedy"

        if self.args.is_train_encoder:
            output_encoder_embedding = True
        else:
            output_encoder_embedding = False

        if self.args.decode_embedding == 1:
            (outs, scores), encoder_outs = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                output_scores=True,
                output_encoder_embedding=output_encoder_embedding,
                cluster_set=self.cluster
            )
            dec = [numerical_decoder(self.args, ids, output=True) for ids in outs]
        elif self.args.decode_embedding == 2:
            if self.args.multiple_decoder:
                target_mask = batch['target_mask'][i].cuda()
            else:
                target_mask = batch['target_mask'].cuda()
            (outs, scores), encoder_outs = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=target_mask,
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                decoder_index=i,
                output_scores=True,
                output_encoder_embedding=output_encoder_embedding,
                cluster_constraint=self.cluster
            )
            dec = decode_token(self.args, outs.cpu().numpy())  # num = 10*len(pred)
        else:
            (outs, scores), encoder_outs = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                # no_repeat_ngram_size=2,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,  # False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                output_scores=True,
                output_encoder_embedding=output_encoder_embedding,
                cluster_constraint=self.cluster
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]

        texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]

        dec = dec_2d(dec, self.args.num_return_sequences)
        for r in batch['rank']:
            if self.args.label_length_cutoff:
                gt = [s[:self.args.max_output_length - 2] for s in list(r[0])]
            else:
                gt = list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])
        if self.args.is_train_encoder:
            cluster_candidates = []
            for i in range(len(dec)):
                cluster_candidates.extend(dec[i])
            cluster_candidates_index = []
            cluster_candidates_doc = []
            cluster_num = []
            for cluster_id in cluster_candidates:
                cluster_candidates_index.extend(self.id_mapping[cluster_id])
                cluster_num.append(len(self.id_mapping[cluster_id]))

            if self.epoch > self.args.train_encoder_epoch:
                for index in cluster_candidates_index:
                    cluster_candidates_doc.append(self.doc[index][0])
                candidates_doc_inputs = self.encoder_tokenizer.batch_encode_plus(cluster_candidates_doc,
                                                                                 max_length=self.args.encoder_max_len,
                                                                                 padding='max_length', truncation=True,
                                                                                 return_tensors="pt")

                for key in candidates_doc_inputs.keys():
                    candidates_doc_inputs[key] = candidates_doc_inputs[key].cuda()
                candidates_doc_embeds = self.encoder(passage=candidates_doc_inputs)
            else:
                candidates_doc_embeds = None
                for index in cluster_candidates_index:
                    if candidates_doc_embeds is None:
                        candidates_doc_embeds = self.doc_embed[index].cuda()
                    else:
                        candidates_doc_embeds = torch.cat([candidates_doc_embeds, self.doc_embed[index].cuda()], dim=0)

            query_embeds = None
            if self.args.use_query_embed_encoder:
                query_embeds = self.encoder(query_enc=encoder_outs.last_hidden_state[::self.args.num_return_sequences].cuda())
            if self.args.use_query_embed_decoder_avg:
                decoder_input = outs
                insert_place = torch.sum(~decoder_input.eq(self.model_config.pad_token_id), dim=-1)

                decoder_attention = torch.ones_like(decoder_input).cuda()
                mask_attention = (~decoder_input.eq(self.model_config.pad_token_id)).int()
                mask_insert = torch.zeros_like(insert_place).cuda()
                mask_attention = mask_attention.scatter(-1, mask_insert.unsqueeze(-1), 1)

                decoder_attention_mask = decoder_attention * mask_attention

                input_ids = batch["source_ids"].cuda()
                input_ids = input_ids.repeat(int(outs.size(0)/self.args.eval_batch_size),1)
                attention_mask = batch["source_mask"].cuda()
                attention_mask = attention_mask.repeat(int(outs.size(0)/self.args.eval_batch_size), 1)

                mod_results = torch.remainder(torch.arange(input_ids.size(0)), self.args.eval_batch_size)
                input_ids = [input_ids[mod_results == i] for i in range(self.args.eval_batch_size)]
                input_ids = torch.cat(input_ids, dim=0)
                attention_mask = [attention_mask[mod_results == i] for i in range(self.args.eval_batch_size)]
                attention_mask = torch.cat(attention_mask, dim=0)

                out_decoder_avg = self.model(
                    input_ids,
                    input_mask=None,
                    logit_mask=None,
                    encoder_outputs=None,
                    only_encoder=False,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input,
                    decoder_attention_mask=decoder_attention_mask,
                    # query_embedding=batch["query_embedding"].cuda(),
                    # prefix_embedding=batch["prefix_emb"].cuda(),
                    # prefix_mask=batch["prefix_mask"],
                    lm_labels=None,
                    return_dict=True,
                    output_hidden_states=True,
                )

                query_embeds = out_decoder_avg.decoder_hidden_states[-1]
                query_embeds = torch.sum(query_embeds * decoder_attention_mask.unsqueeze(-1), dim=1) / torch.sum(decoder_attention_mask, dim=1, keepdim=True)

            if self.args.use_query_embed_decoder_special:
                decoder_input = outs
                # pad_tensor = torch.full((outs.size(0), 1), self.model_config.pad_token_id).cuda()

                # decoder_input = torch.cat([decoder_input, pad_tensor], dim=-1)
                bool_tensor = torch.eq(decoder_input, self.model_config.eos_token_id)
                insert_place = torch.nonzero(bool_tensor)

                # decoder_input[insert_place[:, 0], insert_place[:, 1]] = self.model_config.pad_token_id
                decoder_attention = torch.ones_like(decoder_input).cuda()
                mask_attention = (~decoder_input.eq(self.model_config.pad_token_id)).int()

                decoder_attention_mask = decoder_attention * mask_attention
                decoder_attention_mask[:, 0] = 1

                input_ids = batch["source_ids"].cuda()
                input_ids = input_ids.repeat(int(outs.size(0)/self.args.eval_batch_size),1)
                attention_mask = batch["source_mask"].cuda()
                attention_mask = attention_mask.repeat(int(outs.size(0)/self.args.eval_batch_size), 1)

                # why doing mod?
                mod_results = torch.remainder(torch.arange(input_ids.size(0)), self.args.eval_batch_size)
                input_ids = [input_ids[mod_results == i] for i in range(self.args.eval_batch_size)]
                input_ids = torch.cat(input_ids, dim=0)
                attention_mask = [attention_mask[mod_results == i] for i in range(self.args.eval_batch_size)]
                attention_mask = torch.cat(attention_mask, dim=0)

                out_decoder_special = self.model(
                    input_ids,
                    input_mask=None,
                    logit_mask=None,
                    encoder_outputs=None,
                    only_encoder=False,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input,
                    decoder_attention_mask=decoder_attention_mask,
                    # query_embedding=batch["query_embedding"].cuda(),
                    # prefix_embedding=batch["prefix_emb"].cuda(),
                    # prefix_mask=batch["prefix_mask"],
                    lm_labels=None,
                    return_dict=True,
                    output_hidden_states=True,
                )

                if query_embeds is not None and self.args.use_query_embed_encoder:
                    query_embeds_tmp = query_embeds.clone()
                    query_embeds = out_decoder_special.decoder_hidden_states[-1]
                    mask = torch.zeros((query_embeds.size(0), query_embeds.size(1))).cuda()
                    mask[insert_place[:, 0], insert_place[:, 1]] = 1
                    query_embeds = query_embeds[mask.bool()]

                    query_embeds_tmp = query_embeds_tmp.repeat(1, self.args.num_return_sequences, 1).view(-1,query_embeds_tmp.size(-1))
                    if self.args.fusion_strategy == "concate":
                        query_embeds = self.fusion_layer(torch.cat([query_embeds, query_embeds_tmp], dim=-1))

                    if self.args.fusion_strategy == "average":
                        query_embeds = (query_embeds+query_embeds_tmp)/2

                else:
                    query_embeds = out_decoder_special.decoder_hidden_states[-1]
                    mask = torch.zeros((query_embeds.size(0), query_embeds.size(1))).cuda()
                    mask[insert_place[:,0], insert_place[:,1]] = 1
                    query_embeds = query_embeds[mask.bool()]


            if candidates_doc_embeds.ndim < 2:
                candidates_doc_embeds = candidates_doc_embeds.view(-1, self.model.config.hidden_size)

            if self.args.use_query_embed_encoder:
                if self.args.loss_func == "sigmoid":
                    func = torch.sigmoid
                elif self.args.loss_func == "tanh":
                    func = torch.tanh
                doc_similarity = func(torch.mul(query_embeds.unsqueeze(1), candidates_doc_embeds.unsqueeze(0)).sum(-1))
            else:
                if self.args.loss_func == "sigmoid":
                    func = torch.sigmoid
                elif self.args.loss_func == "tanh":
                    func = torch.tanh
                doc_similarity_all = func(torch.mul(query_embeds.unsqueeze(1), candidates_doc_embeds.unsqueeze(0)).sum(-1))
                doc_similarity = None
                for i in range(len(cluster_num)):
                    if doc_similarity is None:
                        doc_similarity = doc_similarity_all[i][:cluster_num[i]]
                    else:
                        doc_similarity = torch.cat([doc_similarity, doc_similarity_all[i][sum(cluster_num[:i]):sum(cluster_num[:i+1])]], dim=0)

            num_rate = self.args.score_rate
            total_score = [[]for i in range(self.args.eval_batch_size)]
            scores = torch.tensor(scores).view(self.args.eval_batch_size, -1).tolist()
            cluster_num = torch.tensor(cluster_num).view(self.args.eval_batch_size, -1).tolist()
            prob_scores = scores
            prob_scores = self.softmax(torch.tensor(prob_scores))
            doc_score = doc_similarity
            doc_score_all = []
            cluster_candidates_index_all = []
            sum_num = 0
            if self.args.use_query_embed_encoder:
                for i in range(self.args.eval_batch_size):
                    doc_score_all.append(doc_score[i][sum_num:sum_num + sum(cluster_num[i])].clone())
                    cluster_candidates_index_all.append(
                        cluster_candidates_index[sum_num:sum_num + sum(cluster_num[i])].copy())
                    sum_num += sum(cluster_num[i])
            else:
                for i in range(self.args.eval_batch_size):
                    doc_score_all.append(doc_score[sum_num:sum_num + sum(cluster_num[i])].clone())
                    cluster_candidates_index_all.append(cluster_candidates_index[sum_num:sum_num + sum(cluster_num[i])].copy())
                    sum_num += sum(cluster_num[i])

            inf_index_batch_all = [[] for i in range(self.args.eval_batch_size)]
            for batch_id in range(self.args.eval_batch_size):
                for rate in num_rate:
                    alpha = rate
                    doc_score_sub = doc_score_all[batch_id].clone()
                    for i in range(len(cluster_num[batch_id])):
                        doc_score_sub[sum(cluster_num[batch_id][:i]):sum(cluster_num[batch_id][:i+1])] = doc_score_sub[sum(cluster_num[batch_id][:i]):sum(cluster_num[batch_id][:i+1])] + alpha*prob_scores[batch_id][i]
                    values, indices = doc_score_sub.topk(self.args.num_return_sequences, dim=0, largest=True, sorted=True)
                    total_score[batch_id].append((values.clone(), indices.clone()))

                predict_answers_all = []
                for (values, indices) in total_score[batch_id]:
                    predict_answers = []
                    for index in indices:
                        predict_answers.append(str(cluster_candidates_index_all[batch_id][index]))
                    predict_answers_all.append(predict_answers.copy())
                    gt_answers = batch["oldid"][0][batch_id]
                    inf_index_batch = []
                    inf_index_batch.append([texts[batch_id], ",".join(predict_answers), gt_answers])
                    inf_index_batch_all[batch_id].append(inf_index_batch.copy())
        else:
            inf_index_batch_all = []

        return {"inf_result_batch": inf_result_cache, 'inf_result_batch_prob': scores,
                "inf_index_batch": inf_index_batch_all}

    def validation_epoch_end(self, outputs):
        if self.epoch < self.args.begin_val_epoch and self.epoch> 0 :
            self.epoch = self.epoch + 1
            self.l1_query_train_dataset.epoch += 1
            self.log("recall1", self.epoch / 10000)
            return None
        if self.args.multiple_decoder:
            reverse_outputs = []
            for j in range(len(outputs[0])):
                reverse_outputs.append([])
            for i in range(len(outputs)):
                for j in range(len(outputs[0])):
                    reverse_outputs[j].append(outputs[i][j])
            outputs = reverse_outputs

        if self.args.multiple_decoder:
            inf_result_cache = []
            inf_result_cache_prob = []
            inf_index_cache = []
            for index in range(self.args.decoder_num):
                cur_inf_result_cache = [item for sublist in outputs[index] for item in sublist['inf_result_batch']]
                cur_inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(
                    len(sublist['inf_result_batch_prob']) / len(outputs[index][0]['inf_result_batch'])): (i + 1) * int(
                    len(sublist['inf_result_batch_prob']) / len(outputs[index][0]['inf_result_batch']))]) for sublist in
                                             outputs[index] for i in range(len(sublist['inf_result_batch']))]
                inf_result_cache.extend(cur_inf_result_cache)
                inf_result_cache_prob.extend(cur_inf_result_cache_prob)
                cur_index_result_cache = [item for sublist in outputs[index] for item in sublist['inf_index_batch']]
                inf_index_cache.extend(cur_index_result_cache)

            res_all = [inf_index_cache.copy(), inf_result_cache.copy()]

        else:
            res_all = []
            for index in range(len(outputs[0]['inf_index_batch'][0])):
                inf_result_cache = [item  for sublist in outputs for item in sublist['inf_result_batch']]
                inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(
                    len(sublist['inf_result_batch_prob']) / len(outputs[0]['inf_result_batch'])): (i + 1) * int(
                    len(sublist['inf_result_batch_prob']) / len(outputs[0]['inf_result_batch']))]) for sublist in outputs
                                      for batch_id in range(self.args.eval_batch_size)   for i in range(len(sublist['inf_result_batch']))]
                inf_index_cache = [item for sublist in outputs for batch_id in range(self.args.eval_batch_size) for item in sublist['inf_index_batch'][batch_id][index]]
                res_all.append((inf_index_cache.copy(), inf_result_cache.copy()))

        for index, (inf_index_cache, inf_result_cache) in enumerate(res_all):
            appendix = self.args.score_rate[index]
            res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
            res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
            res1 = res.loc[res['rank'] == 1]
            # res1.to_csv(self.args.res1_save_path, mode='w', sep="\t", header=None, index=False)
            res1 = res1.values.tolist()


            q_gt, q_pred = {}, {}
            prev_q = ""
            for [query, pred, gt, _] in res1:
                if query != prev_q:
                    q_pred[query] = pred.split(",")
                    # q_pred[query] = q_pred[query][:5]
                    # q_pred[query] = list(set(q_pred[query]))
                    prev_q = query
                if query in q_gt:
                    if len(q_gt[query]) <= 100:
                        q_gt[query].add(gt)
                else:
                    q_gt[query] = gt.split(",")
                    q_gt[query] = list(set(q_gt[query]))

            if self.args.is_train_encoder:
                res_index = pd.DataFrame(inf_index_cache, columns=["query", "pred", "gt"])
                res_index = res_index.values.tolist()

                q_gt_index, q_pred_index = {}, {}
                prev_q_index = ""
                for [query, pred, gt] in res_index:
                    if query != prev_q_index:
                        q_pred_index[query] = pred.split(",")
                        # q_pred[query] = q_pred[query][:5]
                        # q_pred[query] = list(set(q_pred[query]))
                        prev_q_index = query
                    if query in q_gt_index:
                        if len(q_gt_index[query]) <= 100:
                            q_gt_index[query].append(gt)
                    else:
                        q_gt_index[query] = gt.split(",")
                        q_gt_index[query] = list(set(q_gt_index[query]))

            def cal_recall(q_pred, q_gt, k):
                total_hit = 0
                total_positive = 0
                total_recall = 0
                for q in q_pred:
                    is_hit = 0
                    total_positive += len(q_gt[q])
                    for p in q_gt[q]:
                        if p in q_pred[q][:k]:
                            is_hit += 1
                    total_recall += is_hit / len(q_gt[q])
                    total_hit += is_hit
                recall_avg_mic = total_hit / total_positive
                recall_avg_mac = total_recall / len(q_pred)
                return recall_avg_mac, recall_avg_mic

            def cal_accuracy(q_pred, q_gt, k):
                total_accuracy = 0
                for q in q_pred:
                    is_hit = 0
                    for p in q_pred[q][:k]:
                        if p in q_gt[q]:
                            is_hit = 1
                            break
                    total_accuracy += is_hit

                total_accuracy_avg = total_accuracy / len(q_pred)
                return total_accuracy_avg

            def cal_MRR(q_pred, q_gt, k):
                total_MRR = 0
                for q in q_pred:
                    rank = 1
                    # answer = q_gt[q][0]
                    for p in q_pred[q][:k]:
                        if p in q_gt[q]:
                            total_MRR += 1 / rank
                            break
                        else:
                            rank += 1

                total_MRR_avg = total_MRR / len(q_pred)
                return total_MRR_avg

            def cal_MAP(q_pred, q_gt, k):
                total_MAP = 0
                for q in q_pred:
                    rank = 1
                    pred_true = 1
                    local_MAP = 0
                    for p in q_pred[q][:k]:
                        if p in q_gt[q]:
                            local_MAP += pred_true / rank
                            pred_true += 1
                            rank += 1
                        else:
                            rank += 1
                    total_MAP += local_MAP / k
                total_MAP_avg = total_MAP / len(q_pred)
                return total_MAP_avg

            if self.args.is_train_encoder:
                mac_recall_1, mic_recall_1 = cal_recall(q_pred, q_gt, 1)
                mac_recall_5, mic_recall_5 = cal_recall(q_pred, q_gt, 5)
                mac_recall_10, mic_recall_10 = cal_recall(q_pred, q_gt, 10)
                mac_recall_20, mic_recall_20 = cal_recall(q_pred, q_gt, 20)
                mac_recall_50, mic_recall_50 = cal_recall(q_pred, q_gt, 50)
                mac_recall_100, mic_recall_100 = cal_recall(q_pred, q_gt, 100)
                self.log("cluster_recall1", mac_recall_1)
                self.log("cluster_recall5", mac_recall_5)
                self.log("cluster_recall10", mac_recall_10)
                self.log("cluster_recall20", mac_recall_20)
                self.log("cluster_recall50", mac_recall_50)
                self.log("cluster_recall100", mac_recall_100)

                accuracy_1 = cal_accuracy(q_pred, q_gt, 1)
                accuracy_20 = cal_accuracy(q_pred, q_gt, 20)
                accuracy_100 = cal_accuracy(q_pred, q_gt, 100)
                self.log("cluster_accuracy1", accuracy_1)
                self.log("cluster_accuracy20", accuracy_20)
                self.log("cluster_accuracy100", accuracy_100)

                MRR_100 = cal_MRR(q_pred, q_gt, 100)
                self.log("cluster_MRR100", MRR_100)
                MRR_10 = cal_MRR(q_pred, q_gt, 10)
                self.log("cluster_MRR10", MRR_10)

                MAP_100 = cal_MAP(q_pred, q_gt, 100)
                self.log("cluster_MAP100", MAP_100)
                print("Epoch:  {}".format(str(self.epoch)))
                print("cluster_acc@1:{}".format(str(accuracy_1)))
                print("cluster_acc@100:{}".format(str(accuracy_100)))
                print("cluster_recall@1:{}".format(str(mac_recall_1)))
                print("cluster_recall@5:{}".format(str(mac_recall_5)))
                print("cluster_recall@10:{}".format(str(mac_recall_10)))
                print("cluster_recall@20:{}".format(str(mac_recall_20)))
                print("cluster_recall@100:{}".format(str(mac_recall_100)))
                print("cluster_MRR100:{}".format(str(MRR_100)))
                ############################## total result ######################################

                mac_recall_1, mic_recall_1 = cal_recall(q_pred_index, q_gt_index, 1)
                mac_recall_5, mic_recall_5 = cal_recall(q_pred_index, q_gt_index, 5)
                mac_recall_10, mic_recall_10 = cal_recall(q_pred_index, q_gt_index, 10)
                mac_recall_20, mic_recall_20 = cal_recall(q_pred_index, q_gt_index, 20)
                mac_recall_50, mic_recall_50 = cal_recall(q_pred_index, q_gt_index, 50)
                mac_recall_100, mic_recall_100 = cal_recall(q_pred_index, q_gt_index, 100)
                if appendix == 0:
                    self.log("recall1", mac_recall_1)
                self.log("recall1_{}".format(appendix), mac_recall_1)
                self.log("recall5_{}".format(appendix), mac_recall_5)
                self.log("recall10_{}".format(appendix), mac_recall_10)
                self.log("recall20_{}".format(appendix), mac_recall_20)
                self.log("recall50_{}".format(appendix), mac_recall_50)
                self.log("recall100_{}".format(appendix), mac_recall_100)

                accuracy_1 = cal_accuracy(q_pred_index, q_gt_index, 1)
                accuracy_20 = cal_accuracy(q_pred_index, q_gt_index, 20)
                accuracy_100 = cal_accuracy(q_pred_index, q_gt_index, 100)
                self.log("accuracy1_{}".format(appendix), accuracy_1)
                self.log("accuracy20_{}".format(appendix), accuracy_20)
                self.log("accuracy100_{}".format(appendix), accuracy_100)

                MRR_100 = cal_MRR(q_pred_index, q_gt_index, 100)
                self.log("MRR100_{}".format(appendix), MRR_100)
                MRR_10 = cal_MRR(q_pred_index, q_gt_index, 10)
                self.log("MRR10_{}".format(appendix), MRR_10)

                MAP_100 = cal_MAP(q_pred_index, q_gt_index, 100)
                self.log("MAP100_{}".format(appendix), MAP_100)
                print("Epoch:  {}".format(str(self.epoch)))
                print("acc@1_{}:{}".format(appendix, str(accuracy_1)))
                print("acc@100_{}:{}".format(appendix, str(accuracy_100)))
                print("recall@1_{}:{}".format(appendix, str(mac_recall_1)))
                print("recall@5_{}:{}".format(appendix, str(mac_recall_5)))
                print("recall@10_{}:{}".format(appendix, str(mac_recall_10)))
                print("recall@20_{}:{}".format(appendix, str(mac_recall_20)))
                print("recall@50_{}:{}".format(appendix, str(mac_recall_50)))
                print("recall@100_{}:{}".format(appendix, str(mac_recall_100)))
                print("MRR100_{}:{}".format(appendix, str(MRR_100)))


            else:
                mac_recall_1, mic_recall_1 = cal_recall(q_pred, q_gt, 1)
                mac_recall_5, mic_recall_5 = cal_recall(q_pred, q_gt, 5)
                mac_recall_10, mic_recall_10 = cal_recall(q_pred, q_gt, 10)
                mac_recall_20, mic_recall_20 = cal_recall(q_pred, q_gt, 20)
                mac_recall_50, mic_recall_50 = cal_recall(q_pred, q_gt, 50)
                mac_recall_100, mic_recall_100 = cal_recall(q_pred, q_gt, 100)
                self.log("recall1", mac_recall_1)
                self.log("recall5", mac_recall_5)
                self.log("recall10", mac_recall_10)
                self.log("recall20", mac_recall_20)
                self.log("recall50", mac_recall_50)
                self.log("recall100", mac_recall_100)

                accuracy_1 = cal_accuracy(q_pred, q_gt, 1)
                accuracy_20 = cal_accuracy(q_pred, q_gt, 20)
                accuracy_100 = cal_accuracy(q_pred, q_gt, 100)
                self.log("accuracy1", accuracy_1)
                self.log("accuracy20", accuracy_20)
                self.log("accuracy100", accuracy_100)

                MRR_100 = cal_MRR(q_pred, q_gt, 100)
                self.log("MRR100", MRR_100)
                MRR_10 = cal_MRR(q_pred, q_gt, 10)
                self.log("MRR10", MRR_10)

                MAP_100 = cal_MAP(q_pred, q_gt, 100)
                self.log("MAP100", MAP_100)
                print("Epoch:  {}".format(str(self.epoch)))
                print("acc@1:{}".format(str(accuracy_1)))
                print("acc@100:{}".format(str(accuracy_100)))
                print("recall@1:{}".format(str(mac_recall_1)))
                print("recall@5:{}".format(str(mac_recall_5)))
                print("recall@10:{}".format(str(mac_recall_10)))
                print("recall@20:{}".format(str(mac_recall_20)))
                print("recall@50:{}".format(str(mac_recall_20)))
                print("recall@100:{}".format(str(mac_recall_100)))
                print("MRR100:{}".format(str(MRR_100)))
            print("############################################################")
        self.epoch += 1
        self.l1_query_train_dataset.epoch += 1
        if self.epoch == self.args.train_encoder_epoch+1:
            print("start stage2 !!!!!!!")
            self.args.train_batch_size = self.args.stage2_train_batchsize
            self.args.eval_batch_size = self.args.stage2_eval_batchsize
            self.trainer.val_dataloader = self.val_dataloader()
            self.trainer.train_dataloader = self.train_dataloader()


    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        encoder = self.encoder
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in encoder.named_parameters() if
                           (not any(nd in n for nd in no_decay))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.doc_encoder_learning_rate,
            },
            {
                "params": [p for n, p in encoder.named_parameters() if
                           (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
                "lr": self.args.doc_encoder_learning_rate,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        if self.args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
            )

        elif self.args.scheduler == "exp":
            lf = lambda x: math.pow((self.t_total - x - 0.95) / self.t_total, 0.1)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 动态学习率

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        print('load training data and create training loader.')
        n_samples = self.n_obs['train']
        if hasattr(self, 'l1_query_train_dataset'):
            train_dataset = self.l1_query_train_dataset
        else:
            train_dataset = l1_query(self.args, self.tokenizer, self.encoder_tokenizer, n_samples)
        self.prefix_embedding, self.prefix2idx_dict, self.prefix_mask = \
            train_dataset.prefix_embedding, train_dataset.prefix2idx_dict, train_dataset.prefix_mask
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader

    def val_dataloader(self):
        print('load validation data and create validation loader.')
        n_samples = self.n_obs['validation']
        val_dataset = l1_query(self.args, self.tokenizer, self.encoder_tokenizer, n_samples, task='test')
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(val_dataset, sampler=sampler, batch_size=self.args.eval_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader
