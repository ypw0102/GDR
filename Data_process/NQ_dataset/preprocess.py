from tqdm import trange
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nq')
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--num', type=str, default='334314')
parser.add_argument('--k', type=int, default= 10)
parser.add_argument('--c', type=int, default= 100)

args = parser.parse_args()


## merge parallel results
output_bert_base_tensor_nq_qg = []
output_bert_base_id_tensor_nq_qg = []
for num in trange(4):
    with open(f'qg/pkl/{args.dataset}_output_tensor_512_content_{args.max_len}_{args.return_num}_{num}_id.pkl', 'rb') as f:
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