from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
import pickle
import pandas as pd
from AR2model import BiBertEncoder,get_arguments

def main(args):
    device = f'cuda:{args.cuda_device}'
    #device = "cpu"
    id_doc_dict = {}
    if 'NQ' in args.dataset:
        train_file = "../NQ_doc_content_100.tsv"
        with open(train_file, 'r') as f:
            for line in f.readlines():
                docid, _, _, content, _, _, _ = line.split("\t")
                id_doc_dict[docid] = content
    elif 'Trivia' in args.dataset:
        train_file = "../Trivia_doc_content.tsv"
        with open(train_file, 'r') as f:
            for line in f.readlines():
                _, docid, _, _, content = line.split("\t")
                id_doc_dict[docid] = content

    args_ = get_arguments()
    model = BiBertEncoder(args_).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained("bert-base-uncased").to(f'cuda:{args.cuda_device}')

    ids = list(id_doc_dict.keys())
    text_list_all = []
    text_id_all = []
    i = 0
    batch_size = 20

    while (i < int(len(ids) / batch_size) - 1):
        id_list = ids[i * batch_size: (i + 1) * batch_size]
        text_list = [id_doc_dict[id_] for id_ in id_list]
        text_list_all.append(text_list)
        text_id_all.append(id_list)
        i += 1

    id_list = ids[i * batch_size:]
    text_list = [id_doc_dict[id_] for id_ in id_list]
    text_list_all.append(text_list)
    text_id_all.append(id_list)

    text_partitation = []
    text_partitation_id = []

    base = int(len(text_list_all) / args.partition_num)

    text_partitation.append(text_list_all[:base])
    text_partitation_id.append(text_id_all[:base])
    
    for i in range(args.partition_num-2):
        text_partitation.append(text_list_all[(i+1)*base: (i+2)*base])
        text_partitation_id.append(text_id_all[(i+1)*base: (i+2)*base])

    text_partitation.append(text_list_all[(i+2)*base:  ])
    text_partitation_id.append(text_id_all[(i+2)*base:  ])


    output_tensor = []
    output_id_tensor = []
    count = 0

    for elem in tqdm(text_partitation[args.idx]):
        encoded_input = tokenizer(elem, max_length=args.max_len, return_tensors='pt', padding=True, truncation=True).to(
            device)
        output = model.ctx_model(**encoded_input, return_dict=True)[1].detach().cpu().numpy().tolist()
        #output = model(**encoded_input, return_dict=True).last_hidden_state.detach().cpu()[:, 0, :].numpy().tolist()
        output_tensor.extend(output)
        output_id_tensor.extend(text_partitation_id[args.idx][count])
        count += 1

    output = open(f'pkl/{args.dataset}_output_tensor_{args.max_len}_content_{args.idx}.pkl', 'wb', -1)
    pickle.dump(output_tensor, output)
    output.close()

    output = open(f'pkl/{args.dataset}_output_tensor_{args.max_len}_content_{args.idx}_id.pkl', 'wb', -1)
    pickle.dump(output_id_tensor, output)
    output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    parser.add_argument("--idx", type=int, default=0, help="partitation")
    parser.add_argument("--partition_num", type=int, default=1, help="partitation")
    parser.add_argument("--dataset", type=str, default='NQ', help="partitation")
    parser.add_argument("--cuda_device", type=int, default=0, help="cuda")
    parser.add_argument("--max_len", type=int, default=512, help="cuda")

    args = parser.parse_args()
    print(args)

    main(args)
