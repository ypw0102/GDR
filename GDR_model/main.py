import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import pickle

import nltk
import pandas as pd
import time
import torch
import pytorch_lightning as pl

from main_metrics import recall, MRR100
from main_models import T5FineTuner, l1_query, decode_token
from main_utils import set_seed, get_ckpt, dec_2d, numerical_decoder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm
from transformers import T5Tokenizer

# print(1)
# nltk.download('punkt')
# print(2)
print(torch.__version__)  # 1.10.0+cu113
print(pl.__version__)  # 1.4.9

logger = None
YOUR_API_KEY = 'yourapikey'  # wandb token, please get yours from wandb portal

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def train(args):
    model = T5FineTuner(args)

    if args.infer_ckpt != '':
        ckpt_path = args.infer_ckpt
        state_dict = torch.load(ckpt_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if args.ckpt_monitor == 'train_loss':
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename=args.tag_info + '_{epoch}-{avg_train_loss:.6f}',
            save_on_train_epoch_end=True,
            monitor="avg_train_loss",
            mode="min",
            save_top_k=1,
            every_n_val_epochs=args.check_val_every_n_epoch,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=True,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
            plugins=DDPPlugin(find_unused_parameters=False),
            accelerator=args.accelerator
        )
    elif args.ckpt_monitor == 'recall':
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename=args.tag_info + '_{epoch}-{recall1:.6f}',
            monitor="recall1",
            save_on_train_epoch_end=False,
            mode="max",
            save_top_k=1,
            every_n_val_epochs=args.check_val_every_n_epoch,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=list(range(args.n_gpu)),  # args.n_gpu,
            # devices=args.devices,
            # strategy="ddp",
            num_nodes=args.nodes,
            max_epochs=args.num_train_epochs,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=True,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
            plugins=DDPPlugin(find_unused_parameters=True),
            accelerator=args.accelerator,
            amp_backend='apex',
        )
        print("ngpu:{}".format(str(args.n_gpu)))
        print("usable gpu num:{}".format(str(torch.cuda.device_count())))
    else:
        NotImplementedError("This monitor is not implemented!")

    trainer = pl.Trainer(**train_params)
    print("*!*!*!*!*!*!*!*!*!*!*!*!*!*!")
    trainer.fit(model)


def inference(args):
    model = T5FineTuner(args, train=False)
    if args.infer_ckpt:
        ckpt_path = args.infer_ckpt
    else:
        ckpt_path, ckpt_epoch = get_ckpt(args)
    state_dict = torch.load(ckpt_path)
    # print(state_dict)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("tokenizer ok!")
    num_samples = args.n_test if args.n_test >= 0 else None
    dataset = l1_query(args, tokenizer, num_samples=num_samples, task='test')
    model.to("cuda")
    model.eval()

    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    losses = []

    inf_result_cache = []
    # begin
    # time_begin_infer = time.time()
    for batch in tqdm(loader):
        # time1 = time.time()
        lm_labels = batch["target_ids"].numpy().copy()
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100
        lm_labels = torch.from_numpy(lm_labels).cuda()
        if args.decode_embedding:
            if args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12
        else:
            decode_vocab_size = None

        # time_modelbegin = time.time()
        if args.softmax:
            outs = model(
                input_ids=batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                decoder_input_ids=None,
                lm_labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].cuda(),
            )
            loss = model.ce(outs[0], batch["softmax_index"].squeeze(dim=1).cuda())
            losses.append(loss.detach().cpu().numpy())
            pred_index = torch.argmax(outs[0], dim=1).detach().cpu().numpy()
            dec = pred_index.tolist()
        else:
            print(args.gen_method + "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if args.gen_method == "greedy":
                with torch.no_grad():
                    outs = model.model.generate(
                        batch["source_ids"].cuda(),
                        # decoder_index=i,
                        attention_mask=batch["source_mask"].cuda(),
                        use_cache=False,
                        decoder_attention_mask=batch['target_mask'].cuda(),
                        max_length=args.max_output_length,
                        num_beams=args.num_return_sequences,
                        # no_repeat_ngram_size=2,
                        length_penalty=args.length_penalty,
                        num_return_sequences=args.num_return_sequences,
                        early_stopping=False,
                        decode_embedding=args.decode_embedding,
                        decode_vocab_size=decode_vocab_size,
                        decode_tree=model.root,
                        # output_scores=True
                    )
            elif args.gen_method == "top_k":
                outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=False,
                    do_sample=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=args.max_output_length,
                    top_k=1000,
                    # top_p=0.95,
                    num_return_sequences=args.num_return_sequences,
                    length_penalty=args.length_penalty,
                    decode_embedding=args.decode_embedding,
                    decode_vocab_size=decode_vocab_size,
                )
            else:
                outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=False,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=args.max_output_length,
                    num_beams=2,
                    # repetition_penalty=2.5,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                    decode_embedding=args.decode_embedding,
                    decode_vocab_size=decode_vocab_size,
                )

            if args.decode_embedding == 1:
                dec = [numerical_decoder(args, ids, output=True) for ids in outs]
            elif args.decode_embedding == 2:
                dec = decode_token(args, outs.cpu().numpy())  # num = 10*len(pred)
            else:
                dec = [tokenizer.decode(ids) for ids in outs]
        # time_modelend = time.time()
        # print('model_time:', time_modelend - time_modelbegin)

        texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
        dec = dec_2d(dec, args.num_return_sequences)
        for r in batch['rank']:
            if args.label_length_cutoff:
                gt = [s[:args.max_output_length - 2] for s in list(r[0])]
            else:
                gt = list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])

    # time_end_infer = time.time()
    # print('alltime:', time_end_infer - time_begin_infer)

    res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
    res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
    res1 = res.loc[res['rank'] == 1]
    res1.to_csv(args.res1_save_path, mode='w', sep="\t", header=None, index=False)
    recall_value = recall(args)
    mrr_value = MRR100(args)
    return recall_value, mrr_value


def calculate(args):
    recall_value = recall(args)
    mrr_value = MRR100(args)

    return recall_value, mrr_value


def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default="t5-")
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-")
    parser.add_argument('--encoder_name_or_path', type=str, default="bert--uncased")
    parser.add_argument('--encoder_tokenizer_name_or_path', type=str, default="bert--uncased")
    parser.add_argument('--freeze_encoder', type=int, default=0, choices=[0, 1])
    parser.add_argument('--freeze_embeds', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--n_val', type=int, default=-1)
    parser.add_argument('--n_train', type=int, default=-1)
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--early_stop_callback', type=int, default=0, choices=[0, 1])
    parser.add_argument('--fp_16', type=int, default=0, choices=[0, 1])
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_encoder', type=int, default=1, choices=[0, 1])
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--softmax', type=int, default=0, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--accelerator', type=str, default="ddp")
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_cls', type=int, default=1000)
    parser.add_argument('--decode_embedding', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--output_vocab_size', type=int, default=30)
    parser.add_argument('--hierarchic_decode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_word_embedding', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_decode_embedding', type=int, default=1, choices=[0, 1])
    parser.add_argument('--gen_method', type=str, default="greedy")
    parser.add_argument('--length_penalty', type=int, default=0.8)

    parser.add_argument('--recall_num', type=list, default=[1, 5, 10, 20, 50, 100], help='[1,5,10,20,50,100]')
    parser.add_argument('--random_gen', type=int, default=0, choices=[0, 1])
    parser.add_argument('--label_length_cutoff', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    parser.add_argument('--test_set', type=str, default="dev")
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=4)

    parser.add_argument('--max_input_length', type=int, default=40)
    parser.add_argument('--inf_max_input_length', type=int, default=40)
    parser.add_argument('--max_output_length', type=int, default=7)
    parser.add_argument('--doc_length', type=int, default=16)
    parser.add_argument('--contrastive_variant', type=str, default="", help='E_CL, ED_CL, doc_Reweight')
    parser.add_argument('--num_return_sequences', type=int, default=100, help='generated id num (include invalid)')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval', 'calculate'])
    parser.add_argument('--query_type', type=str, default='gtq_doc',
                        help='gtq -- use ground turth query;'
                             'qg -- use qg; '
                             'doc -- just use top64 doc token; '
                             'aug -- use random doc token. ')
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--decoder_learning_rate', type=float, default=1e-4)
    parser.add_argument('--certain_epoch', type=int, default=None)
    parser.add_argument('--given_ckpt', type=str, default='')
    parser.add_argument('--infer_ckpt', type=str, default='')
    parser.add_argument('--model_info', type=str, default='base', choices=['small', 'large', 'base', '3b', '11b'])
    parser.add_argument('--encoder_info', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--id_class', type=str, default='bert_k30_c30_1')
    parser.add_argument('--ckpt_monitor', type=str, default='recall', choices=['recall', 'train_loss'])
    parser.add_argument('--Rdrop', type=float, default=0, help='default to 0-0.3')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--Rdrop_only_decoder', type=int, default=0,
                        help='1-RDrop only for decoder, 0-RDrop only for all model', choices=[0, 1])
    parser.add_argument('--Rdrop_loss', type=str, default='KL', choices=['KL', 'L2'])
    parser.add_argument('--adaptor_decode', type=int, default=1, help='default to 0,1')
    parser.add_argument('--adaptor_efficient', type=int, default=1, help='default to 0,1')
    parser.add_argument('--adaptor_layer_num', type=int, default=4)
    parser.add_argument('--test1000', type=int, default=0, help='default to 0,1')
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--contrastive', type=int, default=0)
    parser.add_argument('--embedding_distillation', type=float, default=0.0)
    parser.add_argument('--weight_distillation', type=float, default=0.0)
    parser.add_argument('--hard_negative', type=int, default=0)
    parser.add_argument('--aug_query', type=int, default=0)
    parser.add_argument('--aug_query_type', type=str, default='corrupted_query', help='aug_query, corrupted_query')
    parser.add_argument('--sample_neg_num', type=int, default=0)
    parser.add_argument('--query_tloss', type=int, default=0)
    parser.add_argument('--weight_tloss', type=int, default=0)
    parser.add_argument('--ranking_loss', type=int, default=0)
    parser.add_argument('--disc_loss', type=int, default=0)
    parser.add_argument('--input_dropout', type=int, default=1)
    parser.add_argument('--denoising', type=int, default=0)
    parser.add_argument('--multiple_decoder', type=int, default=0)
    parser.add_argument('--decoder_num', type=int, default=1)
    parser.add_argument('--loss_weight', type=int, default=0)
    parser.add_argument('--nq', type=int, default=1)
    parser.add_argument('--kary', type=int, default=30)
    parser.add_argument('--tree', type=int, default=1)
    parser.add_argument('--ckpt_info', type=str, default="334314test")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--docnum', type=int, default=190727, help='if -1 then NCI data,else our data')
    parser.add_argument('--project_path', type=str,
                        default="your project path")
    parser.add_argument('--kmeans_model', type=str, default="ar2", help="ar2 or bert or dpr")
    parser.add_argument('--scheduler', type=str, default="linear", help="linear or exp")
    parser.add_argument('--wandb_pro_name', type=str, default="GDR", help="name for wandb project")
    parser.add_argument('--is_train_encoder', type=int, default=1, help="whether train the encoder")
    parser.add_argument('--tau', type=float, default=0.05, help="contrastive learning temperature")
    parser.add_argument('--encoder_max_len', type=int, default=128, help="max doc len")
    parser.add_argument('--max_intraclass_num', type=int, default=10, help="max contrastive learning intra class num")
    parser.add_argument('--use_query_embed_encoder', type=int, default=1,
                        help="use T5 encoder output as query embedding")
    parser.add_argument('--use_query_embed_decoder_special', type=int, default=0,
                        help="use T5 decoder special token output as query embedding")
    parser.add_argument('--use_query_embed_decoder_avg', type=int, default=0,
                        help="use T5 decoder average output as query embedding")
    # parser.add_argument('--max_interclass_num', type=int, default=5, help="max contrastive learning inter class num")
    parser.add_argument('--intra_rate', type=float, default=1.0, help="contrastive learning intra ratio")
    parser.add_argument('--train_encoder_epoch', type=int, default=51)
    parser.add_argument('--stage2_train_batchsize', type=int, default=2)
    parser.add_argument('--stage2_eval_batchsize', type=int, default=2)
    parser.add_argument('--begin_val_epoch', type=int, default=0)
    parser.add_argument('--doc_encoder_learning_rate', type=float, default=2e-4)  ###### new add
    parser.add_argument('--score_rate', nargs = '+',type=float, default=[0,0.5,1,1.5,2,2.5,3], help="final score for all candidates, combind by: score_rate*nll_score+dualencoder_score")  ###### new add
    parser.add_argument('--train_num', type=int, default=256, help="training num, -1 represents all")
    parser.add_argument('--eval_num', type=int, default=-1, help="eval num, -1 represents all")
    parser.add_argument('--data_suffix', type=str, default="_30_2.5", help="suffix of datasets")
    parser.add_argument('--loss_func', type=str, default="tanh", help="sigmoid, tanh")
    parser.add_argument('--fusion_strategy', type=str, default="concate", help="average or concate")
    parser.add_argument('--neg_sample_strategy', type=str, default="random", help="cluster or bm25 or random")
    parser.add_argument('--expand', type=bool, default=True, help="true or false")

    parser_args = parser.parse_args()
    parser_args.dataset_name = "Self_NQ_{}_{}{}".format(parser_args.kmeans_model,
                                                        str(parser_args.docnum),
                                                        parser_args.data_suffix)
    # args post process
    parser_args.tokenizer_name_or_path += parser_args.model_info
    parser_args.model_name_or_path += parser_args.model_info
    k = parser_args.encoder_tokenizer_name_or_path.split('-')
    k[1] = parser_args.encoder_info
    parser_args.encoder_tokenizer_name_or_path = '-'.join(k)
    parser_args.encoder_name_or_path = '-'.join(k)

    parser_args.gradient_accumulation_steps = max(int(8 / parser_args.n_gpu), 1)

    if parser_args.mode == 'train' and 'doc' in parser_args.query_type:
        assert parser_args.contrastive_variant == ''
        parser_args.max_input_length = parser_args.doc_length
        print("change max input length to", parser_args.doc_length)

    # if parser_args.mode == 'train':
    #     # set to small val to prevent CUDA OOM
    #     # parser_args.num_return_sequences = 10
    #     parser_args.eval_batch_size = 1

    if parser_args.model_info == 'base':
        parser_args.num_layers = 12
        parser_args.num_decoder_layers = 6
        parser_args.d_ff = 3072
        parser_args.d_model = 768
        parser_args.num_heads = 12
        parser_args.d_kv = 64
    elif parser_args.model_info == 'large':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_ff = 4096
        parser_args.d_model = 1024
        parser_args.num_heads = 16
        parser_args.d_kv = 64
    elif parser_args.model_info == 'small':
        parser_args.num_layers = 6
        parser_args.num_decoder_layers = 3
        parser_args.d_ff = 2048
        parser_args.d_model = 512
        parser_args.num_heads = 8
        parser_args.d_kv = 64

    if parser_args.test1000:
        parser_args.n_val = 1000
        parser_args.n_train = 1000
        parser_args.n_test = 1000
    return parser_args


if __name__ == "__main__":
    args = parsers_parser()
    set_seed(args.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(dir_path)
    print(parent_path)
    args.logs_dir = dir_path + '/logs/'
    args.project_path = parent_path
    # this is model pkl save dir
    args.output_dir = dir_path + '/logs/'

    time_str = time.strftime("%Y%m%d-%H%M%S")
    # Note -- you can put important info into here, then it will appear to the name of saved ckpt
    important_info_list = ['nq:', str(args.nq), "kary:", str(args.kary), args.query_type,
                           args.model_info, args.id_class,
                           args.test_set, args.ckpt_monitor, 'dem:',
                           str(args.decode_embedding), 'ada:', str(args.adaptor_decode), 'adaeff:',
                           str(args.adaptor_efficient), 'adanum:', str(args.adaptor_layer_num)]


    dataset_name = 'nq'
    if args.use_query_embed_encoder:
        embedding_type = "en"
    elif args.use_query_embed_decoder_special:
        embedding_type = "desp"
    else:
        embedding_type = "deav"
    args.query_info = dataset_name+"_docnum"+str(args.docnum)+ "_kmodel-" + str(args.kmeans_model) + \
                      "_epoch-" + str(args.num_train_epochs) + "_bs-" + str(args.train_batch_size) + \
                      "_seed-" + str(args.seed) + embedding_type + "_intra_rate-" + str(args.intra_rate) + "_info-" + args.ckpt_info
    if YOUR_API_KEY != '':
        os.environ["WANDB_API_KEY"] = YOUR_API_KEY
        logger = WandbLogger(name='{}-{}'.format(time_str, args.query_info), project=args.wandb_pro_name)
    else:
        logger = TensorBoardLogger("logs/")
    ###########################

    args.tag_info = '{}_lre{}d{}'.format(args.query_info, str(float(args.learning_rate * 1e4)),
                                         str(float(args.decoder_learning_rate * 1e4)))
    args.res1_save_path = args.logs_dir + '{}_res1_recall{}_{}_{}.tsv'.format(
        args.tag_info, args.num_return_sequences, time_str, args.ckpt_info)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        args.recall_num = [1, 5, 10, 20, 50, 100]
        inference(args)
    elif args.mode == 'calculate':
        args.res1_save_path = ''  # your result path
        # args.recall_num = [1,5,10,20,50,100]
        calculate(args)
