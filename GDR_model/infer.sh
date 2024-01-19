#!/usr/bin/env bash

# CKPT path
INFER_CKPT='ckpt file'
BEAM_SIZE=100

# Set args to be the same as training. For example, if you set --id_class bert_k30_c30_1 when training, set --id_class bert_k30_c30_1 in infer.sh
# Dataset: set (--nq 0 -- trivia 1 --id_class bert_k30_c30_4) or (--nq 1 -- trivia 0 --id_class bert_k30_c30_5) with our checkpoint.
# CUDA_VISIBLE_DEVICES=1 
python main.py --decode_embedding 2 --n_gpu 1 --mode eval --query_type gtq_doc_aug_qg --adaptor_layer_num 4 \
--infer_ckpt $INFER_CKPT --num_return_sequences $BEAM_SIZE --tree 1 \
--model_info base --train_batch_size 64 --eval_batch_size 1 --test1000 0 --dropout_rate 0.1 --Rdrop 0.1 \
--adaptor_decode 1 --adaptor_efficient 1 --aug_query 1 --aug_query_type corrupted_query --input_dropout 1 --id_class bert_k30_c30_1 \
--kary 30 --output_vocab_size 30 --doc_length 64 --denoising 0 --max_output_length 10 \
--trivia 0 --nq 1
