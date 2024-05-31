#!/bin/bash

### Training scripts for Question Formation Task, using different objectives

echo "Training with LM Objective"
python train_transformers.py --encoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

echo "Training with Seq2Seq Objective"
python train_transformers.py --not_lm --mode enc_dec --encoder_n_layers 6 --decoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

echo "Training with PrefixLM Objective"
python train_transformers.py --is_prefix_lm --encoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

echo "Training with Classification Objective"
python train_transformers.py --mode enc --dataset lm  --encoder_n_layers 6 --callback --eval_every 1000 --callback --causal_encoder --seed 42 --max_train_steps 300000

echo "Training with Cloze Completion Objective"
python train_mlm.py --dataset lm --encoder_n_layers 6 --eval_every 1000 --callback --mask_strategy aux --causal_encoder --seed 42