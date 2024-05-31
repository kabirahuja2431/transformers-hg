#!/bin/bash

### Training scripts for German Question Formation Task, using different objectives

echo "Training with LM Objective"
python train_transformers.py --encoder_n_layers 6 --callback --dataset question_de --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42

echo "Training with Seq2Seq Objective"
python train_transformers.py --not_lm --mode enc_dec --encoder_n_layers 6 --decoder_n_layers 6 --callback --dataset question_de --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42

echo "Training with PrefixLM Objective"
python train_transformers.py --is_prefix_lm --encoder_n_layers 6 --callback --dataset question_de --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42

echo "Training with Classification Objective"
python train_transformers.py --mode enc --dataset question_de  --encoder_n_layers 6 --callback --eval_every 1000 --callback --causal_encoder --seed 42 --max_train_steps 300000

echo "Training with Cloze Completion Objective"
python train_mlm.py --dataset question_de --encoder_n_layers 6 --eval_every 1000 --callback --mask_strategy aux --causal_encoder --seed 42