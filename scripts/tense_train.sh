#!/bin/bash

### Training scripts for Tense Reinflection Task, using different objectives

echo "Training with LM Objective"
python train_transformers.py --dataset tense --encoder_n_layers 4 --callback --eval_every 1000 --seed 42 --max_train_steps 300000 --tied-embedding

echo "Training with Seq2Seq Objective"
python train_transformers.py --dataset tense --not_lm --callback --mode enc_dec --encoder_n_layers 6 --decoder_n_layers 6 --max_train_steps 300000 --seed 42 --tied-embedding --eval_every 1000

echo "Training with PrefixLM Objective"
python train_transformers.py --dataset tense --is_prefix_lm --encoder_n_layers 4 --callback --eval_every 1000 --seed 42 --max_train_steps 300000 --tied-embedding

echo "Training with Classification Objective"
python train_transformers.py --dataset tense --encoder_n_layers 4 --lr 1e-4 --mode enc --callback --eval_every 1000 --save_every 1000 --seed 42 --max_train_steps 300000 --causal_encoder

echo "Training with Cloze Completion Objective"
python train_mlm.py --dataset tense --encoder_n_layers 4 --callback --eval_every 1000 --seed 42 --max_train_steps 300000 --mask_strategy "all-verbs" --causal_encoder