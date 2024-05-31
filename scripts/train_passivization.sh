#!/bin/bash

### Training scripts for Passivization Task, using different objectives

echo "Training with LM Objective"
python train_transformers.py --dataset passiv --callback --max_train_steps 300000 --seed 42 --encoder_n_layers 6

echo "Training with Seq2Seq Objective"
python train_transformers.py --dataset passiv --callback --not_lm --encoder_n_layers 6 --decoder_n_layers 6 --max_train_steps 300000 --seed 42

echo "Training with PrefixLM Objective"
python train_transformers.py --dataset passiv --callback --is_prefix_lm --max_train_steps 300000 --seed 42 --encoder_n_layers 6

echo "Training with Classification Objective"
python train_transformers.py --dataset passiv --callback --is_prefix_lm --max_train_steps 300000 --seed 42 --mode enc --causal_encoder --encoder_n_layers 6