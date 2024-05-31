#!/bin/bash

### Training scripts for Simple Agreement Task, using different objectives

echo "Training with LM Objective on High Diversity Dataset (120 Types)"
python train_transformers.py --dataset simple_agreement --encoder_n_layers 6 --callback --eval_every 1000 --grammar "agreement_hr_v4_agreement_linear_v4" --seed 42 --max_train_steps 200000

echo "Training with LM Objective on Low Diversity Dataset (12 Types)"
python train_transformers.py --dataset simple_agreement --encoder_n_layers 6 --callback --eval_every 1000 --grammar "agreement_hr_agreement_linear" --seed 42 --max_train_steps 200000

echo "Training with Classification Objective on High Diversity Dataset (120 Types)"
python train_transformers.py --dataset simple_agreement --encoder_n_layers 6 --mode enc --callback --eval_every 1000 --save_every 1000 --seed 42 --max_train_steps 200000 --causal_encoder --grammar "agreement_hr_v4_agreement_linear_v4"

echo "Training with Cloze Completion Objective on High Diversity Dataset (120 Types)"
python train_mlm.py --dataset simple_agreement --encoder_n_layers 6 --mask_strategy "all-verbs" --callback --grammar "agreement_hr_v4_agreement_linear_v4" --seed 42 --causal_encoder