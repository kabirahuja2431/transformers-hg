#!/bin/bash

#SBATCH --job-name=MT-base-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=10:00:00 
#SBATCH --gres=gpu:v100:1

source /share/apps/anaconda3/2020.07/bin/activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path /scratch/ss9536/mt5-mccoy-finetuning-question-have/ \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/mccoy2020/question_have.train.json \
    --validation_file ../data/mccoy2020/question_have.test.json \
    --output_dir $SCRATCH/mt5-mccoy-finetuning-question-have/	 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate



python ../models/run_seq2seq.py \
    --model_name_or_path /scratch/ss9536/mt5-mccoy-finetuning-question-have/ \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/mccoy2020/question_have.train.json \
    --validation_file ../data/mccoy2020/question_have.gen.json \
    --output_dir $SCRATCH/mt5-mccoy-finetuning-question-have/  \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate

