#!/bin/bash

#SBATCH --job-name=MBart-large-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=10:00:00 
#SBATCH --gres=gpu:v100:1


source $HOME/miniconda3/bin/activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'facebook/mbart-large-cc25' \
    --do_train \
    --do_eval \
    --task translation_src_to_tgt \
	--source_lang en_XX \
	--target_lang en_XX \
	--use_fast_tokenizer False \
    --train_file ../data/mccoy2020/question_main.train.json \
    --validation_file ../data/mccoy2020/question_main.dev.json \
    --output_dir $SCRATCH/mbart-mccoy-finetuning	 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
