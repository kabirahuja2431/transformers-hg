#!/bin/bash

#SBATCH --job-name=mBART-large-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:1



singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'facebook/mbart-large-cc25' \
    --do_train \
    --task translation_src_to_tgt \
	--source_lang en_XX \
	--target_lang en_XX \
	--use_fast_tokenizer False \
    --train_file ../data/mccoy2020/question_have.en-de.train.json \
    --validation_file ../data/mccoy2020/question_have.de.dev.json \
    --output_dir /scratch/am12057/mbart-mccoy-finetuning-question-have-en-de/  \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 1.0
"
