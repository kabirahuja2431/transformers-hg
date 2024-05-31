#!/bin/bash

#SBATCH --job-name=MT-base-finetune-passiv-en
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:1

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'google/mt5-base' \
    --do_train \
    --random_weights \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_en/passivization.train.json \
    --validation_file ../data/passiv_en/passivization.dev.json \
    --output_dir /scratch/am12057/mt5-random-finetuning-passivization-en-bs128/  \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --learning_rate 5e-4 \
    --save_steps 20000 \
    --num_train_epochs 500.0
"
