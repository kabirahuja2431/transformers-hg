#!/bin/bash

#SBATCH --job-name=MT-base-eval-de-canaux
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:1



singularity exec --nv --overlay /scratch/ss9536/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'google/t5-efficient-base-$1' \
    --do_eval \
    --do_learning_curve \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_en_nps/passiv_en_nps.train.json \
    --validation_file ../data/passiv_en_nps/passiv_en_nps.$2.json \
    --output_dir $SCRATCH/t5-$1-finetuning-passivization-en-nps-bs128/  \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
"
