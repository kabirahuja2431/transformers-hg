#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=72:00:00 
#SBATCH --gres=gpu:v100:1

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'google/mt5-base' \
    --do_train \
    --do_eval \
	--random_weights \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_de_nps/passiv_de_nps.train.json \
    --validation_file ../data/passiv_de_nps/passiv_de_nps.dev.json \
    --output_dir $SCRATCH/mt5-random-mccoy-finetuning-passiv-de-nps-long/	 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
	--learning_rate 5e-2 \
	--save_steps 200000 \
    --num_train_epochs 50
"
