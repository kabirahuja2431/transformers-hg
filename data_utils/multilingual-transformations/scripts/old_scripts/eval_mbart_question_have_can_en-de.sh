#!/bin/bash

#SBATCH --job-name=MT-base-eval-de-canaux
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=28GB 
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:v100:1



singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'facebook/mbart-large-cc25' \
	--do_eval \
	--do_learning_curve \
    --task translation_src_to_tgt \
	--source_prefix de_DE \
	--target_prefix de_DE \
    --train_file ../data/question_have-can_de/question_have_can.en-de.train.lang_id.json \
    --validation_file ../data/question_have-can_de/question_have_can.de.test.lang_id.json \ # gen_rc_o
    --output_dir $SCRATCH/mbart-cc-mccoy-finetuning-question-have-en-de-zs/  \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
"
