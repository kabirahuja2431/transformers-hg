#!/bin/bash

python plot_violin.py --gold_filename ../data/passiv_en_nps/passiv_en_nps.gen.json --checkpoint_dirs "$SCRATCH/t5-finetuning-passivization-en-nps/,$SCRATCH/mt5-finetuning-passivization-en-nps/,$SCRATCH/bart-finetuning-passivization-en/,$SCRATCH/mbart-cc-finetuning-passivization-en/" --metrics "exact_match,second_word" --out_dir violins --move_legend

python plot_violin.py --gold_filename ../data/passiv_de_nps/passiv_de_nps.gen.json --checkpoint_dirs "$SCRATCH/mt5-finetuning-passivization-de-nps/,$SCRATCH/mbart-cc-finetuning-passivization-de/" --metrics "exact_match,second_word" --out_dir violins --move_legend

python plot_violin.py --gold_filename ../data/question_have-havent_en/question_have.gen.json --checkpoint_dirs "$SCRATCH/t5-mccoy-finetuning-question-have/,$SCRATCH/mt5-mccoy-finetuning-question-have/,$SCRATCH/bart-mccoy-finetuning-question-have/,$SCRATCH/mbart-cc-mccoy-finetuning-question-have/" --metrics "exact_match,first_word" --out_dir violins

python plot_violin.py --gold_filename ../data/question_have-can_withquest_de/question_have_can.de.gen.json --checkpoint_dirs "$SCRATCH/mt5-mccoy-finetuning-question-have-can-withquest-de/,$SCRATCH/mbart-cc-mccoy-finetuning-question-have-can-withquest-de/" --metrics "exact_match,first_word" --out_dir violins
