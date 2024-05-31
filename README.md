<h2 align="center">
  Transformers Hierarchical Generalization
</h2>

Official code repository for the paper [**Learning Syntax Without Planting Trees: Understanding When and Why Transformers Generalize Hierarchically**](https://arxiv.org/abs/2404.16367)

### Dependencies
- Compatible with Python 3.11 (might work for 3.9+ but not tested)
- Pytorch 2.1.0
- The necessary packages can be install through requirements.txt

### Setup
We recommend using a conda environment or virtual environment

```bash
conda env create -n hiergenv python=3.11
conda activate hiergenv
conda install pip
```
#### Installing Pytorch
```bash
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Installing Huggingface Transformers
We make some changes to the transformers library to support pruning (adapted from [DSP](https://github.com/rycolab/differentiable-subset-pruning)). To install,

```bash
cd transformers/
pip install -e .
```

#### Installing remaining packages
```bash
pip install -r requirements.txt
```

#### Set your wandb ID
Open [`train_transformers.py`](train_transformers.py) and in Line 65, replace `WANDB_ENTITY_NAME = "<Insert-Your-Entity-Name>"` with your wandb-id.

### Running experiments from the paper

#### How the Training Objective Influences Hierarchical Generalization 
Training models on the question formation dataset using different training objectives

```bash
#"Training with LM Objective"
python train_transformers.py --encoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

#"Training with Seq2Seq Objective"
python train_transformers.py --not_lm --mode enc_dec --encoder_n_layers 6 --decoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

#"Training with PrefixLM Objective"
python train_transformers.py --is_prefix_lm --encoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding

#"Training with Classification Objective"
python train_transformers.py --mode enc --dataset lm  --encoder_n_layers 6 --callback --eval_every 1000 --callback --causal_encoder --seed 42 --max_train_steps 300000

#"Training with Cloze Completion Objective"
python train_mlm.py --dataset lm --encoder_n_layers 6 --eval_every 1000 --callback --mask_strategy aux --causal_encoder --seed 42
```

Please check [`scripts/`](scripts/) for examples for other datasets.

#### Discovering Subnetworks with Different Generalization Behaviors

First make sure you saved model checkpoints while training the LM model

```bash
python train_transformers.py --encoder_n_layers 6 --callback --dataset lm --max_train_steps 300000 --max_grad_norm 1 --eval_every 1000 --seed 42 --tied-embedding --save_every 1000 --save_dir "<PATH TO SAVE CHECKPOINTS>"
```

To prune a specific model checkpoint
```bash
# For Train-Prune
python prune_heads_v2.py --model_path <SAVE_DIR>/checkpoint_<CHECKPOINT>.pkl --dataset qf --n_layer 6 --tied-embedding --split_for_pruning "train" --pruning_steps 10000 --pruning_lr 0.05 --l0_penalty 0.015

# For Gen-Prune
python prune_heads_v2.py --model_path <SAVE_DIR>/checkpoint_<CHECKPOINT>.pkl --dataset qf --n_layer 6 --tied-embedding --split_for_pruning "test" --pruning_steps 10000 --pruning_lr 0.05 --l0_penalty 0.015

# For Train\Gen Prune
python prune_heads_v2.py --model_path <SAVE_DIR>/checkpoint_<CHECKPOINT>.pkl --dataset qf --n_layer 6 --tied-embedding --split_for_pruning "train" --find_overfitted_heads --pruning_steps 10000 --pruning_lr 0.05 --l0_penalty 0.015
```

To analyse the full training dynamics i.e. run the three pruning methods for all the checkpoints

```bash
python analyse_training_dynamics.py --n_layer 6 --model_path  <SAVE_DIR> --last_ckpt 300000 --incr 1000 --tied-embeddings  --pruning_steps 10000 --pruning_lr 0.05 --l0_penalty 0.015 --seed 42
```

#### Computing Posteriors of different grammars (Section 5.3)
Note here G1 refers to the CFG and G2 is regular grammar
```bash
# Comparing small grammars (12 sentence types)
python pcfg.py --g1_name "agreement_hr" --g2_name "agreement_linear" --save_dir <SAVE_DIR>

# Comparing large grammars (120 sentence types)
python pcfg.py --g1_name "agreement_hr_v4" --g2_name "agreement_linear_v4" --save_dir <SAVE_DIR>
```

To compare the posteriors after applying Bayesian Model Merging to minimize the grammars, supply the `--minimize` argument, for e.g.

```bash
python pcfg.py --g1_name "agreement_hr_v4" --g2_name "agreement_linear_v4" --save_dir <SAVE_DIR> --minimize
```

For any clarification, comments, or suggestions feel free to contact me via email at [kahuja@cs.washington.edu](mailto:kahuja@cs.washington.edu).

### Citing this work
```bibtex
@article{ahuja2024learning,
title={Learning Syntax Without Planting Trees: Understanding When and Why Transformers Generalize Hierarchically}, 
author={Kabir Ahuja and Vidhisha Balachandran and Madhur Panwar and Tianxing He and Noah A. Smith and Navin Goyal and Yulia Tsvetkov},
year={2024},
eprint={2404.16367},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```

### Acknowledgements
The code is adapted from the [Structural Grokking](https://github.com/murtyshikhar/structural-grokking) by Shikhar Murty. The question formation and tense reinflection datasets are from [McCoy et al. 2020](https://github.com/tommccoy1/rnn-hierarchical-biases) and German question formation and passivization are from [Mueller et al. 2022](https://github.com/sebschu/multilingual-transformations/tree/main).