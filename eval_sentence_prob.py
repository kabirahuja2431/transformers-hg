import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from training_utils import *
from data_utils import (
    build_datasets_lm,
    build_datasets_tense_inflection,
)
from data_utils.lm_dataset_helpers import read_lm_data
from transformer_helpers import create_lm, create_model_interface
from train_transformers import eval_lm_callback
import collate

N_EMBD = 512
N_LAYERS = 6
N_HEADS = 8
MODEL_PATH = "/gscratch/argon/kahuja/work/repos/structural-grokking/out/qf_finetuned/"

if __name__ == "__main__":
    datasets, in_vocab, _ = build_datasets_lm()
    model = create_lm(
        len(in_vocab),
        vec_dim=N_EMBD,
        n_heads=N_HEADS,
        encoder_n_layers=N_LAYERS,
        mode="enc_dec",
    )
    model_paths = glob(os.path.join(MODEL_PATH, "*.pickle"))
