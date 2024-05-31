import argparse
import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import copy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import torch
import wandb

# from transformers import GatedGPT2Model
# from models.hf_transformers import HFTransformerModel
from data_utils.lm_dataset_helpers import read_lm_data
from data_utils import (
    build_datasets_lm,
    build_datasets_tense_inflection,
)
from train_transformers import WANDB_ENTITY_NAME
from train_hf_transformers import eval_callback_lm
import collate

from prune_heads_v2 import prune_model_heads

matplotlib.rcParams["figure.dpi"] = 300


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def analyse_step(model_path, step, full_seq_for_pruning=False, **kwargs):
    full_model_path = f"{model_path}/checkpoint_{step}.pickle"
    head_mask, pruning_results = prune_model_heads(
        full_model_path,
        n_embd=kwargs.get("n_embd", 512),
        n_layer=kwargs.get("n_layer", 6),
        n_head=kwargs.get("n_head", 8),
        split_for_pruning="train",
        find_overfitted_heads=False,
        full_seq_for_pruning=full_seq_for_pruning,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dropout=kwargs.get("dropout", 0.1),
        tied_embedding=kwargs.get("tied_embedding", False),
        l0_penalty=kwargs.get("l0_penalty", 0.015),
        pruning_steps=kwargs.get("pruning_steps", 1000),
        pruning_lr=kwargs.get("pruning_lr", 0.1),
    )

    head_mask_test_prune, pruning_results_test_prune = prune_model_heads(
        full_model_path,
        n_embd=kwargs.get("n_embd", 512),
        n_layer=kwargs.get("n_layer", 6),
        n_head=kwargs.get("n_head", 8),
        split_for_pruning="test",
        find_overfitted_heads=False,
        full_seq_for_pruning=full_seq_for_pruning,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dropout=kwargs.get("dropout", 0.1),
        tied_embedding=kwargs.get("tied_embedding", False),
        l0_penalty=kwargs.get("l0_penalty", 0.015),
        pruning_steps=kwargs.get("pruning_steps", 1000),
        pruning_lr=kwargs.get("pruning_lr", 0.1),
        skip_bp=True,
    )

    head_mask_spurious_prune, pruning_results_spurious_prune = prune_model_heads(
        full_model_path,
        n_embd=kwargs.get("n_embd", 512),
        n_layer=kwargs.get("n_layer", 6),
        n_head=kwargs.get("n_head", 8),
        split_for_pruning="train",
        find_overfitted_heads=True,
        full_seq_for_pruning=full_seq_for_pruning,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dropout=kwargs.get("dropout", 0.1),
        tied_embedding=kwargs.get("tied_embedding", False),
        l0_penalty=kwargs.get("l0_penalty", 0.015),
        pruning_steps=kwargs.get("pruning_steps", 1000),
        pruning_lr=kwargs.get("pruning_lr", 0.1),
        skip_bp=True,
    )

    return {
        "step": step,
        "metrics": {
            "val": {
                "aux": {
                    "og": pruning_results["before_pruning"]["val_aux"],
                    "train_prune": pruning_results["after_pruning"]["val_aux"],
                    "test_prune": pruning_results_test_prune["after_pruning"][
                        "val_aux"
                    ],
                    "spurious_prune": pruning_results_spurious_prune["after_pruning"][
                        "val_aux"
                    ],
                },
                # "sent_score": {
                #     "og": pruning_results["before_pruning"]["val_sent_prob"],
                #     "train_prune": pruning_results["after_pruning"]["val_sent_prob"],
                #     "test_prune": pruning_results_test_prune["after_pruning"][
                #         "val_sent_prob"
                #     ],
                #     "spurious_prune": pruning_results_spurious_prune["after_pruning"][
                #         "val_sent_prob"
                #     ],
                # },
            },
            "test": {
                "aux": {
                    "og": pruning_results["before_pruning"]["test_aux"],
                    "train_prune": pruning_results["after_pruning"]["test_aux"],
                    "test_prune": pruning_results_test_prune["after_pruning"][
                        "test_aux"
                    ],
                    "spurious_prune": pruning_results_spurious_prune["after_pruning"][
                        "test_aux"
                    ],
                },
                # "sent_score": {
                #     "og": pruning_results["before_pruning"]["test_sent_prob"],
                #     "train_prune": pruning_results["after_pruning"]["test_sent_prob"],
                #     "test_prune": pruning_results_test_prune["after_pruning"][
                #         "test_sent_prob"
                #     ],
                #     "spurious_prune": pruning_results_spurious_prune["after_pruning"][
                #         "test_sent_prob"
                #     ],
                # },
            },
            "train": {
                "loss": {
                    "train_prune": pruning_results["after_pruning"]["avg_ce_loss"],
                    "test_prune": pruning_results_test_prune["after_pruning"][
                        "avg_ce_loss"
                    ],
                    "spurious_prune": pruning_results_spurious_prune["after_pruning"][
                        "avg_ce_loss"
                    ],
                }
            },
        },
        "sparsity": {
            "train": head_mask.sum().item() / head_mask.numel(),
            "test": head_mask_test_prune.sum().item() / head_mask_test_prune.numel(),
            "spurious": head_mask_spurious_prune.sum().item()
            / head_mask_spurious_prune.numel(),
        },
        "masks": {
            "train": head_mask.squeeze().cpu().numpy().tolist(),
            "test": head_mask_test_prune.squeeze().cpu().numpy().tolist(),
            "spurious": head_mask_spurious_prune.squeeze().cpu().numpy().tolist(),
        },
    }


def main(args):
    wandb.init(
        project="structural-pruning-dynamics",
        entity=WANDB_ENTITY_NAME,
        config=vars(args),
    )
    result_logs = []
    for step in range(0, args.last_ckpt, args.incr):
        results = analyse_step(
            args.model_path,
            step,
            full_seq_for_pruning=args.full_seq_for_pruning,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            l0_penalty=args.l0_penalty,
            pruning_steps=args.pruning_steps,
            tied_embedding=args.tied_embedding,
        )
        wandb.log(results["metrics"], step=step)
        wandb.log(results["sparsity"], step=step)
        result_logs.append(results)

        with open(
            f"{args.model_path}/td_results_full_seq_prune{args.full_seq_for_pruning}.json",
            "w",
        ) as f:
            json.dump(result_logs, f, indent=4)

        # wandb.log({"masks": wandb.Histogram(results["masks"])}, step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoints",
    )
    parser.add_argument(
        "--last_ckpt", type=int, help="Last checkpoint to analyse", default=50000
    )
    parser.add_argument(
        "--incr",
        type=int,
        default=1000,
        help="Increment between checkpoints",
    )
    parser.add_argument(
        "--full_seq_for_pruning",
        action="store_true",
        help="Whether to use full sequence for pruning",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=512,
        help="Model embedding dimension",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Model number of layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Model dropout",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=8,
        help="Model number of heads",
    )
    parser.add_argument(
        "--l0_penalty",
        type=float,
        default=0.015,
        help="L0 penalty",
    )
    parser.add_argument(
        "--pruning_steps",
        type=int,
        default=1000,
        help="Number of pruning steps",
    )
    parser.add_argument("--pruning_lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tied-embedding", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
