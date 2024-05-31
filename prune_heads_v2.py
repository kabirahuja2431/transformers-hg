import argparse
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import wandb
from layers.transformer.gated_transformer import GatedTransformer
from layers.transformer.transformer import Transformer
from models.transformer_lm import TransformerLM
from interfaces.transformer.lm_interface import TransformerLMInterface
from interfaces import TransformerPrefixLMInterface
from data_utils.lm_dataset_helpers import read_lm_data
from data_utils import (
    build_datasets_lm,
    build_datasets_tense_inflection,
)
from data_utils.simple_agreement_helpers import (
    build_datasets_simple_agreement,
    eval_callback_simple_agreement,
    eval_all_agreements,
)

# from train_hf_transformers import eval_callback_lm, eval_lm_sent_prob_callback
from data_utils.lm_dataset_helpers import (
    eval_lm_callback,
    eval_cls_callback,
    eval_lm_sent_prob_callback,
)
from data_utils.tense_inflection_helpers import eval_callback_tense_inflection
from train_transformers import WANDB_ENTITY_NAME
import collate


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class PruningArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    l0_penalty: float = field(
        default=0.5, metadata={"help": "coefficient for l0 regularization."}
    )
    num_of_heads: Optional[int] = field(
        default=None, metadata={"help": "number of heads to be kept."}
    )
    joint_pruning: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, train gate variables and other parameters in the original model together."
        },
    )
    pruning_lr: Optional[float] = field(
        default=0.1, metadata={"help": "learning rate for gate variables."}
    )
    pruning_steps: Optional[float] = field(
        default=5000, metadata={"help": "number of prunning steps"}
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_dataset(dataset, in_vocab):
    def process_example(example):
        inp = example["in"]
        qidx = inp.index(in_vocab["quest"])
        example["in"] = inp[: qidx + 2]
        example["in_len"] = len(example["in"])
        return example

    dataset = dataset.filter(lambda x: in_vocab["quest"] in x["in"])

    dataset = dataset.map(process_example)

    return dataset


def convert_gate_to_mask(gates, num_of_heads=None):
    if num_of_heads is not None:
        head_mask = torch.zeros_like(gates)
        current_heads_to_keep = gates.view(-1).sort(descending=True)[1]
        current_heads_to_keep = current_heads_to_keep[:num_of_heads]
        head_mask = head_mask.view(-1)
        head_mask[current_heads_to_keep] = 1.0
        head_mask = head_mask.view_as(gates)
    else:
        head_mask = (gates > 0.5).float()
        head_mask = head_mask.view(head_mask.shape[0], 1, -1, 1, 1)
    return head_mask


def get_auxiliary_loss(logits, batch, device):
    # Select the last token of the sequence as target using batch["in_len"]
    target = batch["in"].transpose(0, 1)
    target = torch.tensor(
        [
            target[i, batch["in_len"][i].long().item() - 1]
            for i in range(target.shape[0])
        ]
    ).to(device)
    pred = torch.vstack(
        [
            logits[i, batch["in_len"][i].long().item() - 1]
            for i in range(logits.shape[0])
        ]
    ).to(device)
    loss = torch.nn.functional.cross_entropy(pred, target)
    return loss


# def get_tense_loss(logits, batch, device):
#     prefix_len = batch["prefix_len"]

#     #Select


def get_verb_loss(logits, batch, device):
    # Select the ids for verbs in each input using batch["all_verb_ids"]

    target = []
    pred = []
    # all_verb_ids = batch["all_verb_ids"].transpose(0, 1)
    all_verb_ids = batch["main_verb_ids"].transpose(0, 1)
    input_ids = batch["in"].transpose(0, 1)
    for bid in range(len(input_ids)):
        for vid in all_verb_ids[bid]:
            if vid != 0:
                target.append(input_ids[bid, vid])
                pred.append(logits[bid, vid])
    target = torch.tensor(target).to(device)
    pred = torch.vstack(pred).to(device)
    loss = torch.nn.functional.cross_entropy(pred, target)
    return loss


def get_specialised_loss(outputs, targets, device, dataset):
    if dataset == "qf":
        loss = get_auxiliary_loss(outputs, targets, device)
    elif dataset == "simple_agreement":
        loss = get_verb_loss(outputs, targets, device)
    else:
        raise NotImplementedError(f"Dataset: {dataset} not implemented!")

    return loss


def train(
    gated_model,
    optimizer,
    train_dataset,
    negative_dataset,
    full_seq_for_pruning=False,
    max_steps=10000,
    train_batch_size=8,
    device="cpu",
    wandb_logger=None,
    dataset="qf",
):
    train_data_collator = collate.VarLengthCollate(None)
    global_step = 0
    gated_model.model.zero_grad()
    avg_ce_loss = 0
    num_steps = 0
    while True:
        try:
            if global_step > max_steps:
                break
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=train_batch_size,
                collate_fn=train_data_collator,
            )
            if negative_dataset is not None:
                negative_dataloader = torch.utils.data.DataLoader(
                    negative_dataset,
                    sampler=torch.utils.data.RandomSampler(negative_dataset),
                    batch_size=train_batch_size,
                    collate_fn=train_data_collator,
                )

            if negative_dataset is None:
                epoch_iterator = tqdm(train_dataloader, desc="It")
            else:
                epoch_iterator = tqdm(
                    zip(train_dataloader, negative_dataloader), desc="It"
                )
            for step, batch in enumerate(epoch_iterator):
                if global_step > max_steps:
                    break
                global_step += 1
                gated_model.model.train()

                if negative_dataset is None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = gated_model(batch)
                    logits = outputs.outputs.transpose(0, 1).contiguous()

                    if full_seq_for_pruning:
                        loss = outputs.loss
                    else:
                        # # Select the last token of the sequence as target using batch["in_len"]
                        # target = batch["in"].transpose(0, 1)
                        # target = torch.tensor(
                        #     [
                        #         target[i, batch["in_len"][i].long().item() - 1]
                        #         for i in range(target.shape[0])
                        #     ]
                        # ).to(device)
                        # pred = torch.vstack(
                        #     [
                        #         logits[i, batch["in_len"][i].long().item() - 1]
                        #         for i in range(logits.shape[0])
                        #     ]
                        # ).to(device)
                        loss = get_specialised_loss(logits, batch, device, dataset)
                    ce_loss = loss.item()
                    loss += outputs.reg
                else:
                    batch_pos, batch_neg = batch
                    batch_pos = {k: v.to(device) for k, v in batch_pos.items()}
                    batch_neg = {k: v.to(device) for k, v in batch_neg.items()}
                    outputs_pos = gated_model(batch_pos)
                    outputs_neg = gated_model(batch_neg)
                    logits_pos = outputs_pos.outputs.transpose(0, 1).contiguous()
                    logits_neg = outputs_neg.outputs.transpose(0, 1).contiguous()
                    if full_seq_for_pruning:
                        loss_pos = outputs_pos.loss
                        loss_neg = outputs_neg.loss
                    else:
                        loss_pos = get_specialised_loss(
                            logits_pos, batch_pos, device, dataset
                        )
                        loss_neg = get_specialised_loss(
                            logits_neg, batch_neg, device, dataset
                        )
                        ce_loss = loss_pos.item()

                        # # Select the last token of the sequence as target using batch["in_len"]
                        # target_pos = batch_pos["in"].transpose(0, 1)
                        # target_pos = torch.tensor(
                        #     [
                        #         target_pos[i, batch_pos["in_len"][i].long().item() - 1]
                        #         for i in range(target_pos.shape[0])
                        #     ]
                        # ).to(device)
                        # target_neg = batch_neg["in"].transpose(0, 1)
                        # target_neg = torch.tensor(
                        #     [
                        #         target_neg[i, batch_neg["in_len"][i].long().item() - 1]
                        #         for i in range(target_neg.shape[0])
                        #     ]
                        # ).to(device)
                        # pred_pos = torch.vstack(
                        #     [
                        #         logits_pos[i, batch_pos["in_len"][i].long().item() - 1]
                        #         for i in range(logits_pos.shape[0])
                        #     ]
                        # ).to(device)
                        # pred_neg = torch.vstack(
                        #     [
                        #         logits_neg[i, batch_neg["in_len"][i].long().item() - 1]
                        #         for i in range(logits_neg.shape[0])
                        #     ]
                        # ).to(device)
                        # loss_pos = torch.nn.functional.cross_entropy(
                        #     pred_pos, target_pos
                        # )
                        # loss_neg = torch.nn.functional.cross_entropy(
                        #     pred_neg, target_neg
                        # )
                    loss = loss_pos - loss_neg + outputs_pos.reg + outputs_neg.reg
                    ce_loss = loss_pos.item()

                loss.backward()
                optimizer.step()
                gated_model.model.zero_grad()

                gates = gated_model.model.trafo.get_gate_values()
                head_mask = convert_gate_to_mask(torch.vstack(gates))
                num_active_heads = head_mask.sum()
                sparsity = 1 - num_active_heads / head_mask.numel()

                epoch_iterator.set_postfix(
                    {
                        "total_loss": loss.item(),
                        "ce_loss": ce_loss,
                        "num_active_heads": num_active_heads.item(),
                        "sparsity": sparsity.item(),
                    }
                )

                if wandb_logger is not None:
                    wandb.log(
                        {
                            "total_loss": loss.item(),
                            "ce_loss": ce_loss,
                            "num_active_heads": num_active_heads.item(),
                            "sparsity": sparsity.item(),
                        },
                        step=global_step,
                    )

                avg_ce_loss += ce_loss
                num_steps += 1
        except KeyboardInterrupt:
            break
    print("Avg CE loss: ", avg_ce_loss / num_steps)
    return avg_ce_loss / num_steps


def prune_model_heads(
    model_path,
    n_embd,
    n_layer,
    n_head,
    split_for_pruning="train",
    find_overfitted_heads=False,
    full_seq_for_pruning=False,
    device="cpu",
    wandb_logger=None,
    skip_bp=False,
    dataset="qf",
    grammar="agreement_hr_v4_agreement_linear_v4",
    dropout=0.1,
    tied_embedding=False,
    is_prefix_lm=False,
    **pruning_kwargs,
):
    if dataset == "qf":
        datasets, in_vocab, _ = build_datasets_lm()
    elif dataset == "simple_agreement":
        datasets, in_vocab, in_sentences = build_datasets_simple_agreement(
            grammar=grammar,
        )
    elif dataset == "tense":
        if find_overfitted_heads:
            datasets, in_vocab, _ = build_datasets_tense_inflection(
                data_dir="tense_inflection_order_data"
            )
        else:
            datasets, in_vocab, _ = build_datasets_tense_inflection()
    else:
        raise NotImplementedError(f"Dataset: {dataset} not implemented!")
    if full_seq_for_pruning:
        processed_datasets = datasets
    else:
        if dataset == "qf":
            processed_datasets = {
                split: process_dataset(dataset, in_vocab)
                for split, dataset in datasets.items()
            }
        else:
            processed_datasets = datasets
    # Select only a subset of test data for prunning
    # Select random 100 examples
    if dataset == "simple_agreement":
        selected_ids = random.sample(
            list(range(len(processed_datasets["g1_test"]))), k=100
        )
        processed_datasets["g1_test"] = processed_datasets["g1_test"].select(
            selected_ids
        )
        processed_datasets["g2_test"] = processed_datasets["g2_test"].select(
            selected_ids
        )
    else:
        selected_ids = random.sample(
            list(range(len(processed_datasets["test"]))), k=100
        )
        processed_datasets["test"] = processed_datasets["test"].select(selected_ids)
    # breakpoint()
    model = TransformerLM(
        n_input_tokens=len(in_vocab),
        state_size=n_embd,
        nheads=n_head,
        num_encoder_layers=n_layer,
        pos_scale=1.0,
        transformer=GatedTransformer,
        dropout=dropout,
        tied_embedding=tied_embedding,
    )

    # model = HFTransformerModel(
    #     n_input_tokens=len(in_vocab),
    #     n_embd=n_embd,
    #     n_layer=n_layer,
    #     n_head=n_head,
    #     gpt2_model=GatedGPT2Model,
    # )
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu")),  # strict=False
    )

    model.float()
    model.to(device)
    if is_prefix_lm:
        model = TransformerLMInterface(model)
    else:
        model = TransformerPrefixLMInterface(model)

    if dataset == "qf":
        callback_fn = {
            "aux": lambda split, lm_interface: eval_lm_callback(
                lm_interface.model, in_vocab=in_vocab, split=split
            ),
            # "sent_prob": lambda split, lm_interface: eval_lm_sent_prob_callback(
            #     lm_interface, in_vocab=in_vocab, split=split
            # ),
        }
    elif dataset == "simple_agreement":
        callback_fn = {
            "main_verb_acc": lambda split, lm_interface: eval_callback_simple_agreement(
                lm_interface.model, in_vocab, split, grammar
            ),
            "all_verb_agreement_acc": lambda split, lm_interface: eval_all_agreements(
                lm_interface.model, in_vocab, split, grammar
            ),
        }
    elif dataset == "tense":
        callback_fn = {
            "aux": lambda split, lm_interface: eval_callback_tense_inflection(
                lm_interface.model, in_vocab, split
            )
        }
    else:
        raise NotImplementedError(f"Dataset: {dataset} not implemented!")

    eval_keys = ["val", "test"] if dataset != "simple_agreement" else ["val", "g1_test"]
    bp_metrics = {}
    if not skip_bp:
        model.model.eval()
        print("Before pruning:")
        for split in eval_keys:
            for metric, callback in callback_fn.items():
                score = callback(split, model)
                print(f"{split} {metric}: ", score)
                bp_metrics[f"{split}_{metric}"] = score
                if wandb_logger is not None:
                    wandb.log(
                        {f"{split}_{metric}": score},
                    )
        # bp_val_score = eval_lm_callback(model.model, in_vocab=in_vocab, split="val")
        # bp_test_score = eval_lm_callback(model.model, in_vocab=in_vocab, split="test")

        # bp_val_sent_score = eval_lm_sent_prob_callback(
        #     model, in_vocab=in_vocab, split="val"
        # )
        # bp_test_sent_score = eval_lm_sent_prob_callback(
        #     model, in_vocab=in_vocab, split="test"
        # )

        # print("Val score: ", bp_val_score)
        # print("Test score: ", bp_test_score)
        # print("Val sent score: ", bp_val_sent_score)
        # print("Test sent score: ", bp_test_sent_score)
        # if wandb_logger is not None:
        #     wandb.log(
        #         {"bp_val_score": bp_val_score, "bp_test_score": bp_test_score}, step=0
        #     )

    gated_model = copy.deepcopy(model)
    gated_model.model.trafo.apply_gates(pruning_kwargs.get("l0_penalty", 0.5))
    gated_model.model.to(device)
    gated_model.model.train()

    for n, p in gated_model.model.named_parameters():
        if not "log_a" in n:
            p.requires_grad = False
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in gated_model.model.named_parameters() if "log_a" in n
            ],
            "lr": pruning_kwargs.get("pruning_lr", 0.1),
        },
    ]
    print("Pruning heads!")
    if find_overfitted_heads and dataset != "tense" and split_for_pruning == "train":
        negative_dataset = (
            processed_datasets["test"]
            if dataset != "simple_agreement"
            else processed_datasets["g1_test"]
        )
    else:
        negative_dataset = None
    avg_ce_loss = train(
        gated_model=gated_model,
        optimizer=torch.optim.AdamW(optimizer_grouped_parameters),
        train_dataset=processed_datasets[split_for_pruning],
        negative_dataset=negative_dataset,
        full_seq_for_pruning=full_seq_for_pruning,
        train_batch_size=pruning_kwargs.get("train_batch_size", 8),
        device=device,
        max_steps=pruning_kwargs.get("pruning_steps", 100000),
        wandb_logger=wandb_logger,
        dataset=dataset,
    )

    # Mask pruned heads
    print("Masking pruned heads!")
    gates = gated_model.model.trafo.get_gate_values()
    gates = torch.vstack(gates)
    head_mask = convert_gate_to_mask(gates)
    gated_model.model.trafo.remove_gates()
    gated_model.model.trafo.apply_masks(head_mask)

    # Evaluate after pruning
    print("After pruning:")
    gated_model.model.eval()
    ap_metrics = {"avg_ce_loss": avg_ce_loss}
    for split in eval_keys:
        for metric, callback in callback_fn.items():
            score = callback(split, gated_model)
            print(f"{split} {metric}: ", score)
            ap_metrics[f"{split}_{metric}"] = score
            if wandb_logger is not None:
                wandb.log(
                    {f"ap_{split}_{metric}": score},
                )
    # val_score = eval_lm_callback(gated_model.model, in_vocab=in_vocab, split="val")
    # test_score = eval_lm_callback(gated_model.model, in_vocab=in_vocab, split="test")
    # val_sent_score = eval_lm_sent_prob_callback(
    #     gated_model, in_vocab=in_vocab, split="val"
    # )
    # test_sent_score = eval_lm_sent_prob_callback(
    #     gated_model, in_vocab=in_vocab, split="test"
    # )

    # print("Val score: ", val_score)
    # print("Test score: ", test_score)
    # print("Val sent score: ", val_sent_score)
    # print("Test sent score: ", test_sent_score)
    # print("Avg CE loss: ", avg_ce_loss)

    # if wandb_logger is not None:
    #     wandb.log(
    #         {
    #             "ap_val_score": val_score,
    #             "ap_test_score": test_score,
    #             "ap_val_sent_score": val_sent_score,
    #             "ap_test_sent_score": test_sent_score,
    #             "avg_ce_loss": avg_ce_loss,
    #         },
    #     )

    return head_mask, {
        "before_pruning": bp_metrics if not skip_bp else None,
        "after_pruning": ap_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--full_seq_for_pruning", action="store_true")
    parser.add_argument("--split_for_pruning", type=str, default="train")
    parser.add_argument("--find_overfitted_heads", action="store_true")
    parser.add_argument("--l0_penalty", type=float, default=0.015)
    parser.add_argument("--pruning_lr", type=float, default=0.05)
    parser.add_argument("--pruning_steps", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="qf")
    parser.add_argument("--tied-embedding", action="store_true")
    parser.add_argument(
        "--grammar", type=str, default="agreement_hr_v4_agreement_linear_v4"
    )
    parser.add_argument(
        "--skip_bp", action="store_true", help="Skip before pruning eval"
    )
    parser.add_argument("--is_prefix_lm", action="store_true")

    args = parser.parse_args()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if torch.backends.mps.is_available():
    #     device = "mps"
    wandb_logger = wandb.init(
        project="structural-pruning",
        entity=WANDB_ENTITY_NAME,
        config=vars(args),
    )
    args = AttrDict((wandb_logger.config))
    head_mask, scores = prune_model_heads(
        model_path=args.model_path,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        split_for_pruning=args.split_for_pruning,
        find_overfitted_heads=args.find_overfitted_heads,
        full_seq_for_pruning=args.full_seq_for_pruning,
        device=device,
        wandb_logger=wandb_logger,
        dataset=args.dataset,
        grammar=args.grammar,
        skip_bp=args.skip_bp,
        tied_embedding=args.tied_embedding,
        l0_penalty=args.l0_penalty,
        pruning_lr=args.pruning_lr,
        pruning_steps=args.pruning_steps,
        train_batch_size=args.train_batch_size,
        is_prefix_lm=args.is_prefix_lm,
    )

    #     split_for_pruning="train",
    # find_overfitted_heads=False,
    # full_seq_for_pruning=full_seq_for_pruning,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    # dropout=kwargs.get("dropout", 0.1),
    # l0_penalty=kwargs.get("l0_penalty", 0.015),
    # pruning_steps=kwargs.get("pruning_steps", 1000),
    # pruning_lr=kwargs.get("pruning_lr", 0.1),
    if args.save_dir == "":
        save_dir = os.path.dirname(args.model_path)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(
        save_dir,
        f"split_for_pruning_{args.split_for_pruning}_overfitted_heads{args.find_overfitted_heads}_full_seq_for_pruning{args.full_seq_for_pruning}",
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(head_mask, os.path.join(save_path, "head_mask.pt"))
