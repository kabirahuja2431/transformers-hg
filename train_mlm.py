import numpy as np
import random
from tqdm import tqdm
import os
import torch
import wandb
import argparse
from data_utils import build_datasets_mlm
from data_utils.simple_agreement_helpers import build_datasets_simple_agreement_mlm
from transformer_helpers import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_utils.lm_dataset_helpers import eval_mlm_callback
from data_utils.simple_agreement_helpers import eval_mlm_callback_main_verb
from data_utils.tense_inflection_helpers import (
    build_datasets_ti_mlm,
    eval_ti_mlm_callback,
)
from data_utils.qf_de_helpers import (
    build_datasets_quest_de_mlm,
    eval_qf_de_mlm_callback,
)
from training_utils import get_grad_norm, get_opt, get_scheduler, plotting_util
import collate


### Change this for your own system as appropriate
def working_dir():
    # breakpoint()
    dir_name = os.getcwd()
    # USER = os.environ["USER"]
    # dir_name = f"/scr/biggest"

    # def helper(dir_name):
    #     if os.path.exists(dir_name):
    #         sub_dir = "{}/{}/compositionality".format(dir_name, USER)
    #         if not os.path.exists(sub_dir):
    #             os.makedirs(sub_dir)
    #         return sub_dir
    #     else:
    #         return ""

    # try:
    #     return helper(dir_name)
    # except:
    #     dir_name = f"/scr/smurty/biggest"
    #     return helper(dir_name)
    return dir_name


def get_base_transformer_mlm(args, in_vocab, model_name=None):
    model = create_mlm(
        len(in_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_n_layers,
        use_pos_embedding=not args.no_pos_enc,
        causal_encoder=args.causal_encoder,
    )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    interface = create_mlm_interface(model)
    return model, interface


def eval_loop(model, val_datasets, best_losses, device, collator, num_steps):
    def helper(validation):
        model.model.eval()
        loss_curr = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(validation):
                batch = {key: val.to(device) for key, val in batch.items()}
                loss_curr += model(batch, normalize=True).loss.item()
                total += 1
        return loss_curr / total

    eval_batch_size = 64
    plots = {}
    curr_losses = {}
    for key, val_dataset in val_datasets.items():
        validation = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=eval_batch_size,
            collate_fn=collator,
        )
        curr_losses[key] = helper(validation)
        plots["curr-{}-loss".format(key)] = curr_losses[key]
    best_losses = {key: min(curr_losses[key], best_losses[key]) for key in curr_losses}
    plots.update({f"best/{k}": v for k, v in best_losses.items()})
    plotting_util(plots, num_steps)
    return best_losses, curr_losses


def train_loop(
    args,
    model,
    train_dataset,
    val_datasets,
    device,
    save_dir,
    callback_fn=None,
    train_batch_size=8,
    eval_every=1000,
    max_steps=300000,
):
    num_steps = 0
    max_grad_norm = 1
    accum_steps = 1
    opt = get_opt(
        args.lr,
        model,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(opt, max_steps)

    train_data_collator = collate.VarLengthCollate(None)

    best_losses = {key: 100000.0 for key in val_datasets}

    while True:
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        total_train_sz = len(train_dataset)
        if num_steps > max_steps:
            break
        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            losses = []
            for curr_batch_dict in train_dataloader:
                if num_steps % eval_every == 0:
                    print("Evaluating at step {}".format(num_steps))
                    best_losses, curr_losses = eval_loop(
                        model,
                        val_datasets,
                        best_losses,
                        device,
                        train_data_collator,
                        num_steps,
                    )

                    if callback_fn is not None:
                        wandbdict = {
                            "iteration": num_steps,
                        }
                        for split in val_datasets:
                            score = callback_fn(split)
                            print("Score for {}: {}".format(split, score))
                            wandbdict["{}-acc".format(split)] = score
                        # val_score = callback_fn("val")

                        # test_score = callback_fn("test")
                        # print("Val score: {}".format(val_score))
                        # print("Test score: {}".format(test_score))
                        # wandbdict = {
                        #     "val_aux": val_score,
                        #     "test_aux": test_score,
                        #     "iteration": num_steps,
                        # }
                        wandb.log(wandbdict)

                    if len(save_dir) > 0:
                        torch.save(
                            model.model.state_dict(),
                            os.path.join(
                                save_dir, "checkpoint_{}.pickle".format(num_steps)
                            ),
                        )

                if type(model) != torch.nn.Module:
                    model.model.train()
                else:
                    model.train()
                curr_batch_dict = {
                    key: val.to(device) for key, val in curr_batch_dict.items()
                }
                loss_curr = model(curr_batch_dict).loss
                loss_curr.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                grad_norm = get_grad_norm(model.model)
                num_steps += 1
                progress_bar.update(curr_batch_dict["in"].shape[1])
                progress_bar.set_postfix(
                    {
                        "loss": loss_curr.item(),
                        "num_steps": num_steps,
                    }
                )
                wandb.log(
                    {
                        "loss": loss_curr.item(),
                        "grad_norm": grad_norm,
                        "iteration": num_steps,
                    }
                )
                opt.step()
                scheduler.step()
                model.model.zero_grad()
                if num_steps > max_steps:
                    break


def main_lm(args):
    out_vocab = None
    if args.dataset == "simple_agreement":
        dataset, in_vocab, _ = build_datasets_simple_agreement_mlm(
            grammar=args.grammar, mask_strategy=args.mask_strategy
        )
    elif args.dataset == "tense":
        dataset, in_vocab, in_sentences = build_datasets_ti_mlm(
            mask_strategy=args.mask_strategy
        )
    elif args.dataset == "question_de":
        dataset, in_vocab, in_sentences = build_datasets_quest_de_mlm(
            mask_strategy=args.mask_strategy
        )
    else:
        dataset, in_vocab, _ = build_datasets_mlm(mask_strategy=args.mask_strategy)
    model, interface = get_base_transformer_mlm(
        args, in_vocab, model_name=args.model_load_path
    )

    if args.callback:
        if args.dataset == "simple_agreement":
            callback_fn = lambda split: eval_mlm_callback_main_verb(
                mlm=model,
                eval_dataset=dataset[split],
                in_vocab=in_vocab,
                grammar=args.grammar,
                split=split,
            )
        elif args.dataset == "tense":
            callback_fn = lambda split: eval_ti_mlm_callback(
                mlm=model, eval_dataset=dataset[split], in_vocab=in_vocab, split=split
            )
        elif args.dataset == "question_de":
            callback_fn = lambda split: eval_qf_de_mlm_callback(
                mlm=model, eval_dataset=dataset[split], in_vocab=in_vocab, split=split
            )
        else:
            callback_fn = lambda split: eval_mlm_callback(
                model, dataset[split], in_vocab, split
            )
    else:
        callback_fn = None
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = "cpu"
    model.to(device)
    if len(args.save_dir) > 0:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.dataset == "simple_agreement":
        eval_keys = ["val", "g1_test", "g2_test"]
    elif args.dataset == "question_de":
        eval_keys = ["dev", "gen"]
    else:
        eval_keys = ["val", "test"]
    train_loop(
        args,
        interface,
        dataset["train"],
        {key: dataset[key] for key in eval_keys},
        device,
        args.save_dir,
        callback_fn=callback_fn,
        eval_every=args.eval_every,
        max_steps=args.max_train_steps,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lm",
        choices=["lm", "simple_agreement", "tense", "question_de"],
    )
    # only applicable for simple_agreement
    parser.add_argument(
        "--grammar",
        type=str,
        default="agreement_hr_agreement_linear",
        choices=[
            "agreement_hr_agreement_linear",
            "agreement_hr_v4_agreement_linear_v4",
        ],
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--encoder_n_layers", type=int, default=6)
    parser.add_argument("--mode", type=str, default="enc_dec")
    parser.add_argument("--decoder_n_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=300000)
    parser.add_argument(
        "--no-pos-enc",
        action="store_true",
        help="Whether to not use positional encoding",
    )
    parser.add_argument("--mask_strategy", type=str, default="aux")
    parser.add_argument("--callback", action="store_true")
    parser.add_argument(
        "--causal_encoder",
        action="store_true",
        help="Whether to use causal encoder",
    )
    parser.add_argument("--lm_mode", type=str, default="mlm")

    args = parser.parse_args()
    set_seed(args)

    wandb.init(
        project="structural-grokking", entity="kabirahuja2431", config=vars(args)
    )

    if args.save_dir != "":
        wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()

    main_lm(args)
