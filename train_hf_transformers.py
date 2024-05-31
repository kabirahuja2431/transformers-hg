import os
import argparse
import random
import numpy as np
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from models.hf_transformers import HFTransformerModel
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from data_utils.lm_dataset_helpers import read_lm_data
from data_utils import (
    build_datasets_lm,
    build_datasets_tense_inflection,
)
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


def initialize_model(args, in_vocab, model_name=None):
    model = HFTransformerModel(
        n_input_tokens=len(in_vocab),
        n_embd=args.vec_dim,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        pos_scale=args.pos_scale,
    )

    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    return model


def evaluate(model, val_datasets, eval_batch_size, device, collate_fn):
    def helper(validation):
        model.eval()
        loss_curr = 0
        total = 1
        with torch.no_grad():
            for batch in tqdm(validation):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(src=batch["in"], src_len=batch["in_len"])
                loss_curr += outputs.loss.item()
                total += 1
        return loss_curr / total

    results = {}
    for split, val_dataset in val_datasets.items():
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=eval_batch_size,
            collate_fn=collate_fn,
        )
        results[f"{split}_loss"] = helper(val_dataloader)
    return results


def test_continuations(tokenizer, lm, prefixes, gpu_id):
    data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    batch_size = 32
    st = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    final_states = []
    while st < len(prefixes):
        en = min(len(prefixes), st + batch_size)
        cslice = prefixes[st:en]
        inputs, input_lens = tokenizer_helper(cslice)
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        with torch.no_grad():
            outputs = lm(inputs.transpose(0, 1), input_lens)
            final_states += [
                outputs.logits[idx][l - 1] for idx, l in enumerate(input_lens)
            ]
        st = en

    final_states = torch.stack(final_states, dim=0)
    return F.softmax(final_states, dim=1)


def eval_callback_lm(model, in_vocab, split):
    def tokenizer(s):
        return in_vocab(s)

    model.eval()
    sents, _ = read_lm_data([split])
    split_into_words = [sent.split(" ") for sent in sents if "quest" in sent]
    q_words = []
    prefixes = []
    for sent_words in split_into_words:
        idx = sent_words.index("quest")
        q_word = sent_words[idx + 1]
        q_words.append(q_word)
        prefixes.append(" ".join(sent_words[: idx + 1]))

    out = test_continuations(tokenizer, model, prefixes, 0)
    closing_words = ["doesn't", "does", "do", "don't"]
    closing_word_idxs = [in_vocab[w] for w in closing_words]
    out = out[:, closing_word_idxs]

    acc = [closing_words[i] == q_word for i, q_word in zip(out.argmax(dim=1), q_words)]
    agg_acc = sum(acc) / len(out)
    return agg_acc


def eval_lm_sent_prob_callback(model, in_vocab, split):
    data_collator = collate.VarLengthCollate(None)

    def get_neg_output(inp, pos_out):
        auxs = ["do", "don't", "does", "doesn't"]
        q_word = pos_out.split()[0]
        other_aux = [word for word in inp.split() if word in auxs and word != q_word][0]
        other_aux_id = [
            idx
            for idx, word in enumerate(inp.split())
            if word in auxs and word != q_word
        ][0]
        inp_words = inp.split()[:-2]
        neg_out_words = (
            [other_aux]
            + inp_words[:other_aux_id]
            + inp_words[other_aux_id + 1 :]
            + ["?"]
        )
        return " ".join(neg_out_words)

    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    def tokenizer(s):
        return in_vocab(s)

    sents, _ = read_lm_data([split])
    sents = [sent for sent in sents if "quest" in sent]

    inputs = []
    pos_outputs = []
    neg_outputs = []
    inp_pos_out_pairs = []
    inp_neg_out_pairs = []

    for sent in sents:
        sent_words = sent.split(" ")
        idx = sent_words.index("quest")
        inp = " ".join(sent_words[: idx + 1])
        pos_out = " ".join(sent_words[idx + 1 :])
        auxs = ["do", "don't", "does", "doesn't"]
        num_unq_auxs = len(set([word for word in sent_words if word in auxs]))
        if num_unq_auxs < 2:
            continue
        neg_out = get_neg_output(inp, pos_out)
        inputs.append(inp)
        pos_outputs.append(pos_out)
        neg_outputs.append(neg_out)
        inp_pos_out_pairs.append(inp + " " + pos_out)
        inp_neg_out_pairs.append(inp + " " + neg_out)

    num_corrects = 0
    correct_or_not = []
    pos_scores = []
    neg_scores = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Until I implement a more efficient way to do this
    inp_pos_out_pairs = inp_pos_out_pairs[: min(1000, len(inp_pos_out_pairs))]

    with tqdm(total=len(inp_pos_out_pairs)) as pbar:
        for i, (inp_pos, out_pos) in enumerate(
            zip(inp_pos_out_pairs, inp_neg_out_pairs)
        ):
            inp_pos_tokens, inp_pos_len = tokenizer_helper([inp_pos])
            inp_neg_tokens, inp_neg_len = tokenizer_helper([out_pos])
            inp_pos_tokens = inp_pos_tokens.to(device)
            inp_pos_len = inp_pos_len.to(device)
            inp_neg_tokens = inp_neg_tokens.to(device)
            inp_neg_len = inp_neg_len.to(device)

            pos_score = model(
                src=inp_pos_tokens.transpose(0, 1), src_len=inp_pos_len
            ).loss
            neg_score = model(
                src=inp_neg_tokens.transpose(0, 1), src_len=inp_neg_len
            ).loss

            is_correct = float(pos_score.item() < neg_score.item())
            num_corrects += is_correct
            correct_or_not.append(is_correct)
            pos_scores.append(pos_score.item())
            neg_scores.append(neg_score.item())
            pbar.update(1)
            pbar.set_description("Accuracy: {}".format(num_corrects / (i + 1)))

    acc = num_corrects / len(inp_pos_out_pairs)

    return acc


def train_loop(
    args,
    model,
    train_dataset,
    val_datasets,
    device,
    save_dir,
    in_vocab,
    callback_fn=None,
    train_batch_size=8,
    eval_batch_size=32,
    max_grad_norm=1,
    eval_every=1000,
    save_every=1000,
    max_steps=2000000,
    num_warmup_steps=10000,
):
    val_dataloaders = {}
    for split, val_dataset in val_datasets.items():
        val_sampler = SequentialSampler(val_dataset)
        val_dataloaders[split] = DataLoader(
            val_dataset, sampler=val_sampler, batch_size=eval_batch_size
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps
    )

    train_data_collator = collate.VarLengthCollate(None)

    global_step = 0
    model.zero_grad()

    while True:
        if global_step > max_steps:
            break
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if global_step % eval_every == 0:
                results = evaluate(
                    model,
                    val_datasets,
                    eval_batch_size,
                    device,
                    collate_fn=train_data_collator,
                )
                wandb.log(results)

                if callback_fn is not None:
                    val_score = callback_fn("val")
                    test_score = callback_fn("test")
                    print(val_score, test_score)
                    wandbdict = {
                        "iteration": global_step,
                        "val_aux": val_score,
                        "test_aux": test_score,
                    }
                    wandb.log(wandbdict)
                model.train()
            if len(save_dir) > 0 and global_step % save_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "checkpoint_{}.pickle".format(global_step)),
                )

            global_step += 1
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(src=batch["in"], src_len=batch["in_len"])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            epoch_iterator.set_postfix({"loss": loss.item(), "num_steps": global_step})
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            wandb.log(
                {
                    "loss": loss.item(),
                    "iteration": global_step,
                }
            )


def main_lm(args):
    out_vocab = None
    if args.dataset == "dyck":
        datasets, in_vocab, _ = build_datasets_dyck(vocab=args.dyck_vocab)
    elif args.dataset == "tense":
        datasets, in_vocab, _ = build_datasets_tense_inflection(
            include_only_present=args.exclude_identity
        )
    else:
        datasets, in_vocab, _ = build_datasets_lm(
            include_only_quest=args.exclude_identity,
            include_only_decls_nd_simpl_ques=args.pretrain,
        )
    model = initialize_model(args, in_vocab, model_name=args.model_load_path)

    if args.callback:
        if args.dataset == "lm":
            callback_fn = lambda split: eval_callback_lm(model, in_vocab, split)
    else:
        callback_fn = None
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = "cpu"
    model.to(device)

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

    eval_keys = ["val", "test"]

    train_loop(
        args,
        model,
        datasets["train"],
        {k: datasets[k] for k in eval_keys},
        device,
        args.save_dir,
        in_vocab,
        eval_every=args.eval_every,
        save_every=args.eval_every,
        max_steps=args.max_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        callback_fn=callback_fn,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="lm")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--pos_scale", type=float, default=1.0)
    parser.add_argument(
        "--no-pos-enc",
        action="store_true",
        help="Whether to not use positional encoding",
    )
    parser.add_argument(
        "--exclude_identity",
        action="store_true",
        help="Whether to only include sequences with 'quest' token in training data!",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Whether to pretrain the model on the LM task!",
    )

    parser.add_argument("--callback", action="store_true")

    args = parser.parse_args()
    set_seed(args)
    wandb.init(
        project="structural-grokking", entity="kabirahuja2431", config=vars(args)
    )
    if args.save_dir != "":
        wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()

    main_lm(args)
