import os
import torch
import string
from tqdm import tqdm
import numpy as np
from datasets import concatenate_datasets
from datasets import Dataset as HFDataset
from models.transformer_lm import TransformerLM
from util import run_lm_decoding
from vocabulary import WordVocabulary

DATA_DIR = os.path.join(os.getcwd(), "data_utils/COGS/data")


def read_cogs_data(splits):
    in_sentences = []
    out_sentences = []
    index_map = {split: [] for split in splits}

    for split in splits:
        with open(os.path.join(DATA_DIR, split.split("-")[0] + ".tsv"), "r") as f:
            for line in f:
                if line == "":
                    continue
                if split != "gen-struct":
                    inp, out, _ = line.split("\t")
                else:
                    inp, out, type_ = line.split("\t")
                    type_ = type_.strip()
                    if type_ not in ["obj_pp_to_subj_pp", "cp_recursion", "pp_recursion"]:
                        continue
                in_sentences.append(inp)
                out_sentences.append(out)
                index_map[split].append(len(in_sentences) - 1)

    return in_sentences, out_sentences, index_map


def build_datasets_cogs_lm(input_only=False, gen_inputs_for_train=False):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "dev", "test", "gen", "gen-struct"]
    in_sentences, out_sentences, index_map = read_cogs_data(splits)
    sent_pairs = [
        f"[INP] {inp} [OUT] {out} [EOS]"
        for inp, out in zip(in_sentences, out_sentences)
    ]
    print("num examples: {}".format(len(sent_pairs)))

    vocab = WordVocabulary(sent_pairs, split_punctuation=False)
    
    if input_only:
        # Only use input sentences in sent_pairs
        sent_pairs = [f"[INP] {inp} [EOS]" for inp in in_sentences]
    
    dataset = {}
    for split in splits:
        subset = get_subset(sent_pairs, index_map[split])
        subset_tokenized = [vocab(pair) for pair in subset]
        lens = [len(sent) for sent in subset_tokenized]
        data = {
            "in": subset_tokenized,
            "in_len": lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        if split == "gen" and input_only and gen_inputs_for_train:
            dataset["train"] = concatenate_datasets([dataset["train"], dataset_curr])
        else:
            dataset[split] = dataset_curr

    return (
        dataset,
        vocab,
        sent_pairs,
    )
    
def build_datasets_cogs_enc_dec():
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]
    
    splits = ["train", "dev", "test", "gen", "gen-struct"]
    in_sentences, out_sentences, index_map = read_cogs_data(splits)
    print("num examples: {}".format(len(in_sentences)))
    
    sent_pairs = [
        f"{inp} {out}"
        for inp, out in zip(in_sentences, out_sentences)
    ]
    vocab = WordVocabulary(sent_pairs, split_punctuation=False)
    dataset = {}
    
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_sentences, index_map[split])
        in_subset_tokenized = [vocab(inp) for inp in in_subset]
        out_subset_tokenized = [vocab(out) for out in out_subset]
        in_lens = [len(sent) for sent in in_subset_tokenized]
        out_lens = [len(sent) for sent in out_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "out": out_subset_tokenized,
            "out_len": out_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
        
    return dataset, vocab, in_sentences, out_sentences
    


def eval_callback_cogs(lm, in_vocab, split, max_decoding_steps=100, generation_file=None):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    in_sents, out_sents, _ = read_cogs_data([split])
    if len(in_sents) > 2000:
        in_sents = in_sents[:2000]
        out_sents = out_sents[:2000]

    input_sents = []
    target_sents = []
    for in_sent, out_sent in zip(in_sents, out_sents):
        in_sent = f"[INP] {in_sent} [OUT]" if isinstance(lm, TransformerLM) else in_sent
        input_sents.append(in_sent)
        target_sents.append(out_sent)

    out = run_lm_decoding(
        tokenizer, lm, input_sents, 0, max_decoding_steps=max_decoding_steps
    )
    main_correct = 0.0
    preds = []
    for target_sent, out_pred in zip(target_sents, out):
        pred_tokens = in_vocab(out_pred)
        pred_tokens = pred_tokens[
            : len(pred_tokens)
            if "[EOS]" not in pred_tokens
            else pred_tokens.index("[EOS]")
        ]
        pred = " ".join(pred_tokens)
        preds.append(pred)
        main_correct += int(pred == target_sent)
    acc = main_correct / len(target_sents)
    if generation_file is not None:
        with open(generation_file, "a") as f:
            f.write("Split: {}\n".format(split))
            f.write("-" * 50 + "\n")
            
            #Select random 10 examples
            np.random.seed(0) # ToDo: Instead of hardcoding, pass it as a parameter maybe 
            indices = np.random.choice(len(input_sents), 10, replace=False)
            rand_input_sents = [input_sents[i] for i in indices]
            rand_preds = [preds[i] for i in indices]
            rand_target_sents = [target_sents[i] for i in indices]
            for inp, pred, target in zip(rand_input_sents, rand_preds, rand_target_sents):
                f.write(f"Input: {inp}\nPrediction:{pred}\nTarget:{target}\n")
                f.write("*" * 50 + "\n")

            f.write("\n\n")
    
    return acc