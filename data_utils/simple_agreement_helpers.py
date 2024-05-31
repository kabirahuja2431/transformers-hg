import os
import torch
import string
from tqdm import tqdm
import random
import numpy as np
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
from util import test_continuations, test_classification, test_infillings
import collate
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from data_utils.tense_inflection_helpers import (
    sent_to_pos,
    get_main_verb,
    eval_correct_agreement,
)


DATA_DIR = os.path.join(os.getcwd(), "data_utils")


def get_main_verb(sent):
    pos_tags = sent_to_pos(sent)
    # Check if there is a relative clause in the beginning
    # If yes second verb is the main verb
    # Else it is the first verb
    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1
    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break

    main_verb = sent.split()[ind_v]
    return main_verb, ind_v


def switch_verb_plurality(verb):
    if verb[-1] == "s":
        return verb[:-1]
    else:
        return verb + "s"


def process(line):
    return line.replace("\t", " ").strip()


def check_simple_sent(sent):
    # check if before the main verb there is only a single noun
    pos_tags = sent_to_pos(sent)
    # main_verb, mv_idx = get_main_verb(sent)
    # num_nouns = len([tag for tag in pos_tags[:mv_idx] if tag == "N"])
    # return num_nouns == 1
    return "P" not in pos_tags and "R" not in pos_tags


def read_grammar_gen_data(
    splits,
    grammar,
    include_simple_sents_only=False,
    grammar_tgt=None,
    do_process=True,
):
    in_sentences = []
    index_map = {split: [] for split in splits}
    for split in splits:
        if grammar_tgt is None:
            with open(f"{DATA_DIR}/grammar_gen_data/{grammar}.{split}") as reader:
                if do_process:
                    sents = [process(line) for line in reader.readlines()]
                else:
                    sents = [line.strip() for line in reader.readlines()]

                if include_simple_sents_only:
                    sents = [sent for sent in sents if check_simple_sent(sent)]

                for sent in sents:
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(sent)

        else:
            if split not in ["test", "g1_test", "g2_test"]:
                with open(f"{DATA_DIR}/grammar_gen_data/{grammar}.{split}") as reader:
                    if do_process:
                        sents = [process(line) for line in reader.readlines()]
                    else:
                        sents = [line.strip() for line in reader.readlines()]

                    if include_simple_sents_only:
                        sents = [sent for sent in sents if check_simple_sent(sent)]

                    for sent in sents:
                        index_map[split].append(len(in_sentences))
                        in_sentences.append(sent)
            else:
                with open(
                    f"{DATA_DIR}/grammar_gen_data/{grammar_tgt}.{split}"
                ) as reader:
                    if do_process:
                        sents = [process(line) for line in reader.readlines()]
                    else:
                        sents = [line.strip() for line in reader.readlines()]

                    for sent in sents:
                        index_map[split].append(len(in_sentences))
                        in_sentences.append(sent)

    return in_sentences, index_map


def read_grammar_gen_cls_data(splits, grammar, do_process=True):
    in_sentences = []
    out_verbs = []
    out_verb_ids = []
    index_map = {split: [] for split in splits}

    for split in splits:
        with open(f"{DATA_DIR}/grammar_gen_data/{grammar}.{split}") as reader:
            if do_process:
                sents = [process(line) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]

            for sent in sents:
                index_map[split].append(len(in_sentences))
                out_main_verb, out_main_verb_id = get_main_verb(sent)
                in_sent = " ".join(sent.split(" ")[:out_main_verb_id])
                in_sentences.append(in_sent)
                out_verbs.append(out_main_verb)
                out_verb_ids.append(out_main_verb_id)

    return in_sentences, out_verbs, out_verb_ids, index_map


def build_datasets_simple_agreement(
    grammar, grammar_tgt=None, eval_keys=[], include_simple_sents_only=False
):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    if eval_keys == []:
        splits = ["train", "val", "g1_test", "g2_test"]
    else:
        splits = ["train"] + eval_keys
    in_sentences, index_map = read_grammar_gen_data(
        splits,
        grammar,
        include_simple_sents_only=include_simple_sents_only,
        grammar_tgt=grammar_tgt,
    )
    print("num examples: {}".format(len(in_sentences)))
    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)

    dataset = {}

    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]

        main_verb_ids = [[get_main_verb(sent)[1]] for sent in in_subset]
        all_verb_ids = []
        for sent in in_subset:
            pos_tags = sent_to_pos(sent)
            verb_ids = [i for i, tag in enumerate(pos_tags) if tag == "V"]
            all_verb_ids.append(verb_ids)

        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
            "main_verb_ids": main_verb_ids,
            "all_verb_ids": all_verb_ids,
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, in_vocab, in_sentences


def build_datasets_simple_agreement_cls(grammar):

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "g1_test", "g2_test"]

    in_sentences, out_verbs, out_verb_ids, index_map = read_grammar_gen_cls_data(
        splits, grammar=grammar
    )
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    out_vocab = WordVocabulary(out_verbs, split_punctuation=False)

    dataset = {}

    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_verbs, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        out_subset_tokenized = [out_vocab(s)[0] for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "out": out_subset_tokenized,
            "in_len": in_lens,
            "cls_head_idx": [in_len - 1 for in_len in in_lens],
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, out_vocab, in_sentences, out_verbs


def build_datasets_simple_agreement_mlm(grammar, mask_strategy):
    assert mask_strategy in ["main-verb", "all-verbs"]

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    def mask_verbs(sent):
        tokens = sent.split(" ")
        pos_tags = sent_to_pos(sent)
        if mask_strategy == "all-verbs":
            verb_ids = [i for i, tag in enumerate(pos_tags) if tag == "V"]
        else:
            main_verb, ind_v = get_main_verb(sent)
            verb_ids = [ind_v]

        for verb_id in verb_ids:
            tokens[verb_id] = "<mask>"
        return " ".join(tokens)

    def tokenize_masked_sent(masked_sent):
        tokenized_sent = []
        for word in masked_sent.split(" "):
            if word == "<mask>":
                tokenized_sent.append(len(in_vocab))
            else:
                tokenized_sent.append(in_vocab[word])
        return tokenized_sent

    def get_target_ids(og_sent_ids, masked_sent_ids):
        target_ids = []
        for og_id, masked_id in zip(og_sent_ids, masked_sent_ids):
            if og_id == masked_id:
                target_ids.append(-100)
            else:
                target_ids.append(og_id)
        return target_ids

    splits = ["train", "val", "g1_test", "g2_test"]
    in_sentences, index_map = read_grammar_gen_data(splits, grammar)
    print("num examples: {}".format(len(in_sentences)))

    # Mask verbs in all sentences
    masked_in_sentences = [mask_verbs(sent) for sent in in_sentences]
    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        masked_in_subset = get_subset(masked_in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        masked_in_subset_tokenized = [tokenize_masked_sent(s) for s in masked_in_subset]
        target_ids = [
            get_target_ids(og, masked)
            for og, masked in zip(in_subset_tokenized, masked_in_subset_tokenized)
        ]
        data = {
            "in": masked_in_subset_tokenized,
            "out": target_ids,
            "in_len": [len(s) for s in masked_in_subset_tokenized],
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


def eval_callback_simple_agreement(lm, in_vocab, split, grammar, grammar_tgt=None):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    lm.eval()
    sents, _ = read_grammar_gen_data([split], grammar, grammar_tgt=grammar_tgt)

    prefixes = []
    verb_words = []
    for sent in sents:
        tokens = sent.split(" ")
        main_verb, ind_v = get_main_verb(sent)
        verb_words.append(main_verb)
        prefix = " ".join(tokens[:ind_v])
        prefixes.append(prefix)
    out = test_continuations(tokenizer, lm, prefixes, 0)
    closing_words = [
        [main_verb, switch_verb_plurality(main_verb)] for main_verb in verb_words
    ]
    closing_word_ids = [[in_vocab[w] for w in cw] for cw in closing_words]
    acc = []
    for i, cw in enumerate(closing_word_ids):
        logits = out[i][cw]
        pred_id = logits.argmax(dim=-1).item()
        pred_word = closing_words[i][pred_id]
        acc.append(pred_word == verb_words[i])

    # preds = out.argmax(dim=-1).tolist()
    # verb_word_ids = [in_vocab(w) for w in verb_words]
    # acc = [pred == label for pred, label in zip(preds, verb_word_ids)]
    agg_acc = sum(acc) / len(acc)
    print(f"Accuracy on {split}: {agg_acc}")
    return agg_acc


def eval_all_agreements(lm, in_vocab, split, grammar, grammar_tgt=None):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    if split == "g2_test":
        return 0.0
    lm.eval()
    sents, _ = read_grammar_gen_data([split], grammar, grammar_tgt=grammar_tgt)

    # Choose only those sentences which have another verb after the main verb
    def object_rc_verb_filter(sent):
        pos_tags = sent_to_pos(sent)
        main_verb, ind_v = get_main_verb(sent)
        return "V" in pos_tags[ind_v + 1 :]

    sents = [sent for sent in sents if object_rc_verb_filter(sent)]

    if len(sents) == 0:
        return 0.0

    all_agreements_accs = []
    for sent in tqdm(sents):
        tokens = sent.split(" ")

        # Iterate over each verb in the sentence
        pos_tags = sent_to_pos(sent)
        verb_ids = [i for i, tag in enumerate(pos_tags) if tag == "V"]
        prefixes = []
        verb_words = []
        for verb_id in verb_ids:
            prefix = " ".join(tokens[:verb_id])
            prefixes.append(prefix)
            verb_words.append(tokens[verb_id])
        out = test_continuations(tokenizer, lm, prefixes, 0)
        closing_words = [[w, switch_verb_plurality(w)] for w in verb_words]
        closing_word_ids = [[in_vocab[w] for w in cw] for cw in closing_words]
        acc = []
        for i, cw in enumerate(closing_word_ids):
            logits = out[i][cw]
            pred_id = logits.argmax(dim=-1).item()
            pred_word = closing_words[i][pred_id]
            acc.append(pred_word == verb_words[i])
        all_agreements_accs.append(all(acc))

    return sum(all_agreements_accs) / len(all_agreements_accs)


def eval_cls_callback_main_verb(cls_model, grammar, in_vocab, out_vocab, split):

    def tokenizer(s):
        return [cls_model.sos] + in_vocab(s)

    cls_model.eval()

    in_sents, out_verbs, out_verb_ids, _ = read_grammar_gen_cls_data([split], grammar)
    preds, targets, logits = test_classification(
        tokenizer,
        out_vocab,
        cls_model,
        in_sents,
        out_verbs,
        0,
        cls_head_idx=[idx - 1 for idx in out_verb_ids],
        return_logits=True,
    )

    acc = eval_correct_agreement(targets, logits, out_vocab)
    print(acc)
    return acc


def eval_mlm_callback_main_verb(mlm, eval_dataset, in_vocab, grammar, split):
    MAX_SAMPLES_EVAL = len(eval_dataset)  # 1000
    eval_dataset = eval_dataset.select(range(MAX_SAMPLES_EVAL))
    in_sentences, _ = read_grammar_gen_data([split], grammar)
    in_sentences = in_sentences[:MAX_SAMPLES_EVAL]
    infill_idxs = [get_main_verb(sent)[1] for sent in in_sentences]

    # check if inflill_idxs are masked
    # for i, idx in enumerate(infill_idxs):
    #     assert eval_dataset["in"][i][idx] == len(in_vocab)
    out, gold_labels = test_infillings(
        mlm,
        eval_dataset["in"],
        eval_dataset["in_len"],
        eval_dataset["out"],
        infill_idxs=infill_idxs,
        gpu_id=0,
    )

    closing_word_ids = []
    closing_words = []
    gold_verbs = []
    for i in range(len(gold_labels)):
        gold_verb_id = gold_labels[i].item()
        gold_verb = in_vocab[gold_verb_id]
        distractor_verb = switch_verb_plurality(gold_verb)
        distractor_verb_id = in_vocab[distractor_verb]
        closing_word_ids.append([gold_verb_id, distractor_verb_id])
        closing_words.append([gold_verb, distractor_verb])
        gold_verbs.append(gold_verb)

    acc = []
    for i, cw in enumerate(closing_word_ids):
        logits = out[i][cw]
        pred_id = logits.argmax(dim=-1).item()
        pred_word = closing_words[i][pred_id]
        acc.append(pred_word == gold_verbs[i])

    # preds = out.argmax(dåim=-1).tolist()
    # verb_word_ids = [in_vocab(w) for w in verb_words]
    # acc = [pred == label for pred, label in zip(preds, verb_word_ids)]
    agg_acc = sum(acc) / len(acc)
    print(f"Accuracy on {split}: {agg_acc}")
    return agg_acc
