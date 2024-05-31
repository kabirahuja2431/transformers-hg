import os
import torch
import string
from tqdm import tqdm
import random
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
from util import test_continuations, test_classification, test_infillings
from data_utils.lm_dataset_helpers import (
    process,
    read_lm_data,
    build_datasets_lm,
    build_datasets_enc_dec,
    build_datasets_mlm,
    eval_mlm_callback
)
import collate

DATA_DIR = os.path.join(os.getcwd(), "data_utils")


def read_qf_de_cls_data(splits, do_process=True):
    in_sentences = []
    out_auxs = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            "{}/multilingual-transformations/data/question_have-can_withquest_de/question_have_can.de.{}".format(
                DATA_DIR, split
            ),
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]

            sents = [sent for sent in sents if "quest" in sent]
            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sent, out_sent = sent.split("quest")
                out_aux = out_sent.strip().split()[0]
                in_sentences.append(in_sent.strip())
                out_auxs.append(out_aux)

    return in_sentences, out_auxs, index_map


def build_datasets_quest_de_lm():

    return build_datasets_lm(
        data_name="question_have-can_withquest_de",
        filename_prefix="question_have_can.de",
        data_dir=f"{DATA_DIR}/multilingual-transformations/data/",
        splits=["train", "dev", "gen"],
    )


def build_datasets_quest_de_enc_dec():

    return build_datasets_enc_dec(
        data_name="question_have-can_withquest_de",
        filename_prefix="question_have_can.de",
        data_dir=f"{DATA_DIR}/multilingual-transformations/data/",
        splits=["train", "dev", "gen"],
    )


def build_datasets_quest_de_cls():

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "dev", "gen"]
    in_sentences, out_auxs, index_map = read_qf_de_cls_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    out_vocab = WordVocabulary(out_auxs, split_punctuation=False, use_pad=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_auxs, index_map[split])
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
    return dataset, in_vocab, out_vocab, in_sentences, out_auxs


def build_datasets_quest_de_mlm(mask_strategy):

    return build_datasets_mlm(
        mask_strategy=mask_strategy,
        data_name="question_have-can_withquest_de",
        filename_prefix="question_have_can.de",
        data_dir=f"{DATA_DIR}/multilingual-transformations/data/",
        splits=["train", "dev", "gen"],
    )


def eval_qf_de_lm_callback(
    lm,
    in_vocab,
    split,
    is_prefix_lm=False,
):

    def tokenizer(s):
        try:
            return [lm.encoder_sos] + in_vocab(s)
        except AttributeError:
            return [lm.sos] + in_vocab(s)

    sents, _ = read_lm_data(
        [split],
        data_name="question_have-can_withquest_de",
        filename_prefix="question_have_can.de",
        data_dir=f"{DATA_DIR}/multilingual-transformations/data/",
    )
    split_into_words = [sent.split(" ") for sent in sents if "quest" in sent]
    q_words = []
    prefixes = []
    for sent_words in split_into_words:
        idx = sent_words.index("quest")
        q_word = sent_words[idx + 1]
        q_words.append(q_word)
        prefixes.append(" ".join(sent_words[: idx + 1]))

    out = test_continuations(tokenizer, lm, prefixes, 0, prefix_no_pos=is_prefix_lm)
    # out = test_continuations_gpt2(tokenizer, lm, prefixes[:100], args.gpu_id)
    closing_words = ["haben", "kann", "können", "hat"]
    closing_word_idxs = [in_vocab[w] for w in closing_words]
    out = out[:, closing_word_idxs]

    acc = [closing_words[i] == q_word for i, q_word in zip(out.argmax(dim=1), q_words)]
    agg_acc = sum(acc) / len(out)
    print(agg_acc)
    return agg_acc


def eval_qf_de_cls_callback(
    cls_model,
    in_vocab,
    out_vocab,
    split,
):

    def tokenizer(s):
        return in_vocab(s)

    in_sents, out_auxs, _ = read_qf_de_cls_data([split])
    preds, targets = test_classification(
        tokenizer, out_vocab, cls_model, in_sents, out_auxs, 0
    )
    acc = sum(preds == targets) / len(preds)
    print(acc)
    return acc


def eval_qf_de_mlm_callback(
    mlm,
    eval_dataset,
    in_vocab,
    split
):
    return eval_mlm_callback(
        mlm,
        eval_dataset,
        in_vocab,
        split
    )