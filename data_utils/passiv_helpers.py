from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
from util import run_lm_decoding, test_classification, test_infillings
import random
import os
import numpy as np

DATA_DIR = os.path.join(os.getcwd(), "data_utils")


def process(line):
    return line.replace("\t", " ")


def read_passiv_data(splits, do_process=True):

    in_sentences = []
    index_map = {split: [] for split in splits}

    for split in splits:
        with open(
            f"{DATA_DIR}/multilingual-transformations/data/passiv_en_nps/passiv_en_nps.{split}",
            "r",
        ) as reader:

            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]

            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sentences.append(sent)
    return in_sentences, index_map


def read_passiv_cls_data(splits, do_process=True):
    in_sentences = []
    out_subjs = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            f"{DATA_DIR}/multilingual-transformations/data/passiv_en_nps/passiv_en_nps.{split}",
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]

            # sents = [sent for sent in sents if "passiv" in sent]
            for sent in sents:
                index_map[split].append(len(in_sentences))
                if "passiv" in sent:
                    in_sent, out_sent = sent.split("passiv")
                    in_sent = f"{in_sent} passiv"
                else:
                    in_sent, out_sent = sent.split("decl")
                    in_sent = f"{in_sent} decl"
                out_subj = out_sent.strip().split()[1]
                in_sentences.append(in_sent.strip())
                out_subjs.append(out_subj)

    return in_sentences, out_subjs, index_map


def build_datasets_passivization():

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "dev", "gen"]

    in_sentences, index_map = read_passiv_data(splits)

    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)

    dataset = {}

    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        prefix_lens = [
            len(sent.split("passiv")[0].split("decl")[0].split()) for sent in in_subset
        ]
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "prefix_len": prefix_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


def build_datasets_passivization_enc_dec():

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "dev", "gen"]

    sentence_pairs, index_map = read_passiv_data(splits)

    print("num examples: {}".format(len(sentence_pairs)))

    in_sentences = []
    out_sentences = []
    for sentence_pair in sentence_pairs:
        if "passiv" in sentence_pair:
            in_sentence, out_sentence = sentence_pair.split("passiv")
            in_sentence = in_sentence.strip() + " passiv"
        else:
            in_sentence, out_sentence = sentence_pair.split("decl")
            in_sentence = in_sentence.strip() + " decl"
        in_sentences.append(in_sentence.strip())
        out_sentences.append(out_sentence.strip())

    vocab = WordVocabulary(sentence_pairs, split_punctuation=False)

    dataset = {}

    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_sentences, index_map[split])
        in_subset_tokenized = [vocab(s) for s in in_subset]
        out_subset_tokenized = [vocab(s) for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        out_lens = [len(s) for s in out_subset_tokenized]
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


def build_datasets_passivization_cls():

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "dev", "gen"]

    in_sentences, out_subjs, index_map = read_passiv_cls_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    out_vocab = WordVocabulary(out_subjs, split_punctuation=False, use_pad=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_subjs, index_map[split])
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
    return dataset, in_vocab, out_vocab, in_sentences, out_subjs


def eval_callback_passivization(lm, in_vocab, split):

    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    sents, _ = read_passiv_data([split])
    if len(sents) > 2000:
        sents = random.sample(sents, k=2000)

    split_into_words = [sent.split(" ") for sent in sents if "passiv" in sent]
    input_sents = []
    target_sents = []

    for sent_words in split_into_words:
        idx = sent_words.index("passiv")
        input_sent = " ".join(sent_words[: idx + 1])
        target_sent = " ".join(sent_words[idx + 1 :])
        input_sents.append(input_sent)
        target_sents.append(target_sent)

    out = run_lm_decoding(tokenizer, lm, input_sents, 0)

    main_correct = 0.0

    for target_sent, out_pred in zip(target_sents, out):

        # Get second word in both the prediction and target
        try:
            pred_subj = in_vocab(out_pred)[1]
            tgt_subj = target_sent.split(" ")[1]
            main_correct += int(pred_subj == tgt_subj)
        except IndexError:
            main_correct += 0
    return main_correct / len(target_sents)


def eval_passivization_cls_callback(cls_model, in_vocab, out_vocab, split):
    def tokenizer(s):
        return [cls_model.sos] + in_vocab(s)

    in_sents, out_subjs, _ = read_passiv_cls_data([split])
    preds, targets = test_classification(
        tokenizer, out_vocab, cls_model, in_sents, out_subjs, 0
    )
    acc = sum(preds == targets) / len(preds)
    print(acc)
    return acc
