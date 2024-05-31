### Code borrowed from McCoy et al: Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence-to-sequence networks

from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
from util import run_lm_decoding, test_classification, test_infillings
import random
import os
import numpy as np


# Determines whether the main verb of the sentence is correct
### target is first, pred is second
def main_right_tense(senta, sentb):
    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

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

    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]

    return verbb == verba


# Converting a sentence to a list of part-of-speech tags
DATA_DIR = os.path.abspath(os.path.dirname(__file__))
posDictTense = {}
fi = open("{}/tense_inflection_data/pos_tense.txt".format(DATA_DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDictTense[parts[0].strip()] = parts[1].strip()

posDict = {}
fi = open("{}/tense_inflection_data/pos.txt".format(DATA_DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDict[parts[0].strip()] = parts[1].strip()


def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags


def get_main_verb(sent):

    words = sent.split()
    pos_tags = sent_to_pos(sent)

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

    return words[ind_v], ind_v


def switch_verb_plurality(verb):
    if verb[-1] == "s":
        return verb[:-1]
    else:
        return verb + "s"


# Does the sentence have the correct sequence of
# part of speech tags?
def right_pos(senta, sentb):
    try:
        pos_tags_a = sent_to_pos(senta)
        pos_tags_b = sent_to_pos(sentb)
    except KeyError:
        return False
    if pos_tags_a == pos_tags_b:
        return True
    else:
        return False


nouns_sg = [
    "newt",
    "orangutan",
    "peacock",
    "quail",
    "raven",
    "salamander",
    "tyrannosaurus",
    "unicorn",
    "vulture",
    "walrus",
    "xylophone",
    "yak",
    "zebra",
]
nouns_pl = [
    "newts",
    "orangutans",
    "peacocks",
    "quails",
    "ravens",
    "salamanders",
    "tyrannosauruses",
    "unicorns",
    "vultures",
    "walruses",
    "xylophones",
    "yaks",
    "zebras",
]

verbs_sg = [
    "giggles",
    "smiles",
    "sleeps",
    "swims",
    "waits",
    "moves",
    "changes",
    "reads",
    "eats",
    "entertains",
    "amuses",
    "high_fives",
    "applauds",
    "confuses",
    "admires",
    "accepts",
    "remembers",
    "comforts",
]
verbs_pl = [
    "giggle",
    "smile",
    "sleep",
    "swim",
    "wait",
    "move",
    "change",
    "read",
    "eat",
    "entertain",
    "amuse",
    "high_five",
    "applaud",
    "confuse",
    "admire",
    "accept",
    "remember",
    "comfort",
]

auxes_sg = ["does", "doesn't"]
auxes_pl = ["do", "don't"]


# Given an input past tense sentence, outputs
# what the present-tense version would be if
# verbs agreed with the most recent noun instead
# of with their subjects.
# def tense_nearest(sent):
# 	new_words = []
# 	words = sent.split()
# 	tense_agr = "sg"
# 	for word in words:
#      if word in nouns_sg:
#         	tense_agr = "sg"
#         	new_words.append(word)
# 		elif word in nouns_pl:
#       		tense_agr = "pl"
#         	new_words.append(word)
#         elif word in verbs_sg:
#           	verb_ind = verbs_sg.index(word)
# 			if tense_agr == "sg":
#             	new_words.append(verbs_sg[verb_ind])
# 			else:
#             	new_words.append(verbs_pl[verb_ind])
#         elif word in verbs_pl:
#             verb_ind = verbs_pl.index(word)
#             if tense_agr == "sg":
#                 new_words.append(verbs_sg[verb_ind])
#             else:
#                 new_words.append(verbs_pl[verb_ind])
#         else:
#             new_words.append(word)
# 	return " ".join(new_words)


def process(line):
    return line.replace("\t", " ")


def is_simple(sent):
    pos_tags = sent_to_pos(sent)
    # get index of first verb in sentence
    verb_index = pos_tags.index("V")
    # check how many "N" appear before the verb
    num_nouns = pos_tags[:verb_index].count("N")
    if num_nouns == 1:
        return True
    else:
        return False


def read_ti_data(
    splits,
    do_process=True,
    include_only_present=False,
    include_only_past=False,
    include_only_past_and_simple_present=False,
    data_dir="tense_inflection_data",
):
    in_sentences = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            "{}/{}/tense.{}".format(DATA_DIR, data_dir, split),
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]
            # if split == "train":
            if include_only_present:
                sents = [sent for sent in sents if "PRESENT" in sent]
            elif include_only_past_and_simple_present:
                past_sents = [sent for sent in sents if "PAST" in sent]
                present_simple_sents = [
                    sent for sent in sents if "PRESENT" in sent and is_simple(sent)
                ]
                sents = past_sents + present_simple_sents

            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sentences.append(sent)
    return in_sentences, index_map


def read_ti_cls_data(
    splits,
    do_process=True,
):
    in_sentences = []
    out_verbs = []
    out_verb_ids = []
    index_map = {split: [] for split in splits}

    for split in splits:
        with open(
            "{}/tense_inflection_data/tense.{}".format(DATA_DIR, split),
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]

            sents = [sent for sent in sents if "PRESENT" in sent]
            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sent, out_sent = sent.split("PRESENT")
                out_main_verb, out_main_verb_id = get_main_verb(out_sent)
                in_sentences.append(in_sent)
                out_verbs.append(out_main_verb)
                # out_verb_ids.append(out_main_verb_id)
                out_verb_ids.append(out_main_verb_id)
                # breakpoint()

    return in_sentences, out_verbs, out_verb_ids, index_map


def build_datasets_tense_inflection(
    include_only_present=False,
    include_only_past_and_simple_present=False,
    data_dir="tense_inflection_data",
):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_ti_data(
        splits,
        include_only_present=include_only_present,
        include_only_past_and_simple_present=include_only_past_and_simple_present,
        data_dir=data_dir,
    )

    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        prefix_lens = [
            len(sent.split("PRESENT")[0].split("PAST")[0].split()) for sent in in_subset
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


def build_datasets_ti_enc_dec(
    include_only_present=False, include_only_past_and_simple_present=False
):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    sentence_pairs, index_map = read_ti_data(
        splits,
        include_only_present=include_only_present,
        include_only_past_and_simple_present=include_only_past_and_simple_present,
    )

    print("num examples: {}".format(len(sentence_pairs)))
    # Split sentence pairs into in_sentences and out_sentences
    in_sentences = []
    out_sentences = []
    for sentence_pair in sentence_pairs:
        if "PRESENT" in sentence_pair:
            in_sentence, out_sentence = sentence_pair.split("PRESENT")
            in_sentence = in_sentence.strip() + " PRESENT"
        else:
            in_sentence, out_sentence = sentence_pair.split("PAST")
            in_sentence = in_sentence.strip() + " PAST"
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


def build_datasets_ti_cls():

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, out_verbs, out_verb_ids, index_map = read_ti_cls_data(splits)
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
        main_verb_idxs = [get_main_verb(s)[1] for s in in_subset]
        data = {
            "in": in_subset_tokenized,
            "out": out_subset_tokenized,
            "in_len": in_lens,
            "cls_head_idx": main_verb_idxs,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, out_vocab, in_sentences, out_verbs


def build_datasets_ti_mlm(mask_strategy, **kwargs):

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

    def mask_verb_sent_pair(sent_pair):
        in_sent, out_sent = sent_pair.split("PRESENT")
        in_sent = in_sent.strip()
        out_sent = out_sent.strip()
        # mask out_sent
        out_sent_masked = mask_verbs(out_sent)

        return f"{in_sent.strip()} PRESENT {out_sent_masked.strip()}"

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
            if og_id != masked_id:
                target_ids.append(og_id)
            else:
                target_ids.append(-100)
        return target_ids

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_ti_data(splits, include_only_present=True)
    masked_in_sentences = [mask_verb_sent_pair(sent) for sent in in_sentences]

    print("num examples: {}".format(len(in_sentences)))
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


def eval_callback_tense_inflection(lm, in_vocab, split):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    def remove_fullstop(s):
        if len(s) != 0 and s[-1] == ".":
            return " ".join(s.split(" ")[:-1])
        else:
            return s

    sents, _ = read_ti_data([split])
    if len(sents) > 2000:
        sents = random.sample(sents, k=2000)

    split_into_words = [sent.split(" ") for sent in sents if "PRESENT" in sent]
    input_sents = []
    target_sents = []
    for sent_words in split_into_words:
        if "PRESENT" in sent_words:
            word = "PRESENT"
        elif "PAST" in sent_words:
            word = "PAST"
        idx = sent_words.index(word)
        input_sents.append(" ".join(sent_words[: idx + 1]))
        target_sents.append(" ".join(sent_words[idx + 1 :]))

    out = run_lm_decoding(tokenizer, lm, input_sents, 0)

    main_correct = 0.0
    for target_sent, our_pred in zip(target_sents, out):
        pred = remove_fullstop(" ".join(in_vocab(our_pred)))
        target = remove_fullstop(target_sent)
        main_correct += main_right_tense(target, pred)
    ### check exact match acc
    return main_correct / len(target_sents)


def eval_correct_agreement(gold_labels, logits, out_vocab):

    closing_word_ids = []
    closing_words = []
    gold_verbs = []
    # breakpoint()
    for i in range(len(gold_labels)):
        gold_verb_id = gold_labels[i].item()
        gold_verb = out_vocab[gold_verb_id]
        distractor_verb = switch_verb_plurality(gold_verb)
        distractor_verb_id = out_vocab[distractor_verb]
        closing_word_ids.append([gold_verb_id, distractor_verb_id])
        closing_words.append([gold_verb, distractor_verb])
        gold_verbs.append(gold_verb)

    acc = []
    for i, cw in enumerate(closing_word_ids):
        cw_logits = logits[i][cw]
        pred_id = cw_logits.argmax(-1).item()
        pred_word = closing_words[i][pred_id]
        acc.append(pred_word == gold_verbs[i])

    agg_acc = sum(acc) / len(acc)
    return agg_acc


def eval_ti_cls_callback(cls_model, in_vocab, out_vocab, split):

    def tokenizer(s):
        return [cls_model.sos] + in_vocab(s)

    cls_model.eval()
    in_sents, out_verbs, out_verb_ids, _ = read_ti_cls_data([split])
    preds, targets, logits = test_classification(
        tokenizer,
        out_vocab,
        cls_model,
        in_sents,
        out_verbs,
        0,
        cls_head_idx=out_verb_ids,
        return_logits=True,
    )

    # acc = eval_correct_agreement(targets, logits, out_vocab)
    acc = sum(preds == targets) / len(preds)
    # breakpoint()
    print(acc)
    return acc


def eval_ti_mlm_callback(mlm, eval_dataset, in_vocab, split):

    def tokenizer(s):
        return in_vocab(s)

    in_sentences, _ = read_ti_data([split], include_only_present=True)
    infill_idxs = []
    for sent_pair in in_sentences:
        past_sent, present_sent = sent_pair.split("PRESENT")
        past_sent = past_sent.strip()
        present_sent = present_sent.strip()
        infill_idx = get_main_verb(present_sent)[1] + len(past_sent.split(" ")) + 1
        infill_idxs.append(infill_idx)

    out, gold_labels = test_infillings(
        mlm,
        eval_dataset["in"],
        eval_dataset["in_len"],
        eval_dataset["out"],
        infill_idxs=infill_idxs,
        gpu_id=0,
    )

    # breakpoint()
    preds = out.argmax(dim=-1)
    agg_acc = (preds.detach().cpu().numpy() == np.array(gold_labels)).mean()
    # agg_acc = eval_correct_agreement(gold_labels, out, in_vocab)

    # closing_word_ids = []
    # closing_words = []
    # gold_verbs = []

    # for i in range(len(gold_labels)):
    #     gold_verb_id = gold_labels[i].item()
    #     gold_verb = in_vocab[gold_verb_id]
    #     distractor_verb = switch_verb_plurality(gold_verb)
    #     distractor_verb_id = in_vocab[distractor_verb]
    #     closing_word_ids.append([gold_verb_id, distractor_verb_id])
    #     closing_words.append([gold_verb, distractor_verb])
    #     gold_verbs.append(gold_verb)

    # acc = []
    # for i, cw in enumerate(closing_word_ids):
    #     logits = out[i][cw]
    #     pred_id = logits.argmax(dim=-1).item()
    #     pred_word = closing_words[i][pred_id]
    #     acc.append(pred_word == gold_verbs[i])

    # agg_acc = sum(acc) / len(acc)
    print(f"Accuracy on {split}: {agg_acc}")
    return agg_acc
