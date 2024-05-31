import os
import torch
import string
from tqdm import tqdm
import random
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
from util import test_continuations, test_classification, test_infillings
import collate

DATA_DIR = os.path.join(os.getcwd(), "data_utils")


def process(line):
    return line.replace("\t", " ")


#### define a callback function for how to handle model predictions?


def count_num_auxs(text):
    auxs = ["doesn't", "does", "do", "don't"]
    return sum([text.count(aux) for aux in auxs])


def read_lm_data(
    splits,
    data_name="question_formation_data",
    filename_prefix="question",
    test_filename_prefix=None,
    do_process=True,
    include_only_quest=False,
    include_only_decls=False,
    include_only_decls_nd_simpl_ques=False,
    include_only_complex_sents=False,
    till_first_out_token=False,
    data_dir=DATA_DIR,
):
    in_sentences = []
    index_map = {split: [] for split in splits}
    for split in splits:

        filename = (
            test_filename_prefix
            if split == "test" and test_filename_prefix
            else filename_prefix
        )
        with open(
            "{}/{}/{}.{}".format(data_dir, data_name, filename, split),
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]
            if include_only_quest:
                sents = [sent for sent in sents if "quest" in sent]
                if till_first_out_token:
                    sents = [
                        " ".join(sent.split()[: sent.split().index("quest") + 2])
                        for sent in sents
                    ]

            if split == "train":
                if include_only_decls:
                    sents = [sent for sent in sents if "quest" not in sent]
                elif include_only_decls_nd_simpl_ques:
                    sents = [
                        sent
                        for sent in sents
                        if count_num_auxs(sent) <= 2 or "quest" not in sent
                    ]
                elif include_only_complex_sents:
                    sents = [
                        sent
                        for sent in sents
                        if count_num_auxs(sent.split(" quest ")[0]) > 2
                    ]
            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sentences.append(sent)
    return in_sentences, index_map


def read_lm_cls_data(splits, do_process=True):
    in_sentences = []
    out_auxs = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            "{}/question_formation_data/question.{}".format(DATA_DIR, split),
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


def build_datasets_lm(
    data_name="question_formation_data",
    filename_prefix="question",
    test_filename_prefix=None,
    include_only_quest=False,
    include_only_decls=False,
    include_only_decls_nd_simpl_ques=False,
    till_first_out_token=False,
    include_only_complex_sents=False,
    data_dir=DATA_DIR,
    splits=["train", "val", "test"],
):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    in_sentences, index_map = read_lm_data(
        splits,
        data_name=data_name,
        filename_prefix=filename_prefix,
        test_filename_prefix=test_filename_prefix,
        include_only_quest=include_only_quest,
        include_only_decls=include_only_decls,
        include_only_decls_nd_simpl_ques=include_only_decls_nd_simpl_ques,
        include_only_complex_sents=include_only_complex_sents,
        till_first_out_token=till_first_out_token,
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
            len(sent.split("quest")[0].split("decl")[0].split()) for sent in in_subset
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


def build_datasets_enc_dec(
    data_name="question_formation_data",
    filename_prefix="question",
    include_only_quest=False,
    include_only_decls=False,
    include_only_decls_nd_simpl_ques=False,
    till_first_out_token=False,
    data_dir=DATA_DIR,
    splits=["train", "val", "test"],
):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    sentence_pairs, index_map = read_lm_data(
        splits,
        data_name=data_name,
        filename_prefix=filename_prefix,
        include_only_quest=include_only_quest,
        include_only_decls=include_only_decls,
        include_only_decls_nd_simpl_ques=include_only_decls_nd_simpl_ques,
        till_first_out_token=till_first_out_token,
        data_dir=data_dir,
    )
    print("num examples: {}".format(len(sentence_pairs)))

    # Split sentence pairs into in_sentences and out_sentences
    in_sentences = []
    out_sentences = []
    for sentence_pair in sentence_pairs:
        if "quest" in sentence_pair:
            in_sentence, out_sentence = sentence_pair.split("quest")
            in_sentence = in_sentence.strip() + " quest"
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
        out_subset_tokenized = [[len(vocab)] + vocab(s) for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        out_lens = [len(s) for s in out_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "out": out_subset_tokenized,
            "in_len": in_lens,
            "out_len": out_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, vocab, in_sentences, out_sentences


def build_datasets_lm_cls():
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, out_auxs, index_map = read_lm_cls_data(splits)
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


def build_datasets_mlm(
    mask_strategy,
    data_name="question_formation_data",
    filename_prefix="question",
    data_dir=DATA_DIR,
    splits=["train", "val", "test"],
    **kwargs,
):
    assert mask_strategy in ["aux", "random"]

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    def mask_auxs(token_ids, pair_type="quest"):
        if pair_type == "quest":
            sep_token_id = token_ids.index(quest_idx)
        else:
            sep_token_id = token_ids.index(decl_idx)
        decl_token_ids = token_ids[: sep_token_id + 1]
        quest_token_ids = token_ids[sep_token_id + 1 :]
        masked_token_ids = (
            decl_token_ids
            + [len(in_vocab)]
            + [
                len(in_vocab) if token_id in aux_idxs else token_id
                for token_id in decl_token_ids[:-1]
            ]
        )

        if pair_type == "quest":
            target_ids = [-100 for _ in decl_token_ids] + quest_token_ids[0:1]

            qid = 1
            did = 0

            while qid < len(quest_token_ids) - 1 and did < len(decl_token_ids) - 1:
                if quest_token_ids[qid] == decl_token_ids[did]:
                    if quest_token_ids[qid] in aux_idxs:
                        # Target will be the token id of the aux
                        target_ids.append(quest_token_ids[qid])
                    else:
                        # Target will be -100 i.e. to be ignored
                        target_ids.append(-100)
                    qid += 1
                    did += 1

                else:
                    # Add empty token id
                    target_ids.append(len(in_vocab) + 1)
                    did += 1
            target_ids.append(-100)
        else:
            target_ids = [-100 for _ in decl_token_ids] + [
                len(in_vocab) + 1
            ]  # EMPTY TOKEN as the first token
            for token_id in decl_token_ids[:-1]:
                if token_id in aux_idxs:
                    target_ids.append(token_id)
                else:
                    target_ids.append(-100)

        # for i in range(len(quest_token_ids) - 1):
        #     if quest_token_ids[i + 1] == decl_token_ids[i]:
        #         # Add empty token id
        #         target_ids.append(len(in_vocab) + 1)
        #     elif quest_token_ids[i + 1] in aux_idxs:
        #         # Target will be the token id of the aux
        #         target_ids.append(quest_token_ids[i + 1])
        #     else:
        #         # Target will be -100 i.e. to be ignored
        #         target_ids.append(-100)
        # Add -100 for ? token

        return masked_token_ids, target_ids

    def mask_random(token_ids, frac=0.15):
        # select random tokens to mask
        num_mask = int(frac * len(token_ids))
        mask_idxs = random.sample(range(len(token_ids)), num_mask)
        masked_token_ids = [
            len(in_vocab) if i in mask_idxs else token_id
            for i, token_id in enumerate(token_ids)
        ]
        target_ids = [
            -100 if i not in mask_idxs else token_id
            for i, token_id in enumerate(token_ids)
        ]
        return masked_token_ids, target_ids

    def mask(token_ids, pair_type="quest"):
        if mask_strategy == "aux":
            return mask_auxs(token_ids, pair_type)
        elif mask_strategy == "random":
            return mask_random(token_ids, frac=kwargs.get("frac", 0.15))
        else:
            raise NotImplementedError(f"mask strategy {mask_strategy} not implemented")

    in_sentences, index_map = read_lm_data(
        splits,
        data_name=data_name,
        filename_prefix=filename_prefix,
        data_dir=data_dir,
        # include_only_quest=(mask_strategy == "aux"),
    )
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    auxs = list(
        set(
            [
                sent.split("quest")[1].strip().split()[0]
                for sent in in_sentences
                if "quest" in sent
            ]
        )
    )  # ["doesn't", "does", "do", "don't"]
    aux_idxs = [in_vocab(w)[0] for w in auxs]
    quest_idx = in_vocab("quest")[0]
    decl_idx = in_vocab("decl")[0]
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        pair_types = ["decl" if "decl" in sent else "quest" for sent in in_subset]
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        masked_token_ids, target_ids = zip(
            *[
                mask(token_ids, pair_type)
                for token_ids, pair_type in zip(in_subset_tokenized, pair_types)
            ]
        )
        data = {
            "in": masked_token_ids,
            "out": target_ids,
            "in_len": in_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


def eval_lm_callback(
    lm,
    in_vocab,
    split,
    is_prefix_lm=False,
    data_name="question_formation_data",
    filename_prefix="question",
):
    def tokenizer(s):
        try:
            return [lm.encoder_sos] + in_vocab(s)
        except AttributeError:
            return [lm.sos] + in_vocab(s)

    sents, _ = read_lm_data(
        [split], data_name=data_name, filename_prefix=filename_prefix
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
    closing_words = ["doesn't", "does", "do", "don't"]
    closing_word_idxs = [in_vocab[w] for w in closing_words]
    out = out[:, closing_word_idxs]

    acc = [closing_words[i] == q_word for i, q_word in zip(out.argmax(dim=1), q_words)]
    agg_acc = sum(acc) / len(out)
    print(agg_acc)
    return agg_acc


def eval_lm_sent_prob_callback(lm, in_vocab, split):
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
        try:
            return [lm.model.encoder_sos] + in_vocab(s)
        except AttributeError:
            return [lm.model.sos] + in_vocab(s)

    batch_size = 32
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
            pos_score = lm(
                {"in": inp_pos_tokens.transpose(0, 1), "in_len": inp_pos_len}
            ).loss
            neg_score = lm(
                {"in": inp_neg_tokens.transpose(0, 1), "in_len": inp_neg_len}
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


def eval_cls_callback(cls_model, in_vocab, out_vocab, split):
    def tokenizer(s):
        return [cls_model.sos] + in_vocab(s)

    in_sents, out_auxs, _ = read_lm_cls_data([split])
    preds, targets = test_classification(
        tokenizer, out_vocab, cls_model, in_sents, out_auxs, 0
    )
    acc = sum(preds == targets) / len(preds)
    print(acc)
    return acc


def eval_mlm_callback(mlm, eval_dataset, in_vocab, split):
    def tokenizer(s):
        return in_vocab(s)

    quest_id = in_vocab("quest")[0]

    eval_dataset = eval_dataset.filter(lambda x: quest_id in x["in"])

    infill_idxs = [inp.index(quest_id) + 1 for inp in eval_dataset["in"]]
    out, gold_labels = test_infillings(
        mlm,
        eval_dataset["in"],
        eval_dataset["in_len"],
        eval_dataset["out"],
        infill_idxs=infill_idxs,
        gpu_id=0,
    )
    # gold_labels = [eval_dataset["out"][i][idx] for i, idx in enumerate(infill_idxs)]
    acc = [out[i].argmax().item() == gold_labels[i].item() for i in range(len(out))]
    agg_acc = sum(acc) / len(out)
    print(agg_acc)
    return agg_acc


if __name__ == "__main__":
    dataset, in_vocab, in_sentences = build_datasets_lm()
