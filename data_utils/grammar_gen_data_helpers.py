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
from scipy.spatial.distance import cosine
from parse_q_and_tense import parse_question, parse_tense, convert_to_parse
from tree_projections import (
    get_tree_projection,
    get_parsing_accuracy,
    get_sparsity_scores_helper,
)
from data_utils.tense_inflection_helpers import sent_to_pos


DATA_DIR = os.path.join(os.getcwd(), "data_utils")


def process(line):
    return line.replace("\t", " ").strip()


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
                    sents = [
                        sent
                        for sent in sents
                        if "P" not in sent_to_pos(sent) and "R" not in sent_to_pos(sent)
                    ]

                for sent in sents:
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(sent)

        else:
            if split != "test":
                with open(f"{DATA_DIR}/grammar_gen_data/{grammar}.{split}") as reader:
                    if do_process:
                        sents = [process(line) for line in reader.readlines()]
                    else:
                        sents = [line.strip() for line in reader.readlines()]

                    if include_simple_sents_only:
                        sents = [
                            sent
                            for sent in sents
                            if "P" not in sent_to_pos(sent)
                            and "R" not in sent_to_pos(sent)
                        ]

                    for sent in sents:
                        index_map[split].append(len(in_sentences))
                        in_sentences.append(sent)

            with open(f"{DATA_DIR}/grammar_gen_data/{grammar_tgt}.{split}") as reader:
                if do_process:
                    sents = [process(line) for line in reader.readlines()]
                else:
                    sents = [line.strip() for line in reader.readlines()]

                for sent in sents:
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(sent)

    return in_sentences, index_map


def build_datasets_grammar_gen(grammar, grammar_tgt=None):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "test"]
    in_sentences, index_map = read_grammar_gen_data(
        splits, grammar, grammar_tgt=grammar_tgt
    )
    print(f"Number of sentences: {len(in_sentences)}")
    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(sent) for sent in in_subset]
        in_lens = [len(sent) for sent in in_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


def get_gold_parse(sent):
    parse = convert_to_parse(sent, parse_question(sent))
    return parse


def compute_sci(model, in_sentences, tokenizer, gold_parses):
    total_sci_score = 0.0
    pred_parses = []
    batch_size = 1
    st = 0

    with tqdm(total=len(in_sentences)) as progress_bar:
        while st < len(in_sentences):
            en = min(len(in_sentences), st + batch_size)
            output = get_tree_projection(
                in_sentences[st:en][0],
                model,
                tokenizer,
                st_threshold=0,
                verbose=True,
                sim_fn="cosine",
                normalize=True,
                layer_id=-1,
                is_lm=True,
            )
            pred_parses += [output["pred_parse"]]
            total_sci_score += np.sum([output["pred_parse_score"]])
            progress_bar.update(en - st)
            st = en
        # break

    score = total_sci_score / len(in_sentences)
    parsing_acc = get_parsing_accuracy(pred_parses, gold_parses)["f1"]

    return score, parsing_acc, pred_parses, gold_parses


def compute_lsci(model, test_dataset):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstruct_scores = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_dataset)) as pbar:
            for i in range(len(test_dataset)):
                example = torch.tensor(test_dataset[i]["in"]).unsqueeze(0)
                example = example.to(device)
                example_len = (
                    torch.tensor(test_dataset[i]["in_len"]).unsqueeze(0).to(device)
                )

                init_state = {}
                for idx in range(model.trafo.num_encoder_layers):
                    init_state[idx] = torch.zeros(
                        1, example.shape[1], model.state_size
                    ).to(device)
                state = State(step=0, state=init_state)
                patched_reps = torch.zeros(1, example.shape[1], model.state_size).to(
                    device
                )
                for step in range(example.shape[1]):
                    data = model.pos_embed(model.input_embed(example), 0)[
                        :, step : step + 1
                    ]
                    out = model.trafo.encoder.one_step_forward(state, data)
                    patched_reps[:, step] = out[:, -1]

                    # keep only the last step of the state
                    for k, v in state.state.items():
                        state.state[k][:, 0] = v[:, min(step, state.step)]
                    state.step = 1

                # Get original representations
                data = model.pos_embed(model.input_embed(example), 0)
                in_len_mask = model.generate_len_mask(data.shape[1], example_len)
                og_reps = model.trafo.get_hidden_states(
                    data, src_length_mask=in_len_mask, is_lm=True, layer_id=-1
                )
                cos_sims = [
                    1
                    - cosine(
                        patched_reps[0, i].detach().to("cpu").numpy(),
                        og_reps[0, i].detach().to("cpu").numpy(),
                    )
                    for i in range(patched_reps.shape[1])
                ]
                cos_sim = np.mean(cos_sims[2:])
                lstruct_scores.append(cos_sim)
                pbar.set_postfix({"cos_sim": np.mean(lstruct_scores)})
                pbar.update(1)
    return np.mean(lstruct_scores)


def tree_structuredness_callback(grammar, lm, in_vocab, split):
    def tokenizer(s, add_special_tokens=True):
        if add_special_tokens:
            return [lm.encoder_sos] + in_vocab(s)
        else:
            return in_vocab(s)

    in_sentences, _ = read_grammar_gen_data([split], grammar=grammar)
    gold_parses = [get_gold_parse("{} . quest".format(sent)) for sent in in_sentences]

    score, parsing_acc, pred_parses, gold_parses = compute_sci(
        lm, in_sentences, tokenizer, gold_parses
    )

    return score


def linear_structuredness_callback(lm, datasets, split):
    score = compute_lsci(lm, datasets[split])
    return score
