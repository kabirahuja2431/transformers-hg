import os
import copy
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

CFG_DIR = "cfgs/"
QUES_FORM_DATA_DIR = "data_utils/question_formation_data"
DATA_DIR = "data_utils/"


def load_question_formation_data(
    split, include_identity=True, exclude_simple_interrogatives=False
):
    filename = f"{QUES_FORM_DATA_DIR}/question.{split}"

    with open(filename) as f:
        lines = f.read().split("\n")
        if lines[-1] == "":
            lines = lines[:-1]

    in_sents, out_sents = [], []
    for line in lines:
        in_sent, out_sent = line.split("\t")
        if not include_identity and "quest" not in in_sent:
            continue
        in_sent = " ".join(in_sent.split()[:-1])
        auxs = ["do", "does", "doesn't", "don't"]
        num_auxs = len(set([aux for aux in auxs if aux in in_sent.split()]))

        if exclude_simple_interrogatives and num_auxs < 2:
            continue
        #         if not include_task_token:
        #             in_sent = in_sent.replace("quest", "").replace("decl", "").strip()
        in_sents.append(in_sent)
        out_sents.append(out_sent)

    return pd.DataFrame({"input": in_sents, "output": out_sents})


def token_seq_to_type_seq(token_seq, token2tag):
    return " ".join(
        [token2tag.get(token, token) for token in token_seq.split()]
    ).strip()


def main(args):
    assert args.num_types % 2 == 0

    np.random.seed(args.seed)

    # Load Type to token Map
    type_to_token_map = defaultdict(list)
    token2tag = {}
    with open(f"{CFG_DIR}/tag_token_map.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            type_, token = line.split("\t")
            type_to_token_map[type_].append(token)
            token2tag[token] = type_
    token2tag["."] = ""
    token2tag["?"] = ""

    train_quest_form_data = load_question_formation_data(
        "train", include_identity=False
    )
    train_quest_form_data_with_decl = load_question_formation_data(
        "train", include_identity=True
    )
    val_quest_form_data = load_question_formation_data(
        "val", include_identity=False, exclude_simple_interrogatives=True
    )
    test_quest_form_data = load_question_formation_data("test", include_identity=False)

    token_seq_to_type_seq_fn = lambda token_seq: token_seq_to_type_seq(
        token_seq, token2tag
    )
    input_types = (
        train_quest_form_data_with_decl["input"]
        .apply(token_seq_to_type_seq_fn)
        .values.tolist()
    )

    output_types = (
        train_quest_form_data_with_decl["output"]
        .apply(token_seq_to_type_seq_fn)
        .values.tolist()
    )

    input_type2output_types = {}

    for inp, out in zip(input_types, output_types):
        if out.startswith("aux"):
            input_type2output_types[inp] = out
        else:
            if inp not in input_type2output_types:
                input_type2output_types[inp] = ""

    input_types = set(input_types)
    output_types = set(output_types)

    all_train_types = input_types.union(output_types)

    types_list = []

    for inp, out in input_type2output_types.items():
        if out != "":
            types_list += [inp, out]

    ques_cutoff = len(types_list)

    for inp, out in input_type2output_types.items():
        if out == "":
            types_list += [inp]

    final_types_list = types_list[: args.num_types]
    sentence_pairs = []
    while len(sentence_pairs) < args.num_samples:
        for idx in range(0, len(final_types_list), 2):
            inp = final_types_list[idx]
            out = final_types_list[idx + 1]

            inp_types = inp.split()
            inp_tokens = copy.copy(inp_types)
            for i, type_ in enumerate(inp_types):
                if type_ in type_to_token_map:
                    inp_tokens[i] = np.random.choice(type_to_token_map[type_])

            if out.startswith("aux"):
                # Align output to input so same substitutions can be used
                out_types = out.split()
                out_id_to_inp_id = [0 for _ in range(len(out_types))]
                # First token is always the auxiliary, it will be aligned to the first token that differs between the input and output[1:]
                for idx, (inp_type, out_type) in enumerate(
                    zip(inp_types, out_types[1:])
                ):
                    if inp_type != out_type:
                        out_id_to_inp_id[0] = idx
                        break
                    else:
                        out_id_to_inp_id[idx + 1] = idx
                # Align the rest of the tokens
                for idx in range(out_id_to_inp_id[0] + 1, len(out_types)):
                    out_id_to_inp_id[idx] = idx
                out_tokens = [
                    inp_tokens[out_id_to_inp_id[idx]] for idx in range(len(out_types))
                ]

                sentence_pair = " ".join(inp_tokens) + " quest " + " ".join(out_tokens)
                sentence_pairs.append(sentence_pair)

            else:
                out_types = inp.split()
                out_tokens = copy.copy(out_types)
                for i, type_ in enumerate(out_types):
                    if type_ in type_to_token_map:
                        out_tokens[i] = np.random.choice(type_to_token_map[type_])

                sentence_pairs += [
                    " ".join(inp_tokens) + " decl " + " ".join(inp_tokens),
                    " ".join(out_tokens) + " quest " + " ".join(out_tokens),
                ]

    # Split into train and test
    sentence_pairs = np.random.permutation(sentence_pairs)
    n_train = int(args.train_frac * len(sentence_pairs))
    train_pairs = sentence_pairs[:n_train]
    test_pairs = sentence_pairs[n_train:]

    data_dir = f"{DATA_DIR}/cfg_gen_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(f"{data_dir}/cfg_{args.num_types}_types.train", "w") as f:
        f.write("\n".join(train_pairs))
    with open(f"{data_dir}/cfg_{args.num_types}_types.val", "w") as f:
        f.write("\n".join(test_pairs))

    test_pairs = []
    for _, row in test_quest_form_data.iterrows():
        test_pairs.append(
            row.input.replace(".", "").strip()
            + " quest "
            + row.output.replace(".", "").replace("?", "").strip()
        )

    with open(f"{data_dir}/cfg_{args.num_types}_types.test", "w") as f:
        f.write("\n".join(test_pairs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_types", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--train_frac", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
