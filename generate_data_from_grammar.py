import argparse
import copy
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from cfg import CFG

CFG_DIR = "cfgs/"
DATA_DIR = "data_utils/"


def main(args):
    np.random.seed(args.seed)

    grammar = CFG(filename=CFG_DIR + f"{args.grammar}.gr")

    # Generate sentence types
    sentence_types = grammar.generate()
    sentence_types = list(set(sentence_types))

    # Load Type to token Map
    type_to_token_map = defaultdict(list)
    with open(f"{CFG_DIR}/tag_token_map.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            type_, token = line.split("\t")
            type_to_token_map[type_].append(token)

    # Generate sentences
    sentences = set()
    while len(sentences) < args.n_samples:
        sentence_type = np.random.choice(sentence_types)
        types = sentence_type.split()
        tokens = copy.copy(types)
        for i, type_ in enumerate(types):
            if type_ in type_to_token_map:
                tokens[i] = np.random.choice(type_to_token_map[type_])
        sentence = " ".join(tokens)
        sentences.add(sentence)

    sentences = list(sentences)

    # Split into train and test
    n_train = int(args.train_frac * len(sentences))
    sentences = np.random.permutation(sentences)
    train_sentences = sentences[:n_train]
    test_sentences = sentences[n_train:]

    # Save
    data_dir = f"{DATA_DIR}/grammar_gen_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(f"{data_dir}/{args.grammar}.train", "w") as f:
        f.write("\n".join(train_sentences))
    with open(f"{data_dir}/{args.grammar}.test", "w") as f:
        f.write("\n".join(test_sentences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grammar", type=str, default="decl_simple_rlg")
    parser.add_argument("-n", "--n_samples", type=int, default=10000)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
