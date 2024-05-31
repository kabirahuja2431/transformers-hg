import argparse
import os
import json
import copy
import random
import numpy as np
import pandas as pd
from data_utils.lm_dataset_helpers import read_lm_data


def generate_order_rule_generalization_output(input_sent):
    auxs = ["do", "does", "doesn't", "don't"]
    input_tokens = input_sent.split()

    # According to the order rule, the auxiliary verb should be the first token in the output.
    # So, we will first find the auxiliary verb in the input sentence and then move it to the first position.
    # If there is no auxiliary verb in the input sentence, we will throw an error.

    output_tokens = ["" for _ in range(len(input_tokens))]
    first_aux, first_aux_idx = None, None
    for idx, token in enumerate(input_tokens):
        if token in auxs:
            first_aux = token
            first_aux_idx = idx
            break
    if first_aux is None:
        raise ValueError("No auxiliary verb found in the input sentence.")

    for idx in range(len(input_tokens)):
        if idx == 0:
            output_tokens[idx] = first_aux
        elif idx <= first_aux_idx:
            output_tokens[idx] = input_tokens[idx - 1]
        else:
            output_tokens[idx] = input_tokens[idx]

    return " ".join(output_tokens).replace(".", "?")


def get_subsample(in_sentences, index_map, subsample_size, data_type):
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    if data_type == "ambiguous":
        # Just select subsample_size sentences from `in_sentences`
        subsample_indices = np.random.choice(
            len(in_sentences), size=subsample_size, replace=False
        )
        subsample = [in_sentences[idx] for idx in subsample_indices]
        return subsample

    elif data_type == "hr_gen":
        # Select subsample_size/2 sentences from train and test split of `in_sentences` each.
        train_sentences = get_subset(in_sentences, index_map["train"])
        test_sentences = get_subset(in_sentences, index_map["test"])
        train_subsample_indices = np.random.choice(
            len(train_sentences), size=subsample_size // 2, replace=False
        )
        test_subsample_indices = np.random.choice(
            len(test_sentences), size=subsample_size // 2, replace=False
        )
        train_subsample = [train_sentences[idx] for idx in train_subsample_indices]
        test_subsample = [test_sentences[idx] for idx in test_subsample_indices]
        subsample = train_subsample + test_subsample
        # Shuffle the subsample
        random.shuffle(subsample)
        return subsample

    elif data_type == "or_gen":
        # First select subsample_size/2 sentences from train split of `in_sentences`.
        train_sentences = get_subset(in_sentences, index_map["train"])
        train_subsample_indices = np.random.choice(
            len(train_sentences), size=subsample_size // 2, replace=False
        )
        train_subsample = [train_sentences[idx] for idx in train_subsample_indices]

        # For test split, we will first create order rule generalization outputs for each input in in_sentences.
        test_sentences = get_subset(in_sentences, index_map["test"])
        test_inputs = [sent.split(" quest ")[0] for sent in test_sentences]
        test_or_gen_outputs = [
            generate_order_rule_generalization_output(sent) for sent in test_inputs
        ]
        test_orgen_sentences = [
            test_inputs[idx] + " quest " + test_or_gen_outputs[idx]
            for idx in range(len(test_inputs))
        ]

        # Then we will select subsample_size/2 sentences from the new test split.
        test_subsample_indices = np.random.choice(
            len(test_orgen_sentences), size=subsample_size // 2, replace=False
        )
        test_subsample = [test_orgen_sentences[idx] for idx in test_subsample_indices]
        subsample = train_subsample + test_subsample
        # Shuffle the subsample
        random.shuffle(subsample)
        return subsample

    else:
        raise ValueError("Invalid data_type.")


def convert_to_shortcut_grammar_format(sentences):
    """
    The format is:
    `sentence1`\t`sentence2`
    """

    sent1s = []
    sent2s = []
    for sentence in sentences:
        sent1s.append(sentence.split(" quest ")[0])
        sent2s.append(sentence.split(" quest ")[1])
        # shortcut_grammar_inp = {
        #     "a": {"text": sentence.split(" quest ")[0]},
        #     "b": {"text": sentence.split(" quest ")[1]},
        # }
        # shortcut_grammar_inps.append(shortcut_grammar_inp)
    # return shortcut_grammar_inps

    df = pd.DataFrame({"sentence1": sent1s, "sentence2": sent2s})
    return df

def save(df, out_file):
    dir_name = os.path.dirname(out_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df.to_csv(out_file, index=False, sep="\t")

def save_to_json(shortcut_grammar_inps, out_file):
    dir_name = os.path.dirname(out_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(out_file, "w") as f:
        json.dump(shortcut_grammar_inps, f, indent=4)


def main(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    in_sentences, index_map = read_lm_data(
        splits=["train"] if args.type == "ambiguous" else ["train", "test"],
        include_only_quest=True,
    )

    # Get the subsample
    subsample = get_subsample(in_sentences, index_map, args.subsample_size, args.type)

    # Convert in the form used by `ShorcutGrammar`
    sg_form_inps = convert_to_shortcut_grammar_format(subsample)

    # Save to json
    # save_to_json(
    #     sg_form_inps,
    #     out_file=f"{os.getcwd()}/ShortcutGrammar/data/question/{args.type}_{args.subsample_size}.json",
    # )
    
    save(sg_form_inps, f"{os.getcwd()}/ShortcutGrammar/data/qf_{args.type}/train.tsv")
    save(sg_form_inps.sample(frac=0.1, random_state=args.seed, replace=False), f"{os.getcwd()}/ShortcutGrammar/data/qf_{args.type}/dev.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="ambiguous",
        choices=["ambiguous", "hr_gen", "or_gen"],
    )
    parser.add_argument("--subsample_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
