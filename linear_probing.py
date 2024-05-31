import os
import numpy as np
import random
from tqdm import tqdm
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import (
    build_datasets_lm,
    build_datasets_enc_dec,
    build_datasets_tense_inflection,
    build_datasets_ti_enc_dec,
    build_datasets_lm_cls,
    build_datasets_grammar_gen,
)
from data_utils.tense_inflection_helpers import (
    build_datasets_ti_cls,
    build_datasets_ti_mlm,
)
from transformer_helpers import *
from data_utils.tense_inflection_helpers import sent_to_pos


token2type = {}
with open("cfgs/tag_token_map.txt", "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        tag, token = line.strip().split("\t")
        token2type[token] = tag


def sent_to_pos_with_morph(sent):

    tokens = sent.split(" ")

    tags = [token2type.get(token, token) for token in tokens]

    return tags


def get_base_transformer_model(
    args, in_vocab, out_vocab, num_roles=None, model_name=None
):
    model = create_model(
        len(in_vocab),
        len(out_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_n_layers,
        args.decoder_n_layers,
        mode=args.mode,
        tied_embedding=args.tied_embedding,
        dropout=args.dropout,
    )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    interface = create_model_interface(model, label_smoothing=args.label_smoothing)
    return model, interface


def get_base_transformer_lm(args, in_vocab, model_name=None):

    try:
        model = create_lm(
            len(in_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            mode=args.mode,
            use_pos_embeddig=not args.no_pos_enc,
            pos_scale=args.pos_scale,
            gated_model=args.gated_model,
            dropout=args.dropout,
            tied_embedding=args.tied_embedding,
        )
    except AttributeError:
        model = create_lm(
            len(in_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            tied_embedding=args.tied_embedding,
        )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    try:
        interface = create_model_interface(
            model,
            is_lm=True,
            is_null_encoder=(args.mode != "enc_dec"),
            label_smoothing=args.label_smoothing,
        )
    except AttributeError:
        interface = create_model_interface(
            model,
            is_lm=True,
            is_null_encoder=False,
            # label_smoothing=args.label_smoothing,
        )
    return model, interface


def get_hidden_states_single(model, input_ids, layer_id=-1):
    model.eval()

    src = [model.encoder_sos] + input_ids
    src = torch.tensor(src).unsqueeze(0)

    with torch.no_grad():
        hidden_states = model.encoder_only(
            src, mask=torch.zeros(src.size(0), src.size(1)).bool(), layer_id=layer_id
        )
    return hidden_states


def get_model_hidden_states(model, input_ids_batch, layer_id=-1):
    model.eval()
    hidden_states_batch = []

    for input_ids in tqdm(input_ids_batch):
        hidden_states = get_hidden_states_single(model, input_ids, layer_id=-1)
        hidden_states_batch.append(hidden_states)
    return hidden_states_batch


def get_input_from_sent_pair(sent_pair):

    return (
        sent_pair.split("quest")[0]
        .split("decl")[0]
        .split("PRESENT")[0]
        .split("PAST")[0]
        .replace(".", "")
        .strip()
    )


def tag_single_sent_data(sentence, hidden_states, sent_tag_fn):
    in_sent = get_input_from_sent_pair(sentence)
    tokens = in_sent.split(" ")
    tags = sent_tag_fn(in_sent)

    token_wise_hs = []
    labels = []

    for i, tag in enumerate(tags):
        token_wise_hs.append(hidden_states[0, i + 1].cpu().numpy())
        labels.append(tag)

    return token_wise_hs, labels, tokens


def tag_data(sentences, all_hidden_states, sent_tag_fn):

    all_token_wise_hs = []
    all_labels = []
    all_tokens = []
    for i, sentence in enumerate(sentences):
        hidden_states = all_hidden_states[i]
        token_wise_hs, labels, tokens = tag_single_sent_data(
            sentence, hidden_states, sent_tag_fn
        )
        all_token_wise_hs.extend(token_wise_hs)
        all_labels.extend(labels)
        all_tokens.extend(tokens)

    return np.array(all_token_wise_hs), np.array(all_labels), all_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="lm",
        choices=["lm", "seq2seq"],
        help="Model type, lm or seq2seq",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tense",
        choices=["lm", "tense"],
        help="Dataset type: lm or tense",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=512,
        help="Dimensionality of the embeddings and hidden states",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of layers in the encoder and decoder",
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--tied_embedding",
        action="store_true",
        help="Whether to use tied embeddings in the encoder and decoder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the pretrained model checkpoint",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to select for training",
    )
    parser.add_argument(
        "--layer_id", type=int, default=0, help="Layer ID for extracting hidden states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="probing_results/",
        help="Output directory to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "lm":
        if args.model_type == "lm":
            datasets, in_vocab, in_sentences = build_datasets_lm()
        else:
            datasets, in_vocab, in_sentences, _ = build_datasets_enc_dec()

    elif args.dataset == "tense":
        if args.model_type == "lm":
            datasets, in_vocab, in_sentences = build_datasets_tense_inflection()
        else:
            datasets, in_vocab, in_sentences, _ = build_datasets_ti_enc_dec()

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_type}_{model_name}_{args.layer_id}",
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.model_type == "lm":
        model = create_lm(
            len(in_vocab),
            vec_dim=args.n_embd,
            n_heads=args.n_heads,
            encoder_n_layers=args.n_layers,
            mode="enc_dec",
            tied_embedding=args.tied_embedding,
        )
    else:
        model = create_model(
            len(in_vocab),
            len(in_vocab),
            vec_dim=args.n_embd,
            n_heads=args.n_heads,
            encoder_n_layers=args.n_layers,
            decoder_n_layers=args.n_layers,
            mode="enc_dec",
            tied_embedding=args.tied_embedding,
        )

    if args.model_path:
        print("Loading pretrained model from:", args.model_path)
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device("cpu"))
        )

    input_ids = datasets["train"]["in"]
    random_idxs = random.sample(range(len(input_ids)), args.n_samples)
    input_ids_batch = [input_ids[idx] for idx in random_idxs]
    in_sentences_batch = [in_sentences[idx] for idx in random_idxs]

    train_hidden_states = get_model_hidden_states(
        model, input_ids_batch, layer_id=args.layer_id
    )

    X_v1, y_v1, tokens_v1 = tag_data(
        in_sentences_batch, train_hidden_states, sent_to_pos
    )
    X_v2, y_v2, tokens_v2 = tag_data(
        in_sentences_batch, train_hidden_states, sent_to_pos_with_morph
    )

    tokens_unq = list(set(tokens_v1))
    test_tokens = random.sample(tokens_unq, int(0.2 * len(tokens_unq)))
    train_tokens = [token for token in tokens_unq if token not in test_tokens]

    train_idxs_v1 = [i for i, token in enumerate(tokens_v1) if token in train_tokens]
    train_idxs_v2 = [i for i, token in enumerate(tokens_v2) if token in train_tokens]

    test_idxs_v1 = [i for i, token in enumerate(tokens_v1) if token in test_tokens]
    test_idxs_v2 = [i for i, token in enumerate(tokens_v2) if token in test_tokens]

    X_train_v1 = X_v1[train_idxs_v1]
    X_train_v2 = X_v2[train_idxs_v2]

    y_train_v1 = [y_v1[i] for i in train_idxs_v1]
    y_train_v2 = [y_v2[i] for i in train_idxs_v2]

    X_test_v1 = X_v1[test_idxs_v1]
    X_test_v2 = X_v2[test_idxs_v2]

    y_test_v1 = [y_v1[i] for i in test_idxs_v1]
    y_test_v2 = [y_v2[i] for i in test_idxs_v2]

    v1_model = RidgeClassifier()
    v1_model.fit(X_train_v1, y_train_v1)
    v1_train_acc = v1_model.score(X_train_v1, y_train_v1)
    v1_test_acc = v1_model.score(X_test_v1, y_test_v1)
    print(f"Train Accuracy according to POS labeling: {v1_train_acc}")
    print(f"Test Accuracy according to POS labeling: {v1_test_acc}")

    conf_mat = confusion_matrix(y_test_v1, v1_model.predict(X_test_v1))
    fig = plt.figure(figsize=(8, 4))
    sns.heatmap(
        conf_mat,
        annot=True,
        xticklabels=sorted(set(y_test_v1)),
        yticklabels=sorted(set(y_test_v1)),
    )
    plt.savefig(os.path.join(output_dir, "confusion_matrix_pos.png"))
    plt.close()

    v2_model = RidgeClassifierCV()
    v2_model.fit(X_train_v2, y_train_v2)
    v2_train_acc = v2_model.score(X_train_v2, y_train_v2)
    v2_test_acc = v2_model.score(X_test_v2, y_test_v2)
    print(f"Train Accuracy according to POS and Morph labeling: {v2_train_acc}")
    print(f"Test Accuracy according to POS and Morph labeling: {v2_test_acc}")

    conf_mat = confusion_matrix(y_test_v2, v2_model.predict(X_test_v2))
    conf_mat = conf_mat.round(2)
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(
        conf_mat,
        annot=True,
        xticklabels=sorted(set(y_test_v2)),
        yticklabels=sorted(set(y_test_v2)),
    )
    plt.savefig(os.path.join(output_dir, "confusion_matrix_pos_morph.png"))
    plt.close()

    with open(os.path.join(output_dir, "accuracies.txt"), "w") as f:
        f.write(f"Train Accuracy according to POS labeling: {v1_train_acc}\n")
        f.write(f"Test Accuracy according to POS labeling: {v1_test_acc}\n")
        f.write(f"Train Accuracy according to POS and Morph labeling: {v2_train_acc}\n")
        f.write(f"Test Accuracy according to POS and Morph labeling: {v2_test_acc}\n")


if __name__ == "__main__":
    main()
