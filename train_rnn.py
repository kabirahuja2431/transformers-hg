import numpy as np
import random
import os
import torch
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from training_utils import *

import argparse
from data_utils import (
    build_datasets_lm,
    build_datasets_enc_dec,
    build_datasets_tense_inflection,
    build_datasets_ti_enc_dec,
    build_datasets_lm_cls,
    build_datasets_grammar_gen,
)

import torch.nn.functional as F
from data_utils.lm_dataset_helpers import (
    eval_lm_callback,
    eval_cls_callback,
    eval_lm_sent_prob_callback,
)
from data_utils.grammar_gen_data_helpers import (
    tree_structuredness_callback,
    linear_structuredness_callback,
)
from data_utils.tense_inflection_helpers import eval_callback_tense_inflection
from data_utils.simple_agreement_helpers import (
    build_datasets_simple_agreement,
    eval_callback_simple_agreement,
    eval_all_agreements,
)
from data_utils.cogs_helpers import (
    build_datasets_cogs_lm,
    build_datasets_cogs_enc_dec,
    eval_callback_cogs,
)
from models.rnn_lm import RNNLM
from models.rnn_enc_dec import RNNEncDec
from interfaces.rnn.lm_interface import RNNLMInterface
from interfaces.rnn.enc_dec_interface import RNNEncDecInterface


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


### Change this for your own system as appropriate
def working_dir():
    # breakpoint()
    dir_name = os.getcwd()
    # USER = os.environ["USER"]
    # dir_name = f"/scr/biggest"

    # def helper(dir_name):
    #     if os.path.exists(dir_name):
    #         sub_dir = "{}/{}/compositionality".format(dir_name, USER)
    #         if not os.path.exists(sub_dir):
    #             os.makedirs(sub_dir)
    #         return sub_dir
    #     else:
    #         return ""

    # try:
    #     return helper(dir_name)
    # except:
    #     dir_name = f"/scr/smurty/biggest"
    #     return helper(dir_name)
    return dir_name


def create_lm(
    in_vocab_size,
    vec_dim,
    encoder_n_layers,
    rnn_type="LSTM",
    dropout=0.1,
    tied_embedding=False,
):
    args = dict(
        embedding_init="xavier", scale_mode="opennmt", mode="lm", dropout=dropout
    )
    return RNNLM(
        rnn_type=rnn_type,
        n_input_tokens=in_vocab_size,
        state_size=vec_dim,
        num_layers=encoder_n_layers,
        tied_embedding=tied_embedding,
        **args,
    )


def create_enc_dec(
    in_vocab_size,
    out_vocab_size,
    vec_dim,
    encoder_n_layers,
    rnn_type="LSTM",
    dropout=0.1,
    tied_embedding=False,
):
    return RNNEncDec(
        rnn_type=rnn_type,
        n_input_tokens=in_vocab_size,
        n_out_tokens=out_vocab_size,
        state_size=vec_dim,
        num_layers=encoder_n_layers,
        tied_embedding=tied_embedding,
        dropout=dropout,
    )


def create_model_interface(model, label_smoothing=0.0):

    return RNNLMInterface(
        model,
        label_smoothing=label_smoothing,
    )


def get_rnn_lm(args, in_vocab, model_name=None):

    model = create_lm(
        in_vocab_size=len(in_vocab),
        vec_dim=args.vec_dim,
        encoder_n_layers=args.encoder_n_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        tied_embedding=args.tied_embedding,
    )

    if model_name:
        print(f"Loading model from {model_name}")
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    interface = create_model_interface(model, label_smoothing=args.label_smoothing)

    return model, interface


def get_rnn_enc_dec(args, in_vocab, out_vocab, model_name=None):
    model = create_enc_dec(
        in_vocab_size=len(in_vocab),
        out_vocab_size=len(out_vocab),
        vec_dim=args.vec_dim,
        encoder_n_layers=args.encoder_n_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        tied_embedding=args.tied_embedding,
    )
    
    if model_name:
        print(f"Loading model from {model_name}")
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    
    interface = RNNEncDecInterface(
        model,
        label_smoothing=args.label_smoothing,
    )
    return model, interface
    


def main_lm(args):
    out_vocab = None
    if args.dataset == "dyck":
        datasets, in_vocab, _ = build_datasets_dyck(vocab=args.dyck_vocab)
    elif args.dataset == "tense":
        if not args.not_lm:
            datasets, in_vocab, _ = build_datasets_tense_inflection(
                include_only_present=args.exclude_identity,
                include_only_past_and_simple_present=args.pretrain,
            )
        else:
            datasets, in_vocab, in_sents, out_sents = build_datasets_ti_enc_dec(
                include_only_present=args.exclude_identity,
                include_only_past_and_simple_present=args.pretrain,
            )

    elif args.dataset == "grammar_gen":
        datasets, in_vocab, _ = build_datasets_grammar_gen(args.grammar)
    elif args.dataset == "grammar_genv2":
        datasets, in_vocab, _ = build_datasets_grammar_gen(
            args.grammar,
            grammar_tgt=args.grammar_tgt if args.grammar != args.grammar_tgt else None,
        )
    elif args.dataset == "cfg_gen":
        datasets, in_vocab, in_sentences = build_datasets_lm(
            data_name="cfg_gen_data",
            filename_prefix=f"cfg_{args.num_types}_types",
            include_only_quest=args.exclude_identity,
            include_only_decls_nd_simpl_ques=args.pretrain,
        )
    elif args.dataset == "simple_agreement":
        datasets, in_vocab, in_sentences = build_datasets_simple_agreement(
            args.grammar,
            eval_keys=args.eval_keys.split(",") if args.eval_keys != "" else [],
            include_simple_sents_only=args.pretrain,
            grammar_tgt=args.grammar_tgt if args.grammar_tgt != "" else None,
        )
    elif args.dataset == "cogs":
        if not args.not_lm:
            datasets, in_vocab, in_sentences = build_datasets_cogs_lm(
                input_only=args.pretrain, gen_inputs_for_train=args.gen_inputs_for_train
            )
        else:
            datasets, in_vocab, in_sentences, out_sentences = (
                build_datasets_cogs_enc_dec()
            )

    elif args.dataset == "qf_disamb":
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix="question_disamb",
        )
    elif args.dataset == "qf_disamb_order":
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix="question_disamb_order",
        )
    elif "question_D" in args.dataset:
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix=args.dataset, test_filename_prefix="question"
        )

    else:
        if args.mode != "enc":
            if not args.not_lm:
                datasets, in_vocab, _ = build_datasets_lm(
                    include_only_quest=args.exclude_identity,
                    include_only_decls_nd_simpl_ques=args.pretrain,
                    include_only_complex_sents=args.train_on_compl_only,
                )
            else:
                datasets, in_vocab, in_sents, out_sents = build_datasets_enc_dec(
                    include_only_quest=args.exclude_identity,
                    include_only_decls_nd_simpl_ques=args.pretrain,
                )
                out_vocab = in_vocab
        else:
            (
                datasets,
                in_vocab,
                out_vocab,
                in_sentences,
                out_auxs,
            ) = build_datasets_lm_cls()

    if not args.not_lm:
        model, interface = get_rnn_lm(
            args,
            in_vocab,
            model_name=args.model_load_path if args.model_load_path != "" else None,
        )
    else:
        model, interface = get_rnn_enc_dec(
            args,
            in_vocab,
            out_vocab,
            model_name=args.model_load_path if args.model_load_path != "" else None,
        )

    if len(args.save_dir) > 0:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.callback:
        if (
            args.dataset in ["lm", "qf_disamb", "qf_disamb_order", "qf"]
            or "question_D" in args.dataset
        ):
            if args.mode != "enc":
                if not args.not_lm:
                    callback_fn = {
                        "aux": lambda split: eval_lm_callback(model, in_vocab, split),
                        # "sent_prob": lambda split: eval_lm_sent_prob_callback(
                        #     interface, in_vocab, split
                        # ),
                    }
                else:
                    callback_fn = lambda split: eval_lm_callback(model, in_vocab, split)
            else:
                assert out_vocab is not None
                callback_fn = lambda split: eval_cls_callback(
                    model, in_vocab, out_vocab, split
                )
        elif args.dataset == "cfg_gen":
            callback_fn = lambda split: eval_lm_callback(
                model,
                in_vocab,
                split,
                data_name="cfg_gen_data",
                filename_prefix=f"cfg_{args.num_types}_types",
            )
        elif args.dataset == "tense":
            callback_fn = lambda split: eval_callback_tense_inflection(
                model, in_vocab, split
            )
        elif args.dataset == "dyck":
            callback_fn = lambda split: eval_callback_dyck(model, in_vocab, split)
        elif args.dataset in ["grammar_gen", "grammar_genv2"]:
            callback_fn = {
                "linear_eval": lambda split: linear_structuredness_callback(
                    model, datasets, split
                ),
                "parse_eval": lambda split: tree_structuredness_callback(
                    args.grammar, model, in_vocab, split
                ),
            }
        elif args.dataset == "simple_agreement":
            callback_fn = {
                "main_verb_acc": lambda split: eval_callback_simple_agreement(
                    model,
                    in_vocab,
                    split,
                    args.grammar,
                    grammar_tgt=args.grammar_tgt if args.grammar_tgt != "" else None,
                ),
                "all_verb_agreement_acc": lambda split: eval_all_agreements(
                    model,
                    in_vocab,
                    split,
                    args.grammar,
                    grammar_tgt=args.grammar_tgt if args.grammar_tgt != "" else None,
                ),
            }
        elif args.dataset == "cogs":

            gen_file = None
            if args.save_dir != "":
                gen_file = os.path.join(args.save_dir, "gen.txt")
            else:
                gen_file = os.path.join(
                    working_dir(), f"cogs_lm{not args.not_lm}_gen.txt"
                )
            # Overwrite the file if it exists
            with open(gen_file, "w") as f:
                f.write("")
            callback_fn = {
                "em": lambda split: eval_callback_cogs(
                    model, in_vocab, split, generation_file=gen_file
                ),
            }
        else:
            raise Exception
    else:
        callback_fn = None

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device("cpu")

    model.to(device)

    if args.dataset in ["grammar_gen", "grammar_genv2"]:
        eval_keys = ["test"]

    if args.eval_keys == "":
        if args.dataset == "simple_agreement":
            if not args.pretrain:
                eval_keys = ["val", "g1_test", "g2_test"]
            else:
                eval_keys = ["val"]
        elif args.dataset == "cogs":
            if not args.gen_inputs_for_train:
                eval_keys = ["dev", "test", "gen", "gen-struct"]
            else:
                eval_keys = ["dev", "test"]
        else:
            eval_keys = ["val", "test"]
    else:
        eval_keys = args.eval_keys.split(",")
    train_loop(
        args,
        interface,
        datasets["train"],
        {key: datasets[key] for key in eval_keys},
        device,
        args.save_dir,
        in_vocab=in_vocab,
        callback_fn=callback_fn,
        eval_every=args.eval_every,
        max_steps=args.max_train_steps,
        train_batch_size=args.batch_size,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="lm")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--callback", action="store_true")
    parser.add_argument("--mode", type=str, default="enc_dec")

    # Model args
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--encoder_n_layers", type=int, default=2)
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU", "RNN_TANH", "RNN_RELU"],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tied_embedding", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--not_lm",
        action="store_true",
        help="Whether to train on the encoder-decoder task",
    )

    # Training args
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--no_decay_lr", action="store_true", help="Whether to not decay lr!"
    )

    # Dataset args
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Whether to pretrain the model on the LM task!",
    )
    parser.add_argument(
        "--exclude_identity",
        action="store_true",
        help="Whether to exclude identity sentences",
    )
    parser.add_argument(
        "--train_on_compl_only",
        action="store_true",
        help="Whether to train on complex sentences only",
    )
    parser.add_argument(
        "--gen_inputs_for_train",
        action="store_true",
        help="Whether to generate inputs for training",
    )
    parser.add_argument("--dyck_vocab", type=int, default=20)
    parser.add_argument("--eval_keys", type=str, default="")
    parser.add_argument("--grammar", type=str, default="simple_agreement")
    parser.add_argument("--grammar_tgt", type=str, default="")

    args = parser.parse_args()

    set_seed(args)

    wandb_logger = wandb.init(
        project="structural-grokking", entity="kabirahuja2431", config=vars(args)
    )
    args = AttrDict((wandb_logger.config))

    if args.save_dir != "":
        wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()

    main_lm(args)
