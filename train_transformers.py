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
from data_utils.tense_inflection_helpers import (
    build_datasets_ti_cls,
    build_datasets_ti_mlm,
)
from transformer_helpers import *
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
from data_utils.tense_inflection_helpers import (
    eval_callback_tense_inflection,
    eval_ti_cls_callback,
    eval_ti_mlm_callback,
)
from data_utils.simple_agreement_helpers import (
    build_datasets_simple_agreement,
    build_datasets_simple_agreement_cls,
    eval_callback_simple_agreement,
    eval_all_agreements,
    eval_cls_callback_main_verb,
)
from data_utils.cogs_helpers import (
    build_datasets_cogs_lm,
    build_datasets_cogs_enc_dec,
    eval_callback_cogs,
)

from data_utils.qf_de_helpers import (
    build_datasets_quest_de_lm,
    build_datasets_quest_de_enc_dec,
    build_datasets_quest_de_cls,
    eval_qf_de_lm_callback,
    eval_qf_de_cls_callback,
)
from data_utils.passiv_helpers import (
    build_datasets_passivization,
    build_datasets_passivization_enc_dec,
    build_datasets_passivization_cls,
    eval_callback_passivization,
    eval_passivization_cls_callback,
)

WANDB_ENTITY_NAME = "<Insert-Your-Entity-Name>"


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
            is_prefix_lm=args.is_prefix_lm,
        )
    except AttributeError:
        interface = create_model_interface(
            model,
            is_lm=True,
            is_null_encoder=False,
            # label_smoothing=args.label_smoothing,
        )
    return model, interface


def get_base_transformer_cls(args, in_vocab, out_vocab, model_name=None):
    try:
        model = create_cls(
            len(in_vocab),
            len(out_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            use_pos_embeddig=not args.no_pos_enc,
            causal_encoder=args.causal_encoder,
        )
    except AttributeError:
        model = create_cls(
            len(in_vocab),
            len(out_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            causal_encoder=args.causal_encoder,
        )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    interface = create_model_interface(model, is_cls=True, is_lm=False)
    return model, interface


def main_lm(args):
    out_vocab = None
    if args.dataset == "dyck":
        datasets, in_vocab, _ = build_datasets_dyck(vocab=args.dyck_vocab)
    elif args.dataset == "tense":
        if args.mode != "enc":
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
        else:
            (
                datasets,
                in_vocab,
                out_vocab,
                in_sentences,
                out_mvs,
            ) = build_datasets_ti_cls()
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
        if args.mode != "enc":
            datasets, in_vocab, in_sentences = build_datasets_simple_agreement(
                args.grammar,
                eval_keys=args.eval_keys.split(",") if args.eval_keys != "" else [],
                include_simple_sents_only=args.pretrain,
                grammar_tgt=args.grammar_tgt if args.grammar_tgt != "" else None,
            )
        else:
            datasets, in_vocab, out_vocab, in_sentences, out_verbs = (
                build_datasets_simple_agreement_cls(args.grammar)
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
        if args.disamb_num == 0:
            filename = "question"
        else:
            filename = (
                "question_disamb"
                if args.disamb_num == -1
                else f"question_disamb_{args.disamb_num}"
            )
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix=filename,
        )
    elif args.dataset == "qf_disamb_order":
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix="question_disamb_order",
        )
    elif args.dataset == "qf_w_qid":
        if not args.not_lm:
            datasets, in_vocab, in_sentences = build_datasets_lm(
                filename_prefix="question_all_pairs",
            )
        else:
            datasets, in_vocab, in_sents, out_sents = build_datasets_enc_dec(
                filename_prefix="question_all_pairs",
            )
    elif "question_D" in args.dataset:
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix=args.dataset, test_filename_prefix="question"
        )

    elif args.dataset == "question_de":
        if args.mode != "enc":
            if not args.not_lm:
                datasets, in_vocab, in_sentences = build_datasets_quest_de_lm()
            else:
                datasets, in_vocab, in_sents, out_sents = (
                    build_datasets_quest_de_enc_dec()
                )
        else:
            datasets, in_vocab, out_vocab, in_sentences, out_auxs = (
                build_datasets_quest_de_cls()
            )

    elif args.dataset == "passiv":
        if args.mode != "enc":
            if not args.not_lm:
                datasets, in_vocab, in_sentences = build_datasets_passivization()
            else:
                datasets, in_vocab, in_sents, out_sents = (
                    build_datasets_passivization_enc_dec()
                )
        else:
            datasets, in_vocab, out_vocab, in_sentences, out_auxs = (
                build_datasets_passivization_cls()
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
        else:
            (
                datasets,
                in_vocab,
                out_vocab,
                in_sentences,
                out_auxs,
            ) = build_datasets_lm_cls()

    if args.mode != "enc":
        if not args.not_lm:
            model, interface = get_base_transformer_lm(
                args, in_vocab, model_name=args.model_load_path
            )
        else:
            model, interface = get_base_transformer_model(
                args, in_vocab, in_vocab, model_name=args.model_load_path
            )
    else:
        assert out_vocab is not None
        model, interface = get_base_transformer_cls(
            args, in_vocab, out_vocab, model_name=args.model_load_path
        )
    if len(args.save_dir) > 0:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.callback:
        if (
            args.dataset in ["lm", "qf_disamb", "qf_disamb_order", "qf", "qf_w_qid"]
            or "question_D" in args.dataset
        ):
            if args.mode != "enc":
                if not args.not_lm:
                    callback_fn = {
                        "aux": lambda split: eval_lm_callback(
                            model, in_vocab, split, is_prefix_lm=args.is_prefix_lm
                        ),
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

        elif args.dataset == "question_de":
            if args.mode != "enc":
                callback_fn = {
                    "aux": lambda split: eval_qf_de_lm_callback(model, in_vocab, split)
                }
            else:
                assert out_vocab is not None
                callback_fn = {
                    "aux": lambda split: eval_qf_de_cls_callback(
                        model, in_vocab, out_vocab, split
                    )
                }

        elif args.dataset == "passiv":
            if args.mode != "enc":
                callback_fn = {
                    "aux": lambda split: eval_callback_passivization(
                        model, in_vocab, split
                    )
                }
            else:
                callback_fn = {
                    "aux": lambda split: eval_passivization_cls_callback(
                        model, in_vocab, out_vocab, split
                    )
                }

        elif args.dataset == "cfg_gen":
            callback_fn = lambda split: eval_lm_callback(
                model,
                in_vocab,
                split,
                data_name="cfg_gen_data",
                filename_prefix=f"cfg_{args.num_types}_types",
            )
        elif args.dataset == "tense":
            if args.mode != "enc":
                callback_fn = lambda split: eval_callback_tense_inflection(
                    model, in_vocab, split
                )
            else:
                assert out_vocab is not None
                callback_fn = lambda split: eval_ti_cls_callback(
                    model, in_vocab, out_vocab, split
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
            if args.mode != "enc":
                callback_fn = {
                    "main_verb_acc": lambda split: eval_callback_simple_agreement(
                        model,
                        in_vocab,
                        split,
                        args.grammar,
                        grammar_tgt=(
                            args.grammar_tgt if args.grammar_tgt != "" else None
                        ),
                    ),
                    "all_verb_agreement_acc": lambda split: eval_all_agreements(
                        model,
                        in_vocab,
                        split,
                        args.grammar,
                        grammar_tgt=(
                            args.grammar_tgt if args.grammar_tgt != "" else None
                        ),
                    ),
                }
            else:
                callback_fn = {
                    "main_verb_acc": lambda split: eval_cls_callback_main_verb(
                        model, args.grammar, in_vocab, out_vocab, split
                    )
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
        device = "cpu"
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
        elif args.dataset in ["question_de", "passiv"]:
            eval_keys = ["dev", "gen"]
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
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--dataset", type=str, default="lm")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_multiplier", default=4, type=int)
    parser.add_argument("--encoder_n_layers", type=int, default=6)
    parser.add_argument("--mode", type=str, default="enc_dec")
    parser.add_argument("--not_lm", action="store_true")
    parser.add_argument("--decoder_n_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=200000)
    parser.add_argument("--pos_scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-pos-enc",
        action="store_true",
        help="Whether to not use positional encoding",
    )
    # this is only used if args.mode == "enc"
    parser.add_argument(
        "--causal_encoder",
        action="store_true",
        help="Whether to use causal encoder",
    )
    # this is only used if args.dataset == pcfg
    parser.add_argument("--base_folder", type=str, default="m-pcfgset")
    parser.add_argument("--tree_transform", action="store_true")
    # this is only used if args.dataset == "lm"
    parser.add_argument(
        "--exclude_identity",
        action="store_true",
        help="Whether to only include sequences with 'quest' token in training data!",
    )
    parser.add_argument(
        "--train-on-compl-only",
        action="store_true",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Whether to pretrain the model on the LM task!",
    )
    parser.add_argument(
        "--gated-model",
        action="store_true",
        help="Whether to use the gated transformer model!",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Whether to use label smoothing in the loss!",
    )
    parser.add_argument("--tied-embedding", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--no_decay_lr", action="store_true", help="Whether to not decay lr!"
    )
    # this is only used if args.dataset == grammar_gen
    parser.add_argument(
        "--grammar", type=str, default="agreement_hr_v4_agreement_linear_v4"
    )

    # this is only used if args.dataset == grammar_genv2
    parser.add_argument("--grammar_tgt", type=str, default="")

    # this is only used if args.dataset == cfg_gen
    parser.add_argument("--num_types", type=int, default=50)

    # this argument is only used if args.dataset == cogs
    parser.add_argument("--gen_inputs_for_train", action="store_true")

    # this argument is only used if args.dataset == "qf_disamb"
    parser.add_argument(
        "--disamb_num",
        type=int,
        default=-1,
        help="Number of disambiguating examples to include in training data",
    )

    #### evaluating can be time consuming so we can do that later...
    parser.add_argument("--dyck_vocab", type=int, default=20)
    parser.add_argument("--callback", action="store_true")

    parser.add_argument("--eval_keys", type=str, default="")

    parser.add_argument("--is_prefix_lm", action="store_true")

    args = parser.parse_args()
    set_seed(args)
    ### NOTE: change this to your own wandb project and entity!
    wandb_logger = wandb.init(
        project="structural-grokking", entity=WANDB_ENTITY_NAME, config=vars(args)
    )
    # To work with wandb sweeps
    args = AttrDict((wandb_logger.config))

    if args.save_prefix != "":
        args.save_dir = f"{args.save_prefix}-encL{args.encoder_n_layers}-decL{args.decoder_n_layers}-LR{args.lr}-Nheads{args.n_heads}-EmbSize{args.vec_dim}-TiedEmb{args.tied_embedding}-Seq2Seq{args.not_lm}-Mode{args.mode}-PrefixLM{args.is_prefix_lm}-{args.seed}/"

    if args.save_dir != "":
        wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()

    main_lm(args)
