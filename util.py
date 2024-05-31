from transformer_helpers import create_model
import torch
from transformers import AutoTokenizer, RobertaForMaskedLM
from scipy.spatial import distance
import random
import numpy as np
import torch
import random
import collate

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn.functional as F


def run_lm_decoding(tokenizer, lm, prefixes, gpu_id, max_decoding_steps=50):
    data_collator = collate.VarLengthCollate(None)
    # max_decoding_steps = 50

    # the tokenizer just adds an <SOS> token to the front of the string
    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    batch_size = 128
    st = 0
    device = (
        torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    )
    decoded_sents = []
    while st < len(prefixes):
        en = min(len(prefixes), st + batch_size)
        cslice = prefixes[st:en]
        inputs, input_lens = tokenizer_helper(cslice)
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        with torch.no_grad():
            outputs = lm.run_greedy(inputs, input_lens, max_decoding_steps)
            preds = outputs["data"].argmax(axis=-1)
            out_lens = outputs["length"]
            for pred, out_len in zip(preds, out_lens):
                decoded_sents.append(pred[:out_len].tolist())
        st = en
    return decoded_sents


def test_continuations(
    tokenizer,
    lm,
    prefixes,
    gpu_id,
    get_attn_scores=False,
    attn_layer=-1,
    prefix_no_pos=False,
):
    data_collator = collate.VarLengthCollate(None)

    # the tokenizer just adds an <SOS> token to the front of the string
    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    batch_size = 32
    st = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = "cpu"

    if get_attn_scores:
        attn_flows_agg = []
        attn_avg_agg = []
    else:
        final_states = []
    while st < len(prefixes):
        en = min(len(prefixes), st + batch_size)
        cslice = prefixes[st:en]
        inputs, input_lens = tokenizer_helper(cslice)
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        with torch.no_grad():
            if get_attn_scores:
                outputs = lm.get_attention_sparsity(inputs, input_lens)
                attn_flows = get_attn_flows(outputs, en - st)
                attn_avg = get_average_attn(outputs, en - st, attn_layer)
                attn_flows_agg += attn_flows
                attn_avg_agg += attn_avg
            else:
                outputs = lm(
                    inputs,
                    input_lens,  # prefix_len=input_lens if prefix_no_pos else None
                )
                if outputs["data"].shape[1] != 1:
                    final_states += [
                        outputs["data"][idx][l - 1] for idx, l in enumerate(input_lens)
                    ]
                else:
                    final_states += [outputs["data"][idx][0] for idx in range(en - st)]

        st = en
    if get_attn_scores:
        return attn_flows_agg, attn_avg_agg
    else:
        final_states = torch.stack(final_states, dim=0)
        return F.softmax(final_states, dim=1)


def test_classification(
    in_tokenizer,
    out_tokenizer,
    cls_model,
    in_sents,
    out_auxs,
    gpu_id,
    cls_head_idx=None,
    return_logits=False,
):
    data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(slice, tokenizer):
        tokenized = [tokenizer(s) for s in slice]
        lens = [len(s) for s in tokenized]

        to_collate = [{"in": x, "in_len": y} for x, y in zip(tokenized, lens)]
        collated = data_collator(to_collate)
        lens = collated["in_len"].long()
        return collated["in"].transpose(0, 1), lens

    batch_size = 32
    st = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = "cpu"
    all_preds = []
    all_tgts = []
    all_logits = []
    while st < len(in_sents):
        en = min(len(in_sents), st + batch_size)
        inp_slice = in_sents[st:en]
        out_slice = out_auxs[st:en]
        inputs, input_lens = tokenizer_helper(inp_slice, in_tokenizer)
        cls_head_idx_batch = (
            torch.tensor(cls_head_idx[st:en]) + 1
            if cls_head_idx is not None
            else input_lens - 1
        )
        targets, _ = tokenizer_helper(out_slice, out_tokenizer)
        targets = targets.squeeze(1)
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        cls_head_idx_batch = cls_head_idx_batch.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = cls_model(inputs, input_lens, cls_head_idx=cls_head_idx_batch)
            all_preds += [
                outputs["logits"][idx].argmax().detach().cpu().numpy()
                for idx in range(len(outputs["logits"]))
            ]
            all_tgts += [
                targets[idx].detach().cpu().numpy() for idx in range(len(targets))
            ]
            all_logits += [
                outputs["logits"][idx].detach().cpu().numpy()
                for idx in range(len(outputs["logits"]))
            ]
        st = en

    if return_logits:
        return np.array(all_preds), np.array(all_tgts), all_logits
    else:
        return np.array(all_preds), np.array(all_tgts)


def test_infillings(mlm, input_ids, input_lens, target_ids, infill_idxs, gpu_id):
    data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(
        input_ids_slice, input_lens_slice, target_ids_slice, infill_idxs_slice
    ):
        inp_to_collate = [
            {"in": x, "in_len": y, "infill_id": z, "tgt": t}
            for x, y, t, z in zip(
                input_ids_slice, input_lens_slice, target_ids_slice, infill_idxs_slice
            )
        ]
        inp = data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        infill_idxs = inp["infill_id"].long()
        tgt = inp["tgt"].long()
        return inp["in"].transpose(0, 1), in_len, tgt.transpose(0, 1), infill_idxs

    batch_size = 64
    st = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = "cpu"
    final_states = []
    targets = []

    with tqdm(total=len(range(0, len(input_ids), batch_size))) as pbar:
        for st in range(0, len(input_ids), batch_size):
            en = min(len(input_ids), st + batch_size)
            input_ids_slice = input_ids[st:en]
            input_lens_slice = input_lens[st:en]
            tgt_slice = target_ids[st:en]
            infill_idxs_slice = infill_idxs[st:en]
            (
                input_ids_slice,
                input_lens_slice,
                tgt_slice,
                infill_idxs_slice,
            ) = tokenizer_helper(
                input_ids_slice, input_lens_slice, tgt_slice, infill_idxs_slice
            )
            input_ids_slice = input_ids_slice.to(device)
            input_lens_slice = input_lens_slice.to(device)
            with torch.no_grad():
                outputs = mlm(input_ids_slice, input_lens_slice)
                final_states += [
                    outputs["data"][idx][l] for idx, l in enumerate(infill_idxs_slice)
                ]
                targets += [
                    tgt_slice[idx][l] for idx, l in enumerate(infill_idxs_slice)
                ]
            pbar.update(1)

    return F.softmax(torch.stack(final_states, dim=0), dim=1), targets


def get_attn_flows(attn_list, bs):
    attn_flow = [attn_list[0][idx] for idx in range(bs)]
    for attn_mat in attn_list[1:]:
        attn_flow = [torch.matmul(attn_mat[idx], attn_flow[idx]) for idx in range(bs)]
    return attn_flow


def get_average_attn(attn_list, bs, layer):
    if layer != -1:
        return [attn_list[layer][idx] for idx in range(bs)]
    else:
        attn_avg = [attn_list[0][idx] for idx in range(bs)]
        for attn_mat in attn_list[1:]:
            attn_avg = [attn_avg[idx] + attn_mat[idx] for idx in range(bs)]
        return [x / len(attn_list) for x in attn_avg]


def get_gpt2_lm(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer


def test_continuations_gpt2(tokenizer, lm, prefixes, gpu_id):
    all_logits = []
    all_chars = "abcdefghijklmnopqrst"
    all_brackets = ["(" + c for c in all_chars] + [c + ")" for c in all_chars]
    for prefix in tqdm(prefixes):
        curr_logits = get_gpt2_pred_helper(prefix, lm, tokenizer, all_brackets, gpu_id)
        all_logits.append(curr_logits)
    return torch.cat(all_logits, dim=0)


def get_gpt2_pred_helper(prefix, model, tokenizer, all_brackets, gpu_id=-1):
    """
    model: GPT2Model
    tokenizer: GPT2tokenizer
    prefix: a dyck prefix to get predictions for
    """
    ### try out all brackets and collect probabilities of every possible ending

    curr_score = []
    all_continuations = [prefix + " {}".format(bracket) for bracket in all_brackets]
    ei = tokenizer(all_continuations, return_tensors="pt")
    if gpu_id != -1:
        device = torch.device("cuda:{}".format(gpu_id))
        ei = {key: val.to(device) for key, val in ei.items()}
    model.eval()
    with torch.no_grad():
        out = model(**ei)["logits"]
    return torch.tensor(
        [
            [
                out[idx][-3][ei["input_ids"][idx][-2]]
                + out[idx][-2][ei["input_ids"][idx][-1]]
                for idx, _ in enumerate(all_brackets)
            ]
        ]
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measure_elemwise_dist(distance_fn):
    def measure_dist(m1, m2):
        if m1.ndim == 2:
            assert len(m1) == len(m2)
            return [measure_dist(m1[idx], m2[idx]) for idx in range(len(m1))]
        elif distance_fn == distance.cosine:
            return distance_fn(m1, m2)
        else:
            return distance_fn(m1, m2)

    return measure_dist


def get_masking_info(tokenizer, input_strs, fn, **kwargs):
    masked_strs = []
    curr = 0
    sentence2idx_tuple = []
    input_masks = []

    for inp in input_strs:
        input_dict = fn(inp, tokenizer, **kwargs)
        curr_keys = [k for k in input_dict]

        masked_strs += [inp] * len(input_dict)
        input_masks += [input_dict[key] for key in curr_keys]

        relative_idxs = [(curr + p, key) for p, key in enumerate(curr_keys)]
        curr += len(curr_keys)
        sentence2idx_tuple.append(relative_idxs)

    return sentence2idx_tuple, masked_strs, input_masks
