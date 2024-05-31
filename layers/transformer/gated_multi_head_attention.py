import torch
from torch._tensor import Tensor
import torch.nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, List, Union, Tuple
from dataclasses import dataclass
from transformers.gated_bert_utilities import ConcreteGate
from layers.transformer.multi_head_attention import (
    MultiHeadAttentionBase,
    MultiHeadAttention,
    AttentionMask,
)


@dataclass
class GatedAttentionResult:
    attn_out: torch.Tensor
    attn_weights: Optional[torch.Tensor]
    reg: Optional[torch.Tensor]


class GatedMultiHeadAttentionBase(MultiHeadAttentionBase):
    def __init__(self, state_size: int, nheads: int, dropout: float = 0.1):
        super().__init__(state_size, nheads, dropout)
        self.has_gates = False
        self.gate = None
        self.head_mask = None

    def __attention(
        self,
        mask: Optional[AttentionMask],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k,v: [n_nbatch * n_heads, n_in]
        # Output data shape [n_batch * n_heads, n_time_dest, data_size]
        # Out attention score shape: [n_batch, n_heads, n_time_dest, n_time_src]
        reg = None
        logits = torch.bmm(q, k.transpose(1, 2))
        scores = self._masked_softmax(logits * self.scale, mask)
        scores = self.dropout(scores)
        if self.head_mask is not None:
            if len(self.head_mask.shape) == 1:
                # Repeat head mask for all batches
                n_batches = scores.shape[0] // self.n_heads
                head_mask = self.head_mask.repeat(n_batches, 1)
                head_mask = head_mask.view(-1, 1, 1)
                scores = scores * head_mask
            else:
                n_batches = scores.shape[0] // self.n_heads
                head_mask = self.head_mask.repeat(n_batches, 1, 1, 1).view(
                    scores.shape[0], 1, 1
                )
                scores = scores * head_mask

        if self.has_gates:
            scores_heads = scores.view(
                -1, self.n_heads, scores.shape[-2], scores.shape[-1]
            ).contiguous()
            scores_head_masked = self.gate(scores_heads).view(*scores.shape)
            scores = scores_head_masked
            reg = self.gate.get_penalty()
        attn_out = torch.bmm(scores, v)
        attn_weights = scores.view(-1, self.n_heads, *scores.shape[1:])
        if reg is None:
            return attn_out, attn_weights
        return attn_out, attn_weights, reg

    def merged_attention(
        self, n_batch: int, n_out_steps: int, *args, **kwargs
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self.has_gates:
            attn_out, attn_weights, reg = self.__attention(*args, **kwargs)
        else:
            attn_out, attn_weights = self.__attention(*args, **kwargs)
            reg = None

        attn_out = (
            attn_out.view(n_batch, self.n_heads, n_out_steps, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(n_batch, n_out_steps, -1)
        )
        if reg is not None:
            return self.multi_head_merge(attn_out), attn_weights, reg
        return self.multi_head_merge(attn_out), attn_weights

    def get_gate_values(self):
        gate_values = None
        if self.gate is not None:
            gate_values = self.gate.get_gates(False).flatten()
        return gate_values

    def apply_gates(self, l0_penalty):
        if not self.has_gates:
            self.has_gates = True
            self.gate = ConcreteGate([1, self.n_heads, 1, 1], l0_penalty=l0_penalty)

    def remove_gates(self):
        self.has_gates = False

    def apply_masks(self, head_mask):
        self.head_mask = head_mask

    def get_masks(self):
        masks = None
        if self.head_mask is not None:
            masks = self.head_mask.flatten()
        return masks


class GatedMultiHeadAttention(GatedMultiHeadAttentionBase):
    def __init__(
        self,
        state_size: int,
        n_heads: int,
        dropout: float = 0.1,
        input_size: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            state_size=state_size,
            nheads=n_heads,
            dropout=dropout,
        )
        self.data_to_kv = torch.nn.Linear(
            state_size, 2 * n_heads * self.projection_size, bias=False
        )
        self.data_to_q = torch.nn.Linear(
            state_size if input_size is None else input_size,
            n_heads * self.projection_size,
            bias=False,
        )

        self.reset_parameters()

    def forward(
        self,
        curr_state: torch.Tensor,
        attend_to: torch.Tensor,
        mask: Optional[AttentionMask],
        need_weights: bool = False,
    ):
        # Input and output shape: [n_batch, n_steps, data_size]
        k, v = self.transform_data(attend_to, self.data_to_kv, 2)
        (q,) = self.transform_data(curr_state, self.data_to_q, 1)

        attn_result = self.merged_attention(
            curr_state.shape[0], q.shape[1], mask, q, k, v
        )

        if len(attn_result) == 3:
            data, scores, reg = attn_result
        else:
            data, scores = attn_result
            reg = None

        result = GatedAttentionResult(
            attn_out=data,
            attn_weights=scores.mean(1) if need_weights else None,
            reg=reg,
        )

        return result

        # ret = (data,)
        # if need_weights:
        #     # Calculate the mean over the heads
        #     ret += (scores.mean(1),)
        # if reg is not None:
        #     ret += (reg,)

        # return ret
        # if need_weights:
        #     # Calculate the mean over the heads
        #     return data, scores.mean(1)
        # else:
        #     return data

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(
            self.data_to_kv.weight[: self.data_to_kv.weight.shape[0] // 2]
        )
        torch.nn.init.xavier_uniform_(
            self.data_to_kv.weight[self.data_to_kv.weight.shape[0] // 2 :]
        )
