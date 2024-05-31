import torch
import torch.nn as nn
import random
import math
from layers.rnn.rnn import RNNModel
from layers import TiedEmbedding
from typing import Callable, Optional


class DotDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RNNResult(DotDict):
    data: torch.Tensor
    hidden: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(data: torch.Tensor, hidden: torch.Tensor, length: torch.Tensor):
        return RNNResult({"data": data, "hidden": hidden, "length": length})


class RNNLM(torch.nn.Module):

    def __init__(
        self,
        rnn_type: str,
        n_input_tokens: int,
        state_size: int = 512,
        num_layers: int = 2,
        encoder_sos: bool = True,
        embedding_init: str = "xavier",
        tied_embedding: bool = False,
        max_len: int = 5000,
        **kwargs
    ):
        """
        RNN Language Model.

        :param rnn_type: Type of RNN to use
        :param n_input_tokens: Number of channels for the input vectors
        :param state_size: The size of the internal state of the RNN
        :param num_layers: Number of layers in the RNN

        """

        super().__init__()
        assert embedding_init in ["pytorch", "xavier", "kaiming"]
        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
        self.state_size = state_size
        self.num_layers = num_layers
        self.embedding_init = embedding_init
        self.n_input_tokens = n_input_tokens
        self.rnn_type = rnn_type
        self.tied_embedding = tied_embedding
        self.mode = "lm"

        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(**kwargs)
        self.reset_parameters()

    def construct(self, **kwargs):

        self.input_embedding = torch.nn.Embedding(
            self.n_input_tokens + 1 + int(self.encoder_sos is not None), self.state_size
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_input_tokens + 1 + int(self.encoder_sos is not None),
            )

        self.rnn = RNNModel(
            rnn_type=self.rnn_type,
            state_size=self.state_size,
            num_layers=self.num_layers,
            **kwargs
        )

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def input_embed(self, x: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(x.long())
        return src

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def run_teacher_forcing(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
    ) -> RNNResult:

        output, hidden = self.rnn(src, src_len)
        return RNNResult.create(self.output_map(output), hidden, src_len)

    def run_greedy(
        self, src: torch.Tensor, src_len: torch.Tensor, max_len: int
    ) -> RNNResult:

        batch_size = src.shape[0]
        n_steps = src.shape[1]

        src = self.input_embed(src)
        output, hidden = self.rnn(src, src_len)

        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(src_len, dtype=torch.long)

        last_embeddings = self.output_map(
            torch.cat([curr[l - 1].unsqueeze(0) for curr, l in zip(output, src_len)])
        )
        last_hidden = hidden

        pred_words = torch.argmax(last_embeddings, dim=-1)

        next_tgt = torch.cat(
            [
                self.input_embed(pred_words[idx : idx + 1]).unsqueeze(1)
                for idx in range(batch_size)
            ]
        )

        all_outputs = [last_embeddings.unsqueeze(1)]

        for i in range(max_len):
            output, last_hidden = self.rnn(
                next_tgt, torch.ones_like(next_tgt).squeeze(1), last_hidden
            )
            output = self.output_map(output)
            all_outputs.append(output)
            out_token = torch.argmax(output[:, -1], -1)
            running &= out_token != self.encoder_eos
            out_len[running] = i + 1
            next_tgt = self.input_embed(out_token.unsqueeze(1))

        return RNNResult.create(torch.cat(all_outputs, dim=1), last_hidden, out_len)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> RNNResult:

        src = self.input_embed(src)
        return self.run_teacher_forcing(src, src_len)
