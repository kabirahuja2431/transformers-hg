import torch
import torch.nn as nn
import random
import math
from layers.rnn.rnn import RNNModel
from layers import TiedEmbedding
from models.rnn_lm import RNNResult
from typing import Callable, Optional


class RNNEncDec(torch.nn.Module):

    def __init__(
        self,
        rnn_type: str,
        n_input_tokens: int,
        n_out_tokens: int,
        state_size: int = 512,
        num_layers: int = 2,
        embedding_init: str = "xavier",
        tied_embedding: bool = False,
        max_len: int = 5000,
        **kwargs
    ):
        """
        RNN Encoder-Decoder.



        """

        super().__init__()

        assert embedding_init in ["pytorch", "xavier", "kaiming"]
        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1
        self.decoder_eos = n_out_tokens
        self.decoder_sos = n_out_tokens + 1
        self.state_size = state_size
        self.num_layers = num_layers
        self.embedding_init = embedding_init
        self.max_len = max_len
        self.n_input_tokens = n_input_tokens
        self.n_out_tokens = n_out_tokens
        self.tied_embedding = tied_embedding
        self.rnn_type = rnn_type
        self.mode = "enc_dec"

        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(**kwargs)
        self.reset_parameters()

    def construct(self, **kwargs):

        self.input_embedding = torch.nn.Embedding(
            self.n_input_tokens + 2, self.state_size
        )
        self.output_embedding = torch.nn.Embedding(
            self.n_out_tokens + 2, self.state_size
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.output_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_out_tokens + 2,
            )

        self.rnn_encoder = RNNModel(
            rnn_type=self.rnn_type,
            d_model=self.state_size,
            num_layers=self.num_layers,
            bidirectional=False,
            **kwargs
        )

        self.rnn_decoder = RNNModel(
            rnn_type=self.rnn_type,
            d_model=self.state_size,
            num_layers=self.num_layers,
            bidirectional=False,
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

    def output_embed(self, x: torch.Tensor) -> torch.Tensor:
        tgt = self.output_embedding(x.long())
        return tgt

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def run_teacher_forcing(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        tgt: torch.Tensor,
        tgt_len: torch.Tensor,
    ) -> RNNResult:
        """
        Run teacher forcing on the model.

        Args:
            src: Source sequence.
            src_len: Length of source sequence.

        Returns:
            RNNResult: Result of the teacher forcing.
        """

        encoder_output, encoder_hidden = self.rnn_encoder(src, src_len)
        decoder_output, decoder_hidden = self.rnn_decoder(tgt, tgt_len, encoder_hidden)

        return RNNResult.create(
            self.output_map(decoder_output), decoder_hidden, src_len
        )

    def run_greedy(
        self, src: torch.Tensor, src_len: torch.Tensor, max_len: int
    ) -> RNNResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        src = self.input_embed(src)
        encoder_output, encoder_hidden = self.rnn_encoder(src, src_len)

        decoder_input = torch.full(
            (batch_size, 1), self.decoder_sos, dtype=torch.long, device=src.device
        )
        decoder_input = self.output_embed(decoder_input)
        decoder_hidden = encoder_hidden
        output = []
        out_len = torch.full((batch_size,), 1, dtype=torch.long, device=src.device)
        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)

        for i in range(max_len):
            decoder_output, decoder_hidden = self.rnn_decoder(
                decoder_input, torch.ones(batch_size), decoder_hidden
            )
            decoder_input = torch.argmax(self.output_map(decoder_output), dim=-1)
            running = running & (decoder_input != self.encoder_eos)
            decoder_input = self.output_embed(decoder_input)
            output.append(decoder_input)

            out_len += running.long()

            if not running.any():
                break

        return RNNResult.create(torch.cat(output, dim=1), decoder_hidden, out_len)

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        tgt: torch.Tensor = None,
        tgt_len: torch.Tensor = None,
    ) -> RNNResult:

        if tgt is None:
            tgt = (
                torch.ones((src.shape[0], 1), dtype=torch.long, device=src.device)
                * self.decoder_sos
            )
            tgt_len = torch.ones((src.shape[0],), dtype=torch.long, device=src.device)

        src = self.input_embed(src)
        tgt = self.output_embed(tgt)
        return self.run_teacher_forcing(src, src_len, tgt, tgt_len)
