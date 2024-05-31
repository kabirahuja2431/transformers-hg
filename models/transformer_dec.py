import math
from typing import Optional, Callable
import torch
import torch.nn as nn
from layers import TiedEmbedding, PositionalEncoding
from models.transformer_lm import TransformerResult


class TransformerDecoderLM(nn.Module):
    def __init__(
        self,
        n_input_tokens: int,
        state_size: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        ff_multiplier: float = 4,
        max_len: int = 5000,
        tied_embedding: bool = False,
        use_sos: bool = True,
        use_pos_embeddig: bool = True,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        dropout: float = 0.1,
        **kwargs
    ):
        super(TransformerDecoderLM, self).__init__()

        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        self.tied_embedding = tied_embedding
        self.eos = n_input_tokens
        self.sos = n_input_tokens + 1 if use_sos else None
        self.state_size = state_size
        self.nhead = nhead
        self.nlayers = nlayers
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.max_len = max_len
        self.use_pos_embeddig = use_pos_embeddig
        if self.use_pos_embeddig:
            self.pos = PositionalEncoding(
                state_size,
                max_len=max_len,
                batch_first=True,
                scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0,
            )
        self.embedding_init = embedding_init
        self.scale_mode = scale_mode
        self.dropout = dropout
        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(**kwargs)
        self.reset_parameters()
        self.mode = "lm"

    def construct(self, **kwargs):
        self.input_embedding = nn.Embedding(
            self.n_input_tokens + 1 + int(self.sos is not None), self.state_size
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_input_tokens + 1 + int(self.sos is not None),
            )

        self.trafo_layer = nn.TransformerEncoderLayer(
            d_model=self.state_size,
            nhead=self.nhead,
            dim_feedforward=int(self.state_size * self.ff_multiplier),
            dropout=self.dropout,
            batch_first=True,
            **kwargs
        )

        self.trafo = nn.TransformerEncoder(
            self.trafo_layer,
            num_layers=self.nlayers,
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

    def pos_embed(self, t: torch.Tensor, offset: int) -> torch.Tensor:
        if not self.use_pos_embeddig:
            return t

        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def run_teacher_forcing(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
    ) -> TransformerResult:
        src = self.pos_embed(self.input_embed(src), 0)

        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        res = self.trafo(
            src=src,
            mask=nn.Transformer.generate_square_subsequent_mask(
                src.shape[1], device=src.device
            ),
            src_key_padding_mask=in_len_mask,
            is_causal=True,
        )
        return TransformerResult.create(self.output_map(res), src_len)

    def run_greedy(
        self, src: torch.Tensor, src_len: torch.Tensor, max_len: int
    ) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        # src = self.pos_embed(self.input_embed(src), 0)
        # in_len_mask = self.generate_len_mask(n_steps, src_len)

        # A loop over max_steps that greedily decodes next token and appends the result to the input to be used in next step
        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(running, dtype=torch.long)
        all_outputs = []
        tgt = src
        tgt_len = src_len
        for i in range(max_len):
            tgt_embed = self.pos_embed(self.input_embed(tgt), 0)
            in_len_mask = self.generate_len_mask(n_steps + i, tgt_len)
            res = self.trafo(
                src=tgt_embed,
                mask=nn.Transformer.generate_square_subsequent_mask(
                    tgt_embed.shape[1], device=src.device
                ),
                src_key_padding_mask=in_len_mask,
                is_causal=True,
            )
            # Get pred of last token
            out_token = torch.argmax(res.data[:, tgt_len], axis=-1)

            # Update output lens
            running &= out_token != self.encoder_eos
            out_len = running * (out_len + 1) + (1 - running) * out_len

            # Append the predicted tokens to tgt, remember the tokens are to be appended before padding becomes, so can't just directly append to the end
            # First we will add a new column to tgt and fill it with padding tokens
            tgt = torch.cat(
                [
                    tgt,
                    torch.zeros([batch_size, 1], dtype=torch.long, device=src.device),
                ],
                axis=1,
            )
            # Now we will fill the tokens at len with the predicted tokens
            tgt[:, tgt_len] = out_token
            tgt_len = tgt + i

            all_outputs.append(out_token.unsqueeze(-1))

        return TransformerResult.create(torch.cat(all_outputs, 1), out_len)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> TransformerResult:
        return self.run_teacher_forcing(src, src_len)
