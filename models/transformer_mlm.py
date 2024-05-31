import torch
import torch.nn as nn
import random
import math
from layers import TiedEmbedding, PositionalEncoding
from models.transformer_lm import TransformerResult


class TransformerMLM(nn.Module):
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
        use_pos_embedding: bool = True,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        dropout: float = 0.1,
        causal_encoder: bool = False,
        **kwargs
    ):
        super(TransformerMLM, self).__init__()
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        self.tied_embedding = tied_embedding
        self.tgt_ignore_idx = -100
        self.mask_idx = n_input_tokens
        self.empty_idx = n_input_tokens + 1
        self.sos = n_input_tokens + 2 if use_sos else None
        self.state_size = state_size
        self.nhead = nhead
        self.nlayers = nlayers
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.max_len = max_len
        self.use_pos_embedding = use_pos_embedding
        self.causal_encoder = causal_encoder
        if self.use_pos_embedding:
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
        self.mode = "mlm"

    def construct(self, **kwargs):
        self.input_embedding = nn.Embedding(
            self.n_input_tokens + 2 + int(self.sos is not None), self.state_size
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding)
        else:
            self.output_map = nn.Linear(
                self.state_size,
                self.n_input_tokens + 2 + int(self.sos is not None),
            )

        self.trafo_layer = nn.TransformerEncoderLayer(
            d_model=self.state_size,
            nhead=self.nhead,
            dim_feedforward=int(self.ff_multiplier * self.state_size),
            dropout=self.dropout,
            batch_first=True,
            **kwargs
        )
        self.trafo = nn.TransformerEncoder(self.trafo_layer, self.nlayers)

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

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
        if not self.use_pos_embedding:
            return t

        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        # mask out padding
        len_mask = self.int_seq[:max_len] >= len.unsqueeze(-1)

        return len_mask

    def generate_mask(
        self, src: torch.Tensor, max_len: int, len: torch.Tensor
    ) -> torch.Tensor:
        # mask out padding
        len_mask = self.generate_len_mask(max_len, len)

        # mask out [MASK] token in src
        # src_mask = src == self.mask_idx

        # return len_mask | src_mask
        return len_mask

    def get_tgt_mask(
        self, tgt: torch.Tensor, max_len: int, len: torch.Tensor
    ) -> torch.Tensor:
        len_mask = self.generate_len_mask(max_len, len)
        mlm_mask = tgt == -100
        return len_mask | mlm_mask

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> TransformerResult:
        src_embeds = self.pos_embed(self.input_embed(src), 0)
        mask = self.generate_mask(src, src.shape[1], src_len)
        causal_mask = (
            nn.Transformer.generate_square_subsequent_mask(
                src_embeds.shape[1], device=src.device
            )
            if self.causal_encoder
            else None
        )
        res = self.trafo(
            src_embeds,
            mask=causal_mask,
            src_key_padding_mask=mask,
            is_causal=self.causal_encoder,
        )
        return TransformerResult.create(self.output_map(res), src_len)
