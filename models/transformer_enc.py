import math
from typing import Optional, Callable
import torch
import torch.nn as nn
from layers import PositionalEncoding
from layers.transformer.transformer import (
    TransformerEncoderLayer,
    TransformerEncoderWithLayer,
    TransformerDecoderWithLayer,
    TransformerEncoder,
)
from models.transformer_lm import DotDict


class TransformerEncoderResult(DotDict):
    logits: torch.Tensor
    hidden_states: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(logits: torch.Tensor, hidden_states: torch.Tensor, length: torch.Tensor):
        return TransformerEncoderResult(
            {"logits": logits, "hidden_states": hidden_states, "length": length}
        )


class TransformerEncoderCLS(nn.Module):
    def __init__(
        self,
        n_input_tokens: int,
        n_outputs: int,
        state_size: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        ff_multiplier: float = 2,
        max_len: int = 5000,
        use_sos: bool = True,
        use_pos_embeddig: bool = True,
        causal_encoder: bool = False,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        dropout: float = 0,
        **kwargs
    ):

        super(TransformerEncoderCLS, self).__init__()

        assert embedding_init in ["pytorch", "xavier", "kaiming"]
        self.sos = n_input_tokens if use_sos else None
        self.n_input_tokens = n_input_tokens
        self.n_outputs = n_outputs
        self.state_size = state_size
        self.nhead = nhead
        self.nlayers = nlayers
        self.ff_multiplier = ff_multiplier
        self.max_len = max_len
        self.use_pos_embeddig = use_pos_embeddig
        if self.use_pos_embeddig:
            self.pos = PositionalEncoding(
                state_size,
                max_len=max_len,
                batch_first=True,
                scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0,
            )
        self.causal_encoder = causal_encoder
        self.embedding_init = embedding_init
        self.scale_mode = scale_mode
        self.dropout = dropout
        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(**kwargs)
        self.reset_parameters()
        self.mode = "cls"

    def construct(self, **kwargs):
        self.input_embedding = nn.Embedding(
            self.n_input_tokens + int(self.sos is not None), self.state_size
        )

        self.output_layer = nn.Linear(self.state_size, self.n_outputs)

        # self.trafo_layer = nn.TransformerEncoderLayer(
        #     d_model=self.state_size,
        #     nhead=self.nhead,
        #     dim_feedforward=int(self.state_size * self.ff_multiplier),
        #     dropout=self.dropout,
        #     batch_first=True,
        #     **kwargs
        # )

        # self.trafo = nn.TransformerEncoder(self.trafo_layer, num_layers=self.nlayers)

        self.trafo_layer = TransformerEncoderLayer(
            d_model=self.state_size,
            nhead=self.nhead,
            dim_feedforward=int(self.state_size * self.ff_multiplier),
            dropout=self.dropout,
        )
        # breakpoint()
        # self.trafo = TransformerEncoder(self.trafo_layer, n_layers=self.nlayers)
        self.trafo = TransformerEncoderWithLayer()(
            self.nlayers,
            self.state_size,
            self.nhead,
            int(self.state_size * self.ff_multiplier),
            self.dropout,
        )

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

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

    def get_hidden_states(self, src, src_length_mask=None, layer_id=-1):

        memory = self.trafo(
            src,
            src_length_mask=src_length_mask,
            layer_id=layer_id,
        )

        return memory

    def encoder_only(self, src, mask, layer_id=-1, gaussian_noise=None):
        src = self.pos_embed(self.input_embed(src), 0)
        if gaussian_noise is not None:
            src += gaussian_noise

        return self.get_hidden_states(src, mask, layer_id=layer_id)

        # assert layer_id == -1
        # # breakpoint()
        # src = self.pos_embed(self.input_embed(src), 0)
        # if gaussian_noise is not None:
        #     src += gaussian_noise
        # # in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        # causal_mask = (
        #     nn.Transformer.generate_square_subsequent_mask(
        #         src.shape[1], device=src.device
        #     )
        #     if self.causal_encoder
        #     else None
        # )
        # hidden_states = self.trafo(
        #     src=src,
        #     mask=causal_mask,
        #     src_key_padding_mask=mask,
        #     is_causal=self.causal_encoder,
        # )
        # return hidden_states

    def get_encoder_layers(self):
        return self.nlayers

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self, src: torch.Tensor, src_len: torch.Tensor, cls_head_idx: torch.Tensor
    ) -> TransformerEncoderResult:
        src = self.input_embed(src)
        src = self.pos_embed(src, 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        # causal_mask = (
        #     nn.Transformer.generate_square_subsequent_mask(
        #         src.shape[1], device=src.device
        #     )
        #     if self.causal_encoder
        #     else None
        # )
        causal_mask = (
            self.generate_square_subsequent_mask(
                in_len_mask.shape[1], device=src.device
            )
            if self.causal_encoder
            else None
        )

        # hidden_states = self.trafo(
        #     src=src,
        #     mask=causal_mask,
        #     src_key_padding_mask=in_len_mask,
        #     is_causal=self.causal_encoder,
        # )

        hidden_states = self.trafo(
            data=src,
            pos_mask=causal_mask,
            src_length_mask=in_len_mask,
            # s_causal=self.causal_encoder,
        )

        # Gather the predictions at the last token using src_len - 1 as index
        B, T, D = hidden_states.shape
        indices = cls_head_idx.view(B, 1, 1)  # - 1
        indices = indices.long()
        logits = self.output_layer(
            hidden_states.gather(1, indices.expand(B, 1, D)).squeeze(1)
        )

        # logits = self.output_layer(hidden_states[:, src_len - 1, :])

        return TransformerEncoderResult.create(logits, hidden_states, src_len)
