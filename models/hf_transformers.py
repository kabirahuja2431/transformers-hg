import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2Config,
    GPT2Model,
    GatedGPT2Model,
    GatedGPT2SinusoidalPosModel,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    GatedBaseModelOutputWithPastAndCrossAttentions,
)
from layers import TiedEmbedding


class HFTransformerModel(nn.Module):
    def __init__(
        self,
        n_input_tokens: int,
        n_embd: int = 512,
        n_layer: int = 6,
        n_head: int = 4,
        dropout: float = 0.1,
        n_positions: int = 512,
        tied_embedding: bool = False,
        sos: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        embedding_init: str = "xavier",
        scale_mode: str = "opennmt",
        pos_scale: float = 1,
        gpt2_model: nn.Module = GatedGPT2SinusoidalPosModel,
    ):
        super(HFTransformerModel, self).__init__()
        config = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.eos = n_input_tokens
        self.sos = n_input_tokens + 1 if sos else None
        self.n_input_tokens = n_input_tokens
        self.state_size = n_embd
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.tied_embedding = tied_embedding

        self.input_embeddings = nn.Embedding(n_input_tokens, n_embd)
        self._backbone = gpt2_model(config, pos_scale)
        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_input_tokens + 1 + int(self.sos is not None),
            )
        self.register_buffer("int_seq", torch.arange(n_positions, dtype=torch.long))
        self.embedding_init = embedding_init
        self.scale_mode = scale_mode
        self.reset_parameters()

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embeddings.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embeddings.weight)
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] < len.unsqueeze(-1)

    def compute_masked_loss(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss on the target tokens only masking out padded tokens
        """
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            target.contiguous().view(-1),
            reduction="none",
        )
        loss = loss * mask.contiguous().view(-1)
        loss = loss.sum() / mask.sum()
        return loss

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
    ):
        src = src.transpose(0, 1)
        inp_embeddings = self.input_embeddings(src)
        if self.scale_mode == "opennmt":
            inp_embeddings = inp_embeddings * math.sqrt(self.state_size)

        attn_mask = self.generate_len_mask(src.shape[1], src_len)
        result = self._backbone(inputs_embeds=inp_embeddings, attention_mask=attn_mask)
        logits = self.output_map(result.last_hidden_state)
        loss = self.compute_masked_loss(logits[:, :-1], src[:, 1:], attn_mask[:, :-1])
        try:
            return GatedBaseModelOutputWithPastAndCrossAttentions(
                loss=loss,
                logits=logits,
                hidden_states=result.hidden_states
                if self.output_hidden_states
                else None,
                attentions=result.attentions if self.output_attentions else None,
                total_reg=result.total_reg,
            )
        except AttributeError:
            return GatedBaseModelOutputWithPastAndCrossAttentions(
                loss=loss,
                logits=logits,
                hidden_states=result.hidden_states
                if self.output_hidden_states
                else None,
                attentions=result.attentions if self.output_attentions else None,
            )
