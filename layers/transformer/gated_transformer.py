import torch
import torch.nn
import torch.nn.functional as F
from .gated_multi_head_attention import GatedMultiHeadAttention
from .multi_head_attention import AttentionMask
from .transformer import TransformerDecoder, TransformerDecoderWithLayer
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GatedTransformerLayerResult:
    data: torch.Tensor
    attn_weights: Optional[torch.Tensor]
    reg: Optional[torch.Tensor]


@dataclass
class GatedTransformerEncoderResult:
    data: torch.Tensor
    all_hidden_states: Optional[List[torch.Tensor]]
    reg: Optional[torch.Tensor]


class GatedTransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
    ):
        super(GatedTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = GatedMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

        self.has_gates = False
        self.gate = None
        self.head_mask = None

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        full_target=None,
        **kwargs,
    ) -> torch.Tensor:
        need_weights = "get_attn_scores" in kwargs
        attn_result = self.self_attn(
            src,
            src if full_target is None else full_target,
            AttentionMask(mask, pos_mask),
            need_weights=need_weights,
        )

        attn_out = attn_result.attn_out

        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        result = GatedTransformerLayerResult(
            data=src,
            attn_weights=attn_result.attn_weights,
            reg=attn_result.reg,
        )

        return result

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=(
                torch.nn.init.calculate_gain("relu")
                if self.activation is F.relu
                else 1.0
            ),
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def get_gate_values(self):
        return self.self_attn.get_gate_values()

    def apply_gates(self, l0_penalty):
        self.self_attn.apply_gates(l0_penalty=l0_penalty)

    def remove_gates(self):
        self.self_attn.remove_gates()

    def apply_masks(self, head_mask):
        self.self_attn.apply_masks(head_mask)

    def get_masks(self):
        return self.self_attn.get_masks()


class GatedTransformerEncoder(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [layer(*args, **kwargs) for _ in range(n_layers)]
        )
        self.n_layers = n_layers
        self.d_model = self.layers[0].d_model
        self.has_gates = False
        self.gate = None
        self.head_mask = None

    def create_state(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> State:
        return self.State(
            0,
            {
                i: torch.zeros([batch_size, max_length, self.d_model], device=device)
                for i in range(len(self.layers))
            },
        )

    def attn_matrices(self, data: torch.Tensor, src_length_mask, pos_mask):
        attn_matrices = []
        for idx, l in enumerate(self.layers):
            out = l(data, mask=src_length_mask, get_attn_scores=True, pos_mask=pos_mask)
            attn_matrices.append(out.attn_weights)
        return attn_matrices

    def forward(self, data: torch.Tensor, *args, **kwargs):
        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        elif len(args) > 0:
            mask = args[0]
        else:
            mask = None

        if "layer_id" in kwargs:
            layer_id = kwargs["layer_id"]
        else:
            layer_id = -1

        if "get_all_layers" in kwargs:
            all_data = [data]
        else:
            all_data = None

        reg = None
        for idx, l in enumerate(self.layers):
            if type(mask) == list:
                mask_curr = mask[idx]
            else:
                mask_curr = mask

            layer_result = l(data, mask=mask_curr, **kwargs)
            data = layer_result.data
            if layer_result.reg is not None:
                if reg is None:
                    reg = layer_result.reg
                else:
                    reg += layer_result.reg
            if layer_id == idx:
                break
            if "get_all_layers" in kwargs:
                all_data.append(data)

        return GatedTransformerEncoderResult(
            data=data if "get_all_layers" not in kwargs else all_data,
            all_hidden_states=all_data,
            reg=reg,
        )

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert (
            data.shape[1] == 1
        ), f"For one-step forward should have one timesteps, but shape is {data.shape}"

        assert state.step < state.state[0].shape[1]
        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        else:
            mask = None

        for i, l in enumerate(self.layers):
            state.state[i][:, state.step : state.step + 1] = data
            full_target = state.state[i][:, : state.step + 1]
            out = l(
                data,
                # *args,
                # **kwargs,
                mask=mask,
                full_target=full_target,
                pos_offset=state.step,
            )
            data = out.data

        state.step += 1
        return data

    def get_gate_values(self):
        gate_values = []
        for l in self.layers:
            gate_values.append(l.get_gate_values())
        return gate_values

    def apply_gates(self, l0_penalty):
        self.has_gates = True
        for l in self.layers:
            l.apply_gates(l0_penalty=l0_penalty)

    def remove_gates(self):
        self.has_gates = False
        for l in self.layers:
            l.remove_gates()

    def apply_masks(self, head_mask):
        self.has_gates = False
        for i, l in enumerate(self.layers):
            layer_head_mask = head_mask[i]
            l.apply_masks(layer_head_mask)

    def get_masks(self):
        masks = []
        for l in self.layers:
            masks.append(l.get_masks())
        return masks

    def apply_dsp(self, num_of_heads, temperature=None, use_ste=False):
        self.num_of_heads = num_of_heads
        self.use_dsp = True
        self.use_ste = use_ste

        if not use_ste:
            self.temperature = temperature

    def get_w(self):
        return self.w


def GatedTransformerEncoderWithLayer(layer=GatedTransformerEncoderLayer):
    return lambda *args, **kwargs: GatedTransformerEncoder(layer, *args, **kwargs)


class GatedTransformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationFunction = F.relu,
        encoder_layer=GatedTransformerEncoderWithLayer(),
        decoder_layer=TransformerDecoderWithLayer(),
        is_null_encoder=False,
        **kwargs,
    ):
        super().__init__()

        if is_null_encoder:
            self.encoder = lambda src, src_length_mask: src
            self.num_encoder_layers = 0

        else:
            self.encoder = encoder_layer(
                num_encoder_layers,
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
            )
            self.num_encoder_layers = num_encoder_layers

        self.decoder = decoder_layer(
            num_decoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
        )

        self.has_gates = False
        self.gate = None
        self.head_mask = None

    def get_hidden_states(
        self,
        src,
        src_length_mask=None,
        layer_id=-1,
        is_lm=False,
    ):
        if is_lm:
            if type(src_length_mask) == list:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask[0].shape[1], device=src.device
                )
            else:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask.shape[1], device=src.device
                )

            encoder_out = self.encoder(
                src,
                src_length_mask=src_length_mask,
                pos_mask=pos_mask,
                layer_id=layer_id,
            )

        else:
            encoder_out = self.encoder(
                src,
                src_length_mask=src_length_mask,
                layer_id=layer_id,
            )

        memory = encoder_out.data
        return memory

    def get_attn_matrices(
        self,
        src,
        tgt,
        src_length_mask=None,
    ):
        if tgt is None:
            pos_mask = self.generate_square_subsequent_mask(
                src_length_mask.shape[1], device=src.device
            )
        else:
            pos_mask = None

        attn_matrices = self.encoder.attn_matrices(
            src, src_length_mask=src_length_mask, pos_mask=pos_mask
        )
        return attn_matrices

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_length_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if tgt is None:
            pos_mask = self.generate_square_subsequent_mask(
                src_length_mask.shape[1], device=src.device
            )
            encoder_out = self.encoder(
                src, src_length_mask=src_length_mask, pos_mask=pos_mask, **kwargs
            )
            if self.has_gates:
                return encoder_out.data, encoder_out.reg
            else:
                return encoder_out.data

        else:
            encoder_out = self.encoder(src, src_length_mask)
            memory = encoder_out.data
            decoder_out = self.decoder(
                tgt,
                memory,
                tgt_mask,
                src_length_mask,
            )
            if self.has_gates:
                return decoder_out, encoder_out.reg
            else:
                return decoder_out

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )

    def get_gate_values(self):
        return self.encoder.get_gate_values()

    def apply_gates(self, l0_penalty):
        self.has_gates = True
        self.encoder.apply_gates(l0_penalty=l0_penalty)

    def remove_gates(self):
        self.has_gates = False
        self.encoder.remove_gates()

    def apply_masks(self, head_mask):
        self.has_gates = False
        self.encoder.apply_masks(head_mask)

    def get_masks(self):
        return self.encoder.get_masks()

    def apply_dsp(self, num_of_heads, temperature=None, use_ste=False):
        self.num_of_heads = num_of_heads
        self.use_dsp = True
        self.use_ste = use_ste

        if not use_ste:
            self.temperature = temperature
