from typing import Any, Dict, Tuple
from ..result import Result
import torch
import torch.nn
from ..model_interface import ModelInterface
from ..decoder import DecoderResult
from models.transformer_dec import TransformerResult
import layers


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input


class TransformerDecoderLMInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask

        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(
        self, outputs: DecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(self, data: Dict[str, torch.Tensor], normalize=True) -> DecoderResult:
        in_len = data["in_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat([sos_tensor, data["in"]], dim=0).transpose(0, 1)

        out_data = add_eos(data["in"], data["in_len"], self.model.eos).transpose(0, 1)

        res = self.model(inp_data, in_len)

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(inp_data.shape[1], in_len).transpose(
            0, 1
        )

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)

        return DecoderResult(res.data, res.length, loss)
