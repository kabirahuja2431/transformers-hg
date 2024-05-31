import torch
import torch.nn as nn
from typing import Dict, Tuple
from ..model_interface import ModelInterface
from ..mlm import MLMResult
from models.transformer_mlm import TransformerMLM
import layers


class TransformerMLMInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self,
        outputs: MLMResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(self, outputs: MLMResult) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(self, data: Dict[str, torch.Tensor], normalize=True):
        in_len = data["in_len"].long() + 1
        inp_data = data["in"].transpose(0, 1)
        tgt_data = data["out"]
        res = self.model(inp_data, in_len)
        res.data = res.data.transpose(0, 1)
        tgt_mask = ~self.model.get_tgt_mask(
            tgt=tgt_data.transpose(0, 1), max_len=inp_data.shape[1], len=in_len
        ).transpose(0, 1)
        loss = self.loss(res, tgt_data, tgt_mask, normalize)
        return MLMResult(res.data, res.length, tgt_mask, loss)
