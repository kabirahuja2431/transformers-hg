from typing import Any, Dict, Tuple
from ..result import Result
import torch
import torch.nn as nn
from ..model_interface import ModelInterface
from ..encoder import EncoderResult
from models.transformer_enc import TransformerEncoderResult


class TransformerEncoderCLSInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def loss(
        self,
        outputs: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:

        l = self.criterion(outputs, ref.long())
        return l

    def decode_outputs(self, outputs: EncoderResult) -> torch.Tensor:
        return outputs.outputs

    def __call__(self, data: Dict[str, torch.Tensor]) -> EncoderResult:
        in_len = data["in_len"].long() + 1
        cls_head_idx = data["cls_head_idx"] + 1
        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat([sos_tensor, data["in"]], dim=0).transpose(0, 1)
        res = self.model(inp_data, in_len, cls_head_idx=cls_head_idx)

        logits = res.logits
        targets = data["out"]

        loss = self.loss(logits, targets)

        return EncoderResult(outputs=logits, loss=loss)
