import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from interfaces.result import Result
from ..model_interface import ModelInterface
from models.rnn_lm import RNNResult
from models.encoder_decoder import add_eos
from interfaces.rnn.lm_interface import RNNInterfaceResult
import layers


class RNNEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self,
        outputs: RNNResult,
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

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True
    ) -> RNNInterfaceResult:

        in_len = data["in_len"].long() + 1
        out_len = data["out_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
        sos_tensor = sos_tensor.to(data["in"].device)
        enc_inp_data = torch.cat([sos_tensor, data["in"]], dim=0).transpose(0, 1)

        dec_inp_data = torch.cat([sos_tensor, data["out"]], dim=0).transpose(0, 1)

        out_data = add_eos(
            data["out"],
            data["out_len"],
            self.model.encoder_eos,
        ).transpose(0, 1)

        res = self.model(enc_inp_data, in_len, dec_inp_data, out_len)

        res.data = res.data.transpose(0, 1)

        len_mask = ~self.model.generate_len_mask(
            dec_inp_data.shape[1], out_len
        ).transpose(0, 1)
        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)

        return RNNInterfaceResult(res.data, res.length, loss)
