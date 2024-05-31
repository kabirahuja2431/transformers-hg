import torch
import torch.nn
from typing import Dict, Tuple
from ..model_interface import ModelInterface
from ..encoder_decoder import EncoderDecoderResult
from models.transformer_enc_dec import TransformerResult
from models.encoder_decoder import add_eos
from interfaces.transformer.lm_interface import TransformerLMInterface
import layers


class TransformerPrefixLMInterface(TransformerLMInterface):

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True
    ) -> EncoderDecoderResult:

        in_len = data["in_len"].long() + 1
        prefix_len = data["prefix_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat(
            [sos_tensor, data["in"]],
            dim=0,
        ).transpose(0, 1)

        out_data = add_eos(
            data["in"], data["in_len"], self.model.encoder_eos
        ).transpose(0, 1)

        res = self.model(inp_data, in_len, prefix_len)
        res.data = res.data.transpose(0, 1)

        len_mask = ~self.model.generate_len_mask(inp_data.shape[1], in_len).transpose(
            0, 1
        )

        prefix_mask = ~self.model.generate_len_mask(
            inp_data.shape[1], prefix_len
        ).transpose(0, 1)

        # final mask should be everything after prefix_len and before in_len
        mask = (~prefix_mask) & len_mask

        loss = self.loss(res, out_data.transpose(0, 1), mask, normalize)

        if "reg" in res:
            return EncoderDecoderResult(res.data, res.length, loss, res.reg)
        else:
            return EncoderDecoderResult(res.data, res.length, loss, None)
