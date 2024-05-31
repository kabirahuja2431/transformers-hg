from dataclasses import dataclass
from typing import List, Optional
import torch
from .result import Result


@dataclass
class EncoderResult(Result):
    outputs: torch.Tensor
    loss: torch.Tensor

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            return l[0]
        
        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.stack([r.outputs for r in l], axis=0)

        return l[0].__class__(out, loss)