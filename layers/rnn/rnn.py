import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


class RNNModel(nn.Module):

    def __init__(
        self,
        rnn_type: str,
        d_model: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(
                d_model, d_model, num_layers, dropout=dropout, batch_first=True
            )
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']"
                )
            self.rnn = nn.RNN(
                d_model,
                d_model,
                num_layers,
                nonlinearity=nonlinearity,
                dropout=dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )

            # self.init_weights()

        self.rnn_type = rnn_type
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(self, src, lengths, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(src.size(0))
        src_packed = nn.utils.rnn.pack_padded_sequence(
            src, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, hidden = self.rnn(src_packed, hidden)
        output_padded, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output_padded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.num_layers, bsz, self.d_model),
                weight.new_zeros(self.num_layers, bsz, self.d_model),
            )
        else:
            return weight.new_zeros(self.num_layers, bsz, self.d_model)
