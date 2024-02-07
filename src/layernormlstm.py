from torch.nn import RNNCellBase
from torch.nn import LayerNorm
from torch import Tensor
import torch
from typing import Tuple


class LayerNormLSTMCell(RNNCellBase):
    """
    LayerNorm LSTM Cell from https://arxiv.org/abs/1607.06450
    based on implementation from pytorch fastrnn benchmark
    https://github.com/pytorch/pytorch/blob/cbcb2b5ad767622cf5ec04263018609bde3c974a/benchmarks/fastrnns/custom_lstms.py#L149
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        # we only use a single bias, double bias from RNNCellBase would only be necessary for
        # cuDNN compatibility, see https://pytorch.org/docs/2.0/_modules/torch/nn/modules/rnn.html#RNNBase
        del self.bias_hh
        self.ln_input = LayerNorm(4 * hidden_size)
        self.ln_hidden = LayerNorm(4 * hidden_size)
        self.ln_cell = LayerNorm(hidden_size)

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        hx, cx = state
        i_gates = self.ln_input(torch.mm(input, self.weight_ih.t()))
        h_gates = self.ln_hidden(torch.mm(hx, self.weight_hh.t()))
        # add bias after layer norm
        gates = i_gates + h_gates + self.bias_ih
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = self.ln_cell((forget_gate * cx) + (in_gate * cell_gate))
        hy = out_gate * torch.tanh(cy)

        return hy, cy
