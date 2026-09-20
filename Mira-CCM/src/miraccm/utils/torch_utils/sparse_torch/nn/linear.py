import torch
import torch.nn as nn

from ..tensor import SparseTensor

__all__ = ["Linear"]


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
