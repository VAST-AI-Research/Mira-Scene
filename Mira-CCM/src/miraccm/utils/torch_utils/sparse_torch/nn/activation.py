import torch
import torch.nn as nn

from ..tensor import SparseTensor

__all__ = ["ReLU", "SiLU", "GELU", "Activation"]


class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class GELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class Activation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(self.activation(input.feats))
