import torch
import torch.nn as nn

from ..tensor import SparseTensor

__all__ = [
    "GroupNorm",
    "LayerNorm",
    "GroupNorm32",
    "LayerNorm32",
]


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class GroupNorm32(GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.weight is not None:
            self.weight.data = self.weight.data.float()
        if self.bias is not None:
            self.bias.data = self.bias.data.float()

        return super().forward(x.float()).type(x.dtype)


class LayerNorm32(LayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.weight is not None:
            self.weight.data = self.weight.data.float()
        if self.bias is not None:
            self.bias.data = self.bias.data.float()

        return super().forward(x.float()).type(x.dtype)
