import torch
import torch.nn as nn

from ...tensor import SparseTensor

__all__ = ["Conv3d", "ConvTranspose3d"]


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(Conv3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, 0, dilation, bias
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s * stride for s, stride in zip(x._scale, self.conv.stride)]
        )
        return out


class ConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(ConvTranspose3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            bias,
            transposed=True,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s // stride for s, stride in zip(x._scale, self.conv.stride)]
        )
        return out
