from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange

from .vae import DiagonalGaussianDistribution


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype

        dim = inputs.dim()
        inputs = inputs.permute(0, *range(2, dim), 1).contiguous()
        outputs = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)
        outputs = outputs.permute(0, dim - 1, *range(1, dim - 1)).contiguous()

        return outputs.to(origin_dtype)


class BasicAttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w, d = q.shape
        q = rearrange(q, "b c h w d -> b 1 (h w d) c").contiguous()
        k = rearrange(k, "b c h w d -> b 1 (h w d) c").contiguous()
        v = rearrange(v, "b c h w d -> b 1 (h w d) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)

        attn_output = rearrange(
            h_, "b 1 (h w d) c -> b c h w d", h=h, w=w, d=d, c=c, b=b
        )

        return x + self.proj_out(attn_output)


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H * scale_factor, W * scale_factor, D * scale_factor)
    return x


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = FP32LayerNorm(channels)
        self.norm2 = FP32LayerNorm(self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

        # zero out the conv2 parameters
        for p in self.conv2.parameters():
            p.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert (
                in_channels == out_channels
            ), "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            assert (
                in_channels == out_channels
            ), "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).

    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,  # not used
        mid_attn_block: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.mid_attn_block = mid_attn_block

        self.input_layer = nn.Conv3d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1
        )

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))

        middle_block_list = []
        middle_block_list.append(ResBlock3d(channels[-1], channels[-1]))
        if self.mid_attn_block:
            middle_block_list.append(BasicAttnBlock(channels[-1]))
        middle_block_list.append(ResBlock3d(channels[-1], channels[-1]))
        self.middle_block = nn.Sequential(*middle_block_list)

        self.out_layer = nn.Sequential(
            FP32LayerNorm(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

        self.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_layer(sample)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.middle_block(hidden_states)
        hidden_states = self.out_layer(hidden_states)
        return hidden_states


class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).

    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,  # not used
        mid_attn_block: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.mid_attn_block = mid_attn_block

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        middle_block_list = []
        middle_block_list.append(ResBlock3d(channels[0], channels[0]))
        if self.mid_attn_block:
            middle_block_list.append(BasicAttnBlock(channels[0]))
        middle_block_list.append(ResBlock3d(channels[0], channels[0]))
        self.middle_block = nn.Sequential(*middle_block_list)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            FP32LayerNorm(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        self.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_layer(sample)

        hidden_states = self.middle_block(hidden_states)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.out_layer(hidden_states)
        return hidden_states


class AutoencoderKLVoxel(ModelMixin, ConfigMixin):
    """
    Sparse Structure VAE model.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,  # not used
        encoder_channels: List[int] = [32, 128, 512],
        decoder_channels: List[int] = [512, 128, 32],
        mid_attn_block: bool = False,
    ):
        super().__init__()

        self.encoder = SparseStructureEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=encoder_channels,
            num_res_blocks_middle=num_res_blocks_middle,
            mid_attn_block=mid_attn_block,
        )
        self.decoder = SparseStructureDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=decoder_channels,
            num_res_blocks_middle=num_res_blocks_middle,
            mid_attn_block=mid_attn_block,
        )

        self.use_slicing = False
        self.slicing_length = 1

        self.gradient_checkpointing = False

    def enable_slicing(self, slicing_length: int = 1) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True
        self.slicing_length = slicing_length

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(self, logits: torch.Tensor, return_dict=True) -> torch.Tensor:
        """
        Encode a batch of point features into latents.
        """
        if self.use_slicing and logits.shape[0] > 1:
            encoded_slices = [
                self.encoder(logits_slice)
                for logits_slice in logits.split(self.slicing_length)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(logits)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict=True) -> torch.Tensor:
        """
        Decode a batch of latents into point features.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [
                self.decoder(z_slice) for z_slice in z.split(self.slicing_length)
            ]
            h = torch.cat(decoded_slices)
        else:
            h = self.decoder(z)

        if not return_dict:
            return (h,)
        return DecoderOutput(sample=h)

    def forward(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        latent_dist = self.encode(logits).latent_dist
        sample = self.decode(latent_dist.sample()).sample

        return sample, latent_dist
