from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AttnProcessor2_0,
    HunyuanAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph

from miraccm.models.attention_processor import ShapeLayoutSelfAttnProcessor


def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert (
            x.shape[d] % patch_size == 0
        ), f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(
        *x.shape[:2],
        *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []),
    )
    x = x.permute(
        0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)])
    )
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size**DIM), *(x.shape[-DIM:]))
    return x


def unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    assert (
        x.shape[1] % (patch_size**DIM) == 0
    ), f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"

    x = x.reshape(
        x.shape[0],
        x.shape[1] // (patch_size**DIM),
        *([patch_size] * DIM),
        *(x.shape[-DIM:]),
    )
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))
    x = x.reshape(
        x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)]
    )
    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            t.device, t.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert (
            D == self.in_channels
        ), "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(N, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, axes_dim: List[int], theta: float = 10000.0):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)
        x = x.permute(0, 2, 1, 3)
        return x


@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    DiT block used in Trellis Sparse Structure Transformer.
    Dual-stream (shape + layout) with self-attention, cross-attention to
    encoder hidden states, and feed-forward.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        qk_norm_self: bool = True,
        qk_norm_cross: bool = True,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu",  # gelu-approximate
    ):
        super().__init__()
        self.inner_dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.cross_attention_dim = cross_attention_dim
        self.qkv_bias = qkv_bias
        self.qk_norm_cross = qk_norm_cross

        # Modulation
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.modulation_layout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # 1. Self-Attention
        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_layout = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim, # Enable added KV projections for layout stream
            context_pre_only=False,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm=None,
            bias=qkv_bias,
            processor=ShapeLayoutSelfAttnProcessor(),
        )
        # Set qk norm manually
        if qk_norm_self:
            self.attn1.norm_q = MultiHeadRMSNorm(self.head_dim, num_attention_heads)
            self.attn1.norm_k = MultiHeadRMSNorm(self.head_dim, num_attention_heads)
            self.attn1.norm_q_layout = MultiHeadRMSNorm(self.head_dim, num_attention_heads)
            self.attn1.norm_k_layout = MultiHeadRMSNorm(self.head_dim, num_attention_heads)

        # 2. Cross-Attention
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.norm2_layout = FP32LayerNorm(dim, elementwise_affine=True, eps=1e-6)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm=None,
            bias=qkv_bias,
            processor=HunyuanAttnProcessor2_0(),
        )

        self.attn2_layout = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm=None,
            bias=qkv_bias,
            processor=HunyuanAttnProcessor2_0(),
        )

        # Set qk norm manually
        if qk_norm_cross:
            self.attn2.norm_q = MultiHeadRMSNorm(self.head_dim, num_attention_heads)
            self.attn2.norm_k = MultiHeadRMSNorm(self.head_dim, num_attention_heads)

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3_layout = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=int(dim * mlp_ratio),
            bias=True,
        )
        self.ff_layout = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=int(dim * mlp_ratio),
            bias=True,
        )

    def forward(
        self,
        hidden_states: Dict[str, torch.Tensor],
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        layout_encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        # Prepare attention kwargs
        attention_kwargs = (
            attention_kwargs.copy() if attention_kwargs is not None else {}
        )
        cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)

        # Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(temb).chunk(6, dim=1)
        )
        (
            shift_msa_layout, scale_msa_layout, gate_msa_layout,
            shift_mlp_layout, scale_mlp_layout, gate_mlp_layout,
        ) = self.modulation_layout(temb).chunk(6, dim=1)

        # normalization before Self-Attention
        norm_hidden_states_shape = self.norm1(hidden_states['shape'])
        norm_hidden_states_layout = self.norm1_layout(hidden_states['layout'])

        norm_hidden_states_shape = norm_hidden_states_shape * (
            1 + scale_msa.unsqueeze(1)
        ) + shift_msa.unsqueeze(1)
        norm_hidden_states_layout = norm_hidden_states_layout * (
            1 + scale_msa_layout.unsqueeze(1)
        ) + shift_msa_layout.unsqueeze(1)

        # Self-Attention
        self_attn_output = self.attn1(
            {'shape': norm_hidden_states_shape, 'layout': norm_hidden_states_layout},
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **attention_kwargs,
        )

        # Gate and Residual connection / Cross Attn / Feed-forward
        for key in self_attn_output:
            if key == 'shape':
                self_attn_output[key] = self_attn_output[key] * gate_msa.unsqueeze(1)
            elif key == 'layout':
                self_attn_output[key] = self_attn_output[key] * gate_msa_layout.unsqueeze(1)
            hidden_states[key] = hidden_states[key] + self_attn_output[key]

            if key == 'shape':
                cross_attn_output = self.attn2(
                                        self.norm2(hidden_states[key]),
                                        encoder_hidden_states=encoder_hidden_states,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_mask=encoder_attention_mask,
                                        **attention_kwargs,
                                    )
            elif key == 'layout':
                # Use dedicated layout encoder hidden states when provided
                _layout_enc = layout_encoder_hidden_states if layout_encoder_hidden_states is not None else encoder_hidden_states
                cross_attn_output = self.attn2_layout(
                                        self.norm2_layout(hidden_states[key]),
                                        encoder_hidden_states=_layout_enc,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_mask=encoder_attention_mask,
                                        **attention_kwargs,
                                    )

            hidden_states[key] = hidden_states[key] + cross_attn_output * cross_attention_scale

            if key == 'shape':
                mlp_inputs = self.norm3(hidden_states[key])
                mlp_inputs = mlp_inputs * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                mlp_outputs = self.ff(mlp_inputs) * gate_mlp.unsqueeze(1)
            elif key == 'layout':
                mlp_inputs = self.norm3_layout(hidden_states[key])
                mlp_inputs = mlp_inputs * (1 + scale_mlp_layout.unsqueeze(1)) + shift_mlp_layout.unsqueeze(1)
                mlp_outputs = self.ff_layout(mlp_inputs) * gate_mlp_layout.unsqueeze(1)

            hidden_states[key] = hidden_states[key] + mlp_outputs

        return hidden_states
