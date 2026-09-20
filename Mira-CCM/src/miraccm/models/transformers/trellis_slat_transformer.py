from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
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

from ...utils.torch_utils import sparse_torch
from .trellis_sparse_structure_transformer import (
    AbsolutePositionEmbedder,
    RotaryPositionEmbedder,
    TimestepEmbedder,
)


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample_factor: Optional[int] = None,
        upsample_factor: Optional[int] = None,
    ):
        super().__init__()

        if downsample_factor is not None and upsample_factor is not None:
            raise ValueError("Cannot downsample and upsample at the same time")

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels, bias=True),
        )

        self.norm1 = FP32LayerNorm(in_channels, elementwise_affine=True, eps=1e-6)
        self.conv1 = sparse_torch.nn.Conv3d(in_channels, out_channels, 3)

        self.norm2 = FP32LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.conv2 = sparse_torch.nn.Conv3d(out_channels, out_channels, 3)

        self.skip_connection = (
            sparse_torch.nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.updown = None
        if downsample_factor is not None:
            self.updown = sparse_torch.nn.Downsample(downsample_factor)
        elif upsample_factor is not None:
            self.updown = sparse_torch.nn.Upsample(upsample_factor)

    def _updown(self, x: sparse_torch.SparseTensor) -> sparse_torch.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(
        self, x: sparse_torch.SparseTensor, emb: torch.Tensor
    ) -> sparse_torch.SparseTensor:
        # Prepare modulation
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        # Apply up/downsampling
        x = self._updown(x)

        # Apply residual blocks
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        # Apply skip connection
        h = h + self.skip_connection(x)

        return h


class SparseTensorAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Trellis SLat Transformer model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    _attention_backend = None
    _window_size = None
    _shift_window = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    @staticmethod
    def _reshape_tensor(
        x: Union[torch.Tensor, sparse_torch.SparseTensor], shape: Tuple[int, ...]
    ) -> Union[torch.Tensor, sparse_torch.SparseTensor]:
        """
        Note that the `shape` should not include the batch dimension,
        and SparseTensor use batch size * sequence length as the first dimension.
        """
        if isinstance(x, sparse_torch.SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def __call__(
        self,
        attn: Attention,
        hidden_states: sparse_torch.SparseTensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[torch.Tensor] = None,
    ) -> sparse_torch.SparseTensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = self._reshape_tensor(query, (attn.heads, -1))
        key = self._reshape_tensor(key, (attn.heads, -1))
        value = self._reshape_tensor(value, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # TODO: Apply RoPE if needed
        if rotary_emb is not None:

            def _apply_rotary_emb(
                x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]
            ):
                cos, sin = freqs_cis

                if len(x.shape) == 3:  # sparse tensor
                    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)  # [S, 1, D]
                elif len(x.shape) == 4:  # dense tensor
                    cos = cos[None, :, None, :]
                    sin = sin[None, :, None, :]  # [1, S, 1, D]
                cos, sin = cos.to(x.device), sin.to(x.device)

                # [S, H, D//2] or [B, S, H, D//2]
                x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
                # [S, H, D] or [B, S, H, D]
                x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)

                out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
                return out

            def apply_rotary_emb(
                x: Union[torch.Tensor, sparse_torch.SparseTensor],
                freqs_cis: Tuple[torch.Tensor, torch.Tensor],
            ):
                if isinstance(x, sparse_torch.SparseTensor):
                    return x.replace(_apply_rotary_emb(x.feats, freqs_cis))
                else:
                    return _apply_rotary_emb(x, freqs_cis)

            if query_rotary_emb is not None:
                query = apply_rotary_emb(query, query_rotary_emb)
            else:
                query = apply_rotary_emb(query, rotary_emb)

            key = apply_rotary_emb(key, rotary_emb)

        # Execute sparse scaled dot-product attention
        hidden_states = sparse_torch.sparse_scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self._attention_backend,
            window_size=self._window_size,
            shift_window=self._shift_window,
        )

        hidden_states = self._reshape_tensor(hidden_states, (-1,))
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(
        self, x: Union[torch.Tensor, sparse_torch.SparseTensor]
    ) -> Union[torch.Tensor, sparse_torch.SparseTensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, sparse_torch.SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)
        return (x * self.gamma * self.scale).to(x_type)


class SparseAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = SparseTensorAttnProcessor2_0
    _available_processors = [SparseTensorAttnProcessor2_0]

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 12,
        qk_norm: Optional[str] = None,
        bias: bool = True,
        dropout: float = 0.0,
        processor: Optional[AttentionProcessor] = None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim = (
            query_dim if cross_attention_dim is None else cross_attention_dim
        )
        self.dim_head = dim_head
        self.kv_inner_dim = self.inner_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.kv_inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.kv_inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=bias),
                nn.Dropout(dropout),
            ]
        )

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = SparseMultiHeadRMSNorm(self.dim_head, self.heads)
            self.norm_k = SparseMultiHeadRMSNorm(self.dim_head, self.heads)
        else:
            raise ValueError(f"Invalid qk_norm: {qk_norm}")

        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> sparse_torch.SparseTensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            **kwargs,
        )


@maybe_allow_in_graph
class SparseTransformerBlock(nn.Module):
    r"""
    Sparse Transformer block used in Trellis SLat Transformer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
        qk_norm_self: Optional[str] = None,
        qk_norm_cross: Optional[str] = None,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu",  # gelu-approximate
        use_modulation: bool = False,
        norm_elementwise_affine: bool = False,
        norm_cross_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.inner_dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        # Modulation
        if use_modulation:
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True),
            )
        else:
            self.modulation = None

        # 1. Self-Attention
        if use_self_attention:
            self.norm1 = FP32LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )
            self.attn1 = SparseAttention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm=qk_norm_self,
                bias=qkv_bias,
                processor=SparseTensorAttnProcessor2_0(),
            )

        # 2. Cross-Attention
        if use_cross_attention:
            self.norm2 = FP32LayerNorm(
                dim, elementwise_affine=norm_cross_elementwise_affine, eps=norm_eps
            )
            self.attn2 = SparseAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm=qk_norm_cross,
                bias=qkv_bias,
                processor=SparseTensorAttnProcessor2_0(),
            )

        self.norm3 = FP32LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=int(dim * mlp_ratio),
            bias=True,
        )

    def forward(
        self,
        hidden_states: Union[torch.Tensor, sparse_torch.SparseTensor],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        # Prepare attention kwargs
        attention_kwargs = (
            attention_kwargs.copy() if attention_kwargs is not None else {}
        )
        cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)

        # Modulation
        if self.modulation is not None and temb is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation(temb).chunk(6, dim=1)
            )
        else:
            hs_shape = hidden_states.shape
            hs_device, hs_dtype = hidden_states.device, hidden_states.dtype
            shift_msa = torch.zeros(hs_shape, device=hs_device, dtype=hs_dtype)
            scale_msa = torch.zeros(hs_shape, device=hs_device, dtype=hs_dtype)
            gate_msa = torch.ones(hs_shape, device=hs_device, dtype=hs_dtype)
            shift_mlp = torch.zeros(hs_shape, device=hs_device, dtype=hs_dtype)
            scale_mlp = torch.zeros(hs_shape, device=hs_device, dtype=hs_dtype)
            gate_mlp = torch.ones(hs_shape, device=hs_device, dtype=hs_dtype)

        # Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            attn_output = (
                self.attn1(
                    norm_hidden_states,
                    rotary_emb=rotary_emb,
                    **attention_kwargs,
                )
                * gate_msa
            )
            hidden_states = hidden_states + attn_output

        # Cross-Attention
        if self.use_cross_attention:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    rotary_emb=None,
                    **attention_kwargs,
                )
                * cross_attention_scale
            )
            hidden_states = hidden_states + attn_output

        # FFN Layer
        mlp_inputs = self.norm3(hidden_states)
        mlp_inputs = mlp_inputs * (1 + scale_mlp) + shift_mlp
        mlp_outputs = self.ff(mlp_inputs) * gate_mlp
        hidden_states = hidden_states + mlp_outputs

        return hidden_states


class TrellisSLatDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        inner_dim: int = 1280,
        in_channels: int = 8,
        out_channels: int = 8,
        num_layers: int = 21,
        resolution: int = 16,
        patch_size: int = 2,  # seems not used
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 768,
        qk_norm_self: Optional[str] = None,
        qk_norm_cross: Optional[str] = None,
        block_out_channels: List[int] = None,
        resnet_num_blocks: int = 2,
        resnet_scale_factor: int = 2,
        resnet_skip_connection: bool = True,
        pos_embed_type: Literal["ape", "rope"] = "ape",
        axes_dims_rope: List[int] = [256, 256, 256],
    ):
        super().__init__()

        self.resolution = resolution
        self.resnet_skip_connection = resnet_skip_connection
        self.pos_embed_type = pos_embed_type

        self.time_embedder = TimestepEmbedder(inner_dim)

        if pos_embed_type == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(inner_dim)
        elif pos_embed_type == "rope":
            self.pos_embedder = RotaryPositionEmbedder(axes_dims_rope)

        self.proj_in = sparse_torch.nn.Linear(
            in_channels,
            inner_dim if block_out_channels is None else block_out_channels[0],
        )

        # ResNet Input Blocks
        self.input_blocks = nn.ModuleList([])
        if block_out_channels is not None:
            for chs, next_chs in zip(
                block_out_channels, block_out_channels[1:] + [inner_dim]
            ):
                self.input_blocks.extend(
                    [
                        SparseResBlock3d(chs, emb_channels=inner_dim, out_channels=chs)
                        for _ in range(resnet_num_blocks - 1)
                    ]
                    + [
                        SparseResBlock3d(
                            chs,
                            emb_channels=inner_dim,
                            out_channels=next_chs,
                            downsample_factor=resnet_scale_factor,
                        )
                    ]
                )

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm_self=qk_norm_self,
                    qk_norm_cross=qk_norm_cross,
                    activation_fn="gelu-approximate",
                    use_modulation=True,
                    norm_elementwise_affine=False,
                    norm_cross_elementwise_affine=True,
                    norm_eps=1e-6,
                )
                for _ in range(num_layers)
            ]
        )

        # ResNet Output Blocks
        self.output_blocks = nn.ModuleList([])
        if block_out_channels is not None:
            for chs, prev_chs in zip(
                reversed(block_out_channels),
                [inner_dim] + list(reversed(block_out_channels[1:])),
            ):
                self.output_blocks.extend(
                    [
                        SparseResBlock3d(
                            prev_chs * 2 if resnet_skip_connection else prev_chs,
                            emb_channels=inner_dim,
                            out_channels=chs,
                            upsample_factor=resnet_scale_factor,
                        )
                    ]
                    + [
                        SparseResBlock3d(
                            chs * 2 if resnet_skip_connection else chs,
                            emb_channels=inner_dim,
                            out_channels=chs,
                        )
                        for _ in range(resnet_num_blocks - 1)
                    ]
                )

        self.proj_out = sparse_torch.nn.Linear(
            inner_dim if block_out_channels is None else block_out_channels[0],
            out_channels,
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Transformer blocks
        for block in self.blocks:
            block: SparseTransformerBlock
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        hidden_states: sparse_torch.SparseTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        # Time embedding
        temb = self.time_embedder(timestep.to(hidden_states.dtype))

        # Project in
        hidden_states = self.proj_in(hidden_states)

        # Pack with ResNet Input Blocks
        skips = []
        for block in self.input_blocks:
            hidden_states = block(hidden_states, temb)
            skips.append(hidden_states.feats)

        # Position embedding
        if self.pos_embed_type == "ape":
            hidden_states = hidden_states + self.pos_embedder(
                hidden_states.coords[:, 1:]
            ).to(hidden_states.dtype)
            rotary_emb = None
        elif self.pos_embed_type == "rope":
            rotary_emb = self.pos_embedder(hidden_states.coords[:, 1:])

        # Transformer Blocks
        for layer, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb, encoder_hidden_states, rotary_emb
                )
            else:
                hidden_states = block(
                    hidden_states, temb, encoder_hidden_states, rotary_emb
                )

        # Unpack with ResNet Output Blocks
        for block, skip in zip(self.output_blocks, reversed(skips)):
            if self.resnet_skip_connection:
                hidden_states = block(
                    hidden_states.replace(
                        torch.cat([hidden_states.feats, skip], dim=1)
                    ),
                    temb,
                )
            else:
                hidden_states = block(hidden_states, temb)

        # Project out
        hidden_states = F.layer_norm(hidden_states, hidden_states.shape[-1:])
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
