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

        # Modulation
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # 1. Self-Attention
        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm=None,
            bias=qkv_bias,
            processor=HunyuanAttnProcessor2_0(),
        )
        # Set qk norm manually
        if qk_norm_self:
            self.attn1.norm_q = MultiHeadRMSNorm(self.head_dim, num_attention_heads)
            self.attn1.norm_k = MultiHeadRMSNorm(self.head_dim, num_attention_heads)

        # 2. Cross-Attention
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.attn2 = Attention(
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
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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

        # Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + scale_msa.unsqueeze(1)
        ) + shift_msa.unsqueeze(1)
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **attention_kwargs,
        ) * gate_msa.unsqueeze(1)
        hidden_states = hidden_states + attn_output

        # Cross-Attention
        hidden_states = (
            hidden_states
            + self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mask=encoder_attention_mask,
                **attention_kwargs,
            )
            * cross_attention_scale
        )

        # Feed-forward
        mlp_inputs = self.norm3(hidden_states)
        mlp_inputs = mlp_inputs * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_outputs = self.ff(mlp_inputs) * gate_mlp.unsqueeze(1)
        hidden_states = hidden_states + mlp_outputs

        return hidden_states


class TrellisSparseStructureDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):

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
        patch_size: int = 1,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 768,
        qk_norm_self: bool = True,
        qk_norm_cross: bool = True,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.inner_dim = inner_dim
        self.num_heads = num_attention_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        # Timestep embedding
        self.time_embedder = TimestepEmbedder(inner_dim)

        # Position embedding
        self.register_buffer(
            "pos_embed",
            self._set_position_embed(
                self.inner_dim, self.resolution // self.patch_size
            ),
        )

        self.proj_in = nn.Linear(self.in_channels * self.patch_size**3, self.inner_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_heads,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm_self=qk_norm_self,
                    qk_norm_cross=qk_norm_cross,
                    qkv_bias=True,
                    mlp_ratio=self.mlp_ratio,
                    dropout=0.0,
                    activation_fn="gelu-approximate",
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(
            self.inner_dim, self.out_channels * self.patch_size**3
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def _set_position_embed(self, dim: int, resolution: int) -> None:
        pos_embedder = AbsolutePositionEmbedder(dim, 3)
        coords = torch.meshgrid(
            *[torch.arange(res) for res in [resolution] * 3],
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        pos_embed = pos_embedder(coords)

        return pos_embed

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(HunyuanAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)
                # NEW
                if (
                    not module.is_cross_attention
                ):  # is self attention, delete to_q to_k to_v
                    print("delete param for self attention layer")
                    if hasattr(module, "to_q"):
                        delattr(module, "to_q")
                    if hasattr(module, "to_k"):
                        delattr(module, "to_k")
                    if hasattr(module, "to_v"):
                        delattr(module, "to_v")
                else:  # is cross attention, delete to_k to_v
                    print("delete param for self cross_attention layer")
                    if hasattr(module, "to_k"):
                        delattr(module, "to_k")
                    if hasattr(module, "to_v"):
                        delattr(module, "to_v")

        self.set_attn_processor(FusedHunyuanAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        assert [*hidden_states.shape] == [
            hidden_states.shape[0],
            self.in_channels,
            *[self.resolution] * 3,
        ], f"Input shape mismatch, got {hidden_states.shape}, expected {[hidden_states.shape[0], self.in_channels, *[self.resolution] * 3]}"

        # Patchify
        hidden_states = patchify(hidden_states, self.patch_size)
        hidden_states = (
            hidden_states.view(*hidden_states.shape[:2], -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        # Time embedding
        temb = self.time_embedder(timestep.to(hidden_states.dtype))

        # Project in
        hidden_states = self.proj_in(hidden_states)

        # Position embedding
        self.pos_embed = self.pos_embed.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + self.pos_embed[None]

        # DiT Blocks
        for layer, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb, encoder_hidden_states
                )
            else:
                hidden_states = block(hidden_states, temb, encoder_hidden_states)

        # Norm out
        hidden_states = F.layer_norm(hidden_states, hidden_states.shape[-1:])

        # Project out
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.permute(0, 2, 1).view(
            hidden_states.shape[0],
            hidden_states.shape[2],
            *[self.resolution // self.patch_size] * 3,
        )
        hidden_states = unpatchify(hidden_states, self.patch_size).contiguous()

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
