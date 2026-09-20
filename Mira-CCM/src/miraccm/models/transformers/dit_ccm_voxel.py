"""
TrellisSparseStructureDiTModel variant that uses a 2-D canonical coordinate map
as the layout latent instead of a 3-D voxel grid.

Layout stream details:
  - Layout latent shape: [B, 3, H', W']  (2-D spatial)
  - Layout position embedding: 2-D absolute position embedding (via offset 3-D)
  - Forward unpatchify: reshape to [B, C, H', W'] instead of [B, C, X, Y, Z]
"""

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

from miraccm.models.attention_processor import (
    ShapeLayoutSelfAttnProcessor,
)

# Re-use the block implementation from the simplified context variant
from .dit_block import (
    patchify,
    unpatchify,
    TimestepEmbedder,
    AbsolutePositionEmbedder,
    RotaryPositionEmbedder,
    MultiHeadRMSNorm,
    DiTBlock,
)


# ---------------------------------------------------------------------------
# 2-D patchify / unpatchify helpers
# ---------------------------------------------------------------------------

def patchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchify a [B, C, H, W] tensor. Wrapper around the generic patchify."""
    assert x.dim() == 4, f"Expected 4-D tensor, got {x.dim()}-D"
    return patchify(x, patch_size)


def unpatchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Unpatchify a [B, C*p^2, H//p, W//p] tensor back to [B, C, H, W]."""
    assert x.dim() == 4, f"Expected 4-D tensor, got {x.dim()}-D"
    return unpatchify(x, patch_size)


# ---------------------------------------------------------------------------
# Cascaded layout stream helpers
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulate: x * (1 + scale) + shift, broadcasting over token dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayerLayout(nn.Module):
    """
    Final output layer for the layout (CCM) stream.

    Two modes:
      - use_adaLN=True  (cascaded): LayerNorm + adaLN modulation (shift/scale from
        timestep embedding) + Linear. Matches DiT FinalLayer design.
      - use_adaLN=False (non-cascaded): LayerNorm + Linear. Simple projection.

    In both cases, forward signature is (x, temb) for a unified calling convention.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 use_adaLN: bool = True):
        super().__init__()
        self.use_adaLN = use_adaLN
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        if use_adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, hidden_size]  - layout token sequence
        temb: [B, hidden_size]     - timestep embedding
        """
        if self.use_adaLN:
            shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = F.layer_norm(x, x.shape[-1:])
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CCMVoxelDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    Dual-stream (shape + layout) DiT model where the layout stream operates on
    a 2-D canonical coordinate map instead of a 3-D voxel grid.

    latent_config must have 'layout' with:
      in_channels: 3
      pos_embedder:
        resolution: <H'=W'> (integer, H' == W')
        patch_size: <int>
    """

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
        latent_config: Optional[Dict[str, Any]] = None,
        # ---------------------------------------------------------------
        # Cascaded layout stream:
        #   1. Input-side concat of image + mask tokens onto CCM
        #   2. First half of blocks process at given layout_ps (coarse)
        #   3. After block depth//2: fuse with image semantics via
        #      proj_fusion_layout, then pixel-shuffle upsample 2x
        #   4. Second half of blocks process at fine resolution
        #   5. FinalLayerLayout (adaLN) -> unpatchify with layout_ps // 2
        # ---------------------------------------------------------------
        use_cascaded_layout: bool = False,
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
        self.use_cascaded_layout = use_cascaded_layout
        self.patch_size_dict = {
            k: latent_config[k]['pos_embedder']['patch_size'] for k in latent_config
        }

        # ---------------------------------------------------------------
        # Timestep embedding
        # ---------------------------------------------------------------
        self.time_embedder = TimestepEmbedder(inner_dim)

        # ---------------------------------------------------------------
        # Shape stream position embedding (3-D, unchanged)
        # ---------------------------------------------------------------
        self.register_buffer(
            "pos_embed",
            self._set_position_embed_3d(
                self.inner_dim, self.resolution // self.patch_size
            ),
        )

        # ---------------------------------------------------------------
        # Layout stream position embedding (2-D via offset 3-D)
        # ---------------------------------------------------------------
        layout_resolution = latent_config['layout']['pos_embedder']['resolution']
        self.layout_ps_in = self.patch_size_dict['layout']
        shape_pos_resolution = self.resolution // self.patch_size
        # After patchify the spatial dims become (layout_resolution // layout_ps_in)^2
        self.register_buffer(
            "layout_pos_embed",
            self._set_position_embed_2d(
                inner_dim,
                layout_resolution // self.layout_ps_in,
                offset=shape_pos_resolution,
            ),
        )

        # ---------------------------------------------------------------
        # Input / output projections
        # ---------------------------------------------------------------
        self.proj_in = nn.Linear(
            self.in_channels * self.patch_size_dict['shape'] ** 3, self.inner_dim
        )

        # Conv2d-based patchify for the layout (CCM) stream.
        # concat_img_mask_to_layout=True: CCM channels + RGB image (3) + binary mask (1)
        in_chans = latent_config['layout']['in_channels']
        conv_in_chans = in_chans + 3 + 1  # CCM + RGB image + mask
        self.conv_patchify_layout = nn.Conv2d(
            conv_in_chans, self.inner_dim,
            kernel_size=self.layout_ps_in, stride=self.layout_ps_in,
        )

        self.proj_out = nn.Linear(
            self.inner_dim, self.out_channels * self.patch_size_dict['shape'] ** 3
        )
        # ---------------------------------------------------------------
        # Layout output layer (unified for both cascaded and non-cascaded)
        # ---------------------------------------------------------------
        out_ch = latent_config['layout']['in_channels']
        if use_cascaded_layout:
            assert self.layout_ps_in >= 2, (
                f"use_cascaded_layout requires layout patch_size >= 2, got {self.layout_ps_in}"
            )
            self.layout_ps_out = self.layout_ps_in // 2
            self.final_layer_layout = FinalLayerLayout(
                inner_dim, self.layout_ps_out, out_ch, use_adaLN=True,
            )

            # proj_fusion_layout: fuse coarse layout tokens with semantics, then
            # expand channels 4x to allow pixel-shuffle 2x upsample.
            mask_ps2 = self.layout_ps_in ** 2
            self.proj_fusion_layout = nn.Sequential(
                nn.Linear(inner_dim + cross_attention_dim + mask_ps2, inner_dim * 4),
                nn.SiLU(),
                nn.Linear(inner_dim * 4, inner_dim * 4),
                nn.SiLU(),
                nn.Linear(inner_dim * 4, inner_dim * 4),
            )

            # Position embedding for the fine-resolution tokens after pixel-shuffle.
            fine_res = layout_resolution // self.layout_ps_out
            self.register_buffer(
                "layout_pos_embed_fine",
                self._set_position_embed_2d(
                    inner_dim,
                    fine_res,
                    step=0.5,
                    offset=shape_pos_resolution,
                ),
            )
        else:
            self.layout_ps_out = self.layout_ps_in
            self.final_layer_layout = FinalLayerLayout(
                inner_dim, self.layout_ps_out, out_ch, use_adaLN=False,
            )

        # ---------------------------------------------------------------
        # DiT blocks (always use DiTBlock from simplified context variant)
        # ---------------------------------------------------------------
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

        # Learnable class embedding (image / mask / image_cropped / mask_cropped)
        # 2 class embeddings:
        #   index 0 -> non-cropped keys  ("image", "mask")
        #   index 1 -> cropped keys      ("image_cropped", "mask_cropped")
        self.class_embedding = nn.Parameter(torch.zeros(2, cross_attention_dim))

        self.gradient_checkpointing = False

        self._init_weights()

    # ------------------------------------------------------------------
    # Position embedding helpers
    # ------------------------------------------------------------------

    def _set_position_embed_3d(
        self,
        dim: int,
        resolution: Optional[int] = None,
        interval: int = 1,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """3-D absolute position embedding used for the shape stream."""
        pos_embedder = AbsolutePositionEmbedder(dim, in_channels=3)
        if coords is None:
            assert resolution is not None, "resolution must be provided when coords is None"
            coords = torch.meshgrid(
                *[torch.arange(0, resolution, interval) for _ in range(3)],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3).float()
        else:
            coords = coords.float()
        return pos_embedder(coords)  # [N^3, dim]

    def _set_position_embed_2d(
        self, dim: int, resolution: int, step: float = 1.0, offset: float = 0.0
    ) -> torch.Tensor:
        """2-D absolute position embedding used for the layout (CCM) stream.

        Uses the 3-D position embedder on coordinates [y, x, 0] with offset so
        layout coordinate range does not overlap the shape branch.

        Args:
            dim:        embedding dimension (inner_dim).
            resolution: number of tokens per spatial side (H'//patch_size).
            step:       coordinate spacing between adjacent tokens.
            offset:     coordinate offset applied to the first two axes.

        Returns:
            Tensor of shape [resolution^2, dim].
        """
        coords_1d = offset + torch.arange(resolution, dtype=torch.float32) * step
        ys, xs = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        zs = torch.zeros_like(xs)
        coords = torch.stack([ys, xs, zs], dim=-1).reshape(-1, 3)  # [H*W, 3]
        return self._set_position_embed_3d(dim, coords=coords)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Basic weight initialization (called automatically in __init__).

        - Xavier uniform for all Linear layers
        - Zero-init for adaLN modulation outputs and proj_out
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)
            if hasattr(block, "modulation_layout"):
                nn.init.constant_(block.modulation_layout[-1].weight, 0)
                nn.init.constant_(block.modulation_layout[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # Zero-out final_layer_layout
        if hasattr(self.final_layer_layout, "adaLN_modulation"):
            nn.init.constant_(self.final_layer_layout.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer_layout.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_layout.linear.weight, 0)
        nn.init.constant_(self.final_layer_layout.linear.bias, 0)

    def init_layout_from_shape(self) -> None:
        """Initialize layout branch from pretrained shape branch.

        Two steps:
          1. Xavier-init layout-only modules (conv_patchify, proj_fusion, final_layer)
          2. Copy shape-stream weights → layout-stream counterparts (warm start)

        Call this AFTER loading pretrained weights (from_pretrained).
        """
        print("=== init_layout_from_shape: Initialising layout branch from shape ===")

        # Step 1: Xavier-init layout-only new modules
        if hasattr(self, "conv_patchify_layout"):
            nn.init.xavier_uniform_(self.conv_patchify_layout.weight.view(
                self.conv_patchify_layout.weight.size(0), -1))
            nn.init.constant_(self.conv_patchify_layout.bias, 0)
        if hasattr(self, "proj_fusion_layout"):
            for layer in self.proj_fusion_layout:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        if hasattr(self.final_layer_layout, "adaLN_modulation"):
            nn.init.constant_(self.final_layer_layout.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer_layout.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer_layout.linear.weight, 0)
            nn.init.constant_(self.final_layer_layout.linear.bias, 0)

        # Step 2: Copy shape → layout block weights
        for block in self.blocks:
            if hasattr(block, "modulation_layout") and hasattr(block, "modulation"):
                block.modulation_layout.load_state_dict(block.modulation.state_dict())
            block.norm1_layout.load_state_dict(block.norm1.state_dict())
            if hasattr(block, "norm2_layout"):
                block.norm2_layout.load_state_dict(block.norm2.state_dict())
            block.norm3_layout.load_state_dict(block.norm3.state_dict())
            block.ff_layout.load_state_dict(block.ff.state_dict())
            if hasattr(block, "attn2_layout"):
                block.attn2_layout.load_state_dict(block.attn2.state_dict())
            if hasattr(block.attn1, "add_q_proj") and hasattr(block.attn1, "to_q"):
                block.attn1.add_q_proj.load_state_dict(block.attn1.to_q.state_dict())
            if hasattr(block.attn1, "add_k_proj") and hasattr(block.attn1, "to_k"):
                block.attn1.add_k_proj.load_state_dict(block.attn1.to_k.state_dict())
            if hasattr(block.attn1, "add_v_proj") and hasattr(block.attn1, "to_v"):
                block.attn1.add_v_proj.load_state_dict(block.attn1.to_v.state_dict())
            if hasattr(block.attn1, "to_add_out") and hasattr(block.attn1, "to_out"):
                source = (
                    block.attn1.to_out[0]
                    if isinstance(block.attn1.to_out, nn.ModuleList)
                    else block.attn1.to_out
                )
                block.attn1.to_add_out.load_state_dict(source.state_dict())
            if hasattr(block.attn1, "norm_q_layout") and hasattr(block.attn1, "norm_q"):
                block.attn1.norm_q_layout.load_state_dict(block.attn1.norm_q.state_dict())
            if hasattr(block.attn1, "norm_k_layout") and hasattr(block.attn1, "norm_k"):
                block.attn1.norm_k_layout.load_state_dict(block.attn1.norm_k.state_dict())

    # ------------------------------------------------------------------
    # Misc properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name, module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors "
                f"{len(processor)} does not match the number of attention layers: {count}."
            )

        def fn_recursive_attn_processor(name, module, processor):
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
        self.set_attn_processor(HunyuanAttnProcessor2_0())

    # ------------------------------------------------------------------
    # Layout preprocessing helpers
    # ------------------------------------------------------------------

    def _build_cascaded_semantics(
        self,
        layout_encoder_hidden_states,
        layout_mask_for_concat: torch.Tensor,
        N_layout: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build concatenated semantics tensor for cascaded midpoint fusion.

        Extracts DINOv2 image tokens (L2-normalized) and patchified mask,
        concatenates them into a single tensor ready for proj_fusion_layout.

        Returns:
            sem: [B, N_layout, cross_attention_dim + layout_ps_in^2]
        """
        # Image: DINOv2 tokens from image_cropped (strip CLS + register)
        _img_feat = layout_encoder_hidden_states["image_cropped"]
        _ne = _img_feat.shape[1] - N_layout
        if _ne > 0:
            _img_feat = _img_feat[:, _ne:, :]  # [B, N_layout, cross_attention_dim]
        sem_img = F.normalize(_img_feat.to(dtype), dim=-1)

        # Mask: patchified layout_mask_for_concat → [B, N_layout, ps_in^2]
        _mask_pat = patchify_2d(layout_mask_for_concat.to(dtype), self.layout_ps_in)
        sem_mask = (
            _mask_pat.view(*_mask_pat.shape[:2], -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        return torch.cat([sem_img, sem_mask], dim=-1)

    def _patchify_layout(
        self,
        layout_latent: torch.Tensor,
        layout_image_for_conv: Optional[torch.Tensor],
        layout_mask_for_concat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Conv2d patchify: concat CCM + image + mask in channel dim, then strided conv → tokens.

        Args:
            layout_latent: [B, C, H', W'] noisy CCM latent
            layout_image_for_conv: [B, 3, H', W'] downsampled RGB or None
            layout_mask_for_concat: [B, 1, H', W'] downsampled mask or None

        Returns:
            layout_tokens: [B, N_layout, inner_dim]
        """
        _spatial_parts = [layout_latent]
        if layout_image_for_conv is not None:
            _spatial_parts.append(
                layout_image_for_conv.to(dtype=layout_latent.dtype)
            )
        if layout_mask_for_concat is not None:
            _spatial_parts.append(
                layout_mask_for_concat.to(dtype=layout_latent.dtype)
            )
        ccm_input = torch.cat(_spatial_parts, dim=1)
        return (
            self.conv_patchify_layout(ccm_input)
            .flatten(2).transpose(1, 2).contiguous()
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: dict,                              # {'shape': [B,C,X,Y,Z], 'layout': [B,3,H',W']}
        timestep: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, Dict[str, torch.Tensor]] = None,  # shape branch condition (shape_hs_dict from caller)
        layout_encoder_hidden_states: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,  # layout branch condition (layout_hs_dict from caller)
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # Optional pixel-aligned single-channel mask: [B, 1, layout_resolution, layout_resolution]
        layout_mask_for_concat: Optional[torch.Tensor] = None,
        # Optional RGB image downsampled to layout resolution: [B, 3, layout_resolution, layout_resolution]
        layout_image_for_conv: Optional[torch.Tensor] = None,
    ):
        layout_resolution = self.config.latent_config['layout']['pos_embedder']['resolution']
        N_layout = (layout_resolution // self.layout_ps_in) ** 2

        # ------------------------------------------------------------------
        # [Cascaded layout] Build semantics for midpoint fusion
        # ------------------------------------------------------------------
        cascaded_sem = None
        if self.use_cascaded_layout:
            cascaded_sem = self._build_cascaded_semantics(
                layout_encoder_hidden_states, layout_mask_for_concat,
                N_layout, hidden_states['layout'].dtype,
            )

        # ------------------------------------------------------------------
        # Patchify both streams
        # ------------------------------------------------------------------
        # Shape stream: [B, C, X, Y, Z] -> [B, N_shape, C*p^3]
        hidden_states['shape'] = patchify(hidden_states['shape'], self.patch_size_dict['shape'])
        hidden_states['shape'] = (
            hidden_states['shape'].view(*hidden_states['shape'].shape[:2], -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        # Layout stream: [B, C, H', W'] -> [B, N_layout, inner_dim]
        hidden_states['layout'] = self._patchify_layout(
            hidden_states['layout'], layout_image_for_conv, layout_mask_for_concat,
        )

        # ------------------------------------------------------------------
        # Time embedding
        # ------------------------------------------------------------------
        temb = self.time_embedder(timestep.to(hidden_states['shape'].dtype))

        # ------------------------------------------------------------------
        # Input projections
        # ------------------------------------------------------------------
        hidden_states['shape'] = self.proj_in(hidden_states['shape'])
        # Conv path: tokens are already in inner_dim space; no further projection needed.

        # ------------------------------------------------------------------
        # Position embeddings
        # ------------------------------------------------------------------
        pos_embed = self.pos_embed.to(hidden_states['shape'].device, hidden_states['shape'].dtype)
        hidden_states['shape'] = hidden_states['shape'] + pos_embed[None]

        layout_pos_embed = self.layout_pos_embed.to(
            hidden_states['layout'].device, hidden_states['layout'].dtype
        )
        hidden_states['layout'] = hidden_states['layout'] + layout_pos_embed[None]

        # ------------------------------------------------------------------
        # Class embeddings -> encoder_hidden_states (shape stream cross-attn)
        # index 0 -> non-cropped keys, index 1 -> cropped keys
        # ------------------------------------------------------------------
        _class_emb_keys = ["image", "mask", "image_cropped", "mask_cropped"]
        if isinstance(encoder_hidden_states, dict):
            parts = []
            for k in _class_emb_keys:
                v = encoder_hidden_states.get(k)
                if v is not None:
                    emb_idx = 1 if "cropped" in k else 0
                    parts.append(v + self.class_embedding[emb_idx].to(dtype=v.dtype, device=v.device))
            encoder_hidden_states = torch.cat(parts, dim=1) if parts else None

        if isinstance(layout_encoder_hidden_states, dict):
            parts = []
            for k in _class_emb_keys:
                v = layout_encoder_hidden_states.get(k)
                if v is not None:
                    emb_idx = 1 if "cropped" in k else 0
                    parts.append(v + self.class_embedding[emb_idx].to(dtype=v.dtype, device=v.device))
            layout_encoder_hidden_states = torch.cat(parts, dim=1) if parts else None

        # ------------------------------------------------------------------
        # DiT blocks  (two-half loop for cascaded layout; single loop otherwise)
        # ------------------------------------------------------------------
        depth = len(self.blocks)
        half  = depth // 2

        def _run_block(block, hs):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(
                        shape, layout, temb,
                        encoder_hidden_states, layout_encoder_hidden_states,
                    ):
                        return_d = module(
                            {'shape': shape, 'layout': layout},
                            temb,
                            encoder_hidden_states,
                            layout_encoder_hidden_states,
                        )
                        return return_d['shape'], return_d['layout']
                    return custom_forward

                hs['shape'], hs['layout'] = (
                    self._gradient_checkpointing_func(
                        create_custom_forward(block),
                        hs['shape'],
                        hs['layout'],
                        temb,
                        encoder_hidden_states,
                        layout_encoder_hidden_states,
                    )
                )
            else:
                hs = block(
                    hs, temb,
                    encoder_hidden_states, layout_encoder_hidden_states,
                )
            return hs

        # ---- First half ----
        for block in self.blocks[:half]:
            hidden_states = _run_block(block, hidden_states)

        # ---- Midpoint: cascaded layout pixel-shuffle upsample ----
        if self.use_cascaded_layout:
            B, N_c, D = hidden_states['layout'].shape

            # proj_fusion: [B, N_c, D + sem_dim] -> [B, N_c, D*4]
            fused = self.proj_fusion_layout(
                torch.cat([hidden_states['layout'], cascaded_sem.to(hidden_states['layout'].dtype)], dim=-1)
            )

            # Pixel-shuffle 2x: [B, N_c, D*4] -> [B, 4*N_c, D]
            h_c = int(N_c ** 0.5)
            w_c = h_c
            fused = fused.reshape(B, h_c, w_c, 2, 2, D)
            fused = torch.einsum("nhwpqc->nchpwq", fused)
            fused = fused.reshape(B, D, h_c * 2, w_c * 2)
            hidden_states['layout'] = fused.flatten(2).transpose(1, 2).contiguous()
            # -> [B, N_fine, D]  where N_fine = 4 * N_c

            # Add fine-resolution position embedding
            fine_pe = self.layout_pos_embed_fine.to(
                hidden_states['layout'].device, hidden_states['layout'].dtype
            )
            hidden_states['layout'] = hidden_states['layout'] + fine_pe[None]

        # ---- Second half ----
        for block in self.blocks[half:]:
            hidden_states = _run_block(block, hidden_states)

        # ------------------------------------------------------------------
        # Output norm + projection
        # ------------------------------------------------------------------
        hidden_states['shape'] = F.layer_norm(hidden_states['shape'], hidden_states['shape'].shape[-1:])
        hidden_states['shape'] = self.proj_out(hidden_states['shape'])

        hidden_states['layout'] = self.final_layer_layout(hidden_states['layout'], temb)

        # ------------------------------------------------------------------
        # Unpatchify
        # ------------------------------------------------------------------
        # Shape stream (3-D)
        shape_ps = self.patch_size_dict['shape']
        hidden_states['shape'] = hidden_states['shape'].permute(0, 2, 1).view(
            hidden_states['shape'].shape[0],
            hidden_states['shape'].shape[2],
            *[self.resolution // shape_ps] * 3,
        )
        hidden_states['shape'] = unpatchify(hidden_states['shape'], shape_ps).contiguous()

        # Layout stream (2-D)
        out_ps = self.layout_ps_out
        spatial_res = int(hidden_states['layout'].shape[1] ** 0.5)
        hidden_states['layout'] = hidden_states['layout'].permute(0, 2, 1).view(
            hidden_states['layout'].shape[0],
            hidden_states['layout'].shape[2],
            spatial_res,
            spatial_res,
        )
        hidden_states['layout'] = unpatchify_2d(hidden_states['layout'], out_ps).contiguous()

        return hidden_states
