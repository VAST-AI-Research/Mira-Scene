# Adapted from https://github.com/huggingface/diffusers/blob/87f7d111437e1dad2a25d4653c57886f8f058cd3/src/diffusers/models/attention_processor.py#L3122-L3217

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention


class ShapeLayoutSelfAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a normalization layer and rotary embedding on query and key vector.

    Bidirectional attention only, no context_latent support.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: Dict[str, torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states['shape']  # [B, N, C]
        hidden_states_layout = hidden_states['layout']  # [B, N', C']

        batch_size, sequence_length, _ = (
            hidden_states_shape.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query_shape = attn.to_q(hidden_states_shape)
        key_shape = attn.to_k(hidden_states_shape)
        value_shape = attn.to_v(hidden_states_shape)

        query_layout = attn.add_q_proj(hidden_states_layout)
        key_layout = attn.add_k_proj(hidden_states_layout)
        value_layout = attn.add_v_proj(hidden_states_layout)

        # make sure key_shape and key_layout have the same channel dimension
        assert key_shape.shape[-1] == key_layout.shape[-1], "key_shape and key_layout must have the same channel dimension"

        # Reshape for multi-head attention
        inner_dim = key_shape.shape[-1]
        head_dim = inner_dim // attn.heads

        query_shape = query_shape.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        key_shape = key_shape.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_shape = value_shape.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query_layout = query_layout.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, heads, N', head_dim]
        key_layout = key_layout.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_layout = value_layout.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query_shape = attn.norm_q(query_shape)
        if attn.norm_k is not None:
            key_shape = attn.norm_k(key_shape)
        if attn.norm_q_layout is not None:
            query_layout = attn.norm_q_layout(query_layout)
        if attn.norm_k_layout is not None:
            key_layout = attn.norm_k_layout(key_layout)

        # Bidirectional attention: both streams attend to concatenated shape+layout tokens
        _k_joint = torch.cat([key_shape, key_layout], dim=2)
        _v_joint = torch.cat([value_shape, value_layout], dim=2)
        hidden_states_shape = F.scaled_dot_product_attention(
            query_shape, _k_joint, _v_joint, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states_layout = F.scaled_dot_product_attention(
            query_layout, _k_joint, _v_joint, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states_output = {}
        for name, tensor in zip(
            ["shape", "layout"],
            [hidden_states_shape, hidden_states_layout],
        ):
            # reshape and linear proj
            hidden_states_tensor = tensor.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states_output[name] = hidden_states_tensor.to(query_shape.dtype)

        # linear proj
        hidden_states_output["shape"] = attn.to_out[0](hidden_states_output["shape"])
        # dropout
        hidden_states_output["shape"] = attn.to_out[1](hidden_states_output["shape"])

        hidden_states_output["layout"] = attn.to_add_out(hidden_states_output["layout"])

        return hidden_states_output
