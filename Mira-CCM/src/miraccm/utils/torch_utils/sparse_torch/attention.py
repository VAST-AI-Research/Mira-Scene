import math
from enum import Enum
from typing import *
from typing import Any, Dict, Optional

import torch

from .tensor import SparseTensor


class AttentionBackend(Enum):
    AUTO = "auto"
    XFORMERS = "xformers"
    FLASH_ATTN = "flash"
    TORCH = "torch"


class AttentionBackendBase:
    """Base class for attention backends"""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def is_available(self) -> bool:
        """Check if the backend is available"""
        raise NotImplementedError

    def compute_attention(self, q, k, v, q_seqlen, kv_seqlen, num_all_args, **kwargs):
        """Compute attention"""
        raise NotImplementedError


class XFormersBackend(AttentionBackendBase):
    def __init__(self):
        super().__init__("xformers", priority=3)
        self._xops = None

    def is_available(self) -> bool:
        try:
            import xformers.ops as xops

            self._xops = xops
            return True
        except ImportError:
            return False

    def compute_attention(self, q, k, v, q_seqlen, kv_seqlen, num_all_args, **kwargs):
        if self._xops is None:
            import xformers.ops as xops

            self._xops = xops

        if num_all_args == 1:
            q, k, v = q.unbind(dim=1)
        elif num_all_args == 2:
            k, v = v.unbind(dim=1)

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = self._xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        return self._xops.memory_efficient_attention(q, k, v, mask)[0]


class FlashAttentionBackend(AttentionBackendBase):
    def __init__(self):
        super().__init__("flash", priority=4)
        self._flash_attn = None

    def is_available(self) -> bool:
        try:
            import flash_attn

            self._flash_attn = flash_attn
            return True
        except ImportError:
            return False

    def compute_attention(
        self, q, k, v, q_seqlen, kv_seqlen, num_all_args, device, **kwargs
    ):
        if self._flash_attn is None:
            import flash_attn

            self._flash_attn = flash_attn

        cu_seqlens_q = (
            torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)])
            .int()
            .to(device)
        )
        if num_all_args in [2, 3]:
            cu_seqlens_kv = (
                torch.cat(
                    [torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]
                )
                .int()
                .to(device)
            )

        if num_all_args == 1:
            return self._flash_attn.flash_attn_varlen_qkvpacked_func(
                q, cu_seqlens_q, max(q_seqlen)
            )
        elif num_all_args == 2:
            return self._flash_attn.flash_attn_varlen_kvpacked_func(
                q, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
            )
        elif num_all_args == 3:
            return self._flash_attn.flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
            )


class TorchBackend(AttentionBackendBase):
    def __init__(self):
        super().__init__("torch", priority=1)

    def is_available(self) -> bool:
        return True  # PyTorch is always available

    def compute_attention(self, q, k, v, q_seqlen, kv_seqlen, num_all_args, **kwargs):
        # Simple PyTorch native implementation (as fallback)
        if num_all_args == 1:
            q, k, v = q.unbind(dim=1)
        elif num_all_args == 2:
            k, v = v.unbind(dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

        # Apply mask (simplified handling here)
        # In practice, appropriate masks should be created based on q_seqlen and kv_seqlen

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights
        return torch.matmul(attn_weights, v)


# Backend registry
_ATTENTION_BACKENDS = {
    AttentionBackend.XFORMERS: XFormersBackend(),
    AttentionBackend.FLASH_ATTN: FlashAttentionBackend(),
    AttentionBackend.TORCH: TorchBackend(),
}


def get_best_available_backend() -> AttentionBackendBase:
    """Get the best available backend"""
    available_backends = [
        backend for backend in _ATTENTION_BACKENDS.values() if backend.is_available()
    ]
    if not available_backends:
        raise RuntimeError("No attention backend available")

    # Sort by priority and return the highest priority backend
    return max(available_backends, key=lambda x: x.priority)


def get_backend(
    backend: Optional[Union[AttentionBackend, str]] = None,
) -> AttentionBackendBase:
    """Get the specified backend or the best available backend

    Args:
        backend: Either an AttentionBackend enum or a string representing the backend name.
                Supported strings: "auto", "xformers", "flash", "torch"
    """
    if backend is None or backend == AttentionBackend.AUTO or backend == "auto":
        return get_best_available_backend()

    # Handle string input
    if isinstance(backend, str):
        # Convert string to enum
        try:
            backend_enum = AttentionBackend(backend)
        except ValueError:
            raise ValueError(
                f"Unknown backend string: {backend}. Supported values: {[e.value for e in AttentionBackend]}"
            )
        backend = backend_enum

    # Handle enum input
    if backend not in _ATTENTION_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    return _ATTENTION_BACKENDS[backend]


__all__ = [
    "sparse_scaled_dot_product_attention",
    "sparse_windowed_scaled_dot_product_attention",
    "AttentionBackend",
    "get_backend",
    "get_best_available_backend",
]


@overload
def sparse_scaled_dot_product_attention(
    qkv: SparseTensor,
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor,
    kv: Union[SparseTensor, torch.Tensor],
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    kv: SparseTensor,
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor,
    k: SparseTensor,
    v: SparseTensor,
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    k: SparseTensor,
    v: SparseTensor,
    *,
    backend: Optional[AttentionBackend] = None,
    window_size: Optional[int] = None,
    shift_window: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


def sparse_scaled_dot_product_attention(*args, **kwargs):
    # Extract parameters
    backend = kwargs.pop("backend", None)
    window_size = kwargs.pop("window_size", None)
    shift_window = kwargs.pop("shift_window", None)

    # If window parameters are provided, use windowed attention
    if window_size is not None:
        return sparse_windowed_scaled_dot_product_attention(
            *args,
            window_size=window_size,
            shift_window=shift_window or (0, 0, 0),
            backend=backend,
            **kwargs,
        )

    # Get backend instance
    backend_instance = get_backend(backend)

    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert (
        num_all_args in arg_names_dict
    ), f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(
            qkv, SparseTensor
        ), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert (
            len(qkv.shape) == 4 and qkv.shape[1] == 3
        ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [
            qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])
        ]
        kv_seqlen = q_seqlen
        qkv = qkv.feats  # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(kv, (SparseTensor, torch.Tensor))
            or isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert (
            q.shape[0] == kv.shape[0]
        ), f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert (
                len(q.shape) == 3
            ), f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, C]
        else:
            assert (
                len(q.shape) == 4
            ), f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)  # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert (
                len(kv.shape) == 4 and kv.shape[1] == 2
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [
                kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])
            ]
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            assert (
                len(kv.shape) == 5
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)  # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(k, (SparseTensor, torch.Tensor))
            and type(k) == type(v)
            or isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert (
            q.shape[0] == k.shape[0] == v.shape[0]
        ), f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert (
                len(q.shape) == 3
            ), f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, Ci]
        else:
            assert (
                len(q.shape) == 4
            ), f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert (
                len(k.shape) == 3
            ), f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert (
                len(v.shape) == 3
            ), f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [
                k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])
            ]
            k = k.feats  # [T_KV, H, Ci]
            v = v.feats  # [T_KV, H, Co]
        else:
            assert (
                len(k.shape) == 4
            ), f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert (
                len(v.shape) == 4
            ), f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)  # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)  # [T_KV, H, Co]

    # Use backend to compute attention
    out = backend_instance.compute_attention(
        q, k, v, q_seqlen, kv_seqlen, num_all_args, device=device, **kwargs
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (
        (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    )
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(
        shift_window, device=tensor.device, dtype=torch.int32
    ).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(
        window_size, device=tensor.device, dtype=torch.int32
    ).unsqueeze(0)
    shifted_indices = (
        shifted_coords
        * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    ).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = (
        torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32)
        // OFFSET[0]
    )
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices


@overload
def sparse_windowed_scaled_dot_product_attention(
    qkv: SparseTensor,
    *,
    window_size: int,
    shift_window: Tuple[int, ...] = (0, 0, 0),
    backend: Optional[AttentionBackend] = None,
) -> SparseTensor:
    """
    Apply windowed scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_windowed_scaled_dot_product_attention(
    q: SparseTensor,
    kv: SparseTensor,
    *,
    window_size: int,
    shift_window: Tuple[int, ...] = (0, 0, 0),
    backend: Optional[AttentionBackend] = None,
) -> SparseTensor:
    """
    Apply windowed scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.
    """
    ...


@overload
def sparse_windowed_scaled_dot_product_attention(
    q: SparseTensor,
    k: SparseTensor,
    v: SparseTensor,
    *,
    window_size: int,
    shift_window: Tuple[int, ...] = (0, 0, 0),
    backend: Optional[AttentionBackend] = None,
) -> SparseTensor:
    """
    Apply windowed scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.
        backend (AttentionBackend, optional): Attention backend to use. Defaults to auto.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...


def sparse_windowed_scaled_dot_product_attention(*args, **kwargs):
    # Extract window parameters
    window_size = kwargs.pop("window_size", None)
    shift_window = kwargs.pop("shift_window", (0, 0, 0))
    backend = kwargs.pop("backend", None)

    if window_size is None:
        raise ValueError("window_size is required for windowed attention")

    # Get backend instance
    backend_instance = get_backend(backend)

    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert (
        num_all_args in arg_names_dict
    ), f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(
            qkv, SparseTensor
        ), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert (
            len(qkv.shape) == 4 and qkv.shape[1] == 3
        ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

        # Window partition
        serialization_spatial_cache_name = (
            f"window_partition_{window_size}_{shift_window}"
        )
        serialization_spatial_cache = qkv.get_spatial_cache(
            serialization_spatial_cache_name
        )
        if serialization_spatial_cache is None:
            fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                calc_window_partition(qkv, window_size, shift_window)
            )
            qkv.register_spatial_cache(
                serialization_spatial_cache_name,
                (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
            )
        else:
            fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                serialization_spatial_cache
            )

        M = fwd_indices.shape[0]
        H = qkv.feats.shape[2]
        C = qkv.feats.shape[3]
        qkv_feats = qkv.feats[fwd_indices]  # [M, 3, H, C]

        # Compute attention using backend
        if all([seq_len == window_size for seq_len in seq_lens]):
            B = len(seq_lens)
            N = window_size
            qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
            out = backend_instance.compute_attention(
                qkv_feats, None, None, [N] * B, [N] * B, 1, device=qkv.device
            )
            out = out.reshape(B * N, H, C)  # [M, H, C]
        else:
            out = backend_instance.compute_attention(
                qkv_feats, None, None, seq_lens, seq_lens, 1, device=qkv.device
            )

        out = out[bwd_indices]  # [T, H, C]
        return qkv.replace(out)

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]

        # For windowed attention with q+kv, we need to handle the case where q is sparse
        # and kv might be sparse or dense
        if isinstance(q, SparseTensor) and isinstance(kv, SparseTensor):
            # Both are sparse - use q's coordinates for windowing
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            kv_seqlen = [
                kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])
            ]

            # Window partition based on q
            serialization_spatial_cache_name = (
                f"window_partition_{window_size}_{shift_window}"
            )
            serialization_spatial_cache = q.get_spatial_cache(
                serialization_spatial_cache_name
            )
            if serialization_spatial_cache is None:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                    calc_window_partition(q, window_size, shift_window)
                )
                q.register_spatial_cache(
                    serialization_spatial_cache_name,
                    (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
                )
            else:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                    serialization_spatial_cache
                )

            # Reorder features according to window partition
            q_feats = q.feats[fwd_indices]  # [M, H, C]
            kv_feats = kv.feats[fwd_indices]  # [M, 2, H, C]

            # Compute attention
            out = backend_instance.compute_attention(
                q_feats, kv_feats, None, seq_lens, seq_lens, 2, device=q.device
            )
            out = out[bwd_indices]  # [T, H, C]
            return s.replace(out)

        elif isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor):
            # q is dense, kv is sparse - not supported for windowed attention
            raise NotImplementedError(
                "Windowed attention with dense q and sparse kv is not supported"
            )
        else:
            raise ValueError(
                "Invalid combination of q and kv types for windowed attention"
            )

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]

        # For windowed attention with q+k+v, we need q to be sparse
        if (
            isinstance(q, SparseTensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            kv_seqlen = [
                k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])
            ]

            # Window partition based on q
            serialization_spatial_cache_name = (
                f"window_partition_{window_size}_{shift_window}"
            )
            serialization_spatial_cache = q.get_spatial_cache(
                serialization_spatial_cache_name
            )
            if serialization_spatial_cache is None:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                    calc_window_partition(q, window_size, shift_window)
                )
                q.register_spatial_cache(
                    serialization_spatial_cache_name,
                    (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
                )
            else:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
                    serialization_spatial_cache
                )

            # Reorder features according to window partition
            q_feats = q.feats[fwd_indices]  # [M, H, C]
            k_feats = k.feats[fwd_indices]  # [M, H, C]
            v_feats = v.feats[fwd_indices]  # [M, H, C]

            # Compute attention
            out = backend_instance.compute_attention(
                q_feats, k_feats, v_feats, seq_lens, seq_lens, 3, device=q.device
            )
            out = out[bwd_indices]  # [T, H, C]
            return s.replace(out)

        elif (
            isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ):
            # q is dense, k and v are sparse - not supported for windowed attention
            raise NotImplementedError(
                "Windowed attention with dense q and sparse k,v is not supported"
            )
        else:
            raise ValueError(
                "Invalid combination of q, k, v types for windowed attention"
            )
