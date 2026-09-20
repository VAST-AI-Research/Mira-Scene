"""Geometry metrics: Chamfer Distance, F-score, 3D BBox IoU.

Ported verbatim (behaviour-preserving) from the reference evaluator so the
numbers reproduce exactly. Two conventions worth knowing before you read the
code:

  * ``compute_chamfer_distance`` returns the mean of SQUARED nearest-neighbour
    distances (summed over both directions). This is the convention the
    published numbers use -- do not "fix" it to euclidean.
  * ``compute_fscore`` uses true euclidean distances against ``tau``.

Both are chunked to bound peak memory on large point sets.
"""

from __future__ import annotations

import torch


class ChamferDistance(torch.nn.Module):
    """Chunked bidirectional nearest-neighbour squared distances.

    Returns (min_dist1, min_dist2) where min_dist1[b, i] is the squared
    distance from xyz1[b, i] to its nearest neighbour in xyz2[b] (and
    symmetrically for min_dist2). Halves the chunk size and retries on CUDA
    OOM so that large scenes degrade in speed rather than crashing.
    """

    def __init__(self, chunk_size: int = 1024):
        super().__init__()
        self.chunk_size = chunk_size

    @staticmethod
    def _one_side(query: torch.Tensor, ref: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B, N, _ = query.shape
        M = ref.shape[1]
        out = torch.full((B, N), float("inf"), device=query.device, dtype=query.dtype)
        chunk = min(chunk_size, M)
        j = 0
        while j < M:
            end = min(j + chunk, M)
            try:
                diff = query.unsqueeze(2) - ref[:, j:end, :].unsqueeze(1)
                dist = torch.sum(diff * diff, dim=-1)
                out = torch.minimum(out, dist.min(dim=2).values)
                del diff, dist
                j = end
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                chunk //= 2
                if chunk < 1:
                    raise RuntimeError("Chamfer: cannot reduce chunk size further") from e
        return out

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor):
        assert xyz1.shape[-1] == 3 and xyz2.shape[-1] == 3, "Only 3D points supported"
        assert xyz1.device == xyz2.device, "Devices differ"

        B, N, _ = xyz1.shape
        M = xyz2.shape[1]
        if N == 0 or M == 0:
            return (
                torch.zeros((B, N), dtype=xyz1.dtype, device=xyz1.device),
                torch.zeros((B, M), dtype=xyz2.dtype, device=xyz2.device),
            )

        min_dist1 = self._one_side(xyz1, xyz2, self.chunk_size)
        min_dist2 = self._one_side(xyz2, xyz1, self.chunk_size)

        # An inf survives only if a side was empty, which is handled above;
        # keep the guard so a malformed mesh yields 0 rather than nan.
        for t in (min_dist1, min_dist2):
            mask = torch.isinf(t)
            if mask.any():
                t[mask] = 0

        return min_dist1, min_dist2


def compute_chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, chunk_size: int = 2048):
    """Bidirectional Chamfer Distance over squared distances.

    Args:
        pred, gt: (B, N, 3) / (B, M, 3) point sets.

    Returns:
        (cd, cd_pred_to_gt, cd_gt_to_pred), each (B,).
    """
    cd_fn = ChamferDistance(chunk_size=chunk_size).to(pred.device)
    dist1, dist2 = cd_fn(pred, gt)
    dist1 = dist1.mean(dim=1)
    dist2 = dist2.mean(dim=1)
    return dist1 + dist2, dist1, dist2


def compute_fscore(pred: torch.Tensor, gt: torch.Tensor, tau: float = 0.1, chunk_size: int = 2048):
    """F-score at euclidean distance threshold ``tau``. Returns (B,)."""
    B, N, _ = pred.shape
    M = gt.shape[1]

    min_pred_to_gt = torch.zeros(B, N, device=pred.device)
    min_gt_to_pred = torch.zeros(B, M, device=gt.device)

    for b in range(B):
        pred_b, gt_b = pred[b], gt[b]
        for i in range(0, N, chunk_size):
            n = min(chunk_size, N - i)
            d = torch.cdist(pred_b[i : i + n].unsqueeze(0), gt_b.unsqueeze(0), p=2)
            min_pred_to_gt[b, i : i + n] = d.min(dim=2).values.squeeze(0)[:n]
        for i in range(0, M, chunk_size):
            n = min(chunk_size, M - i)
            d = torch.cdist(gt_b[i : i + n].unsqueeze(0), pred_b.unsqueeze(0), p=2)
            min_gt_to_pred[b, i : i + n] = d.min(dim=2).values.squeeze(0)[:n]

    precision = (min_pred_to_gt < tau).float().sum(dim=1) / N
    recall = (min_gt_to_pred < tau).float().sum(dim=1) / M
    return 2 * precision * recall / (precision + recall + 1e-8)


def compute_volume_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Axis-aligned 3D bounding-box IoU per object. Returns (B,)."""
    pred_min, pred_max = pred.min(dim=1).values, pred.max(dim=1).values
    gt_min, gt_max = gt.min(dim=1).values, gt.max(dim=1).values

    inter = (torch.min(pred_max, gt_max) - torch.max(pred_min, gt_min)).clamp(min=0)
    inter_vol = inter[:, 0] * inter[:, 1] * inter[:, 2]

    pred_dims = (pred_max - pred_min).clamp(min=0)
    gt_dims = (gt_max - gt_min).clamp(min=0)
    pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]
    gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]

    return inter_vol / (pred_vol + gt_vol - inter_vol + 1e-8)


def normalize_points(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize each point set independently into [-0.95, 0.95] by its AABB.

    Object-level CD / F-score are defined in this normalized space, which makes
    them shape metrics insensitive to per-object scale and placement.
    """
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    ranges = max_vals - min_vals
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
    return 1.9 * (tensor - min_vals) / ranges - 0.95
