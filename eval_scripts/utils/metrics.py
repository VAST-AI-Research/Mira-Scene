"""
metrics.py

Shared evaluation metrics for CCM and correspondence evaluation.
"""

import numpy as np
from scipy.spatial import cKDTree


def align_scale_shift(pred_pts, gt_pts):
    """Find s, t such that pred * s + t ≈ gt (least squares).

    Args:
        pred_pts: [N, 3] paired predicted points
        gt_pts:   [N, 3] paired GT points

    Returns:
        s: float scalar
        t: [3] shift vector
    """
    pred_mean = pred_pts.mean(axis=0)
    gt_mean = gt_pts.mean(axis=0)
    pred_c = pred_pts - pred_mean
    gt_c = gt_pts - gt_mean
    s = (pred_c * gt_c).sum() / max((pred_c * pred_c).sum(), 1e-12)
    t = gt_mean - s * pred_mean
    return float(s), t


def chamfer_and_fscore(pts_a, pts_b, thresholds=(0.01, 0.05)):
    """Compute Chamfer Distance and F-scores at multiple thresholds.

    Args:
        pts_a: [N, 3] point set A
        pts_b: [M, 3] point set B
        thresholds: tuple of distance thresholds for F-score

    Returns:
        chamfer: float (symmetric mean distance)
        fscores: dict {threshold: f-score}
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('nan'), {th: 0.0 for th in thresholds}
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)

    chamfer = float(dist_a2b.mean() + dist_b2a.mean()) / 2.0

    fscores = {}
    for th in thresholds:
        precision = float((dist_a2b < th).sum()) / max(len(pts_a), 1)
        recall = float((dist_b2a < th).sum()) / max(len(pts_b), 1)
        if precision + recall > 0:
            fscores[th] = 2.0 * precision * recall / (precision + recall)
        else:
            fscores[th] = 0.0

    return chamfer, fscores


def compute_iou(mask_a, mask_b):
    """Compute IoU between two boolean masks.

    Args:
        mask_a, mask_b: [H, W] bool arrays

    Returns:
        float IoU value
    """
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def load_coord_map_npy(path):
    """Load a coord map npy, return [H, W, 3] float32 + valid mask."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    valid = np.abs(arr).sum(axis=-1) > 1e-6
    return arr, valid


def load_coord_map_exr(path):
    """Load a ccm.exr file, return [H, W, 3] float32 + valid mask."""
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        return None, None
    ccm = arr[:, :, ::-1].copy().astype(np.float32)  # BGR -> RGB
    valid = np.isfinite(ccm).all(axis=-1) & (np.abs(ccm).sum(axis=-1) > 1e-6)
    ccm[~valid] = 0.0
    return ccm, valid


def eval_coord_maps(pred, pred_valid, gt, gt_valid, thresholds=(0.01, 0.05), max_pts=50000):
    """Evaluate coord maps: resize, align on overlap, compute chamfer + f-score.

    Args:
        pred: [H, W, 3] predicted coord map
        pred_valid: [H, W] bool
        gt: [H, W, 3] GT coord map
        gt_valid: [H, W] bool
        thresholds: F-score thresholds
        max_pts: subsample limit

    Returns:
        dict with 'chamfer' and 'F@{th}' keys, or None if insufficient overlap
    """
    from PIL import Image

    # Resize pred to match GT if shapes differ
    if pred.shape[:2] != gt.shape[:2]:
        H, W = gt.shape[:2]
        pred_resized = np.stack([
            np.array(Image.fromarray(pred[:, :, c], mode="F").resize((W, H), Image.BILINEAR))
            for c in range(3)
        ], axis=-1)
        pred_valid = np.array(
            Image.fromarray(pred_valid.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        ) > 127
        pred = pred_resized

    overlap = pred_valid & gt_valid
    if overlap.sum() < 10:
        return None

    s, t = align_scale_shift(pred[overlap], gt[overlap])
    pred_aligned = pred * s + t

    pred_pts = pred_aligned[pred_valid]
    gt_pts = gt[gt_valid]

    if len(pred_pts) > max_pts:
        pred_pts = pred_pts[np.random.choice(len(pred_pts), max_pts, replace=False)]
    if len(gt_pts) > max_pts:
        gt_pts = gt_pts[np.random.choice(len(gt_pts), max_pts, replace=False)]

    chamfer, fscores = chamfer_and_fscore(pred_pts, gt_pts, thresholds)
    return {"chamfer": chamfer, **{f"F@{th}": v for th, v in fscores.items()}}
