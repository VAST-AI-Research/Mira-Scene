"""
metrics.py

Shared evaluation metrics for CCM, voxel, and point cloud evaluation.
Used by both system_ccm_voxel.py (validation_step) and eval_scripts/.
"""

import numpy as np


def chamfer_distance_numpy(pts_a, pts_b):
    """Compute bidirectional Chamfer Distance between two point clouds.

    Uses scipy cKDTree for memory-efficient nearest-neighbor search.

    Args:
        pts_a: [N, 3] numpy float array
        pts_b: [M, 3] numpy float array

    Returns:
        float: symmetric mean distance, or nan if either is empty
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('nan')
    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)
    return float(dist_a2b.mean() + dist_b2a.mean()) / 2.0


def one_way_chamfer_distance_numpy(src_pts, tgt_pts):
    """Mean nearest-neighbour distance from ``src_pts`` to ``tgt_pts``.

    This is useful for visible CCM surface points against the complete GT
    voxel-center cloud: missing invisible GT surfaces are intentionally not
    penalized in the source-to-target direction.
    """
    src_pts = np.asarray(src_pts)
    tgt_pts = np.asarray(tgt_pts)
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return float("nan")
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(tgt_pts).query(src_pts)
    return float(distances.mean())


def chamfer_and_fscore(pts_a, pts_b, thresholds=(0.01, 0.05)):
    """Compute Chamfer Distance and F-scores at multiple thresholds.

    Args:
        pts_a: [N, 3] numpy float array
        pts_b: [M, 3] numpy float array
        thresholds: tuple of distance thresholds for F-score

    Returns:
        chamfer: float (symmetric mean distance)
        fscores: dict {threshold: f-score}
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('nan'), {th: 0.0 for th in thresholds}
    from scipy.spatial import cKDTree
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


def voxel_iou_numpy(pred_occ, gt_occ):
    """Compute IoU between two occupancy grids.

    Args:
        pred_occ: [R, R, R] numpy array (>0 = occupied)
        gt_occ:   [R, R, R] numpy array (>0 = occupied)

    Returns:
        float: IoU value
    """
    pred_bool = pred_occ > 0
    gt_bool = gt_occ > 0
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def voxel_iou_batch(pred_voxel, gt_voxel):
    """Compute mean IoU across a batch of occupancy grids.

    Args:
        pred_voxel: [B, 1, R, R, R] numpy or tensor (>0 = occupied)
        gt_voxel:   [B, 1, R, R, R] numpy or tensor (1 = occupied)

    Returns:
        float: mean IoU across batch
    """
    if hasattr(pred_voxel, 'cpu'):
        pred_voxel = pred_voxel.float().cpu().numpy()
    if hasattr(gt_voxel, 'cpu'):
        gt_voxel = gt_voxel.float().cpu().numpy()
    B = pred_voxel.shape[0]
    ious = [voxel_iou_numpy(pred_voxel[i, 0], gt_voxel[i, 0]) for i in range(B)]
    return float(np.mean(ious))


def ccm_mse_masked(pred_ccm, gt_ccm, mask):
    """Compute MSE between predicted and GT CCM within mask region.

    Args:
        pred_ccm: [3, H, W] or [H, W, 3] numpy float array
        gt_ccm:   [3, H, W] or [H, W, 3] numpy float array
        mask:     [H, W] bool/float array (>0.5 = valid)

    Returns:
        float: mean MSE within mask
    """
    if pred_ccm.shape[0] == 3:
        pred_ccm = pred_ccm.transpose(1, 2, 0)
    if gt_ccm.shape[0] == 3:
        gt_ccm = gt_ccm.transpose(1, 2, 0)
    mask_bool = mask > 0.5 if mask.dtype != bool else mask
    num_valid = mask_bool.sum()
    if num_valid == 0:
        return 0.0
    diff_sq = ((pred_ccm - gt_ccm) ** 2) * mask_bool[..., None]
    return float(diff_sq.sum() / (num_valid * 3.0))


def ccm_mse_masked_batch(pred_ccm, gt_ccm, mask):
    """Compute mean masked MSE across a batch.

    Args:
        pred_ccm: [B, 3, H, W] numpy or tensor
        gt_ccm:   [B, 3, H, W] numpy or tensor
        mask:     [B, 1, H, W] numpy or tensor

    Returns:
        float: mean MSE across batch
    """
    if hasattr(pred_ccm, 'cpu'):
        pred_ccm = pred_ccm.float().cpu().numpy()
    if hasattr(gt_ccm, 'cpu'):
        gt_ccm = gt_ccm.float().cpu().numpy()
    if hasattr(mask, 'cpu'):
        mask = mask.float().cpu().numpy()
    B = pred_ccm.shape[0]
    mses = [ccm_mse_masked(pred_ccm[i], gt_ccm[i], mask[i, 0]) for i in range(B)]
    return float(np.mean(mses))


def ccm_l1_masked(pred_ccm, gt_ccm, mask):
    """Compute L1 between predicted and GT CCM within mask region.

    Args:
        pred_ccm: [3, H, W] or [H, W, 3] numpy float array
        gt_ccm:   [3, H, W] or [H, W, 3] numpy float array
        mask:     [H, W] bool/float array (>0.5 = valid)

    Returns:
        float: mean L1 within mask
    """
    if pred_ccm.shape[0] == 3:
        pred_ccm = pred_ccm.transpose(1, 2, 0)
    if gt_ccm.shape[0] == 3:
        gt_ccm = gt_ccm.transpose(1, 2, 0)
    mask_bool = mask > 0.5 if mask.dtype != bool else mask
    num_valid = mask_bool.sum()
    if num_valid == 0:
        return 0.0
    diff_abs = np.abs(pred_ccm - gt_ccm) * mask_bool[..., None]
    return float(diff_abs.sum() / (num_valid * 3.0))


def ccm_l1_masked_batch(pred_ccm, gt_ccm, mask):
    """Compute mean masked L1 across a batch.

    Args:
        pred_ccm: [B, 3, H, W] numpy or tensor
        gt_ccm:   [B, 3, H, W] numpy or tensor
        mask:     [B, 1, H, W] numpy or tensor

    Returns:
        float: mean L1 across batch
    """
    if hasattr(pred_ccm, 'cpu'):
        pred_ccm = pred_ccm.float().cpu().numpy()
    if hasattr(gt_ccm, 'cpu'):
        gt_ccm = gt_ccm.float().cpu().numpy()
    if hasattr(mask, 'cpu'):
        mask = mask.float().cpu().numpy()
    B = pred_ccm.shape[0]
    l1s = [ccm_l1_masked(pred_ccm[i], gt_ccm[i], mask[i, 0]) for i in range(B)]
    return float(np.mean(l1s))


def align_scale_shift(pred_pts, gt_pts):
    """Find s, t such that pred * s + t ~ gt (least squares).

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


# ---------------------------------------------------------------------------
# Loaders (for eval scripts)
# ---------------------------------------------------------------------------

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
    ccm = arr[:, :, ::-1].copy().astype(np.float32)
    valid = np.isfinite(ccm).all(axis=-1) & (np.abs(ccm).sum(axis=-1) > 1e-6)
    ccm[~valid] = 0.0
    return ccm, valid
