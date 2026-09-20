"""Per-object robust ICP alignment used before object-level CD / F-score.

Ported behaviour-preserving from the reference evaluator. Object CD and F-score
are shape metrics, so each predicted object is first aligned to its GT
counterpart; without this step the numbers also absorb residual pose error,
which the pose metrics (ICP-Rot / ADD-S) already report separately.

Pipeline per object, following I-Scene:
  1. yaw sweep, pre-scored with a trimmed symmetric Chamfer
  2. seed ICP from the top-k yaw candidates
  3. coarse point-to-point on a voxel-downsampled cloud, then fine
     point-to-plane with a Tukey robust loss at full resolution
  4. project the rotation back onto SO(3) and sanity-check it
  5. run in two normalization spaces and keep whichever scores better

Every stage falls back rather than raising: a failed fine ICP degrades to
coarse, and a failed coarse degrades to identity.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

VOXEL_SIZE = 0.03
TOTAL_ITERATIONS = 80
YAW_CANDIDATES = 8
TOP_K_YAW = 3
TRIM_RATIO = 0.2


def _ensure_open3d():
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("open3d is required for robust ICP (pip install open3d)") from e
    return o3d


def _to_pcd(pts: np.ndarray):
    o3d = _ensure_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def _project_to_SO3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def _shared_normalize(src: np.ndarray, dst: np.ndarray):
    """Center both clouds on their joint AABB and scale to unit extent."""
    all_pts = np.concatenate([src, dst], axis=0)
    bb_min, bb_max = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    scale = float((bb_max - bb_min).max()) or 1.0
    if scale < 1e-12:
        scale = 1.0
    return (src - center) / scale, (dst - center) / scale, center, scale


def _aabb_recenter_normalize(pts: np.ndarray):
    """Center a cloud on its own AABB and scale it into [-1, 1]."""
    bb_min, bb_max = pts.min(axis=0), pts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    scale = float((bb_max - bb_min).max()) / 2.0
    if scale < 1e-12:
        scale = 1.0
    return (pts - center) / scale, center, scale


def _yaw_rotation(angle: float, up_axis: int = 2) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    if up_axis == 2:
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
    elif up_axis == 1:
        R[0, 0], R[0, 2] = c, s
        R[2, 0], R[2, 2] = -s, c
    else:
        R[1, 1], R[1, 2] = c, -s
        R[2, 1], R[2, 2] = s, c
    return R


def _trimmed_symmetric_chamfer(
    src: np.ndarray, dst: np.ndarray, trim_ratio: float = TRIM_RATIO, max_points: int = 2000
) -> float:
    """Cheap outlier-tolerant score for ranking yaw candidates.

    Downsampling is a deterministic stride, not a random draw, so candidate
    ranking is reproducible across runs.
    """
    if src.shape[0] > max_points:
        src = src[np.linspace(0, src.shape[0] - 1, max_points, dtype=int)]
    if dst.shape[0] > max_points:
        dst = dst[np.linspace(0, dst.shape[0] - 1, max_points, dtype=int)]

    dists = np.sqrt(((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=-1))
    nn_sd, nn_ds = dists.min(axis=1), dists.min(axis=0)

    k_sd = max(1, int(len(nn_sd) * (1.0 - trim_ratio)))
    k_ds = max(1, int(len(nn_ds) * (1.0 - trim_ratio)))
    return float(np.sort(nn_sd)[:k_sd].mean() + np.sort(nn_ds)[:k_ds].mean())


def _icp_score(result) -> float:
    """Combine residual and coverage; lambda matches the normalized voxel size."""
    return result.inlier_rmse + VOXEL_SIZE * (1.0 - result.fitness)


def _sanity_check(T: np.ndarray) -> bool:
    if not np.all(np.isfinite(T)):
        return False
    if abs(np.linalg.det(T[:3, :3]) - 1.0) > 0.5:
        return False
    # In normalized space a huge translation means the fit diverged.
    return bool(np.linalg.norm(T[:3, 3]) <= 10.0)


def _robust_icp_single(src: np.ndarray, dst: np.ndarray, up_axis: int = 2) -> Tuple[np.ndarray, float]:
    o3d = _ensure_open3d()

    yaw_angles = np.linspace(0, 2 * np.pi, YAW_CANDIDATES, endpoint=False)
    yaw_scores = [
        _trimmed_symmetric_chamfer(src @ _yaw_rotation(a, up_axis).T, dst) for a in yaw_angles
    ]
    top_indices = np.argsort(yaw_scores)[:TOP_K_YAW]

    src_down = _to_pcd(src).voxel_down_sample(VOXEL_SIZE)
    dst_down = _to_pcd(dst).voxel_down_sample(VOXEL_SIZE)

    seed_iterations = max(10, TOTAL_ITERATIONS // 4)
    best_seed_T, best_seed_score = np.eye(4), float("inf")
    for idx in top_indices:
        T_init = np.eye(4)
        T_init[:3, :3] = _yaw_rotation(yaw_angles[idx], up_axis)
        result = o3d.pipelines.registration.registration_icp(
            src_down,
            dst_down,
            max_correspondence_distance=2.5 * VOXEL_SIZE,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=seed_iterations),
        )
        score = _icp_score(result)
        if score < best_seed_score:
            best_seed_score = score
            best_seed_T = result.transformation.copy()

    coarse_iters = max(10, TOTAL_ITERATIONS // 2)
    fine_iters = max(10, TOTAL_ITERATIONS - coarse_iters)

    coarse_result = o3d.pipelines.registration.registration_icp(
        src_down,
        dst_down,
        max_correspondence_distance=2.5 * VOXEL_SIZE,
        init=best_seed_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=coarse_iters),
    )
    T_coarse = coarse_result.transformation.copy()

    src_full, dst_full = _to_pcd(src), _to_pcd(dst)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 3, max_nn=30)
    src_full.estimate_normals(search_param)
    dst_full.estimate_normals(search_param)

    try:
        loss = o3d.pipelines.registration.TukeyLoss(k=1.5 * VOXEL_SIZE)
        fine_result = o3d.pipelines.registration.registration_icp(
            src_full,
            dst_full,
            max_correspondence_distance=VOXEL_SIZE,
            init=T_coarse,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=fine_iters),
        )
    except Exception:
        # Point-to-plane needs usable normals; degenerate meshes fall back here.
        fine_result = o3d.pipelines.registration.registration_icp(
            src_full,
            dst_full,
            max_correspondence_distance=1.5 * VOXEL_SIZE,
            init=T_coarse,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=fine_iters),
        )

    T_fine = fine_result.transformation.copy()
    T_fine[:3, :3] = _project_to_SO3(T_fine[:3, :3])
    if _sanity_check(T_fine):
        return T_fine, _icp_score(fine_result)

    T_coarse[:3, :3] = _project_to_SO3(T_coarse[:3, :3])
    if _sanity_check(T_coarse):
        logger.warning("Fine ICP failed sanity check, falling back to coarse")
        return T_coarse, _icp_score(coarse_result)

    logger.warning("Coarse and fine ICP both failed sanity check, using identity")
    return np.eye(4), float("inf")


def robust_icp_align(src, dst, up_axis: int = 2):
    """Align each object in a batch of point sets.

    Args:
        src: (B, N, 3) predicted points.
        dst: (B, N, 3) GT points.
        up_axis: 0=x, 1=y, 2=z. Yaw is swept about this axis.

    Returns:
        (B, N, 3) aligned copy of ``src``.
    """
    import torch

    B = src.shape[0]
    out = torch.empty_like(src)

    for b in range(B):
        src_np = src[b].detach().cpu().numpy().astype(np.float64)
        dst_np = dst[b].detach().cpu().numpy().astype(np.float64)

        best_T, best_score = np.eye(4), float("inf")

        # Space 1: joint AABB. Space 2: each cloud on its own AABB, which
        # additionally absorbs a scale difference between pred and GT.
        src_s, dst_s, center_s, scale_s = _shared_normalize(src_np, dst_np)
        src_r, src_c, src_sc = _aabb_recenter_normalize(src_np)
        dst_r, dst_c, dst_sc = _aabb_recenter_normalize(dst_np)
        spaces = [("shared_aabb", src_s, dst_s), ("per_aabb", src_r, dst_r)]

        for space_name, s_pts, d_pts in spaces:
            try:
                T, score = _robust_icp_single(s_pts, d_pts, up_axis=up_axis)
            except Exception as e:
                logger.warning("Robust ICP object %d space=%s failed: %s", b, space_name, e)
                continue
            if score >= best_score:
                continue
            best_score = score
            # Undo the normalization so the transform applies in input space.
            if space_name == "shared_aabb":
                R = T[:3, :3]
                t = T[:3, 3] * scale_s + center_s - R @ center_s
            else:
                R = T[:3, :3] * (dst_sc / src_sc)
                t = (-R @ src_c) + T[:3, 3] * dst_sc + dst_c
            best_T = np.eye(4)
            best_T[:3, :3] = R
            best_T[:3, 3] = t

        R_t = torch.tensor(best_T[:3, :3], dtype=src.dtype, device=src.device)
        t_t = torch.tensor(best_T[:3, 3], dtype=src.dtype, device=src.device)
        out[b] = src[b] @ R_t.T + t_t

    return out
