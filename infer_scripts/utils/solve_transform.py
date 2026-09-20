"""
solve_transform.py

Standalone similarity transform solver for canonical→camera space.
Extracted from data_processor.ccm_voxel — no dependency on miraccm.

Usage:
    from utils.solve_transform import solve_similarity_transforms

    transforms = solve_similarity_transforms(
        canonical_coord_map,  # [NI, 3, H, W] tensor
        camera_pts_map,       # [NI, H, W, 3] tensor (OpenGL)
        valid_mask,           # [NI, H, W] bool tensor
        masks,                # [NI, 1, H, W] tensor
    )
    # transforms: list of {'s', 'R', 't', 'transform_matrix'} dicts
"""

import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_similarity_transform(src, tgt):
    """Torch-only Umeyama similarity solve (source rows map to target rows).

    This local implementation keeps scene assembly independent of Open3D,
    which the broader ``UniDataset.utils.pcd_utils`` module imports eagerly.
    """
    src = src.float()
    tgt = tgt.float()
    src_mean = src.mean(0, keepdim=True)
    tgt_mean = tgt.mean(0, keepdim=True)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    src_scale = torch.linalg.norm(src_centered)
    tgt_scale = torch.linalg.norm(tgt_centered)
    if src_scale < 1e-8 or tgt_scale < 1e-8:
        raise ValueError("degenerate points in similarity transform")
    scale = tgt_scale / src_scale
    u, _, vh = torch.linalg.svd(src_centered.T @ tgt_centered)
    rotation = vh.T @ u.T
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = tgt_mean[0] - scale * (rotation @ src_mean[0])
    matrix = torch.eye(4, dtype=src.dtype, device=src.device)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {"R": rotation, "t": translation, "s": scale, "transform_matrix": matrix}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depth_edge_mask(
    depth: torch.Tensor,
    dilation_radius: int = 3,
    rel_range_threshold: float = 0.05,
) -> torch.Tensor:
    """Mask out pixels near depth discontinuities. Returns [H, W] bool (True = safe)."""
    k = 2 * dilation_radius + 1
    pad = dilation_radius
    d4 = depth.float().unsqueeze(0).unsqueeze(0)
    d_max = F.max_pool2d(d4, k, stride=1, padding=pad)
    d_min = -F.max_pool2d(-d4, k, stride=1, padding=pad)
    d_mean = F.avg_pool2d(d4, k, stride=1, padding=pad)
    rel_range = (d_max - d_min) / (d_mean.abs() + 1e-6)
    is_edge = rel_range.squeeze() > rel_range_threshold
    is_edge_dilated = F.max_pool2d(
        is_edge.float().unsqueeze(0).unsqueeze(0),
        k, stride=1, padding=pad,
    ).squeeze() > 0.5
    return ~is_edge_dilated


def _ransac_similarity_transform(
    can_pts, cam_pts,
    n_iter=500, sample_size=6,
    inlier_threshold_ratio=0.02, min_inliers=10,
    refine_iters=5, max_pts=3000,
):
    """Robust similarity transform via RANSAC + iterative refinement."""
    N_full = can_pts.shape[0]
    dev = can_pts.device

    if N_full > max_pts:
        perm = torch.randperm(N_full, device=dev)[:max_pts]
        can_sub, cam_sub = can_pts[perm], cam_pts[perm]
    else:
        can_sub, cam_sub = can_pts, cam_pts
    N = can_sub.shape[0]

    cam_extent = (cam_sub.max(0).values - cam_sub.min(0).values).norm()
    thresh = inlier_threshold_ratio * cam_extent

    best_count = 0
    best_inlier_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    best_transform = None

    for iter_idx in range(n_iter):
        idx = torch.randperm(N, device=dev)[:sample_size]
        try:
            tf = compute_similarity_transform(can_sub[idx].float(), cam_sub[idx].float())
        except Exception:
            continue
        s, R, tv = tf["s"], tf["R"].to(dev), tf["t"].to(dev)
        pred = s * (can_sub @ R.T) + tv
        residuals = (pred - cam_sub).norm(dim=-1)
        inliers = residuals < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inlier_mask = inliers
            best_transform = tf
        inlier_rate = best_count / N
        if inlier_rate > 0:
            log_fail = math.log(max(1.0 - inlier_rate ** sample_size, 1e-300))
            if log_fail < 0 and iter_idx + 1 >= math.log(0.01) / log_fail:
                break

    if best_count < min_inliers:
        return best_transform, best_inlier_mask

    inlier_mask = best_inlier_mask.clone()
    try:
        tf = compute_similarity_transform(can_sub[inlier_mask].float(), cam_sub[inlier_mask].float())
    except Exception:
        return best_transform, best_inlier_mask

    for _ in range(refine_iters):
        s, R, tv = tf["s"], tf["R"].to(dev), tf["t"].to(dev)
        pred = s * (can_sub @ R.T) + tv
        residuals = (pred - cam_sub).norm(dim=-1)
        new_mask = residuals < thresh
        if int(new_mask.sum()) <= int(inlier_mask.sum()):
            break
        inlier_mask = new_mask
        try:
            tf = compute_similarity_transform(can_sub[inlier_mask].float(), cam_sub[inlier_mask].float())
        except Exception:
            break

    return tf, inlier_mask


def _extract_correspondences(
    canonical_coord_map, camera_pts_map, valid_mask, masks,
    use_depth_edge_filter=True,
    depth_edge_dilation=3, depth_edge_threshold=0.05,
    edge_filter_min_ratio=0.3,
    depth_edge_dilation_fallback=3, depth_edge_threshold_fallback=0.05,
    canonical_near_origin_threshold=0.02,
):
    """Extract valid 3D-3D correspondences per instance."""
    NI = canonical_coord_map.shape[0]
    dev = canonical_coord_map.device
    camera_pts_map = camera_pts_map.to(dev)
    valid_mask = valid_mask.to(dev).bool()

    corr_list = []
    for i in range(NI):
        obj_mask = masks[i, 0] > 0.5
        can_map_i = canonical_coord_map[i].permute(1, 2, 0).clamp(-0.5, 0.5)
        cam_map_i = camera_pts_map[i]
        vmask_i = valid_mask[i]

        if use_depth_edge_filter:
            depth_for_edge = (-cam_map_i[..., 2]).clamp(min=1e-6)
            edge_safe = _depth_edge_mask(depth_for_edge, depth_edge_dilation, depth_edge_threshold)
        else:
            edge_safe = torch.ones_like(obj_mask)

        can_nonzero = can_map_i.abs().sum(dim=-1) > 1e-6
        can_not_near_origin = can_map_i.norm(dim=-1) > canonical_near_origin_threshold
        valid_base = obj_mask & vmask_i & can_nonzero & can_not_near_origin
        valid_filtered = valid_base & edge_safe

        n_base = int(valid_base.sum())
        n_filtered = int(valid_filtered.sum())
        if use_depth_edge_filter and n_base > 0 and n_filtered < edge_filter_min_ratio * n_base:
            if depth_edge_dilation != depth_edge_dilation_fallback or depth_edge_threshold != depth_edge_threshold_fallback:
                edge_safe_fb = _depth_edge_mask(
                    (-cam_map_i[..., 2]).clamp(min=1e-6),
                    depth_edge_dilation_fallback, depth_edge_threshold_fallback,
                )
                valid_fb = valid_base & edge_safe_fb
                valid = valid_fb if int(valid_fb.sum()) >= edge_filter_min_ratio * n_base else valid_base
            else:
                valid = valid_base
        else:
            valid = valid_filtered

        corr_list.append((can_map_i[valid].float(), cam_map_i[valid].float()))
    return corr_list


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

@torch.no_grad()
def solve_similarity_transforms(
    canonical_coord_map: torch.Tensor,
    camera_pts_map: torch.Tensor,
    valid_mask: torch.Tensor,
    masks: torch.Tensor,
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_n_iter: int = 500,
    ransac_inlier_ratio: float = 0.02,
    ransac_max_pts: int = 3000,
    use_depth_edge_filter: bool = True,
    depth_edge_dilation: int = 3,
    depth_edge_threshold: float = 0.05,
) -> List[Dict]:
    """Solve canonical→camera similarity transforms via RANSAC.

    Args:
        canonical_coord_map: [NI, 3, H, W] canonical coordinates.
        camera_pts_map: [NI, H, W, 3] camera-space points (OpenGL).
        valid_mask: [NI, H, W] bool.
        masks: [NI, 1, H, W] instance masks.

    Returns:
        List of dicts, each with 's', 'R' [3,3], 't' [3], 'transform_matrix' [4,4].
    """
    NI = canonical_coord_map.shape[0]
    dev = canonical_coord_map.device

    _identity = lambda: {
        "s": torch.tensor(1.0, dtype=torch.float32),
        "R": torch.eye(3, dtype=torch.float32),
        "t": torch.zeros(3, dtype=torch.float32),
        "transform_matrix": torch.eye(4, dtype=torch.float32),
    }

    corr_list = _extract_correspondences(
        canonical_coord_map, camera_pts_map, valid_mask, masks,
        use_depth_edge_filter=use_depth_edge_filter,
        depth_edge_dilation=depth_edge_dilation,
        depth_edge_threshold=depth_edge_threshold,
    )

    transform_dict_list = []
    for i in range(NI):
        can_pts, cam_pts = corr_list[i]
        if can_pts.shape[0] < min_points:
            transform_dict_list.append(_identity())
            continue

        if use_ransac:
            transform, _ = _ransac_similarity_transform(
                can_pts, cam_pts,
                n_iter=ransac_n_iter,
                inlier_threshold_ratio=ransac_inlier_ratio,
                min_inliers=min_points,
                max_pts=ransac_max_pts,
            )
            if transform is None:
                transform_dict_list.append(_identity())
                continue
        else:
            transform = compute_similarity_transform(src=can_pts, tgt=cam_pts)

        transform_dict_list.append({
            "s": transform["s"].float().cpu(),
            "R": transform["R"].float().cpu(),
            "t": transform["t"].float().cpu(),
            "transform_matrix": transform["transform_matrix"].float().cpu(),
        })

    return transform_dict_list


# ---------------------------------------------------------------------------
# Joint Transform (shared up-direction constraint)
# ---------------------------------------------------------------------------

def _build_ortho_basis(u):
    """Build orthonormal basis {e1, e2} for the plane perpendicular to u."""
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=u.dtype, device=u.device)
    if u[0].abs() > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=u.dtype, device=u.device)
    e1 = ref - (ref @ u) * u
    e1 = e1 / e1.norm().clamp(min=1e-8)
    e2 = torch.linalg.cross(u, e1)
    return e1, e2


def _closed_form_scale_translation(q, cam_pts):
    """Solve optimal (s, t) given rotated points q and target cam_pts.

    Model: cam = s * q + t. Fully differentiable.
    """
    q_mean = q.mean(0)
    cam_mean = cam_pts.mean(0)
    q_c = q - q_mean
    cam_c = cam_pts - cam_mean
    s = (cam_c * q_c).sum() / (q_c * q_c).sum().clamp(min=1e-8)
    t = cam_mean - s * q_mean
    return s, t


@torch.no_grad()
def solve_similarity_transforms_joint(
    canonical_coord_map: torch.Tensor,
    camera_pts_map: torch.Tensor,
    valid_mask: torch.Tensor,
    masks: torch.Tensor,
    min_points: int = 10,
    use_ransac: bool = True,
    ransac_n_iter: int = 500,
    ransac_inlier_ratio: float = 0.02,
    ransac_max_pts: int = 3000,
    use_depth_edge_filter: bool = True,
    depth_edge_dilation: int = 3,
    depth_edge_threshold: float = 0.05,
    joint_opt_lr: float = 0.01,
    joint_opt_steps: int = 200,
    joint_max_pts_per_obj: int = 2000,
) -> List[Dict]:
    """Jointly solve similarity transforms with shared up-direction constraint.

    All objects' canonical Z+ [0,0,1] are constrained to map to the same
    direction u in camera space: R_i[:, 2] = u for every instance i.

    Algorithm:
      1. Extract correspondences.
      2. Initial per-object RANSAC solve (independent).
      3. Estimate shared up direction from weighted average of R_i[:, 2].
      4. Joint optimization over u (2 DOF) and per-object theta_i
         (in-plane rotation, 1 DOF each), with s_i, t_i solved in
         closed form at each step.
      5. Build final transform dicts.

    Returns:
        List of dicts, each with 's', 'R' [3,3], 't' [3], 'transform_matrix' [4,4].
    """
    NI = canonical_coord_map.shape[0]
    dev = canonical_coord_map.device

    _identity = lambda: {
        "s": torch.tensor(1.0, dtype=torch.float32),
        "R": torch.eye(3, dtype=torch.float32),
        "t": torch.zeros(3, dtype=torch.float32),
        "transform_matrix": torch.eye(4, dtype=torch.float32),
    }

    # Phase 1: Extract correspondences
    corr_list = _extract_correspondences(
        canonical_coord_map, camera_pts_map, valid_mask, masks,
        use_depth_edge_filter=use_depth_edge_filter,
        depth_edge_dilation=depth_edge_dilation,
        depth_edge_threshold=depth_edge_threshold,
    )

    # Phase 2: Initial per-object RANSAC
    initial_transforms = []
    inlier_can_list = []
    inlier_cam_list = []
    valid_obj_indices = []

    for i in range(NI):
        can_pts, cam_pts = corr_list[i]
        if can_pts.shape[0] < min_points:
            initial_transforms.append(None)
            inlier_can_list.append(torch.zeros(0, 3, device=dev))
            inlier_cam_list.append(torch.zeros(0, 3, device=dev))
            continue

        if use_ransac:
            tf, inlier_mask = _ransac_similarity_transform(
                can_pts, cam_pts,
                n_iter=ransac_n_iter,
                inlier_threshold_ratio=ransac_inlier_ratio,
                min_inliers=min_points,
                max_pts=ransac_max_pts,
            )
        else:
            tf = compute_similarity_transform(src=can_pts, tgt=cam_pts)
            inlier_mask = torch.ones(can_pts.shape[0], dtype=torch.bool, device=dev)

        if tf is None:
            initial_transforms.append(None)
            inlier_can_list.append(torch.zeros(0, 3, device=dev))
            inlier_cam_list.append(torch.zeros(0, 3, device=dev))
            continue

        initial_transforms.append(tf)
        valid_obj_indices.append(i)

        # Get inlier points: use the transform to find inliers on the FULL point set
        # (since _ransac works on a subsample, inlier_mask may not match can_pts length)
        s, R, tv = tf["s"], tf["R"].to(dev), tf["t"].to(dev)
        pred_full = s * (can_pts @ R.T) + tv
        residuals_full = (pred_full - cam_pts).norm(dim=-1)
        cam_extent = (cam_pts.max(0).values - cam_pts.min(0).values).norm()
        thresh = ransac_inlier_ratio * cam_extent
        inlier_mask_full = residuals_full < thresh

        inlier_can = can_pts[inlier_mask_full]
        inlier_cam = cam_pts[inlier_mask_full]
        M = inlier_can.shape[0]
        if M < min_points:
            # Fallback: use all points
            inlier_can = can_pts
            inlier_cam = cam_pts
            M = inlier_can.shape[0]
        if M > joint_max_pts_per_obj:
            perm = torch.randperm(M, device=dev)[:joint_max_pts_per_obj]
            inlier_can = inlier_can[perm]
            inlier_cam = inlier_cam[perm]
        inlier_can_list.append(inlier_can)
        inlier_cam_list.append(inlier_cam)

    # If fewer than 2 valid objects, fall back to independent transforms
    if len(valid_obj_indices) < 2:
        result = []
        for i in range(NI):
            if initial_transforms[i] is not None:
                tf = initial_transforms[i]
                result.append({
                    "s": tf["s"].float().cpu(),
                    "R": tf["R"].float().cpu(),
                    "t": tf["t"].float().cpu(),
                    "transform_matrix": tf["transform_matrix"].float().cpu(),
                })
            else:
                result.append(_identity())
        return result

    # Phase 3: Estimate shared up direction
    up_candidates = []
    up_weights = []
    for i in valid_obj_indices:
        R_i = initial_transforms[i]["R"].float().to(dev)
        u_i = R_i[:, 2]  # third column = R @ [0,0,1]
        up_candidates.append(u_i)
        up_weights.append(float(inlier_can_list[i].shape[0]))

    up_candidates = torch.stack(up_candidates, dim=0)  # [K, 3]
    up_weights = torch.tensor(up_weights, dtype=torch.float32, device=dev)

    # Align signs
    dots = (up_candidates * up_candidates[0:1]).sum(dim=-1)
    signs = dots.sign()
    signs[signs == 0] = 1.0
    up_candidates = up_candidates * signs.unsqueeze(-1)

    u_avg = (up_candidates * up_weights.unsqueeze(-1)).sum(0)
    u_norm = u_avg.norm()
    if u_norm < 1e-6:
        best_idx = up_weights.argmax()
        u_init = up_candidates[best_idx]
    else:
        u_init = u_avg / u_norm

    # Phase 4: Joint optimization
    phi_init = torch.acos(u_init[2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    psi_init = torch.atan2(u_init[1], u_init[0])

    e1_init, e2_init = _build_ortho_basis(u_init)
    theta_inits = []
    for i in valid_obj_indices:
        R_i = initial_transforms[i]["R"].float().to(dev)
        r0 = R_i[:, 0]
        cos_th = (r0 * e1_init).sum()
        sin_th = -(r0 * e2_init).sum()
        theta_inits.append(torch.atan2(sin_th, cos_th))
    theta_inits = torch.stack(theta_inits)

    phi_param = phi_init.clone().detach().requires_grad_(True)
    psi_param = psi_init.clone().detach().requires_grad_(True)
    theta_param = theta_inits.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([phi_param, psi_param, theta_param], lr=joint_opt_lr)

    K = len(valid_obj_indices)
    opt_can = [inlier_can_list[i] for i in valid_obj_indices]
    opt_cam = [inlier_cam_list[i] for i in valid_obj_indices]

    with torch.enable_grad():
        for step in range(joint_opt_steps):
            optimizer.zero_grad()

            u = torch.stack([
                torch.sin(phi_param) * torch.cos(psi_param),
                torch.sin(phi_param) * torch.sin(psi_param),
                torch.cos(phi_param),
            ])

            e1, e2 = _build_ortho_basis(u)

            total_loss = torch.tensor(0.0, device=dev)
            for j in range(K):
                theta_j = theta_param[j]
                cos_t = torch.cos(theta_j)
                sin_t = torch.sin(theta_j)

                col0 = cos_t * e1 - sin_t * e2
                col1 = sin_t * e1 + cos_t * e2
                R_j = torch.stack([col0, col1, u], dim=1)

                q = (R_j @ opt_can[j].T).T
                s_j, t_j = _closed_form_scale_translation(q, opt_cam[j])

                pred = s_j * q + t_j
                loss_j = (pred - opt_cam[j]).pow(2).sum(dim=-1).mean()
                total_loss = total_loss + loss_j

            total_loss.backward()
            optimizer.step()

    # Phase 5: Extract final transforms
    with torch.no_grad():
        u_final = torch.stack([
            torch.sin(phi_param) * torch.cos(psi_param),
            torch.sin(phi_param) * torch.sin(psi_param),
            torch.cos(phi_param),
        ])
        e1_f, e2_f = _build_ortho_basis(u_final)

    transform_dict_list = []
    for i in range(NI):
        if initial_transforms[i] is None:
            transform_dict_list.append(_identity())
            continue

        j = valid_obj_indices.index(i)
        with torch.no_grad():
            theta_j = theta_param[j]
            cos_t = torch.cos(theta_j)
            sin_t = torch.sin(theta_j)
            col0 = cos_t * e1_f - sin_t * e2_f
            col1 = sin_t * e1_f + cos_t * e2_f
            R_j = torch.stack([col0, col1, u_final], dim=1)

            q = (R_j @ inlier_can_list[i].T).T
            s_j, t_j = _closed_form_scale_translation(q, inlier_cam_list[i])

            transform_matrix = torch.eye(4, dtype=torch.float32, device=dev)
            transform_matrix[:3, :3] = s_j * R_j
            transform_matrix[:3, 3] = t_j

        transform_dict_list.append({
            "s": s_j.float().cpu(),
            "R": R_j.float().cpu(),
            "t": t_j.float().cpu(),
            "transform_matrix": transform_matrix.float().cpu(),
        })

    return transform_dict_list


# ---------------------------------------------------------------------------
# Gravity-constrained transform (known floor frame)
# ---------------------------------------------------------------------------

def _numpy_transform_dict(matrix: np.ndarray) -> Dict:
    """Convert a uniform-scale 4x4 matrix to the public transform format."""
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = float(np.linalg.norm(matrix[:3, 0]))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    rotation = matrix[:3, :3] / scale
    return {
        "s": torch.tensor(scale, dtype=torch.float32),
        "R": torch.from_numpy(rotation.astype(np.float32)),
        "t": torch.from_numpy(matrix[:3, 3].astype(np.float32)),
        "transform_matrix": torch.from_numpy(matrix.astype(np.float32)),
    }


def _transform_dict_matrix(transform: Dict) -> np.ndarray:
    value = transform.get("transform_matrix")
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)
    scale = float(transform["s"])
    rotation = transform["R"]
    translation = transform["t"]
    if isinstance(rotation, torch.Tensor):
        rotation = rotation.detach().cpu().numpy()
    if isinstance(translation, torch.Tensor):
        translation = translation.detach().cpu().numpy()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * np.asarray(rotation)
    matrix[:3, 3] = np.asarray(translation)
    return matrix


def solve_similarity_transforms_gravity(
    canonical_coord_map: torch.Tensor,
    camera_pts_map: torch.Tensor,
    valid_mask: torch.Tensor,
    masks: torch.Tensor,
    camera_to_floor_transform: np.ndarray,
    constrain_upright: List[bool],
    initial_transforms: Optional[List[Dict]] = None,
    min_points: int = 10,
    use_depth_edge_filter: bool = True,
    depth_edge_dilation: int = 3,
    depth_edge_threshold: float = 0.05,
    max_points_per_object: int = 3000,
    optimization_steps: int = 160,
    optimization_lr: float = 0.03,
) -> List[Dict]:
    """Fit canonical-to-floor transforms with a known hard gravity direction.

    Objects selected by ``constrain_upright`` have canonical +Z mapped exactly
    to floor +Y.  Their yaw is initialized from the independent CCM solve and
    refined against the same dense CCM/depth correspondences; scale and
    translation are solved in closed form at every iteration.  Unselected
    objects preserve the independent pose, merely expressed in floor space.

    This intentionally differs from :func:`solve_similarity_transforms_joint`:
    the latter estimates a shared up vector from the objects, whereas this
    function uses the floor estimate as the authoritative gravity frame.
    """
    object_count = canonical_coord_map.shape[0]
    if len(constrain_upright) != object_count:
        raise ValueError("constrain_upright length must match the number of objects")

    camera_to_floor = np.asarray(camera_to_floor_transform, dtype=np.float64)
    if camera_to_floor.shape != (4, 4) or not np.isfinite(camera_to_floor).all():
        raise ValueError("camera_to_floor_transform must be a finite 4x4 matrix")

    if initial_transforms is None:
        initial_transforms = solve_similarity_transforms(
            canonical_coord_map,
            camera_pts_map,
            valid_mask,
            masks,
            min_points=min_points,
            use_depth_edge_filter=use_depth_edge_filter,
            depth_edge_dilation=depth_edge_dilation,
            depth_edge_threshold=depth_edge_threshold,
        )
    if len(initial_transforms) != object_count:
        raise ValueError("initial_transforms length must match the number of objects")

    # Apply the rigid camera-to-floor transform to the target point map.
    dev = canonical_coord_map.device
    floor_rotation = torch.as_tensor(
        camera_to_floor[:3, :3], dtype=camera_pts_map.dtype, device=dev
    )
    floor_translation = torch.as_tensor(
        camera_to_floor[:3, 3], dtype=camera_pts_map.dtype, device=dev
    )
    camera_pts_map = camera_pts_map.to(dev)
    floor_pts_map = camera_pts_map @ floor_rotation.T + floor_translation
    correspondences = _extract_correspondences(
        canonical_coord_map,
        floor_pts_map,
        valid_mask,
        masks,
        use_depth_edge_filter=use_depth_edge_filter,
        depth_edge_dilation=depth_edge_dilation,
        depth_edge_threshold=depth_edge_threshold,
    )

    result: List[Dict] = []
    for index in range(object_count):
        initial_floor = camera_to_floor @ _transform_dict_matrix(initial_transforms[index])
        if not constrain_upright[index]:
            result.append(_numpy_transform_dict(initial_floor))
            continue

        canonical_points, floor_points = correspondences[index]
        if canonical_points.shape[0] < min_points:
            # Keep the hard gravity constraint even when dense refitting is not
            # possible: retain the independent scale/translation and its
            # projected yaw, but remove pitch and roll.
            initial_scale = float(np.linalg.norm(initial_floor[:3, 0]))
            initial_rotation = initial_floor[:3, :3] / max(initial_scale, 1e-8)
            horizontal_x = initial_rotation[[0, 2], 0]
            if np.linalg.norm(horizontal_x) < 1e-6:
                horizontal_x = np.array([-initial_rotation[2, 1], initial_rotation[0, 1]])
            theta_initial = math.atan2(float(horizontal_x[1]), float(horizontal_x[0]))
            cosine, sine = math.cos(theta_initial), math.sin(theta_initial)
            upright_matrix = np.eye(4, dtype=np.float64)
            upright_matrix[:3, :3] = initial_scale * np.array(
                [[cosine, sine, 0.0], [0.0, 0.0, 1.0], [sine, -cosine, 0.0]]
            )
            upright_matrix[:3, 3] = initial_floor[:3, 3]
            result.append(_numpy_transform_dict(upright_matrix))
            continue

        initial_scale = float(np.linalg.norm(initial_floor[:3, 0]))
        initial_rotation = initial_floor[:3, :3] / max(initial_scale, 1e-8)
        horizontal_x = initial_rotation[[0, 2], 0]
        if np.linalg.norm(horizontal_x) < 1e-6:
            # col1 = up x col0, so col0 can also be recovered from col1.
            horizontal_x = np.array([-initial_rotation[2, 1], initial_rotation[0, 1]])
        theta_initial = math.atan2(float(horizontal_x[1]), float(horizontal_x[0]))

        # Restrict optimization cost while retaining a deterministic sample.
        if canonical_points.shape[0] > max_points_per_object:
            selected = torch.linspace(
                0,
                canonical_points.shape[0] - 1,
                max_points_per_object,
                device=dev,
            ).long()
            canonical_points = canonical_points[selected]
            floor_points = floor_points[selected]

        theta = torch.tensor(theta_initial, dtype=torch.float32, device=dev, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=optimization_lr)
        with torch.enable_grad():
            for _ in range(optimization_steps):
                optimizer.zero_grad()
                cosine, sine = torch.cos(theta), torch.sin(theta)
                col0 = torch.stack([cosine, cosine * 0.0, sine])
                col1 = torch.stack([sine, sine * 0.0, -cosine])
                col2 = torch.stack([sine * 0.0, sine * 0.0 + 1.0, sine * 0.0])
                rotation = torch.stack([col0, col1, col2], dim=1)
                rotated = canonical_points @ rotation.T
                scale, translation = _closed_form_scale_translation(rotated, floor_points)
                # A negative similarity scale is a reflected solution and is
                # invalid for mesh placement.
                scale = scale.clamp(min=1e-6)
                translation = floor_points.mean(0) - scale * rotated.mean(0)
                residual = scale * rotated + translation - floor_points
                loss = residual.square().sum(dim=-1).mean()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            cosine, sine = torch.cos(theta), torch.sin(theta)
            col0 = torch.stack([cosine, cosine * 0.0, sine])
            col1 = torch.stack([sine, sine * 0.0, -cosine])
            col2 = torch.stack([sine * 0.0, sine * 0.0 + 1.0, sine * 0.0])
            rotation = torch.stack([col0, col1, col2], dim=1)
            rotated = canonical_points @ rotation.T
            scale, translation = _closed_form_scale_translation(rotated, floor_points)
            scale = scale.clamp(min=1e-6)
            translation = floor_points.mean(0) - scale * rotated.mean(0)
            matrix = torch.eye(4, dtype=torch.float32, device=dev)
            matrix[:3, :3] = scale * rotation
            matrix[:3, 3] = translation
        result.append(_numpy_transform_dict(matrix.detach().cpu().numpy()))

    return result
