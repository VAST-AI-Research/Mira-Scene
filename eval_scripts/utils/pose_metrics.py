"""Pose metrics: ADD-S, ICP-Rot, EMD, and 2D silhouette IoU.

Ported behaviour-preserving from the reference evaluator. These are the `raw`
report-mode variants, which is what CCM / our-model results use in the paper:

  * ADD-S      -- computed on the posed objects directly, no scene-level
                  similarity transform (that is the `stable` variant).
  * ICP-Rot    -- recenter + GT-diameter scale, then point-to-point ICP from
                  identity; report the rotation angle of the ICP result.
  * EMD        -- per-object bbox-normalize, ICP align, Hungarian assignment.
  * 2D IoU     -- Open3D raycast silhouette of pred vs GT mesh through the GT
                  camera. Geometry-only; no Blender / material render.

Sampling is seeded from (scene_name, object_idx, kind) so reruns are
reproducible without a global seed.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

# Paper-fixed constants (see eval_scripts/README.md).
DIAMETER_POINTS = 4096
EMD_POINTS = 1024
ICP_MAX_ITER = 100
ICP_MAX_CORR_RATIO = 0.1
ICP_MIN_FITNESS = 1e-4
SEED = 0


def ensure_open3d():
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("open3d is required for pose metrics (pip install open3d)") from e
    return o3d


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------

def derive_seed(scene_name: str, object_idx: int, kind: str) -> int:
    payload = f"{SEED}|{scene_name}|{int(object_idx)}|{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="little", signed=False)


@contextmanager
def _temporary_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed) % (2 ** 32))
    try:
        yield
    finally:
        np.random.set_state(state)


def sample_surface_points(mesh: trimesh.Trimesh, num_points: int, seed: int) -> torch.Tensor:
    """Seeded surface sampling, degrading to volume then vertex sampling.

    Degenerate meshes (no faces, non-watertight) are common in predictions, so
    the fallbacks matter: without them a single bad object aborts the scene.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError("Mesh has no vertices to sample from")

    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(num_points), seed=int(seed))
        return torch.from_numpy(np.asarray(points, dtype=np.float32))
    except Exception:
        pass

    try:
        with _temporary_numpy_seed(seed):
            points = trimesh.sample.volume_mesh(mesh, int(num_points))
        if len(points) > 0:
            return torch.from_numpy(np.asarray(points, dtype=np.float32))
    except Exception:
        pass

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(vertices.shape[0], size=int(num_points), replace=vertices.shape[0] < int(num_points))
    return torch.from_numpy(np.asarray(vertices[idx], dtype=np.float32))


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------

def _to_tensor(points, device: torch.device) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        return points.to(device=device, dtype=torch.float32)
    return torch.as_tensor(points, dtype=torch.float32, device=device)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation.shape}")
    cos_theta = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def bbox_center(points: torch.Tensor) -> torch.Tensor:
    return (points.min(dim=0).values + points.max(dim=0).values) * 0.5


def normalize_by_bbox(points: torch.Tensor, target_extent: float = 2.0) -> torch.Tensor:
    """Center on the AABB and scale its longest side to ``target_extent``."""
    mins, maxs = points.min(dim=0).values, points.max(dim=0).values
    center = (mins + maxs) * 0.5
    extent = torch.clamp((maxs - mins).max(), min=1e-8)
    return (points - center) * (float(target_extent) / extent)


def _chunked_min_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int) -> torch.Tensor:
    mins = []
    dst_batch = dst.unsqueeze(0)
    for start in range(0, src.shape[0], chunk_size):
        d = torch.cdist(src[start : start + chunk_size].unsqueeze(0), dst_batch, p=2).squeeze(0)
        mins.append(d.min(dim=1).values.detach().cpu())
    return torch.cat(mins, dim=0)


def _mean_min_distance(src, dst, device: torch.device, chunk_size: int = 2048) -> float:
    mins = _chunked_min_distances(_to_tensor(src, device), _to_tensor(dst, device), chunk_size)
    return float(mins.mean().item())


def compute_gt_diameter(gt_points: torch.Tensor, device: torch.device, chunk_size: int = 1024) -> float:
    """Max pairwise distance within the GT point cloud (the ADD-S normalizer)."""
    pts = _to_tensor(gt_points, device)
    if pts.shape[0] < 2:
        return 0.0
    max_dist = torch.tensor(0.0, device=device, dtype=torch.float32)
    pts_batch = pts.unsqueeze(0)
    for start in range(0, pts.shape[0], chunk_size):
        d = torch.cdist(pts[start : start + chunk_size].unsqueeze(0), pts_batch, p=2).squeeze(0)
        max_dist = torch.maximum(max_dist, d.max())
    return float(max_dist.item())


# ---------------------------------------------------------------------------
# ADD-S
# ---------------------------------------------------------------------------

def compute_add_s(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_diameter: float,
    device: torch.device,
) -> float:
    """Symmetric mean closest-point distance, normalized by GT diameter."""
    diameter = max(float(gt_diameter), 1e-8)
    pred_to_gt = _mean_min_distance(pred_points, gt_points, device)
    gt_to_pred = _mean_min_distance(gt_points, pred_points, device)
    return float((pred_to_gt + gt_to_pred) / (2.0 * diameter))


# ---------------------------------------------------------------------------
# ICP-Rot
# ---------------------------------------------------------------------------

def _center_and_scale_for_icp(
    pred: np.ndarray, gt: np.ndarray, gt_diameter: float
) -> Tuple[np.ndarray, np.ndarray]:
    scale = max(float(gt_diameter), 1e-8)
    return (
        (pred - pred.mean(axis=0, keepdims=True)) / scale,
        (gt - gt.mean(axis=0, keepdims=True)) / scale,
    )


def compute_icp_rot_deg(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_diameter: float,
) -> Dict[str, Optional[float]]:
    """Rotation error via point-to-point ICP from identity init.

    Returns ``icp_rot_deg=None`` when the fit is too poor to trust
    (fitness <= ICP_MIN_FITNESS or no correspondences). Reporting None rather
    than 0.0 matters: a failed fit returns an identity transform, which would
    otherwise be indistinguishable from a perfect 0-degree alignment.
    """
    o3d = ensure_open3d()
    pred_np = np.asarray(pred_points.detach().cpu().numpy(), dtype=np.float64)
    gt_np = np.asarray(gt_points.detach().cpu().numpy(), dtype=np.float64)
    if pred_np.shape[0] == 0 or gt_np.shape[0] == 0:
        raise RuntimeError("ICP requires non-empty point clouds")

    pred_np, gt_np = _center_and_scale_for_icp(pred_np, gt_np, gt_diameter)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pred_np)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(gt_np)

    result = o3d.pipelines.registration.registration_icp(
        src,
        dst,
        max(ICP_MAX_CORR_RATIO, 1e-6),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )

    try:
        num_corr = int(len(result.correspondence_set))
    except Exception:
        num_corr = 0

    valid = bool(num_corr > 0 and float(result.fitness) > ICP_MIN_FITNESS)
    rotation = np.asarray(result.transformation[:3, :3], dtype=np.float64)
    return {
        "icp_rot_deg": rotation_angle_deg(rotation) if valid else None,
        "icp_valid": 1.0 if valid else 0.0,
    }


# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------

def apply_icp_transform(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """Align pred to gt with point-to-point ICP and return transformed points."""
    o3d = ensure_open3d()
    pred_np = np.asarray(pred_points.detach().cpu().numpy(), dtype=np.float64)
    gt_np = np.asarray(gt_points.detach().cpu().numpy(), dtype=np.float64)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pred_np)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(gt_np)

    result = o3d.pipelines.registration.registration_icp(
        src,
        dst,
        max(ICP_MAX_CORR_RATIO, 1e-6),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )

    transform = np.asarray(result.transformation, dtype=np.float64)
    pred_h = np.concatenate([pred_np, np.ones((pred_np.shape[0], 1))], axis=1)
    aligned = (transform @ pred_h.T).T[:, :3]
    return torch.from_numpy(np.asarray(aligned, dtype=np.float32))


def compute_emd_hungarian(pred_points: torch.Tensor, gt_points: torch.Tensor, device: torch.device) -> float:
    """Optimal-assignment (Hungarian) mean matching distance."""
    from scipy.optimize import linear_sum_assignment

    pred = _to_tensor(pred_points, device)
    gt = _to_tensor(gt_points, device)
    n = min(pred.shape[0], gt.shape[0])
    if n == 0:
        raise RuntimeError("EMD requires non-empty point clouds")
    dists = torch.cdist(pred[:n].unsqueeze(0), gt[:n].unsqueeze(0), p=2).squeeze(0)
    dists = dists.detach().cpu().numpy()
    row, col = linear_sum_assignment(dists)
    return float(dists[row, col].mean())


# ---------------------------------------------------------------------------
# 2D silhouette IoU
# ---------------------------------------------------------------------------

def _mesh_to_o3d(mesh: trimesh.Trimesh):
    o3d = ensure_open3d()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        return None
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex["positions"] = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    mesh_t.triangle["indices"] = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    return mesh_t


def raycast_silhouette(
    mesh: trimesh.Trimesh,
    intrinsics: Tuple[float, float, float, float, int, int],
) -> np.ndarray:
    """Boolean hit-mask of ``mesh`` seen from the camera.

    The mesh must already be in camera space, so extrinsics are identity and
    rays originate at the origin.

    Ray directions follow the OpenGL convention this data uses (+X right, +Y up,
    -Z forward), which means the vertical term is NEGATED: image rows increase
    downward while +Y points up. Dropping that negation flips the silhouette
    vertically. It is easy to miss, because pred and GT would flip together and
    their IoU would still look reasonable -- verified instead by projecting the
    GT mesh against its own input/mask_00k.png (0.86 mean IoU with the
    negation, 0.08 without).
    """
    o3d = ensure_open3d()
    fx, fy, cx, cy, width, height = intrinsics
    mesh_t = _mesh_to_o3d(mesh)
    if mesh_t is None:
        return np.zeros((height, width), dtype=bool)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    dirs = np.stack(
        [
            (uu.reshape(-1) - float(cx)) / max(float(fx), 1e-8),
            -(vv.reshape(-1) - float(cy)) / max(float(fy), 1e-8),
            -np.ones(width * height, dtype=np.float32),
        ],
        axis=1,
    )
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    rays = np.concatenate([np.zeros_like(dirs), dirs.astype(np.float32)], axis=1)

    hits = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    return np.isfinite(hits["t_hit"].numpy()).reshape(height, width)


def compute_2d_iou(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    intrinsics: Tuple[float, float, float, float, int, int],
) -> float:
    """Silhouette mask IoU between pred and GT mesh through the GT camera.

    Both meshes must be in camera space.
    """
    pred_mask = raycast_silhouette(pred_mesh, intrinsics)
    gt_mask = raycast_silhouette(gt_mesh, intrinsics)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred_mask, gt_mask).sum() / union)


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def compute_pose_metrics(
    pred_meshes: Sequence[trimesh.Trimesh],
    gt_meshes: Sequence[trimesh.Trimesh],
    scene_name: str,
    device: str = "cuda",
    num_points: int = 20480,
) -> list:
    """Per-object ADD-S / ICP-Rot / EMD.

    ``pred_meshes[i]`` is paired with ``gt_meshes[i]``; the caller is
    responsible for having ordered them consistently. 2D IoU is computed
    separately by the caller, since it must project in camera space while these
    metrics run in the caller's chosen frame.
    """
    if len(pred_meshes) != len(gt_meshes):
        raise ValueError(f"Object count mismatch: pred={len(pred_meshes)}, gt={len(gt_meshes)}")
    if len(pred_meshes) == 0:
        raise RuntimeError("No meshes to evaluate")

    torch_device = torch.device(device)
    rows = []

    for idx, (pred_mesh, gt_mesh) in enumerate(zip(pred_meshes, gt_meshes)):
        pred_points = sample_surface_points(pred_mesh, num_points, derive_seed(scene_name, idx, "pred"))
        gt_points = sample_surface_points(gt_mesh, num_points, derive_seed(scene_name, idx, "gt"))
        diameter_points = sample_surface_points(
            gt_mesh, DIAMETER_POINTS, derive_seed(scene_name, idx, "gt_diameter")
        )
        gt_diameter = compute_gt_diameter(diameter_points, device=torch_device)

        add_s = compute_add_s(pred_points, gt_points, gt_diameter, torch_device)
        icp = compute_icp_rot_deg(pred_points, gt_points, gt_diameter)

        pred_emd = normalize_by_bbox(
            sample_surface_points(pred_mesh, EMD_POINTS, derive_seed(scene_name, idx, "pred_emd"))
        )
        gt_emd = normalize_by_bbox(
            sample_surface_points(gt_mesh, EMD_POINTS, derive_seed(scene_name, idx, "gt_emd"))
        )
        emd = compute_emd_hungarian(apply_icp_transform(pred_emd, gt_emd), gt_emd, torch_device)

        row = {
            "object_idx": idx,
            "add_s": float(add_s),
            "icp_rot_deg": (
                float("nan") if icp["icp_rot_deg"] is None else float(icp["icp_rot_deg"])
            ),
            "icp_valid": float(icp["icp_valid"]),
            "emd": float(emd),
        }

        rows.append(row)

    return rows
