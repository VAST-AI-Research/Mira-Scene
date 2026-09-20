"""
visualization.py

Standalone visualization utilities for scene reconstruction.
No dependency on miraccm.
"""

import os
from typing import List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image


# Default color palette (RGBA, 24 colors)
COLORS = [
    [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
    [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
    [128, 0, 0, 255], [0, 128, 0, 255], [0, 0, 128, 255],
    [128, 128, 0, 255], [128, 0, 128, 255], [0, 128, 128, 255],
    [64, 0, 0, 255], [0, 64, 0, 255], [0, 0, 64, 255],
    [64, 64, 0, 255], [64, 0, 64, 255], [0, 64, 64, 255],
    [192, 192, 192, 255], [128, 128, 128, 255],
    [255, 165, 0, 255], [75, 0, 130, 255], [238, 130, 238, 255],
    [0, 0, 0, 255],
]


def transform_pcd(transform_dict, pcd: np.ndarray) -> np.ndarray:
    """Apply similarity transform: world = s * pcd @ R^T + t.

    Args:
        transform_dict: dict with 's' (scalar), 'R' (3x3), 't' (3,).
        pcd: [N, 3] numpy array.

    Returns:
        [N, 3] transformed points.
    """
    s = transform_dict["s"]
    R = transform_dict["R"]
    t = transform_dict["t"]
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return (s * pcd @ R.T) + t


def save_scene_pcd(
    canonical_pcds: List[np.ndarray],
    transform_dict_list: List[dict],
    save_path: str,
    to_y_up: bool = False,
):
    """Transform canonical PCDs to scene space and save as colored PLY.

    Args:
        canonical_pcds: list of [N, 3] per-instance canonical point clouds.
        transform_dict_list: list of {s, R, t} dicts, one per instance.
        save_path: output PLY path.
        to_y_up: if True, rotate Z-up → Y-up before saving.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_pts, all_colors = [], []
    for i, (pcd, tf) in enumerate(zip(canonical_pcds, transform_dict_list)):
        if len(pcd) == 0:
            continue
        world_pcd = transform_pcd(tf, pcd)
        color = COLORS[i % len(COLORS)]
        all_pts.append(world_pcd)
        all_colors.append(np.tile(color, (len(world_pcd), 1)))

    if not all_pts:
        return

    scene = trimesh.PointCloud(
        np.concatenate(all_pts, axis=0),
        colors=np.concatenate(all_colors, axis=0),
    )
    if to_y_up:
        scene.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
    scene.export(save_path)


def save_projection_image(
    canonical_pcds: List[np.ndarray],
    transform_dict_list: List[dict],
    save_path: str,
    fov_rad: float,
    h: int = 518,
    w: int = 518,
    mask_np: Optional[np.ndarray] = None,
):
    """Project transformed PCDs onto a 2D image and save as PNG.

    Args:
        canonical_pcds: list of [N, 3] per-instance canonical point clouds.
        transform_dict_list: list of {s, R, t} dicts.
        save_path: output PNG path.
        fov_rad: horizontal FOV in radians.
        h, w: image dimensions.
        mask_np: optional [NI, H, W] or [H, W] bool mask for semi-transparent background overlay.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    f = 1.0 / np.tan(fov_rad / 2.0)
    f_y = f * w / h
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, (pcd, tf) in enumerate(zip(canonical_pcds, transform_dict_list)):
        if len(pcd) == 0:
            continue
        pts = transform_pcd(tf, pcd)
        color_rgb = np.array(COLORS[i % len(COLORS)][:3], dtype=np.uint8)
        valid = pts[:, 2] < 0
        if not np.any(valid):
            continue
        pts_v = pts[valid]
        z = -pts_v[:, 2]
        x_ndc = pts_v[:, 0] / z * f
        y_ndc = pts_v[:, 1] / z * f_y
        px = ((x_ndc * 0.5 + 0.5) * w).astype(np.int32)
        py = ((1.0 - (y_ndc * 0.5 + 0.5)) * h).astype(np.int32)
        in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        canvas[py[in_bounds], px[in_bounds]] = color_rgb

    # Overlay instance masks as semi-transparent background
    if mask_np is not None:
        masks = mask_np if mask_np.ndim == 3 else mask_np[np.newaxis]  # [NI, H, W]
        mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(masks.shape[0]):
            color_rgb_m = np.array(COLORS[i % len(COLORS)][:3], dtype=np.uint8)
            m = masks[i]
            if m.shape != (h, w):
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
            mask_overlay[m] = color_rgb_m
        has_pts = canvas.sum(axis=-1) > 0
        blended = (mask_overlay.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
        blended[has_pts] = canvas[has_pts]
        canvas = blended

    Image.fromarray(canvas).save(save_path)
