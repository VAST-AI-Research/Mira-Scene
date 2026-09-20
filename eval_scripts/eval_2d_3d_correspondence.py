"""
eval_2d_3d_correspondence.py

Evaluate 2D-3D correspondence consistency between predicted CCM and
reconstructed mesh (glb).

No GT data required — this measures self-consistency:
  1. Estimate camera pose from CCM via PnP (2D pixel <-> 3D canonical coords)
  2. Raycast the reconstructed mesh with this camera -> mesh-projected coord_map + mask
  3. Compare CCM coord_map vs mesh coord_map -> Chamfer Distance + F-score
  4. Compare CCM mask vs mesh mask -> 2D IoU

Input: {pred_dir}/{scene}/ containing:
  - canonical_coord_map_000.npy (or canonical_coord_map_restored_000.npy)
  - {recon_subdir}/000.glb (reconstructed mesh, e.g. moge_recon/000.glb)

Usage:
    python eval_scripts/eval_2d_3d_correspondence.py \\
        --pred_dir /path/to/predictions \\
        --recon_subdir moge_recon
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils.metrics import compute_iou, chamfer_and_fscore, align_scale_shift


# ---------------------------------------------------------------------------
# Y-up to Z-up transform (glb is Y-up, CCM is Z-up)
# ---------------------------------------------------------------------------

Y_UP_TO_Z_UP = np.array([
    [1,  0,  0,  0],
    [0,  0, -1,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera pose estimation from CCM via PnP
# ---------------------------------------------------------------------------

def estimate_camera_pose(ccm, resolution=518):
    """Estimate camera intrinsic + extrinsic from canonical coord map via PnP.

    Args:
        ccm: [3, H, W] float32, canonical z-up 3D coords (0 where invalid)

    Returns:
        R_cw, t_cw, K, success
    """
    H, W = ccm.shape[1], ccm.shape[2]
    ccm_hw3 = ccm.transpose(1, 2, 0)  # [H, W, 3]

    mask = np.abs(ccm_hw3).sum(axis=-1) > 1e-6
    vs, us = np.where(mask)
    pts_3d = ccm_hw3[vs, us].astype(np.float64)
    pts_2d = np.stack([us + 0.5, vs + 0.5], axis=-1).astype(np.float64)

    if len(pts_3d) < 6:
        return None, None, None, False

    f_init = W / (2.0 * np.tan(np.deg2rad(30)))
    K = np.array([[f_init, 0, W / 2.0], [0, f_init, H / 2.0], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, dist_coeffs,
        iterationsCount=10000, reprojectionError=4.0,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success or inliers is None or len(inliers) < 6:
        return None, None, None, False

    rvec, tvec = cv2.solvePnPRefineLM(
        pts_3d[inliers.ravel()], pts_2d[inliers.ravel()],
        K, dist_coeffs, rvec, tvec,
    )
    R_cw, _ = cv2.Rodrigues(rvec)

    n_inliers = len(inliers)
    pts_2d_reproj, _ = cv2.projectPoints(pts_3d[inliers.ravel()], rvec, tvec, K, dist_coeffs)
    reproj_err = np.linalg.norm(pts_2d_reproj.reshape(-1, 2) - pts_2d[inliers.ravel()], axis=-1).mean()
    print(f"    PnP: {n_inliers}/{len(pts_3d)} inliers, reproj_err={reproj_err:.2f}px")

    return R_cw, tvec, K, True


# ---------------------------------------------------------------------------
# Mesh raycasting
# ---------------------------------------------------------------------------

def raycast_mesh(mesh, R_cw, t_cw, K, resolution=518):
    """Raycast mesh using estimated camera pose.

    Returns:
        hit_mask: [H, W] bool
        coord_map: [H, W, 3] float32, z-up canonical coords at each hit pixel
    """
    H = W = resolution
    rc_scene = o3d.t.geometry.RaycastingScene()
    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(mesh.vertices.astype(np.float32))
    mesh_o3d.triangle.indices = o3d.core.Tensor(mesh.faces.astype(np.int32))
    rc_scene.add_triangles(mesh_o3d)

    R_wc = R_cw.T
    cam_origin = -R_wc @ t_cw.ravel()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy, np.ones_like(uu)], axis=-1)
    dirs_world = (R_wc @ dirs_cam.reshape(-1, 3).T).T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    origins = np.broadcast_to(cam_origin, dirs_world.shape).copy()
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)

    result = rc_scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = result["t_hit"].numpy().reshape(H, W)
    hit_mask = np.isfinite(t_hit)

    coord_map = np.broadcast_to(cam_origin, (H, W, 3)).copy() + t_hit[..., None] * dirs_world.reshape(H, W, 3)
    coord_map[~hit_mask] = 0.0

    return hit_mask, coord_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(scene_dir, recon_subdir, resolution, fscore_thresholds, max_pts):
    """Evaluate 2D-3D correspondence for one scene."""
    # Find CCM
    ccm = None
    for fname in ["canonical_coord_map_000.npy", "canonical_coord_map_restored_000.npy"]:
        path = os.path.join(scene_dir, fname)
        if os.path.exists(path):
            ccm = np.load(path).astype(np.float32)
            if ccm.ndim == 3 and ccm.shape[0] != 3:
                ccm = ccm.transpose(2, 0, 1)
            break
    if ccm is None:
        return None

    # Find glb mesh
    glb_path = os.path.join(scene_dir, recon_subdir, "000.glb")
    if not os.path.exists(glb_path):
        return None

    # Estimate camera from CCM
    R_cw, t_cw, K, success = estimate_camera_pose(ccm, resolution)
    if not success:
        return None

    # Load and convert mesh (Y-up -> Z-up)
    mesh = trimesh.load(glb_path, force="mesh")
    mesh.apply_transform(Y_UP_TO_Z_UP)

    # Raycast
    hit_mask, mesh_coord_map = raycast_mesh(mesh, R_cw, t_cw, K, resolution)

    # CCM mask and points
    ccm_hw3 = ccm.transpose(1, 2, 0)
    ccm_valid = np.abs(ccm_hw3).sum(axis=-1) > 1e-6

    # 2D IoU
    iou = compute_iou(ccm_valid, hit_mask)

    # 3D correspondence
    overlap = ccm_valid & hit_mask
    if overlap.sum() < 10:
        return {"iou": iou, "chamfer": None, "fscores": None}

    s, t = align_scale_shift(mesh_coord_map[overlap], ccm_hw3[overlap])
    mesh_aligned = mesh_coord_map * s + t

    ccm_pts = ccm_hw3[ccm_valid]
    mesh_pts = mesh_aligned[hit_mask]

    if len(ccm_pts) > max_pts:
        ccm_pts = ccm_pts[np.random.choice(len(ccm_pts), max_pts, replace=False)]
    if len(mesh_pts) > max_pts:
        mesh_pts = mesh_pts[np.random.choice(len(mesh_pts), max_pts, replace=False)]

    chamfer, fscores = chamfer_and_fscore(ccm_pts, mesh_pts, fscore_thresholds)

    # Save correspondence outputs for inspection
    corr_dir = os.path.join(scene_dir, recon_subdir, "correspondence")
    os.makedirs(corr_dir, exist_ok=True)
    Image.fromarray((hit_mask.astype(np.uint8) * 255)).save(os.path.join(corr_dir, "mesh_mask.png"))
    np.save(os.path.join(corr_dir, "mesh_coord_map.npy"), mesh_coord_map)

    return {"iou": iou, "chamfer": chamfer, "fscores": fscores}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D-3D correspondence consistency")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Prediction output directory")
    parser.add_argument("--recon_subdir", type=str, default="moge_recon",
                        help="Reconstruction subdirectory containing 000.glb")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--fscore_thresholds", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--max_pts", type=int, default=50000)
    parser.add_argument("--max_cases", type=int, default=-1)
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.pred_dir, f"eval_2d_3d_{args.recon_subdir}.csv")

    np.random.seed(42)
    thresholds = tuple(args.fscore_thresholds)

    scene_names = sorted([
        d for d in os.listdir(args.pred_dir)
        if os.path.isdir(os.path.join(args.pred_dir, d))
    ])
    if args.max_cases > 0:
        scene_names = scene_names[:args.max_cases]

    print(f"Evaluating 2D-3D correspondence: {len(scene_names)} scenes")
    print(f"  Pred: {args.pred_dir}")
    print(f"  Recon: {args.recon_subdir}\n")

    results = []
    for name in scene_names:
        scene_dir = os.path.join(args.pred_dir, name)
        print(f"  {name}")

        metrics = evaluate_scene(scene_dir, args.recon_subdir, args.resolution, thresholds, args.max_pts)
        if metrics is None:
            print(f"    SKIP (missing CCM or glb, or PnP failed)")
            results.append({"scene": name, "status": "skip"})
            continue

        parts = [f"IoU={metrics['iou']:.4f}"]
        if metrics["chamfer"] is not None:
            parts.append(f"CD={metrics['chamfer']:.6f}")
            parts.extend(f"F@{th}={metrics['fscores'][th]:.4f}" for th in thresholds)
        print(f"    {', '.join(parts)}")

        entry = {"scene": name, "status": "ok", "iou": metrics["iou"]}
        if metrics["chamfer"] is not None:
            entry["chamfer"] = metrics["chamfer"]
            for th in thresholds:
                entry[f"F@{th}"] = metrics["fscores"][th]
        results.append(entry)

    # Summary
    valid = [r for r in results if r["status"] == "ok"]
    metric_keys = ["iou", "chamfer"] + [f"F@{th}" for th in thresholds]
    if valid:
        avgs = {}
        for k in metric_keys:
            vals = [r[k] for r in valid if k in r and r[k] is not None]
            if vals:
                avgs[k] = np.mean(vals)
        print(f"\n  AVERAGE ({len(valid)} scenes): " +
              ", ".join(f"{k}={avgs[k]:.6f}" for k in metric_keys if k in avgs))
    else:
        avgs = {}

    # CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene"] + metric_keys)
        for r in results:
            if r["status"] == "ok":
                row = [r["scene"]]
                for k in metric_keys:
                    row.append(f"{r[k]:.6f}" if k in r and r[k] is not None else "N/A")
                writer.writerow(row)
            else:
                writer.writerow([r["scene"]] + ["N/A"] * len(metric_keys))
        if avgs:
            writer.writerow(["AVERAGE"] + [f"{avgs[k]:.6f}" if k in avgs else "N/A" for k in metric_keys])

    print(f"\n  Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
