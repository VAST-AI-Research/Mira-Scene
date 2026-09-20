"""
eval_scene.py

Evaluate scene reconstruction quality: predicted scene.glb against the GT
scene_camera.glb.

Metrics (the paper set):

  Object-level -- each object is normalized into its own [-0.95, 0.95] AABB and
  robust-ICP aligned to its GT counterpart first, so these are pure shape
  metrics and do not double-count pose error:
    object_cd        Bidirectional Chamfer Distance (mean of squared distances)
    object_fscore    F-score at tau = 0.1
    object_emd       Hungarian-assignment EMD after per-object ICP

  Scene-level -- computed on posed objects, so these capture placement:
    iou_3d           Per-object axis-aligned 3D bounding-box IoU
    icp_rot_deg      Rotation error from point-to-point ICP (degrees)
    iou_2d           Silhouette mask IoU via Open3D raycast, GT camera
    add_s            ADD-S normalized by GT diameter

Coordinate frame
----------------
Both scene.glb and scene_camera.glb are stored in camera space (OpenGL: +X
right, +Y up, -Z forward). Metrics are evaluated in WORLD space by default,
applying c2w from gt/camera.json to both sides -- this is the convention the
published numbers use.

The frame is not cosmetic. CD, F-score, 3D IoU and EMD all normalize by
axis-aligned bounding boxes, which are not rotation-invariant, so evaluating in
camera space shifts them measurably (on one 8-scene set, object_cd 0.0185 in
camera space vs 0.0213 in world space). ADD-S and ICP-Rot are unaffected,
because c2w is rigid. Use --frame camera only to reproduce a camera-space run.

2D IoU always projects in camera space with identity extrinsics, which is exact
and needs no matrix inverse.

Data layout:
    GT:   {gt_dir}/{scene}/gt/scene_camera.glb   nodes object_000, object_001, ...
          {gt_dir}/{scene}/gt/camera.json        H, W, fov_x, fov_y, c2w
    Pred: {pred_dir}/{scene}/scene.glb           nodes geometry_0, geometry_1, ...

Objects are paired by index: the k-th pred object against the k-th GT object.
Node names are sorted by numeric suffix, so pred geometry_k pairs with GT
object_00k. A scene whose object counts disagree is reported as a mismatch
rather than silently truncated -- mispaired objects yield plausible-looking but
meaningless numbers.

Usage:
    python eval_scripts/eval_scene.py \
        --pred_dir /path/to/predictions \
        --gt_dir /path/to/blendswap_eval
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.alignment import robust_icp_align
from utils.pose_metrics import (
    compute_2d_iou as compute_2d_iou_fn,
    compute_pose_metrics,
    derive_seed,
    sample_surface_points,
)
from utils.scene_metrics import (
    compute_chamfer_distance,
    compute_fscore,
    compute_volume_iou,
    normalize_points,
)

OBJECT_PAT = re.compile(r"^object_(\d+)$")
GEOMETRY_PAT = re.compile(r"^geometry_(\d+)$")


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def sort_node_names(node_names):
    """Order nodes deterministically by numeric suffix.

    GT uses object_000..., pred uses geometry_0.... Anything else keeps the raw
    graph order: an arbitrary-but-stable order beats an ad-hoc guess, and count
    mismatches are caught downstream.
    """
    if not node_names:
        return node_names
    if all(OBJECT_PAT.match(n) for n in node_names):
        return sorted(node_names, key=lambda n: int(OBJECT_PAT.match(n).group(1)))
    if all(GEOMETRY_PAT.match(n) for n in node_names):
        return sorted(node_names, key=lambda n: int(GEOMETRY_PAT.match(n).group(1)))
    return list(node_names)


def extract_meshes(scene):
    """Per-object meshes with their scene-graph transforms baked in."""
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    meshes = []
    for node_name in sort_node_names(list(scene.graph.nodes_geometry)):
        transform, geometry_name = scene.graph[node_name]
        geom = scene.geometry.get(geometry_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def load_camera(gt_scene_dir):
    """Return (intrinsics, c2w) from gt/camera.json; entries may be None."""
    camera_path = os.path.join(gt_scene_dir, "gt", "camera.json")
    if not os.path.exists(camera_path):
        return None, None
    with open(camera_path) as f:
        cam = json.load(f)

    intrinsics = None
    if all(k in cam for k in ("H", "W", "fov_x", "fov_y")):
        H, W = int(cam["H"]), int(cam["W"])
        fx = W / (2.0 * math.tan(float(cam["fov_x"]) / 2.0))
        fy = H / (2.0 * math.tan(float(cam["fov_y"]) / 2.0))
        intrinsics = (fx, fy, W / 2.0, H / 2.0, W, H)

    c2w = None
    if "c2w" in cam:
        m = np.array(cam["c2w"], dtype=np.float64)
        if m.shape == (4, 4):
            # Blender cameras can carry non-unit scale on the rotation columns;
            # strip it so the transform stays rigid (which is what keeps ADD-S
            # and ICP-Rot frame-invariant).
            scales = np.linalg.norm(m[:3, :3], axis=0)
            if not np.allclose(scales, 1.0, atol=1e-4):
                m = m.copy()
                m[:3, :3] = m[:3, :3] / scales[np.newaxis, :]
            c2w = m

    return intrinsics, c2w


def transform_meshes(meshes, matrix):
    if matrix is None:
        return meshes
    out = []
    for mesh in meshes:
        copied = mesh.copy()
        copied.apply_transform(matrix)
        out.append(copied)
    return out


def sample_object_points(meshes, scene_name, num_points, kind):
    """Stack per-object seeded surface samples into (n_objects, num_points, 3)."""
    return torch.stack(
        [
            sample_surface_points(m, num_points, derive_seed(scene_name, i, kind))
            for i, m in enumerate(meshes)
        ]
    )


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    pred_scene_path,
    gt_scene_dir,
    scene_name,
    num_points=20480,
    chunk_size=10240,
    fscore_threshold=0.1,
    device="cuda",
    frame="world",
    use_alignment=True,
    compute_2d_iou=True,
):
    """Evaluate one scene. Returns (metrics, per_object_rows) or {'error': ...}."""
    gt_path = os.path.join(gt_scene_dir, "gt", "scene_camera.glb")
    if not os.path.exists(gt_path):
        return {"error": "missing_gt_scene_camera"}
    if not os.path.exists(pred_scene_path):
        return {"error": "missing_pred_scene"}

    pred_cam = extract_meshes(trimesh.load(pred_scene_path, force="scene"))
    gt_cam = extract_meshes(trimesh.load(gt_path, force="scene"))

    if len(pred_cam) == 0 or len(gt_cam) == 0:
        return {"error": "no_meshes"}
    if len(pred_cam) != len(gt_cam):
        return {
            "error": f"object_count_mismatch_pred{len(pred_cam)}_gt{len(gt_cam)}",
            "num_pred_objects": len(pred_cam),
            "num_gt_objects": len(gt_cam),
        }

    intrinsics, c2w = load_camera(gt_scene_dir)
    if frame == "world":
        if c2w is None:
            return {"error": "missing_c2w_in_camera_json"}
        pred_meshes = transform_meshes(pred_cam, c2w)
        gt_meshes = transform_meshes(gt_cam, c2w)
    else:
        pred_meshes, gt_meshes = pred_cam, gt_cam

    n_objects = len(gt_meshes)
    pred_pts = sample_object_points(pred_meshes, scene_name, num_points, "pred").to(device)
    gt_pts = sample_object_points(gt_meshes, scene_name, num_points, "gt").to(device)

    # --- Object-level shape metrics, in per-object normalized space ---
    pred_norm = normalize_points(pred_pts)
    gt_norm = normalize_points(gt_pts)
    if use_alignment:
        # Yaw is swept about z: world frame here is z-up.
        pred_norm = robust_icp_align(pred_norm, gt_norm, up_axis=2)
    object_cd, _, _ = compute_chamfer_distance(pred_norm, gt_norm, chunk_size=chunk_size)
    object_fscore = compute_fscore(
        pred_norm, gt_norm, tau=fscore_threshold, chunk_size=chunk_size
    )

    # --- Scene-level placement metric, on posed objects ---
    iou_3d = compute_volume_iou(pred_pts, gt_pts)

    result = {
        "num_objects": n_objects,
        "object_cd": float(object_cd.mean()),
        "object_fscore": float(object_fscore.mean()),
        "iou_3d": float(iou_3d.mean()),
    }

    # --- Pose metrics (ADD-S / ICP-Rot / EMD / 2D IoU) ---
    # 2D IoU is projected from the camera-space meshes with identity
    # extrinsics; the other pose metrics use the selected frame.
    rows = compute_pose_metrics(
        pred_meshes,
        gt_meshes,
        scene_name=scene_name,
        device=device,
        num_points=num_points,
    )
    if compute_2d_iou and intrinsics is not None:
        for i, row in enumerate(rows):
            row["iou_2d"] = compute_2d_iou_fn(pred_cam[i], gt_cam[i], intrinsics)

    def finite_mean(key):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    result["object_emd"] = finite_mean("emd")
    result["icp_rot_deg"] = finite_mean("icp_rot_deg")
    result["add_s"] = finite_mean("add_s")
    # Objects whose ICP fit was too poor to trust are excluded from
    # icp_rot_deg; surface the count so a low angle over 1/7 objects is not
    # mistaken for a good scene-wide result.
    result["num_valid_icp"] = int(sum(r.get("icp_valid", 0.0) for r in rows))
    if "iou_2d" in rows[0]:
        result["iou_2d"] = finite_mean("iou_2d")

    per_object = [
        {
            "scene_name": scene_name,
            "object_idx": i,
            "object_cd": float(object_cd[i]),
            "object_fscore": float(object_fscore[i]),
            "object_emd": rows[i]["emd"],
            "iou_3d": float(iou_3d[i]),
            "icp_rot_deg": rows[i]["icp_rot_deg"],
            "add_s": rows[i]["add_s"],
            **({"iou_2d": rows[i]["iou_2d"]} if "iou_2d" in rows[i] else {}),
        }
        for i in range(n_objects)
    ]

    return result, per_object


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    "object_cd",
    "object_fscore",
    "object_emd",
    "iou_3d",
    "icp_rot_deg",
    "iou_2d",
    "add_s",
]


# Pipeline layouts differ: construct_scene.py writes into a gt_scene/ subdir,
# while the merged eval dumps drop scene.glb at the scene root.
PRED_SCENE_CANDIDATES = ("scene.glb", "gt_scene/scene.glb")


def count_scenes(pred_dir, gt_dir, pred_scene_file):
    """Scenes with both a pred mesh at `pred_scene_file` and a GT dir."""
    return sorted(
        d
        for d in os.listdir(pred_dir)
        if os.path.isfile(os.path.join(pred_dir, d, pred_scene_file))
        and os.path.isdir(os.path.join(gt_dir, d))
    )


def detect_pred_scene_file(pred_dir, gt_dir):
    """Pick the candidate layout matching the most scenes (None if none match)."""
    best, best_scenes = None, []
    for candidate in PRED_SCENE_CANDIDATES:
        scenes = count_scenes(pred_dir, gt_dir, candidate)
        if len(scenes) > len(best_scenes):
            best, best_scenes = candidate, scenes
    return best, best_scenes


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate scene reconstruction")
    p.add_argument("--pred_dir", required=True, help="Prediction dir: {pred_dir}/{scene}/scene.glb")
    p.add_argument("--gt_dir", required=True, help="GT dir: {gt_dir}/{scene}/gt/scene_camera.glb")
    p.add_argument(
        "--pred_scene_file",
        default=None,
        help="Path to the scene mesh relative to {pred_dir}/{scene}/. "
        f"Auto-detected from {PRED_SCENE_CANDIDATES} when omitted.",
    )
    p.add_argument(
        "--frame",
        default="world",
        choices=["world", "camera"],
        help="Frame for metric computation (default: world, the paper convention)",
    )
    p.add_argument(
        "--no_alignment",
        action="store_true",
        help="Skip per-object robust ICP before object CD / F-score",
    )
    p.add_argument("--num_points", type=int, default=20480)
    p.add_argument("--chunk_size", type=int, default=10240)
    p.add_argument("--fscore_threshold", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_2d_iou", action="store_true", help="Skip 2D IoU (slowest metric)")
    p.add_argument("--max_scenes", type=int, default=-1)
    p.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Exact scene ID to evaluate; repeatable. Applied before --scene_filter.",
    )
    p.add_argument("--scene_filter", default=None, help="Only scenes containing this substring")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--per_object_csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.pred_dir, "eval_scene_results.csv")
    if args.per_object_csv is None:
        args.per_object_csv = os.path.join(args.pred_dir, "eval_scene_results_per_object.csv")

    # A scene counts only if both sides exist, so a partial pred dir yields a
    # short run rather than a wall of skips.
    if args.pred_scene_file is None:
        args.pred_scene_file, scenes = detect_pred_scene_file(args.pred_dir, args.gt_dir)
        if args.pred_scene_file is None:
            print(
                f"No scenes found. Expected {args.pred_dir}/{{scene}}/<scene file> "
                f"(tried {', '.join(PRED_SCENE_CANDIDATES)}) "
                f"with a matching {args.gt_dir}/{{scene}}/"
            )
            return
    else:
        scenes = count_scenes(args.pred_dir, args.gt_dir, args.pred_scene_file)

    if args.scene:
        selected = set(args.scene)
        scenes = [s for s in scenes if s in selected]
    if args.scene_filter:
        scenes = [s for s in scenes if args.scene_filter in s]
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    if not scenes:
        print(
            f"No scenes found. Expected {args.pred_dir}/{{scene}}/{args.pred_scene_file} "
            f"with a matching {args.gt_dir}/{{scene}}/"
        )
        return

    print(f"Evaluating {len(scenes)} scenes")
    print(f"  Pred:  {args.pred_dir}/{{scene}}/{args.pred_scene_file}")
    print(f"  GT:    {args.gt_dir}/{{scene}}/gt/scene_camera.glb")
    print(f"  Frame: {args.frame}, per-object ICP: {not args.no_alignment}")
    print(f"  Points/object: {args.num_points}, F-score tau: {args.fscore_threshold}")
    print(f"  2D IoU: {not args.no_2d_iou}, device: {args.device}\n")

    results, per_object_rows, failures = [], [], []
    for scene_name in tqdm(scenes, desc="Eval"):
        pred_path = os.path.join(args.pred_dir, scene_name, args.pred_scene_file)
        gt_scene_dir = os.path.join(args.gt_dir, scene_name)
        try:
            out = evaluate_scene(
                pred_path,
                gt_scene_dir,
                scene_name=scene_name,
                num_points=args.num_points,
                chunk_size=args.chunk_size,
                fscore_threshold=args.fscore_threshold,
                device=args.device,
                frame=args.frame,
                use_alignment=not args.no_alignment,
                compute_2d_iou=not args.no_2d_iou,
            )
        except Exception as exc:
            failures.append((scene_name, f"{type(exc).__name__}: {exc}"))
            continue

        if isinstance(out, dict) and "error" in out:
            failures.append((scene_name, out["error"]))
            continue

        metrics, per_object = out
        metrics["scene_name"] = scene_name
        results.append(metrics)
        per_object_rows.extend(per_object)

    if failures:
        print(f"\n{len(failures)} scene(s) not evaluated:")
        for name, reason in failures:
            print(f"  {name}: {reason}")

    if not results:
        print("\nNo valid results.")
        return

    df = pd.DataFrame(results)
    cols = ["scene_name", "num_objects"] + [c for c in METRIC_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    print("\n=== Per-Scene Results ===")
    print(df.to_string(index=False, float_format="%.4f"))

    # Scene-equal weighting (each scene counts once), matching how the table
    # above reads. Object-weighted means differ when object counts vary, which
    # is what the per-object CSV is for.
    print("\n=== Average (scene-equal) ===")
    for col in [c for c in METRIC_ORDER if c in df.columns]:
        vals = df[col].dropna()
        if len(vals):
            print(f"  {col}: {vals.mean():.4f}")

    avg_row = {"scene_name": "AVERAGE", "num_objects": int(df["num_objects"].sum())}
    for col in df.columns:
        if col in ("scene_name", "num_objects"):
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        avg_row[col] = vals.mean() if len(vals) else float("nan")
    df_out = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    for path, frame_df in (
        (args.output_csv, df_out),
        (args.per_object_csv, pd.DataFrame(per_object_rows)),
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        frame_df.to_csv(path, index=False, float_format="%.6f")

    print(f"\nSaved: {args.output_csv}")
    print(f"Saved: {args.per_object_csv}")


if __name__ == "__main__":
    main()
