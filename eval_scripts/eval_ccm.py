"""
eval_ccm.py

Evaluate predicted CCM quality against GT CCM.

Pred: {pred_dir}/{scene}/canonical_coord_map_restored_000.npy  (preferred)
      {pred_dir}/{scene}/canonical_coord_map_000.npy           (fallback)
GT:   {gt_dir}/{scene}/ccm.exr

Computes Chamfer Distance and F-score after scale+shift alignment.

Usage:
    python eval_scripts/eval_ccm.py \\
        --pred_dir /path/to/predictions \\
        --gt_dir /path/to/ccm_ground_truth
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.metrics import load_coord_map_npy, load_coord_map_exr, eval_coord_maps


def find_pred_ccm(scene_dir):
    """Find predicted CCM. Prefer restored (scene-level)."""
    for fname in ["canonical_coord_map_restored_000.npy", "canonical_coord_map_000.npy"]:
        path = os.path.join(scene_dir, fname)
        if os.path.exists(path):
            return path
    return None


def find_gt_ccm(gt_scene_dir):
    """Find GT CCM. Supports .exr and .npy."""
    for fname, loader in [("ccm.exr", "exr"), ("canonical_coord_map.npy", "npy")]:
        path = os.path.join(gt_scene_dir, fname)
        if os.path.exists(path):
            return path, loader
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate CCM prediction quality")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--fscore_thresholds", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--max_pts", type=int, default=50000)
    parser.add_argument("--max_cases", type=int, default=-1)
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.pred_dir, "eval_ccm_results.csv")

    np.random.seed(42)
    thresholds = tuple(args.fscore_thresholds)

    scene_names = sorted([
        d for d in os.listdir(args.pred_dir)
        if os.path.isdir(os.path.join(args.pred_dir, d))
    ])
    if args.max_cases > 0:
        scene_names = scene_names[:args.max_cases]

    print(f"Evaluating {len(scene_names)} scenes (CCM)")
    print(f"  Pred: {args.pred_dir}")
    print(f"  GT:   {args.gt_dir}\n")

    results = []
    for name in scene_names:
        pred_path = find_pred_ccm(os.path.join(args.pred_dir, name))
        gt_path, gt_loader = find_gt_ccm(os.path.join(args.gt_dir, name))

        if pred_path is None or gt_path is None:
            print(f"  {name}: SKIP")
            results.append({"scene": name, "status": "skip"})
            continue

        pred, pred_valid = load_coord_map_npy(pred_path)
        gt, gt_valid = (load_coord_map_exr if gt_loader == "exr" else load_coord_map_npy)(gt_path)
        if gt is None:
            results.append({"scene": name, "status": "skip"})
            continue

        metrics = eval_coord_maps(pred, pred_valid, gt, gt_valid, thresholds, args.max_pts)
        if metrics is None:
            print(f"  {name}: SKIP (insufficient overlap)")
            results.append({"scene": name, "status": "skip"})
            continue

        fs_str = ", ".join(f"F@{th}={metrics[f'F@{th}']:.4f}" for th in thresholds)
        print(f"  {name}: CD={metrics['chamfer']:.6f}, {fs_str}")
        results.append({"scene": name, "status": "ok", **metrics})

    # Summary
    valid = [r for r in results if r["status"] == "ok"]
    metric_keys = ["chamfer"] + [f"F@{th}" for th in thresholds]
    if valid:
        avgs = {k: np.mean([r[k] for r in valid]) for k in metric_keys}
        print(f"\n  AVERAGE ({len(valid)} scenes): " +
              ", ".join(f"{k}={avgs[k]:.6f}" for k in metric_keys))
    else:
        avgs = {}

    # CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene"] + metric_keys)
        for r in results:
            if r["status"] == "ok":
                writer.writerow([r["scene"]] + [f"{r[k]:.6f}" for k in metric_keys])
            else:
                writer.writerow([r["scene"]] + ["N/A"] * len(metric_keys))
        if avgs:
            writer.writerow(["AVERAGE"] + [f"{avgs[k]:.6f}" for k in metric_keys])

    print(f"\n  Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
