#!/usr/bin/env python3
"""Estimate depth and camera-space points for prepared Mira-Scene cases.

Input format
------------
The default input root is ``demo_data/``. Each case must contain a scene image
at ``<case>/input/scene.png`` (``scene.png`` and ``original_image.png`` are
also accepted for compatibility). Optional object/floor masks can be stored
alongside the scene image::

    demo_data/
    └── <case>/
        └── input/
            ├── scene.png
            ├── scene_fg.png       # optional
            ├── mask_000.png       # optional object masks
            └── floor_mask.png     # optional

Output format
-------------
Results are written to ``<case>/depth/<method>/``::

    depth.npy              # [H, W] float32 positive depth
    camera_pts_map.npy     # [H, W, 3] float32 OpenGL camera-space points
    valid_mask.npy         # [H, W] bool
    intrinsics.npy         # [3, 3] float32 camera intrinsic matrix
    fov_x_rad.txt          # horizontal FOV in radians
    camera_pts.ply         # RGB-colored point cloud for visualization
    metric_alignment.json  # written when --enable_metric is used

With ``--enable_metric``, the selected method remains the geometry source. A
robust scale is estimated against MoGe2 depth over shared valid pixels and is
applied to both depth and XYZ points::

    scale = median(moge2_depth / source_depth)
    depth_metric = scale * source_depth
    camera_pts_metric = scale * source_camera_pts

Only scale is fitted because both point maps use the same camera optical center;
rotation and translation remain identity.

Usage examples
--------------

    python infer_scripts/1_estimation_depth.py --data_dir demo_data --all --method moge
    python infer_scripts/1_estimation_depth.py --data_dir demo_data --all --method moge --enable_metric
    python infer_scripts/1_estimation_depth.py --data_dir demo_data --case 003_home_office --method ppd --enable_metric
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

import sys


INFER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INFER_ROOT.parent
if str(INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(INFER_ROOT))

from utils.depth_estimation import (  # noqa: E402
    DepthEstimationResult,
    create_depth_estimator,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "Mira_Scene_Demo" / "data"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as file:
            np.save(file, array)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def output_complete(output_dir: Path, require_metric: bool) -> bool:
    required = (
        "depth.npy",
        "camera_pts_map.npy",
        "valid_mask.npy",
        "intrinsics.npy",
        "fov_x_rad.txt",
    )
    return all((output_dir / name).is_file() for name in required) and (
        not require_metric or (output_dir / "metric_alignment.json").is_file()
    )


def load_result(output_dir: Path) -> DepthEstimationResult:
    depth = np.asarray(np.load(output_dir / "depth.npy"), dtype=np.float32)
    points = np.asarray(
        np.load(output_dir / "camera_pts_map.npy"), dtype=np.float32
    )
    valid = np.asarray(np.load(output_dir / "valid_mask.npy"), dtype=bool)
    intrinsics = np.asarray(
        np.load(output_dir / "intrinsics.npy"), dtype=np.float32
    )
    fov = float((output_dir / "fov_x_rad.txt").read_text().strip())
    if points.shape != depth.shape + (3,) or valid.shape != depth.shape:
        raise ValueError(
            f"invalid cached depth result shapes in {output_dir}: "
            f"depth={depth.shape}, points={points.shape}, valid={valid.shape}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"invalid intrinsics shape in {output_dir}: {intrinsics.shape}")
    return DepthEstimationResult(points, valid, depth, intrinsics, fov)


def save_result(
    output_dir: Path,
    result: DepthEstimationResult,
    image_rgb_u8: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(output_dir / "depth.npy", result.depth.astype(np.float32))
    _atomic_save_npy(
        output_dir / "camera_pts_map.npy",
        result.camera_pts_map.astype(np.float32),
    )
    _atomic_save_npy(
        output_dir / "valid_mask.npy", result.valid_mask.astype(bool)
    )
    _atomic_save_npy(
        output_dir / "intrinsics.npy", result.intrinsics.astype(np.float32)
    )
    _atomic_write_text(output_dir / "fov_x_rad.txt", f"{result.fov_x_rad:.10f}\n")

    valid = (
        result.valid_mask
        & np.isfinite(result.camera_pts_map).all(axis=-1)
        & np.isfinite(result.depth)
        & (result.depth > 0)
    )
    points = result.camera_pts_map[valid].astype(np.float32, copy=False)
    colors = image_rgb_u8[valid]
    if len(points):
        rgba = np.concatenate(
            [colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1
        )
        # trimesh does not offer an atomic path export, so write to a temporary
        # sibling and publish it only after export succeeds.
        ply_path = output_dir / "camera_pts.ply"
        temporary_path = output_dir / f".{ply_path.name}.tmp"
        try:
            trimesh.PointCloud(points, colors=rgba).export(
                temporary_path, file_type="ply"
            )
            os.replace(temporary_path, ply_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def align_to_moge2_metric(
    source: DepthEstimationResult,
    target: DepthEstimationResult,
    case_name: str,
    source_method: str,
) -> tuple[DepthEstimationResult, dict[str, Any]]:
    if source.depth.shape != target.depth.shape:
        raise ValueError(
            f"source/MoGe2 depth shape mismatch: {source.depth.shape} vs {target.depth.shape}"
        )
    shared_valid = (
        source.valid_mask
        & target.valid_mask
        & np.isfinite(source.depth)
        & np.isfinite(target.depth)
        & (source.depth > 0)
        & (target.depth > 0)
    )
    source_depth = source.depth[shared_valid].astype(np.float64)
    target_depth = target.depth[shared_valid].astype(np.float64)
    if source_depth.size < 32:
        raise ValueError(
            f"only {source_depth.size} shared valid pixels; at least 32 are required"
        )

    ratios = target_depth / source_depth
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 32:
        raise ValueError("insufficient finite positive MoGe2/source depth ratios")
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid metric alignment scale: {scale}")

    aligned_depth = source.depth.astype(np.float32, copy=True)
    aligned_points = source.camera_pts_map.astype(np.float32, copy=True)
    finite_depth = np.isfinite(aligned_depth)
    finite_points = np.isfinite(aligned_points).all(axis=-1)
    aligned_depth[finite_depth] *= scale
    aligned_points[finite_points] *= scale

    post_source = aligned_depth[shared_valid].astype(np.float64)
    relative_error = np.abs(post_source - target_depth) / np.maximum(
        target_depth, 1e-8
    )
    report: dict[str, Any] = {
        "schema": "mira_depth_to_moge2_metric_scale_v1",
        "case": case_name,
        "source_method": source_method,
        "target_method": "moge2",
        "transform": "camera_points_metric = scale * camera_points_source",
        "depth_transform": "depth_metric = scale * depth_source",
        "scale": scale,
        "rotation": "identity",
        "translation": [0.0, 0.0, 0.0],
        "fit_method": "median(moge2_depth / source_depth) over shared valid positive pixels",
        "shared_valid_pixel_count": int(shared_valid.sum()),
        "ratio_median": float(np.median(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_mad": float(np.median(np.abs(ratios - scale))),
        "post_alignment_relative_error_median": float(np.median(relative_error)),
        "post_alignment_relative_error_p95": float(
            np.quantile(relative_error, 0.95)
        ),
        "coordinate_convention": "OpenGL +X right, +Y up, -Z forward",
    }
    return (
        DepthEstimationResult(
            camera_pts_map=aligned_points,
            valid_mask=source.valid_mask.astype(bool, copy=True),
            depth=aligned_depth,
            intrinsics=source.intrinsics.astype(np.float32, copy=True),
            fov_x_rad=float(source.fov_x_rad),
        ),
        report,
    )


def find_scene_image(case_dir: Path) -> Path:
    for relative in ("input/scene.png", "scene.png", "original_image.png"):
        path = case_dir / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"no scene image found in {case_dir}")


def discover_cases(data_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        cases = [data_root / name for name in requested]
    else:
        cases = sorted(
            path
            for path in data_root.iterdir()
            if path.is_dir() and (path / "input" / "scene.png").is_file()
        )
    missing = [str(path) for path in cases if not path.is_dir()]
    if missing:
        raise FileNotFoundError("missing case directories: " + ", ".join(missing))
    return cases


def moge2_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "pretrained": args.moge2_checkpoint,
        "repo_root": args.moge2_repo_root,
        "resolution_level": args.moge2_resolution_level,
        "num_tokens": args.moge2_num_tokens,
        "use_fp16": not args.moge2_no_fp16,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--data_dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case", action="append", help="case name; repeat as needed")
    parser.add_argument(
        "--all", action="store_true", help="process every discoverable case (default if --case is omitted)"
    )
    parser.add_argument("--method", choices=("moge", "moge2", "ppd"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cases", "--max_cases", type=int, default=-1)
    parser.add_argument("--scene-filter", "--scene_filter")
    parser.add_argument(
        "--enable-metric",
        "--enable_metric",
        action="store_true",
        help="scale the selected estimator's depth/XYZ points to MoGe2 metric scale",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--moge2-checkpoint",
        "--moge2_checkpoint",
        default=os.environ.get("MIRA_MOGE2_CHECKPOINT", "/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl"),
    )
    parser.add_argument(
        "--moge2-repo-root",
        "--moge2_repo_root",
        default=os.environ.get("MIRA_MOGE2_ROOT", "/mnt/pfs/users/sunyangtian/projectpp/MoGe"),
    )
    parser.add_argument("--moge2-resolution-level", "--moge2_resolution_level", type=int, default=9)
    parser.add_argument("--moge2-num-tokens", "--moge2_num_tokens", type=int)
    parser.add_argument("--moge2-no-fp16", "--moge2_no_fp16", action="store_true")
    parser.add_argument("--ppd-checkpoint", default=os.environ.get("MIRA_PPD_CHECKPOINT"))
    parser.add_argument("--ppd-moge-checkpoint", default=os.environ.get("MIRA_PPD_MOGE_CHECKPOINT"))
    parser.add_argument("--ppd-da2-checkpoint", default=os.environ.get("MIRA_PPD_DA2_CHECKPOINT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    cases = discover_cases(data_root, args.case)
    if args.scene_filter:
        cases = [path for path in cases if args.scene_filter in path.name]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError(f"no prepared cases selected under {data_root}")

    primary_kwargs: dict[str, Any] = {"device": args.device}
    if args.method == "moge2":
        primary_kwargs = moge2_kwargs(args)
    elif args.method == "ppd":
        missing = [name for name, value in (("--ppd-checkpoint", args.ppd_checkpoint),
                   ("--ppd-moge-checkpoint", args.ppd_moge_checkpoint),
                   ("--ppd-da2-checkpoint", args.ppd_da2_checkpoint)) if not value]
        if missing:
            raise ValueError("PPD requires " + ", ".join(missing))
        primary_kwargs.update(ppd_checkpoint=args.ppd_checkpoint,
                              moge_checkpoint=args.ppd_moge_checkpoint,
                              da2_checkpoint=args.ppd_da2_checkpoint)
    primary_estimator = create_depth_estimator(args.method, **primary_kwargs)
    metric_estimator = None
    if args.enable_metric and args.method != "moge2":
        metric_estimator = create_depth_estimator("moge2", **moge2_kwargs(args))

    failures = 0
    for case_dir in tqdm(cases, desc=f"Depth ({args.method})"):
        output_dir = case_dir / "depth" / args.method
        if output_complete(output_dir, args.enable_metric) and not args.force:
            continue
        try:
            image_path = find_scene_image(case_dir)
            image_u8 = np.asarray(Image.open(image_path).convert("RGB"))
            image = image_u8.astype(np.float32) / 255.0

            # If a metric run was interrupted after publishing scaled arrays
            # but before publishing metric_alignment.json, those arrays are
            # indistinguishable from an old unscaled cache. Re-estimate the
            # primary result in that case rather than risk applying the scale
            # twice. Completed metric runs were already skipped above.
            incomplete_metric_run = (
                args.enable_metric
                and output_complete(output_dir, False)
                and not (output_dir / "metric_alignment.json").is_file()
            )
            if (
                output_complete(output_dir, False)
                and not args.force
                and not incomplete_metric_run
            ):
                result = load_result(output_dir)
            else:
                result = primary_estimator.estimate(image)

            metric_report = None
            if args.enable_metric:
                if args.method == "moge2":
                    metric_result = result
                else:
                    metric_dir = case_dir / "depth" / "moge2"
                    if output_complete(metric_dir, False) and not args.force:
                        metric_result = load_result(metric_dir)
                    else:
                        assert metric_estimator is not None
                        metric_result = metric_estimator.estimate(image)
                        save_result(metric_dir, metric_result, image_u8)
                result, metric_report = align_to_moge2_metric(
                    result, metric_result, case_dir.name, args.method
                )

            save_result(output_dir, result, image_u8)
            if metric_report is not None:
                _atomic_write_text(
                    output_dir / "metric_alignment.json",
                    json.dumps(metric_report, indent=2, ensure_ascii=False) + "\n",
                )
                print(
                    f"{case_dir.name}: metric scale={metric_report['scale']:.8g}, "
                    f"shared_valid={metric_report['shared_valid_pixel_count']}"
                )
        except Exception as error:
            failures += 1
            print(f"{case_dir.name}: ERROR {type(error).__name__}: {error}")

    print(f"Done: {len(cases) - failures}/{len(cases)} cases succeeded")
    if failures:
        raise SystemExit(f"depth estimation failed for {failures} case(s)")


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main, "01_depth.log", primary_root_flags=("--data-root", "--data_dir"))
