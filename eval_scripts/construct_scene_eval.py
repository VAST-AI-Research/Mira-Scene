#!/usr/bin/env python3
"""Construct SAM3D evaluation scenes with GT depth and no scene graph.

This entry point is intentionally narrower than ``5_construct_scene.py``.  It
implements the historical BlendSwap evaluation contract: predicted CCMs and
SAM3D meshes are aligned directly to the GT camera point map with the shared-up
(``joint``) similarity solver.  Floor estimation, support relationships, and a
semantic scene graph are neither loaded nor synthesized.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
INFER_ROOT = REPO_ROOT / "infer_scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_scripts.core.io import atomic_write_json, sha256_file, stable_hash
from infer_scripts.utils.depth_estimation import DepthEstimationResult
from infer_scripts.utils.solve_transform import solve_similarity_transforms_joint


OUTPUT_SCHEMA = "mira_evaluation_scene_v1"
DEPTH_METHOD = "gt"
MESH_BACKEND = "sam3d"
DEPTH_FILES = (
    "depth.npy",
    "camera_pts_map.npy",
    "valid_mask.npy",
    "intrinsics.npy",
    "fov_x_rad.txt",
)
CANONICAL_MESH_ALIGNMENT = trimesh.transformations.rotation_matrix(
    angle=np.pi / 2,
    direction=[1, 0, 0],
    point=[0, 0, 0],
)


def load_gt_depth(data_dir: Path, case_name: str) -> tuple[DepthEstimationResult, Path]:
    directory = data_dir / case_name / "depth" / DEPTH_METHOD
    missing = [name for name in DEPTH_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{case_name}: incomplete depth/{DEPTH_METHOD}: {', '.join(missing)}"
        )
    with (directory / "fov_x_rad.txt").open(encoding="utf-8") as handle:
        fov = float(handle.read().strip())
    return (
        DepthEstimationResult(
            camera_pts_map=np.load(directory / "camera_pts_map.npy"),
            valid_mask=np.load(directory / "valid_mask.npy"),
            depth=np.load(directory / "depth.npy"),
            intrinsics=np.load(directory / "intrinsics.npy"),
            fov_x_rad=fov,
        ),
        directory,
    )


def load_ccm_outputs(directory: Path) -> dict[str, Any]:
    rgb_path = directory / "rgb_mask.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"missing CCM image: {rgb_path}")
    rgb_mask = np.array(Image.open(rgb_path))
    scene_image = rgb_mask[:, : rgb_mask.shape[1] // 2, :3].astype(np.float32) / 255.0

    merged_restored = directory / "canonical_coord_map_restored.npy"
    merged_cropped = directory / "canonical_coord_map.npy"
    restored = sorted(directory.glob("canonical_coord_map_restored_*.npy"))
    cropped = sorted(directory.glob("canonical_coord_map_[0-9]*.npy"))
    if merged_restored.is_file():
        values = np.load(merged_restored)
        ccms = [values[index] for index in range(values.shape[0])]
    elif merged_cropped.is_file():
        values = np.load(merged_cropped)
        ccms = [values[index] for index in range(values.shape[0])] if values.ndim == 4 else [values]
    elif restored:
        ccms = [np.load(path) for path in restored]
    elif cropped:
        ccms = [np.load(path) for path in cropped]
    else:
        raise FileNotFoundError(f"missing canonical coordinate maps under {directory}")
    if not ccms:
        raise ValueError(f"no CCM instances under {directory}")
    return {"scene_image": scene_image, "ccm_list": ccms, "num_instances": len(ccms)}


def _mask_count(data_dir: Path, case_name: str) -> int:
    paths = sorted((data_dir / case_name / "input").glob("mask_[0-9][0-9][0-9].png"))
    expected = [f"mask_{index:03d}.png" for index in range(len(paths))]
    if not paths or [path.name for path in paths] != expected:
        raise ValueError(
            f"{case_name}: evaluation masks must be contiguous from mask_000.png"
        )
    return len(paths)


def load_instance_meshes(directory: Path, count: int) -> tuple[list[trimesh.Trimesh], list[Path]]:
    paths = [directory / f"{index:03d}.glb" for index in range(count)]
    missing = [path.name for path in paths if not path.is_file()]
    indexed = sorted(directory.glob("[0-9][0-9][0-9].glb")) if directory.is_dir() else []
    if missing or len(indexed) != count:
        raise FileNotFoundError(
            f"expected exactly {count} SAM3D meshes under {directory}; "
            f"missing={missing}, found={[path.name for path in indexed]}"
        )
    meshes: list[trimesh.Trimesh] = []
    for path in paths:
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            raise ValueError(f"invalid or empty SAM3D mesh: {path}")
        mesh = deepcopy(mesh)
        mesh.apply_transform(CANONICAL_MESH_ALIGNMENT)
        meshes.append(mesh)
    return meshes, paths


def transform_to_matrix(transform: dict[str, Any]) -> np.ndarray:
    value = transform.get("transform_matrix")
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        matrix = np.asarray(value, dtype=np.float64)
    else:
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
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("similarity solver returned an invalid transform")
    return matrix


def make_scene(meshes: Sequence[trimesh.Trimesh], transforms: Sequence[np.ndarray]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for index, (mesh, transform) in enumerate(zip(meshes, transforms)):
        scene.add_geometry(
            deepcopy(mesh),
            node_name=f"object_{index:03d}",
            geom_name=f"object_{index:03d}_geometry",
            transform=transform,
        )
    return scene


def atomic_export(scene: trimesh.Scene, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    try:
        scene.export(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def input_fingerprint(
    ccm_dir: Path,
    mesh_paths: Sequence[Path],
    depth_dir: Path,
) -> str:
    ccm_paths = [ccm_dir / "rgb_mask.png"]
    for pattern in (
        "canonical_coord_map.npy",
        "canonical_coord_map_restored.npy",
        "canonical_coord_map_[0-9]*.npy",
        "canonical_coord_map_restored_[0-9]*.npy",
    ):
        ccm_paths.extend(Path(path) for path in glob.glob(str(ccm_dir / pattern)))
    inputs = sorted({path.resolve() for path in [*ccm_paths, *mesh_paths, *(depth_dir / name for name in DEPTH_FILES)]})
    return stable_hash([[str(path), sha256_file(path)] for path in inputs if path.is_file()])


def implementation_fingerprint() -> str:
    paths = [Path(__file__), INFER_ROOT / "utils" / "solve_transform.py"]
    return stable_hash(
        [[str(path.relative_to(REPO_ROOT)), sha256_file(path)] for path in paths]
    )


def output_complete(save_dir: Path, fingerprint: str) -> bool:
    scene_path = save_dir / "scene.glb"
    metadata_path = save_dir / "scene_optimization.json"
    if not scene_path.is_file() or scene_path.stat().st_size == 0 or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema") == OUTPUT_SCHEMA
        and metadata.get("mode") == "graph_free_evaluation"
        and metadata.get("mesh_backend") == MESH_BACKEND
        and metadata.get("depth_method") == DEPTH_METHOD
        and metadata.get("solve_method") == "joint"
        and metadata.get("input_fingerprint") == fingerprint
        and metadata.get("implementation_fingerprint") == implementation_fingerprint()
    )


def construct_scene(
    scene_data: dict[str, Any],
    depth: DepthEstimationResult,
    meshes: Sequence[trimesh.Trimesh],
    save_dir: Path,
    case_name: str,
    input_fingerprint_value: str,
    seed: int,
) -> dict[str, Any]:
    count = int(scene_data["num_instances"])
    if count == 0 or len(meshes) != count:
        raise ValueError(f"CCM contains {count} objects but {len(meshes)} meshes were loaded")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = scene_data["scene_image"].shape[:2]

    tensors = []
    for ccm in scene_data["ccm_list"]:
        tensor = torch.from_numpy(ccm).unsqueeze(0).float().to(device)
        if tensor.shape[-2:] != (height, width):
            tensor = F.interpolate(
                tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        tensors.append(tensor)
    ccm_tensor = torch.cat(tensors, dim=0)
    masks = (ccm_tensor.abs().sum(dim=1, keepdim=True) > 1e-6).float()
    camera_points = torch.from_numpy(depth.camera_pts_map).to(device)
    valid = torch.from_numpy(depth.valid_mask).to(device)
    camera_points = camera_points.unsqueeze(0).expand(count, -1, -1, -1)
    valid = valid.unsqueeze(0).expand(count, -1, -1)

    print(f"  Solving {count} object transforms with Legacy Joint...")
    solved = solve_similarity_transforms_joint(ccm_tensor, camera_points, valid, masks)
    transforms = [transform_to_matrix(item) for item in solved]
    atomic_export(make_scene(meshes, transforms), save_dir / "scene.glb")

    metadata = {
        "schema": OUTPUT_SCHEMA,
        "mode": "graph_free_evaluation",
        "case": case_name,
        "mesh_backend": MESH_BACKEND,
        "depth_method": DEPTH_METHOD,
        "solve_method": "joint",
        "coordinate_frame": "camera",
        "object_count": count,
        "node_names": [f"object_{index:03d}" for index in range(count)],
        "input_fingerprint": input_fingerprint_value,
        "implementation_fingerprint": implementation_fingerprint(),
        "seed": seed,
        "transforms": [matrix.tolist() for matrix in transforms],
    }
    atomic_write_json(save_dir / "scene_optimization.json", metadata)
    for stale in ("scene_initial.glb", "scene_with_floor.glb"):
        (save_dir / stale).unlink(missing_ok=True)
    return metadata


def process_case(data_dir: Path, output_dir: Path, case_name: str, force: bool, seed: int) -> bool:
    ccm_dir = output_dir / case_name / "CCM"
    mesh_dir = output_dir / case_name / "mesh" / MESH_BACKEND
    scene_data = load_ccm_outputs(ccm_dir)
    mask_count = _mask_count(data_dir, case_name)
    if scene_data["num_instances"] != mask_count:
        raise ValueError(
            f"{case_name}: CCM object count {scene_data['num_instances']} "
            f"does not match mask count {mask_count}"
        )
    meshes, mesh_paths = load_instance_meshes(mesh_dir, mask_count)
    depth, depth_dir = load_gt_depth(data_dir, case_name)
    fingerprint = input_fingerprint(ccm_dir, mesh_paths, depth_dir)
    save_dir = output_dir / case_name / "scene" / MESH_BACKEND / f"{DEPTH_METHOD}_depth"
    if not force and output_complete(save_dir, fingerprint):
        print("  Skipped (matching graph-free GT scene exists)")
        return False
    construct_scene(
        scene_data,
        depth,
        meshes,
        save_dir,
        case_name,
        fingerprint,
        seed + sum(case_name.encode("utf-8")),
    )
    print(f"  scene.glb -> {save_dir / 'scene.glb'}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Exact case ID; repeatable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        print("Graph-free evaluation scene environment is ready.")
        return
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir() or not output_dir.is_dir():
        raise FileNotFoundError("--data_dir and --output_dir must both exist")
    if args.case:
        names = list(dict.fromkeys(args.case))
    else:
        names = sorted(
            path.name
            for path in data_dir.iterdir()
            if path.is_dir() and (output_dir / path.name / "CCM").is_dir()
        )
    if not names:
        raise ValueError("no evaluation cases selected")

    failures: list[tuple[str, str]] = []
    rebuilt = skipped = 0
    for index, name in enumerate(tqdm(names, desc="GT scene construction")):
        print(f"\n[{index + 1}/{len(names)}] {name}")
        try:
            if process_case(data_dir, output_dir, name, args.force, args.seed):
                rebuilt += 1
            else:
                skipped += 1
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures.append((name, message))
            print(f"  FAILED ({message})")
    print(f"\nDone. {rebuilt} rebuilt, {skipped} reused, {len(failures)} failed.")
    if failures:
        for name, message in failures:
            print(f"  {name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    from infer_scripts.core.stage_logging import run_logged

    run_logged(
        main,
        "05_sam3d_eval_scene.log",
        primary_root_flags=("--output_dir",),
    )
