#!/usr/bin/env python3
"""Generate per-instance meshes for prepared Mira-Scene demo cases.

Input layout::

    demo_data/<case>/
    └── CCM/
        ├── rgb_mask.png
        ├── masks.npy
        └── voxel_coords_*.npy

The CCM files are produced by ``infer_scripts/2_inference_CCM.py``.  This script
runs SAM-3D stage 2 once per instance and writes the resulting meshes to::

    demo_data/<case>/mesh/sam3d/<instance>.glb

With ``--save_overlay``, an additional ``<instance>_overlay.glb`` containing
the predicted canonical-coordinate points is written.  A custom
``--output_dir`` writes ``<output_dir>/<case>/mesh/sam3d`` while still reading CCM
inputs from ``--demo_dir``.

Single-GPU example::

    DATA_DIR=demo_data
    SAM3D_CONFIG=/path/to/sam3d/pipeline.yaml
    CUDA_VISIBLE_DEVICES=0 python infer_scripts/3_inference_mesh.py \
        --demo_dir "$DATA_DIR" \
        --ckpt_dir "$SAM3D_CONFIG" \
        --save_overlay

Multi-GPU example (one independent scene shard per GPU)::

    DATA_DIR=demo_data
    SAM3D_CONFIG=/path/to/sam3d/pipeline.yaml
    NUM_GPUS=8
    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES="$SHARD_ID" python infer_scripts/3_inference_mesh.py \
            --demo_dir "$DATA_DIR" \
            --ckpt_dir "$SAM3D_CONFIG" \
            --save_overlay \
            --num_shards "$NUM_GPUS" \
            --shard_id "$SHARD_ID" &
    done
    wait

Each process exposes one GPU through ``CUDA_VISIBLE_DEVICES`` and writes to
disjoint case directories; no distributed launcher is needed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

INFER_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFER_DIR.parent
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

def load_ccm_outputs(ccm_dir):
    """Load scene image, instance masks, and voxel coordinates from one CCM dir."""
    rgb_mask_path = os.path.join(ccm_dir, "rgb_mask.png")
    if not os.path.exists(rgb_mask_path):
        return None

    rgb_mask_img = np.array(Image.open(rgb_mask_path))
    width_half = rgb_mask_img.shape[1] // 2
    scene_image = rgb_mask_img[:, :width_half, :3].astype(np.uint8)

    masks_path = os.path.join(ccm_dir, "masks.npy")
    if os.path.exists(masks_path):
        masks = np.load(masks_path)
    else:
        mask_image = rgb_mask_img[:, width_half:, 0]
        masks = (mask_image[None] > 127).astype(np.uint8)

    coord_paths = sorted(glob.glob(os.path.join(ccm_dir, "voxel_coords_*.npy")))
    if not coord_paths:
        return None
    return {
        "scene_image": scene_image,
        "masks": masks,
        "voxel_coords_list": [np.load(path) for path in coord_paths],
    }


def save_ccm_overlay(ccm_dir, mesh_dir, glb, instance_index):
    """Save a mesh plus its canonical-coordinate points as green spheres."""
    ccm = None
    merged_path = os.path.join(ccm_dir, "canonical_coord_map.npy")
    per_instance_path = os.path.join(ccm_dir, f"canonical_coord_map_{instance_index:03d}.npy")
    if os.path.exists(merged_path):
        merged = np.load(merged_path)
        if merged.ndim == 4 and instance_index < merged.shape[0]:
            ccm = merged[instance_index]
        elif merged.ndim == 3:
            ccm = merged
    elif os.path.exists(per_instance_path):
        ccm = np.load(per_instance_path)
    if ccm is None:
        return

    ccm_points = ccm.transpose(1, 2, 0).reshape(-1, 3)
    ccm_points = ccm_points[np.abs(ccm_points).sum(axis=-1) > 1e-6]
    if len(ccm_points) == 0:
        return

    # CCM is z-up while SAM-3D meshes are y-up.
    rotation = trimesh.transformations.rotation_matrix(
        angle=-np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
    )
    ccm_points = (rotation[:3, :3] @ ccm_points.T).T
    sphere_template = trimesh.creation.uv_sphere(radius=0.005, count=[8, 8])
    sphere_template.visual.vertex_colors = [0, 255, 0, 255]
    ccm_mesh = trimesh.util.concatenate([
        sphere_template.copy().apply_translation(point) for point in ccm_points
    ])
    overlay = trimesh.Scene()
    overlay.add_geometry(deepcopy(glb), node_name="mesh")
    overlay.add_geometry(ccm_mesh, node_name="ccm_pcd")
    overlay.export(os.path.join(mesh_dir, f"{instance_index:03d}_overlay.glb"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate meshes for demo_data via SAM-3D stage 2")
    parser.add_argument("--demo_dir", type=str, default=str(REPO_ROOT / "Mira_Scene_Demo" / "data"))
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="SAM-3D pipeline.yaml/checkpoint configuration")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to --demo_dir, writing <case>/mesh.")
    parser.add_argument("--with_texture_baking", action="store_true")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--case", action="append",
                        help="Exact case name; repeat as needed")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards")
    return args


def main():
    args = parse_args()
    # Heavy SAM3D/PyTorch3D imports are intentionally delayed so CLI help and
    # the orchestration preflight work outside the dedicated SAM3D environment.
    from utils.sam3d_utils import load_sam3d_pipeline, merge_mask_to_rgba
    demo_dir = os.path.abspath(os.path.expanduser(args.demo_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or demo_dir))
    ckpt_dir = os.path.abspath(os.path.expanduser(args.ckpt_dir))
    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(f"demo directory does not exist: {demo_dir}")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"SAM-3D checkpoint/config does not exist: {ckpt_dir}")

    if args.case:
        scene_names = list(dict.fromkeys(args.case))
        missing = [
            name for name in scene_names
            if not os.path.isdir(os.path.join(demo_dir, name, "CCM"))
        ]
        if missing:
            raise FileNotFoundError(
                "missing CCM case directories: " + ", ".join(missing)
            )
    else:
        scene_names = sorted(
            name for name in os.listdir(demo_dir)
            if os.path.isdir(os.path.join(demo_dir, name, "CCM"))
        )
    if args.max_cases > 0:
        scene_names = scene_names[:args.max_cases]
    total_scenes = len(scene_names)
    scene_names = scene_names[args.shard_id::args.num_shards]
    print(f"Found {len(scene_names)}/{total_scenes} cases in {demo_dir} "
          f"(shard {args.shard_id + 1}/{args.num_shards})")
    if not scene_names:
        print("No cases with CCM outputs assigned to this shard; nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM-3D pipeline from {ckpt_dir}")
    sam3d_pipeline = load_sam3d_pipeline(ckpt_dir)
    print("SAM-3D pipeline loaded.")
    failures = []

    for scene_index, scene_name in enumerate(tqdm(scene_names, desc="Mesh generation")):
        ccm_dir = os.path.join(demo_dir, scene_name, "CCM")
        mesh_dir = os.path.join(output_dir, scene_name, "mesh", "sam3d")
        data = load_ccm_outputs(ccm_dir)
        if data is None:
            print(f"  [{scene_index:03d}] {scene_name}: skipped (missing CCM files)")
            continue
        os.makedirs(mesh_dir, exist_ok=True)

        for instance_index, voxel_coords in enumerate(data["voxel_coords_list"]):
            try:
                masks = data["masks"]
                mask = masks[instance_index] > 0 if instance_index < len(masks) else masks[0] > 0
                rgba = merge_mask_to_rgba(data["scene_image"], mask)
                coords_with_batch = np.concatenate(
                    [np.zeros((len(voxel_coords), 1), dtype=np.int32), voxel_coords], axis=1
                )
                coords_tensor = torch.from_numpy(coords_with_batch).to(device)
                result = sam3d_pipeline.run_stage2(
                    image=rgba, coords=coords_tensor, mask=None,
                    seed=args.eval_seed, with_texture_baking=args.with_texture_baking,
                )
                glb = result.get("glb")
            except Exception as exc:
                failures.append((scene_name, instance_index, repr(exc)))
                print(f"  [{scene_index:03d}] {scene_name} instance {instance_index}: FAILED ({exc})")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if glb is None:
                print(f"  [{scene_index:03d}] {scene_name} instance {instance_index}: no mesh produced")
                continue
            glb_path = os.path.join(mesh_dir, f"{instance_index:03d}.glb")
            try:
                glb.export(glb_path)
            except Exception as exc:
                failures.append((scene_name, instance_index, f"export failed: {exc!r}"))
                print(f"  [{scene_index:03d}] {scene_name} instance {instance_index}: export FAILED ({exc})")
                continue
            print(f"  [{scene_index:03d}] {scene_name} instance {instance_index} -> {glb_path}")
            if args.save_overlay:
                try:
                    save_ccm_overlay(ccm_dir, mesh_dir, glb, instance_index)
                    print(f"  [{scene_index:03d}] {scene_name} instance {instance_index} overlay -> "
                          f"{os.path.join(mesh_dir, f'{instance_index:03d}_overlay.glb')}")
                except Exception as exc:
                    print(f"  [{scene_index:03d}] {scene_name} instance {instance_index}: "
                          f"overlay failed ({exc}), mesh kept")

    print(f"\nDone. {len(scene_names)} cases processed.")
    if failures:
        print(f"{len(failures)} instance(s) failed and were skipped:")
        for scene_name, instance_index, error in failures:
            print(f"  {scene_name} instance {instance_index}: {error}")


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main, "03_sam3d_mesh.log", primary_root_flags=("--output_dir",),
               fallback_root_flags=("--demo_dir",))
