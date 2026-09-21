#!/usr/bin/env python3
"""Construct a gravity-aligned scene from CCM, depth, meshes, and supports.

Input layout (for each ``<case>``)::

    <output_dir>/<case>/CCM/
        rgb_mask.png
        canonical_coord_map*.npy
        canonical_pcd_*.ply                 # optional, for scene.ply/proj.png
    <output_dir>/<case>/mesh/<backend>/<NNN>.glb  # canonical object meshes
    <output_dir>/<case>/floor/
        floor_alignment.json                # written by 4_estimate_floor.py
        floor_texture.png                   # standalone reusable texture
    <data_dir>/<case>/depth/<method>/
        depth.npy, camera_pts_map.npy, valid_mask.npy
        intrinsics.npy, fov_x_rad.txt
    <data_dir>/<case>/scene_graph.json      # written by 0_estimate_scene_graph.py

Output layout::

    <output_dir>/<case>/scene/<backend>/<method>_depth/
        scene_initial.glb                    # initial independent camera-space scene
        scene.glb                             # final foreground-only camera-space scene
        scene_with_floor.glb                 # final foreground + floor in floor space
        scene_optimization.json              # transforms, support decisions, diagnostics
        scene.ply, proj.png                   # debug/visualization outputs
        camera_pts.ply                        # camera points + initial mesh overlay

Example command::

    python infer_scripts/5_construct_scene.py \
        --output_dir Mira_Scene_Demo/data \
        --data_dir Mira_Scene_Demo/data \
        --depth_method auto \
        --solve_method gravity_joint \
        --placement_backend geometry \
        --support_confidence 0.90 \
        --floor_margin_ratio 0.10 \
        --floor_min_margin 0.25 \
        --max_cases -1

Step 5 consumes the independent floor artifacts from ``4_estimate_floor.py``
and the semantic ``scene_graph.json``. It keeps the evaluator-compatible
foreground scene in camera coordinates while also writing a full floor-frame
scene with a floor resized from the final reconstructed mesh projection.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from utils.depth_estimation import DepthEstimationResult
from utils.scene_placement import create_placement_backend, prepare_support_graph, transformed_vertices
from utils.solve_transform import solve_similarity_transforms, solve_similarity_transforms_gravity
from utils.visualization import save_projection_image, save_scene_pcd


OUTPUT_SCHEMA = "mira_scene_optimization_v1"
SUPPORTED_FLOOR_SCHEMAS = {"mira_scene_floor_estimate_v9"}
SUPPORTED_GRAPH_SCHEMAS = {"mira_scene_graph_v1"}
CANONICAL_MESH_ALIGNMENT = trimesh.transformations.rotation_matrix(
    angle=np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
)


def load_precomputed_depth(data_dir, scene_name, method=None):
    """Load one complete depth result from ``<data>/<case>/depth``."""
    root = os.path.join(data_dir, scene_name, "depth")
    priorities = [method] if method and method != "auto" else ["gt", "ppd", "moge2", "moge"]
    for selected in priorities:
        directory = os.path.join(root, selected)
        paths = [os.path.join(directory, name) for name in (
            "depth.npy", "camera_pts_map.npy", "valid_mask.npy", "intrinsics.npy", "fov_x_rad.txt"
        )]
        if not all(os.path.exists(path) for path in paths):
            continue
        with open(paths[4], encoding="utf-8") as handle:
            fov = float(handle.read().strip())
        return DepthEstimationResult(
            camera_pts_map=np.load(paths[1]), valid_mask=np.load(paths[2]),
            depth=np.load(paths[0]), intrinsics=np.load(paths[3]), fov_x_rad=fov,
        ), selected
    return None, None


def load_ccm_outputs(ccm_dir):
    """Load per-instance CCM outputs and optional canonical point clouds."""
    rgb_path = os.path.join(ccm_dir, "rgb_mask.png")
    if not os.path.exists(rgb_path):
        return None
    rgb_mask = np.array(Image.open(rgb_path))
    scene_image = rgb_mask[:, :rgb_mask.shape[1] // 2, :3].astype(np.float32) / 255.0
    merged_restored = os.path.join(ccm_dir, "canonical_coord_map_restored.npy")
    merged_cropped = os.path.join(ccm_dir, "canonical_coord_map.npy")
    restored = sorted(glob.glob(os.path.join(ccm_dir, "canonical_coord_map_restored_*.npy")))
    cropped = sorted(glob.glob(os.path.join(ccm_dir, "canonical_coord_map_[0-9]*.npy")))
    if os.path.exists(merged_restored):
        values = np.load(merged_restored)
        ccms = [values[index] for index in range(values.shape[0])]
    elif os.path.exists(merged_cropped):
        values = np.load(merged_cropped)
        ccms = [values[index] for index in range(values.shape[0])] if values.ndim == 4 else [values]
    elif restored:
        ccms = [np.load(path) for path in restored]
    elif cropped:
        ccms = [np.load(path) for path in cropped]
    else:
        return None
    point_clouds = []
    for index in range(len(ccms)):
        path = os.path.join(ccm_dir, f"canonical_pcd_{index:03d}.ply")
        if os.path.exists(path):
            geometry = trimesh.load(path)
            point_clouds.append(np.asarray(geometry.vertices) if hasattr(geometry, "vertices") else None)
        else:
            point_clouds.append(None)
    return {"scene_image": scene_image, "ccm_list": ccms,
            "pcd_list": point_clouds, "num_instances": len(ccms)}


def load_instance_glbs(mesh_dir, num_instances):
    """Load and convert SAM3D Y-up meshes to canonical CCM Z-up."""
    meshes = []
    for index in range(num_instances):
        path = os.path.join(mesh_dir, f"{index:03d}.glb")
        if not os.path.exists(path):
            meshes.append(None)
            continue
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            meshes.append(None)
            continue
        mesh = deepcopy(mesh)
        mesh.apply_transform(CANONICAL_MESH_ALIGNMENT)
        meshes.append(mesh)
    return meshes


def load_floor_inputs(output_root: Path, case_name: str):
    directory = output_root / case_name / "floor"
    metadata_path, texture_path = directory / "floor_alignment.json", directory / "floor_texture.png"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing Step 4 metadata: {metadata_path}")
    if not texture_path.is_file():
        raise FileNotFoundError(f"missing standalone floor texture: {texture_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema") not in SUPPORTED_FLOOR_SCHEMAS:
        raise ValueError(f"unsupported floor schema {metadata.get('schema')!r}; rerun Step 4")
    camera_to_floor = np.asarray(metadata.get("camera_to_floor_transform"), dtype=np.float64)
    floor_to_camera = np.asarray(metadata.get("floor_to_camera_transform"), dtype=np.float64)
    if camera_to_floor.shape != (4, 4) or floor_to_camera.shape != (4, 4):
        raise ValueError("floor transforms must both be 4x4")
    if not np.allclose(camera_to_floor @ floor_to_camera, np.eye(4), atol=1e-5):
        raise ValueError("camera/floor transforms are not inverses")
    return metadata, metadata_path, texture_path


def load_scene_graph(data_root: Path, case_name: str, object_count: int):
    path = data_root / case_name / "scene_graph.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing scene graph: {path}")
    with path.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    if graph.get("schema") not in SUPPORTED_GRAPH_SCHEMAS:
        raise ValueError(f"unsupported scene graph schema: {graph.get('schema')!r}")
    indices = sorted(int(node["mask_index"]) for node in graph.get("nodes", [])
                     if node.get("kind") == "object" and isinstance(node.get("mask_index"), int))
    if indices != list(range(object_count)):
        raise ValueError(f"scene graph object indices {indices} do not match 0..{object_count - 1}")
    return graph, path


def transform_to_matrix(transform: Dict) -> np.ndarray:
    value = transform.get("transform_matrix")
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)
    scale = float(transform["s"])
    rotation = transform["R"].detach().cpu().numpy() if isinstance(transform["R"], torch.Tensor) else transform["R"]
    translation = transform["t"].detach().cpu().numpy() if isinstance(transform["t"], torch.Tensor) else transform["t"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3], matrix[:3, 3] = scale * np.asarray(rotation), np.asarray(translation)
    return matrix


def matrix_to_transform(matrix: np.ndarray) -> Dict:
    scale = float(np.linalg.norm(matrix[:3, 0]))
    return {"s": scale, "R": matrix[:3, :3] / max(scale, 1e-12), "t": matrix[:3, 3]}


def atomic_export(scene: trimesh.Scene, output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    scene.export(temporary)
    os.replace(temporary, output_path)


def atomic_json(data: Dict, output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    os.replace(temporary, output_path)


def make_mesh_scene(meshes: Sequence[trimesh.Trimesh], transforms: Sequence[np.ndarray]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for index, (mesh, transform) in enumerate(zip(meshes, transforms)):
        scene.add_geometry(deepcopy(mesh), node_name=f"object_{index:03d}",
                           geom_name=f"object_{index:03d}_geometry", transform=transform)
    return scene


def _overlay_vertex_colors(mesh: trimesh.Trimesh, index: int) -> np.ndarray:
    """Return PLY-compatible colors, preserving material colors when possible."""
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)
        if colors.shape == (len(mesh.vertices), 4):
            return colors
    except Exception:
        pass
    palette = np.asarray([
        [230, 74, 25, 255], [0, 114, 178, 255], [0, 158, 115, 255],
        [204, 121, 167, 255], [240, 228, 66, 255], [86, 180, 233, 255],
    ], dtype=np.uint8)
    return np.tile(palette[index % len(palette)], (len(mesh.vertices), 1))


def export_camera_initial_overlay(
    camera_points: np.ndarray,
    camera_colors: np.ndarray,
    meshes: Sequence[trimesh.Trimesh],
    transforms: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    """Write camera points and the initial placed meshes into one mixed PLY.

    A normal ``trimesh.Scene`` PLY export silently drops PointCloud geometry
    when triangle meshes are present. Build one vertex pool explicitly: mesh
    vertices are referenced by faces, while camera samples remain colored loose
    vertices in the same camera coordinate frame.
    """
    vertices, faces, colors = [], [], []
    vertex_offset = 0
    for index, (mesh, transform) in enumerate(zip(meshes, transforms)):
        placed = deepcopy(mesh)
        placed.apply_transform(transform)
        mesh_vertices = np.asarray(placed.vertices, dtype=np.float64)
        vertices.append(mesh_vertices)
        colors.append(_overlay_vertex_colors(placed, index))
        if len(placed.faces):
            faces.append(np.asarray(placed.faces, dtype=np.int64) + vertex_offset)
        vertex_offset += len(mesh_vertices)

    camera_points = np.asarray(camera_points, dtype=np.float64)
    camera_colors = np.asarray(camera_colors, dtype=np.uint8)
    if camera_colors.ndim != 2 or camera_colors.shape[0] != len(camera_points):
        raise ValueError("camera point colors must match camera point count")
    if camera_colors.shape[1] == 3:
        camera_colors = np.concatenate(
            [camera_colors, np.full((len(camera_colors), 1), 255, dtype=np.uint8)],
            axis=1,
        )
    if camera_colors.shape[1] != 4:
        raise ValueError("camera point colors must have RGB or RGBA channels")
    vertices.append(camera_points)
    colors.append(camera_colors)

    overlay = trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=(np.concatenate(faces, axis=0)
               if faces else np.empty((0, 3), dtype=np.int64)),
        vertex_colors=np.concatenate(colors, axis=0),
        process=False,
        maintain_order=True,
    )
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        overlay.export(temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def make_floor_mesh(center_xz, side, texture_path, texture_repeat):
    half, (x, z) = side / 2.0, center_xz
    vertices = np.array([[x-half, 0, z-half], [x+half, 0, z-half],
                         [x+half, 0, z+half], [x-half, 0, z+half]], dtype=np.float64)
    uv = np.array([[0, 0], [texture_repeat, 0], [texture_repeat, texture_repeat],
                   [0, texture_repeat]], dtype=np.float64)
    material = trimesh.visual.material.PBRMaterial(
        name="floor_texture", baseColorTexture=Image.open(texture_path).convert("RGB"),
        baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0,
        roughnessFactor=1.0, doubleSided=True,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 2, 1], [0, 3, 2]]),
                           visual=visual, process=False)


def floor_from_final_meshes(meshes, transforms, metadata, texture_path,
                            margin_ratio, minimum_margin):
    projected = np.concatenate([transformed_vertices(mesh, transform)[:, [0, 2]]
                                for mesh, transform in zip(meshes, transforms)], axis=0)
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    extent = maximum - minimum
    base_side = float(max(extent))
    margin = float(max(minimum_margin, margin_ratio * base_side))
    final_side = max(base_side + 2 * margin, 2 * minimum_margin)
    center = (minimum + maximum) / 2
    placeholder = float(metadata["floor_placeholder_size"])
    base_tiling = float(metadata.get("floor_texture_tiling", 1.0))
    repeat = final_side * base_tiling / placeholder
    return make_floor_mesh(center, final_side, texture_path, repeat), {
        "source": "final_foreground_mesh_projection_xz",
        "projected_min_xz": minimum.tolist(), "projected_max_xz": maximum.tolist(),
        "projected_extent_xz": extent.tolist(), "center_xz": center.tolist(),
        "base_side": base_side, "margin_each_side": margin,
        "margin_ratio": margin_ratio, "minimum_margin": minimum_margin,
        "final_side": final_side, "texture_repeat_each_axis": repeat,
        "texture_density_source": {"placeholder_size": placeholder,
                                   "placeholder_texture_tiling": base_tiling},
    }


def construct_scene(depth_result, depth_method, scene_data, meshes, floor_metadata,
                    floor_metadata_path, floor_texture_path, scene_graph,
                    scene_graph_path, save_dir, solve_method="gravity_joint",
                    placement_backend="geometry", support_confidence=0.90,
                    floor_margin_ratio=0.10, floor_min_margin=0.25, seed=42,
                    mesh_backend="sam3d", mesh_dir=None):
    """Solve, support-adjust, and write one complete scene."""
    save_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = scene_data["scene_image"]
    height, width = image.shape[:2]
    count = scene_data["num_instances"]
    if count == 0 or len(meshes) != count or any(mesh is None for mesh in meshes):
        missing = [index for index, mesh in enumerate(meshes) if mesh is None]
        raise ValueError(f"all {count} instance meshes are required; missing/invalid: {missing}")

    tensors = []
    for ccm in scene_data["ccm_list"]:
        tensor = torch.from_numpy(ccm).unsqueeze(0).float().to(device)
        if tensor.shape[-2:] != (height, width):
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        tensors.append(tensor)
    ccm_tensor = torch.cat(tensors, dim=0)
    masks = (ccm_tensor.abs().sum(dim=1, keepdim=True) > 1e-6).float()
    camera_points = torch.from_numpy(depth_result.camera_pts_map).to(device)
    valid = torch.from_numpy(depth_result.valid_mask).to(device)
    camera_points = camera_points.unsqueeze(0).expand(count, -1, -1, -1)
    valid = valid.unsqueeze(0).expand(count, -1, -1)

    print("  Solving initial independent CCM/depth transforms...")
    initial = solve_similarity_transforms(ccm_tensor, camera_points, valid, masks)
    initial_camera = [transform_to_matrix(item) for item in initial]
    camera_to_floor = np.asarray(floor_metadata["camera_to_floor_transform"], dtype=np.float64)
    floor_to_camera = np.asarray(floor_metadata["floor_to_camera_transform"], dtype=np.float64)
    initial_floor = [camera_to_floor @ matrix for matrix in initial_camera]
    graph_decisions, _, _ = prepare_support_graph(scene_graph, count, support_confidence)
    # ``upright_mode`` is an optional per-object scene-graph override.  Missing
    # values are treated as ``auto`` for backwards compatibility: accepted
    # operational rests_on edges determine the constraint as before. Newly
    # generated hanging objects use ``force`` in the segmentation graph.
    derived_upright = [False] * count
    for decision in graph_decisions:
        if decision["status"] == "accepted":
            derived_upright[int(decision["child"].split("_")[-1])] = True
    nodes = {node["id"]: node for node in scene_graph.get("nodes", [])}
    upright = []
    upright_modes = []
    upright_reasons = []
    for index in range(count):
        mode = nodes.get(f"object_{index:03d}", {}).get("upright_mode", "auto")
        if mode not in {"auto", "force", "free"}:
            raise ValueError(f"invalid upright_mode {mode!r} for object_{index:03d}")
        upright_modes.append(mode)
        if mode == "force":
            upright.append(True)
            upright_reasons.append("graph_override_force")
        elif mode == "free":
            upright.append(False)
            upright_reasons.append("graph_override_free")
        else:
            upright.append(derived_upright[index])
            upright_reasons.append(
                "high_confidence_operational_rests_on" if derived_upright[index]
                else "no_accepted_resting_edge"
            )

    if solve_method in {"joint", "gravity_joint"}:
        print(f"  Fitting gravity-aware transforms ({sum(upright)} upright objects)...")
        gravity = solve_similarity_transforms_gravity(
            ccm_tensor, camera_points, valid, masks, camera_to_floor, upright,
            initial_transforms=initial,
        )
        gravity_floor = [transform_to_matrix(item) for item in gravity]
    else:
        gravity_floor = [matrix.copy() for matrix in initial_floor]

    print(f"  Applying support graph with {placement_backend} backend...")
    backend = create_placement_backend(placement_backend)
    final_floor, support_decisions, collision_diagnostics = backend.optimize(
        meshes, gravity_floor, scene_graph, support_confidence
    )
    gravity_camera = [floor_to_camera @ matrix for matrix in gravity_floor]
    final_camera = [floor_to_camera @ matrix for matrix in final_floor]
    atomic_export(make_mesh_scene(meshes, initial_camera), save_dir / "scene_initial.glb")
    atomic_export(make_mesh_scene(meshes, final_camera), save_dir / "scene.glb")

    floor_mesh, floor_bounds = floor_from_final_meshes(
        meshes, final_floor, floor_metadata, floor_texture_path,
        floor_margin_ratio, floor_min_margin,
    )
    full_scene = make_mesh_scene(meshes, final_floor)
    full_scene.add_geometry(floor_mesh, node_name="floor", geom_name="floor_geometry")
    atomic_export(full_scene, save_dir / "scene_with_floor.glb")

    point_clouds = scene_data["pcd_list"]
    indices = [index for index, points in enumerate(point_clouds) if points is not None and len(points)]
    if indices:
        selected_points = [point_clouds[index] for index in indices]
        selected_transforms = [matrix_to_transform(final_camera[index]) for index in indices]
        save_scene_pcd(selected_points, selected_transforms, str(save_dir / "scene.ply"))
        selected_masks = np.stack([masks[index, 0].cpu().numpy() > 0.5 for index in indices])
        save_projection_image(selected_points, selected_transforms, str(save_dir / "proj.png"),
                              fov_rad=depth_result.fov_x_rad, h=height, w=width,
                              mask_np=selected_masks)

    valid_depth = np.asarray(depth_result.valid_mask, dtype=bool)
    valid_depth &= np.isfinite(depth_result.camera_pts_map).all(axis=-1)
    if valid_depth.any():
        rgb = (image[valid_depth] * 255).astype(np.uint8)
        alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
        export_camera_initial_overlay(
            depth_result.camera_pts_map[valid_depth],
            np.concatenate([rgb, alpha], axis=1),
            meshes,
            initial_camera,
            save_dir / "camera_pts.ply",
        )

    support_by_child = {item["child"]: item for item in support_decisions}
    objects = []
    for index in range(count):
        node_id = f"object_{index:03d}"
        objects.append({
            "id": node_id, "name": nodes.get(node_id, {}).get("name"),
            "motion": nodes.get(node_id, {}).get("motion"),
            "upright_mode": upright_modes[index],
            "upright_constrained": upright[index] and solve_method != "independent",
            "upright_reason": "independent_ablation" if solve_method == "independent"
                              else upright_reasons[index],
            "support_decision": support_by_child.get(node_id),
            "transforms": {
                "initial_camera": initial_camera[index].tolist(),
                "initial_floor": initial_floor[index].tolist(),
                "gravity_camera": gravity_camera[index].tolist(),
                "gravity_floor": gravity_floor[index].tolist(),
                "final_camera": final_camera[index].tolist(),
                "final_floor": final_floor[index].tolist(),
            },
        })
    report = {
        "schema": OUTPUT_SCHEMA, "depth_method": depth_method,
        "mesh_backend": mesh_backend, "solve_method": solve_method,
        "placement_backend": backend.name, "seed": seed,
        "coordinate_frames": {
            "scene_initial.glb": "camera, foreground only",
            "scene.glb": "camera, foreground only",
            "scene_with_floor.glb": "floor (+Y up, y=0 floor), foreground plus floor",
            "camera_pts.ply": "camera, RGB depth points plus scene_initial mesh overlay",
        },
        "inputs": {"floor_metadata": str(floor_metadata_path),
                   "floor_schema": floor_metadata.get("schema"),
                   "floor_texture": str(floor_texture_path),
                   "scene_graph": str(scene_graph_path),
                   "scene_graph_schema": scene_graph.get("schema"),
                   "mesh_backend": mesh_backend,
                   "mesh_directory": str(mesh_dir) if mesh_dir is not None else None},
        "camera_to_floor_transform": camera_to_floor.tolist(),
        "floor_to_camera_transform": floor_to_camera.tolist(),
        "support_confidence_threshold": support_confidence,
        "support_decisions": support_decisions, "floor": floor_bounds,
        "collision_diagnostics": collision_diagnostics, "object_count": count,
        "scene_glb_node_count": count, "scene_with_floor_glb_node_count": count + 1,
        "objects": objects,
    }
    atomic_json(report, save_dir / "scene_optimization.json")
    print(f"  Saved {count} objects; floor side={floor_bounds['final_side']:.3f} to {save_dir}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Gravity- and support-aware scene assembly")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--depth_method", choices=["auto", "gt", "ppd", "moge2", "moge"], default="auto",
                        help="auto uses Step 4's depth method")
    parser.add_argument("--solve_method", choices=["independent", "joint", "gravity_joint"],
                        default="gravity_joint", help="joint aliases gravity_joint")
    parser.add_argument("--placement_backend", choices=["geometry"], default="geometry")
    parser.add_argument("--mesh-backend", choices=["sam3d", "trellis2"], default="sam3d",
                        help="Select meshes from mesh/<backend>/")
    parser.add_argument("--support_confidence", type=float, default=0.90)
    parser.add_argument("--floor_margin_ratio", type=float, default=0.10)
    parser.add_argument("--floor_min_margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--scene_filter", default=None)
    parser.add_argument("--case", action="append",
                        help="Exact case name; repeat as needed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir, args.data_dir = args.output_dir.expanduser().resolve(), args.data_dir.expanduser().resolve()
    if not args.output_dir.is_dir() or not args.data_dir.is_dir():
        raise FileNotFoundError("--output_dir and --data_dir must both exist")
    if not 0 <= args.support_confidence <= 1:
        raise ValueError("--support_confidence must be in [0, 1]")
    if args.floor_margin_ratio < 0 or args.floor_min_margin <= 0:
        raise ValueError("invalid floor margins")
    if args.case:
        names = list(dict.fromkeys(args.case))
        missing = [
            name for name in names
            if not (args.output_dir / name / "CCM").is_dir()
            or not (args.output_dir / name / "mesh" / args.mesh_backend).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing CCM/mesh/{args.mesh_backend} case directories: " + ", ".join(missing)
            )
    else:
        names = sorted(
            path.name
            for path in args.output_dir.iterdir()
            if path.is_dir()
            and (path / "CCM").is_dir()
            and (path / "mesh" / args.mesh_backend).is_dir()
        )
    if args.scene_filter:
        names = [name for name in names if args.scene_filter in name]
    if args.max_cases > 0:
        names = names[:args.max_cases]
    print(f"Found {len(names)} scene(s) in {args.output_dir}")
    failures, successes, skipped = [], 0, 0
    for scene_index, name in enumerate(tqdm(names, desc="Scene construction")):
        print(f"\n[{scene_index + 1}/{len(names)}] {name}")
        try:
            data = load_ccm_outputs(args.output_dir / name / "CCM")
            if data is None:
                raise FileNotFoundError("missing essential CCM outputs")
            mesh_dir = args.output_dir / name / "mesh" / args.mesh_backend
            meshes = load_instance_glbs(mesh_dir, data["num_instances"])
            metadata, metadata_path, texture_path = load_floor_inputs(args.output_dir, name)
            floor_method = metadata.get("depth_method")
            requested = floor_method if args.depth_method == "auto" else args.depth_method
            if args.depth_method != "auto" and requested != floor_method:
                raise ValueError(f"requested depth {requested!r} differs from floor depth {floor_method!r}")
            depth, method = load_precomputed_depth(args.data_dir, name, requested)
            if depth is None:
                raise FileNotFoundError(f"missing complete {requested!r} depth result")
            graph, graph_path = load_scene_graph(args.data_dir, name, data["num_instances"])
            save_dir = args.output_dir / name / "scene" / args.mesh_backend / f"{method}_depth"
            outputs = [save_dir / item for item in ("scene_initial.glb", "scene.glb",
                                                     "scene_with_floor.glb", "scene_optimization.json")]
            if not args.force and all(path.is_file() for path in outputs):
                print("  Skipped (complete outputs exist; pass --force to rebuild)")
                skipped += 1
                continue
            construct_scene(depth, method, data, meshes, metadata, metadata_path,
                            texture_path, graph, graph_path, save_dir,
                            solve_method=args.solve_method,
                            placement_backend=args.placement_backend,
                            support_confidence=args.support_confidence,
                            floor_margin_ratio=args.floor_margin_ratio,
                            floor_min_margin=args.floor_min_margin,
                            seed=args.seed + sum(name.encode("utf-8")),
                            mesh_backend=args.mesh_backend, mesh_dir=mesh_dir)
            successes += 1
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures.append((name, message))
            print(f"  FAILED ({message})")
    print(f"\nDone. {successes} succeeded, {skipped} skipped, {len(failures)} failed.")
    if failures:
        for name, message in failures:
            print(f"  {name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    from core.stage_logging import run_logged
    log_backend = "sam3d"
    for index, token in enumerate(sys.argv[1:]):
        if token == "--mesh-backend" and index + 2 <= len(sys.argv) - 1:
            log_backend = sys.argv[index + 2]
        elif token.startswith("--mesh-backend="):
            log_backend = token.split("=", 1)[1]
    run_logged(main, f"05_{log_backend}_scene.log", primary_root_flags=("--output_dir",))
