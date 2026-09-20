#!/usr/bin/env python3
"""Estimate a floor plane and its orientation for prepared ``demo_data``.

Input layout::

    demo_data/<case>/
    ├── input/
    │   ├── scene.png             # source of the embedded floor texture
    │   ├── floor_mask.png        # used only for plane estimation
    │   └── mask_*.png            # not used by this stage
    └── depth/<method>/
        ├── camera_pts_map.npy       # [H, W, 3], OpenGL camera coordinates
        └── valid_mask.npy            # optional [H, W] validity mask

For every case, the floor pixels are gathered from ``floor_mask.png`` and the
finite camera-space points in the selected depth result.  A deterministic
RANSAC + SVD fit estimates the plane.  The normal is oriented to OpenGL +Y,
and a minimal rotation/translation is exported as the camera-to-floor frame
transform.  The in-plane yaw is inherently ambiguous; the reported transform
uses the minimal rotation that aligns the fitted normal with +Y.

This stage deliberately does not infer the final floor extent.  It writes a
fixed-size square placeholder centered at the floor-frame origin on ``y=0``.
The final floor should be resized later from the projected reconstructed meshes
during scene assembly.  Consequently, noisy foreground depth and
``input/mask_*.png`` do not affect this stage.

Three outputs are written to ``demo_data/<case>/floor/`` (or
``--output_dir``)::

    floor_alignment.json             # plane, normal, transform, fit metrics
    floor_plane.glb                  # fixed textured square at y=0
    floor_texture.png                # standalone final floor texture

The PNG is also embedded in the GLB material.  It is saved separately so scene
assembly can resize the floor and rebuild its UVs without extracting an image
from the GLB.  ``--generate_floor_texture`` optionally replaces the texture
extracted from ``scene.png`` with a Lumina-generated seamless texture;
generation failures fall back to the source texture unless
``--require_generated_floor_texture`` is specified.

Single-directory example::

    python infer_scripts/4_estimate_floor.py --demo_dir demo_data

Select one depth method explicitly::

    python infer_scripts/4_estimate_floor.py --demo_dir demo_data --depth_method ppd

The default depth priority is ``gt`` → ``ppd`` → ``moge2`` → ``moge``.  Cases
without a floor mask, depth point map, or a sufficiently supported plane are
reported and do not stop the remaining cases until the final summary.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_DIR = REPO_ROOT / "Mira_Scene_Demo" / "data"
UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
LUMINA_DEFAULT_BASE_URL = "https://lumina.tripo3d.com/v1"
LUMINA_API_KEY_ENVS = ("CODEX_API_KEY", "LUMINA_API_KEY")
DEFAULT_ENV_FILE = None
OUTPUT_SCHEMA = "mira_scene_floor_estimate_v9"
LEGACY_OUTPUT_NAMES = (
    "camera_to_floor_transform.npy",
    "floor_to_camera_transform.npy",
    "floor_points_camera.ply",
    "floor_points_aligned.ply",
    "floor_plane.ply",
    "floor_mask_valid.png",
    "floor_inlier_mask.png",
    # Names used by an earlier add_floor implementation.
    "floor.glb",
    "floor_texture_source.png",
    # v7 debugging output; foreground depth no longer participates in floor
    # estimation or sizing.
    "foreground_points_aligned.ply",
)


def load_codex_api_key(env_file: Path | None) -> bool:
    """Credentials are accepted only from the inherited process environment."""
    return bool(os.environ.get("CODEX_API_KEY") or os.environ.get("LUMINA_API_KEY"))


def load_depth_points(case_dir: Path, method: str) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Load the first available camera point map in the requested priority."""
    methods = [method] if method != "auto" else ["gt", "ppd", "moge2", "moge"]
    for selected in methods:
        depth_dir = case_dir / "depth" / selected
        points_path = depth_dir / "camera_pts_map.npy"
        if not points_path.is_file():
            continue
        points = np.asarray(np.load(points_path), dtype=np.float64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"invalid camera_pts_map shape in {points_path}: {points.shape}")
        valid_path = depth_dir / "valid_mask.npy"
        if valid_path.is_file():
            valid = np.asarray(np.load(valid_path), dtype=bool)
            if valid.shape != points.shape[:2]:
                raise ValueError(
                    f"valid_mask shape {valid.shape} does not match points {points.shape[:2]}"
                )
        else:
            valid = np.ones(points.shape[:2], dtype=bool)
        return points, valid, selected
    return None, None, None


def load_floor_candidates(case_dir: Path, points: np.ndarray, valid: np.ndarray):
    mask_path = case_dir / "input" / "floor_mask.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing floor mask: {mask_path}")
    floor_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if floor_mask.shape != points.shape[:2]:
        floor_mask = np.asarray(
            Image.fromarray((floor_mask.astype(np.uint8) * 255)).resize(
                (points.shape[1], points.shape[0]), Image.Resampling.NEAREST
            )
        ) > 127
    finite = np.isfinite(points).all(axis=-1)
    candidate_mask = floor_mask & valid & finite
    return floor_mask, candidate_mask, points[candidate_mask]


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper rotation sending normalized ``source`` to ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-10:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(source, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [[0.0, -cross[2], cross[1]],
         [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (cross_norm ** 2))


def fit_floor_plane(
    points: np.ndarray,
    threshold: float,
    iterations: int,
    min_inliers: int,
    seed: int,
    ransac_max_points: int,
):
    """Fit a plane with deterministic RANSAC followed by two SVD refinements."""
    if len(points) < max(3, min_inliers):
        return None

    rng = np.random.default_rng(seed)
    if len(points) > ransac_max_points:
        hypothesis_indices = rng.choice(len(points), size=ransac_max_points, replace=False)
        hypothesis_points = points[hypothesis_indices]
    else:
        hypothesis_points = points

    best_normal = None
    best_point = None
    best_count = 0
    for _ in range(iterations):
        sample = hypothesis_points[rng.choice(len(hypothesis_points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        distances = np.abs((hypothesis_points - sample[0]) @ normal)
        count = int((distances <= threshold).sum())
        if count > best_count:
            best_count = count
            best_normal = normal
            best_point = sample[0]

    if best_normal is None:
        return None
    # Re-score the best hypothesis on every candidate point.
    initial_inliers = np.abs((points - best_point) @ best_normal) <= threshold
    if int(initial_inliers.sum()) < min_inliers:
        return None

    inlier_points = points[initial_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, UP) < 0:
        normal = -normal

    refined_inliers = np.abs((points - centroid) @ normal) <= threshold
    if int(refined_inliers.sum()) < min_inliers:
        return None
    inlier_points = points[refined_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, UP) < 0:
        normal = -normal

    residuals = np.abs((inlier_points - centroid) @ normal)
    return {
        "point": centroid,
        "normal": normal,
        "inlier_mask": refined_inliers,
        "num_input_points": int(len(points)),
        "num_inliers": int(len(inlier_points)),
        "median_residual": float(np.median(residuals)),
        "p95_residual": float(np.percentile(residuals, 95)),
        "max_residual": float(np.max(residuals)),
    }


def make_floor_transform(plane: dict) -> np.ndarray:
    rotation = rotation_between_vectors(plane["normal"], UP)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ plane["point"]
    return transform


def load_scene_image(case_dir: Path, size: tuple[int, int]) -> Image.Image:
    image_path = case_dir / "input" / "scene.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"missing source scene image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def make_source_floor_texture(
    scene_image: Image.Image, floor_mask: np.ndarray, size: int
) -> Image.Image:
    """Crop observed floor pixels and fill non-floor pixels with median colour."""
    rgb = np.asarray(scene_image, dtype=np.uint8)
    rows, cols = np.nonzero(floor_mask)
    if len(rows) == 0:
        raise ValueError("floor mask has no valid source-image pixels")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop_rgb = rgb[top:bottom, left:right]
    crop_mask = floor_mask[top:bottom, left:right]
    fill = np.median(crop_rgb[crop_mask], axis=0).astype(np.uint8)
    texture = np.broadcast_to(fill, crop_rgb.shape).copy()
    texture[crop_mask] = crop_rgb[crop_mask]
    return Image.fromarray(texture, "RGB").resize((size, size), Image.Resampling.LANCZOS)


def ensure_lumina_no_proxy(base_url: str) -> None:
    """Avoid enterprise-proxy failures when directly accessing Lumina."""
    host = urlparse(base_url).hostname
    if not host:
        return
    entries: list[str] = []
    known: set[str] = set()
    for key in ("NO_PROXY", "no_proxy"):
        for value in os.environ.get(key, "").split(","):
            value = value.strip()
            if value and value.lower() not in known:
                entries.append(value)
                known.add(value.lower())
    if host.lower() not in known:
        entries.append(host)
    merged = ",".join(entries)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def generate_floor_texture_with_lumina(
    scene_image: Image.Image, floor_mask: np.ndarray, args: argparse.Namespace
) -> tuple[Image.Image, str]:
    """Generate a seamless extension of the observed floor material."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("--generate_floor_texture requires the openai Python package") from error
    api_key = next(
        (os.environ.get(name) for name in LUMINA_API_KEY_ENVS if os.environ.get(name)), None
    )
    if not api_key:
        raise RuntimeError(
            "--generate_floor_texture requires " + " or ".join(LUMINA_API_KEY_ENVS)
        )
    ensure_lumina_no_proxy(args.floor_texture_base_url)
    reference = make_source_floor_texture(scene_image, floor_mask, args.floor_texture_size)
    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")
    buffer.name = "floor_texture_reference.png"
    buffer.seek(0)
    prompt = (
        "Use case: photorealistic-natural. Asset type: seamless 3D floor texture. "
        "Primary request: create a square seamless tileable texture that extends the floor material, "
        "colour, grain or pattern visible in the supplied reference image. "
        "Constraints: output only a flat top-down floor surface; preserve the observed material and colour; "
        "no furniture, walls, people, objects, shadows, perspective, text, logos, borders, or watermark."
    )
    client = OpenAI(
        api_key=api_key,
        base_url=args.floor_texture_base_url.rstrip("/"),
        max_retries=0,
        timeout=args.floor_texture_timeout,
    )
    response = client.images.edit(
        model=args.floor_texture_model,
        image=buffer,
        prompt=prompt,
        n=1,
        size=f"{args.floor_texture_size}x{args.floor_texture_size}",
        quality=args.floor_texture_quality,
        output_format="png",
        response_format="b64_json",
        background="opaque",
    )
    encoded = response.data[0].b64_json
    if not encoded:
        raise RuntimeError("Lumina returned no inline image bytes")
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return image, prompt


def remove_legacy_outputs(output_dir: Path) -> None:
    """Remove only auxiliary files produced by previous floor implementations."""
    for name in LEGACY_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def output_is_current(output_dir: Path, placeholder_size: float) -> bool:
    """Return whether current deliverables exist for the requested size."""
    metadata_path = output_dir / "floor_alignment.json"
    plane_path = output_dir / "floor_plane.glb"
    texture_path = output_dir / "floor_texture.png"
    if not metadata_path.is_file() or not plane_path.is_file() or not texture_path.is_file():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        return (
            metadata.get("schema") == OUTPUT_SCHEMA
            and np.isclose(
                float(metadata.get("floor_placeholder_size", float("nan"))),
                placeholder_size,
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def make_floor_placeholder_mesh(
    size: float,
    texture_image: Image.Image,
    texture_tiling: float,
) -> trimesh.Trimesh:
    """Create a square placeholder centered at the floor-frame origin."""
    half_size = 0.5 * float(size)
    vertices = np.array(
        [[-half_size, 0.0, -half_size],
         [half_size, 0.0, -half_size],
         [half_size, 0.0, half_size],
         [-half_size, 0.0, half_size]],
        dtype=np.float64,
    )
    uv = np.array(
        [[0.0, 0.0], [texture_tiling, 0.0],
         [texture_tiling, texture_tiling], [0.0, texture_tiling]],
        dtype=np.float64,
    )
    material = trimesh.visual.material.PBRMaterial(
        name="floor_texture",
        baseColorTexture=texture_image,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64),
        visual=visual,
        process=False,
    )


def process_case(args: argparse.Namespace, case_dir: Path) -> tuple[bool, str]:
    points, valid, method = load_depth_points(case_dir, args.depth_method)
    if points is None:
        return False, "no camera_pts_map.npy found for the selected depth methods"
    _, candidate_mask, floor_points = load_floor_candidates(case_dir, points, valid)
    if len(floor_points) < args.min_floor_inliers:
        return False, f"only {len(floor_points)} valid floor pixels (need {args.min_floor_inliers})"

    plane = fit_floor_plane(
        floor_points,
        threshold=args.ransac_threshold,
        iterations=args.ransac_iterations,
        min_inliers=args.min_floor_inliers,
        seed=args.seed + sum(case_dir.name.encode("utf-8")),
        ransac_max_points=args.ransac_max_points,
    )
    if plane is None:
        return False, f"floor plane fit failed ({len(floor_points)} candidate points)"

    transform = make_floor_transform(plane)
    output_dir = Path(args.output_dir or args.demo_dir).expanduser().resolve() / case_dir.name / "floor"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use only geometric inliers as the source-texture reference, avoiding
    # mislabeled foreground pixels that happen to lie inside floor_mask.
    floor_inlier_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidate_rows, candidate_cols = np.nonzero(candidate_mask)
    floor_inlier_mask[
        candidate_rows[plane["inlier_mask"]], candidate_cols[plane["inlier_mask"]]
    ] = True
    scene_image = load_scene_image(case_dir, (points.shape[1], points.shape[0]))
    floor_texture = make_source_floor_texture(
        scene_image, floor_inlier_mask, args.floor_texture_size
    )
    texture_mode = "source"
    texture_prompt = None
    if args.generate_floor_texture:
        try:
            floor_texture, texture_prompt = generate_floor_texture_with_lumina(
                scene_image, floor_inlier_mask, args
            )
            texture_mode = "lumina_generated"
        except Exception as error:
            if args.require_generated_floor_texture:
                return False, f"floor texture generation failed: {type(error).__name__}: {error}"
            print(
                f"  {case_dir.name}: generated floor texture unavailable "
                f"({type(error).__name__}: {error}); using source texture"
            )

    floor_texture_path = output_dir / "floor_texture.png"
    floor_texture.save(floor_texture_path, format="PNG")
    floor_mesh = make_floor_placeholder_mesh(
        args.floor_placeholder_size,
        floor_texture,
        args.floor_texture_tiling,
    )
    floor_mesh.export(output_dir / "floor_plane.glb")

    metadata = {
        "schema": OUTPUT_SCHEMA,
        "case": case_dir.name,
        "depth_method": method,
        "coordinate_convention": "OpenGL camera (+X right, +Y up, -Z forward)",
        "floor_frame_convention": "+Y is floor normal and floor plane is y=0",
        "in_plane_yaw": "underdetermined; transform uses minimal rotation normal->+Y",
        "floor_mask": str(case_dir / "input" / "floor_mask.png"),
        "source_scene_image": str(case_dir / "input" / "scene.png"),
        "camera_pts_map": str(case_dir / "depth" / method / "camera_pts_map.npy"),
        "camera_to_floor_transform": transform.tolist(),
        "floor_to_camera_transform": np.linalg.inv(transform).tolist(),
        "floor_plane_camera_point": plane["point"].tolist(),
        "floor_plane_camera_normal": plane["normal"].tolist(),
        "floor_plane_equation": {
            "normal": plane["normal"].tolist(),
            "offset": float(-np.dot(plane["normal"], plane["point"])),
        },
        "num_input_points": plane["num_input_points"],
        "num_inliers": plane["num_inliers"],
        "inlier_fraction": float(plane["num_inliers"] / max(1, plane["num_input_points"])),
        "ransac_threshold": args.ransac_threshold,
        "median_residual": plane["median_residual"],
        "p95_residual": plane["p95_residual"],
        "max_residual": plane["max_residual"],
        "floor_plane_visualization": str(output_dir / "floor_plane.glb"),
        "floor_vertices": int(len(floor_mesh.vertices)),
        "floor_faces": int(len(floor_mesh.faces)),
        "floor_extent_source": "fixed square placeholder; resize from reconstructed meshes during scene assembly",
        "floor_placeholder_size": args.floor_placeholder_size,
        "floor_placeholder_center_xz": [0.0, 0.0],
        "floor_plane_xz_bounds": {
            "min": floor_mesh.vertices[:, [0, 2]].min(axis=0).tolist(),
            "max": floor_mesh.vertices[:, [0, 2]].max(axis=0).tolist(),
        },
        "floor_texture_mode": texture_mode,
        "floor_texture_path": str(floor_texture_path),
        "floor_texture_size": args.floor_texture_size,
        "floor_texture_tiling": args.floor_texture_tiling,
        "floor_texture_model": args.floor_texture_model if texture_mode == "lumina_generated" else None,
        "floor_texture_base_url": (
            args.floor_texture_base_url if texture_mode == "lumina_generated" else None
        ),
        "floor_texture_prompt": texture_prompt,
    }
    with (output_dir / "floor_alignment.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    remove_legacy_outputs(output_dir)
    return True, (
        f"{method}: {plane['num_inliers']}/{plane['num_input_points']} inliers, "
        f"normal={np.round(plane['normal'], 4).tolist()}, "
        f"median residual={plane['median_residual']:.5f}, texture={texture_mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate a floor frame and fixed square placeholder")
    parser.add_argument("--demo_dir", type=Path, default=DEFAULT_DEMO_DIR,
                        help="Root containing <case>/input and <case>/depth (default: demo_data)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Optional output root; defaults to --demo_dir")
    parser.add_argument("--depth_method", choices=["auto", "gt", "ppd", "moge2", "moge"], default="auto")
    parser.add_argument("--scene_filter", default=None)
    parser.add_argument("--case", action="append",
                        help="Exact case name; repeat as needed")
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--ransac_threshold", type=float, default=0.03,
                        help="Plane inlier distance in depth units")
    parser.add_argument("--ransac_iterations", type=int, default=1000)
    parser.add_argument("--ransac_max_points", type=int, default=50000,
                        help="Maximum points used for RANSAC hypotheses; scoring uses all points")
    parser.add_argument("--min_floor_inliers", type=int, default=500)
    parser.add_argument("--floor_placeholder_size", type=float, default=2.0,
                        help="Side length of the square placeholder in depth units (default: 2.0)")
    parser.add_argument("--floor_texture_tiling", type=float, default=2.0,
                        help="Texture repeats along each side of the square placeholder")
    parser.add_argument("--floor_texture_size", type=int, default=1024,
                        help="Square source/generated texture size embedded in floor_plane.glb")
    parser.add_argument("--generate_floor_texture", action="store_true",
                        help="Use Lumina to synthesize a seamless floor texture")
    parser.add_argument("--require_generated_floor_texture", action="store_true",
                        help="Fail instead of falling back when Lumina generation fails")
    parser.add_argument("--floor_texture_model", default="gpt-image-2")
    parser.add_argument("--floor_texture_base_url", default=LUMINA_DEFAULT_BASE_URL)
    parser.add_argument("--floor_texture_quality", default="high",
                        choices=["low", "medium", "high", "auto"])
    parser.add_argument("--floor_texture_timeout", type=float, default=300.0)
    parser.add_argument("--env_file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing floor outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.demo_dir = args.demo_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.env_file = args.env_file.expanduser().resolve() if args.env_file is not None else None
    if not args.demo_dir.is_dir():
        raise FileNotFoundError(f"demo directory does not exist: {args.demo_dir}")
    if args.ransac_threshold <= 0 or args.ransac_iterations < 1 or args.ransac_max_points < 3:
        raise ValueError("invalid RANSAC parameters")
    if (
        args.min_floor_inliers < 3
        or args.floor_placeholder_size <= 0
        or args.floor_texture_tiling <= 0
        or args.floor_texture_size < 256
        or args.floor_texture_timeout <= 0
    ):
        raise ValueError("invalid floor parameters")
    if args.require_generated_floor_texture and not args.generate_floor_texture:
        raise ValueError("--require_generated_floor_texture requires --generate_floor_texture")
    if args.generate_floor_texture:
        load_codex_api_key(args.env_file)

    if args.case:
        cases = [args.demo_dir / name for name in dict.fromkeys(args.case)]
        missing = [str(path) for path in cases if not path.is_dir()]
        if missing:
            raise FileNotFoundError("missing case directories: " + ", ".join(missing))
    else:
        cases = sorted(path for path in args.demo_dir.iterdir() if path.is_dir())
    if args.scene_filter:
        cases = [path for path in cases if args.scene_filter in path.name]
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if not cases:
        raise RuntimeError(f"no cases found under {args.demo_dir}")

    output_root = args.output_dir or args.demo_dir
    print(f"Estimating floor for {len(cases)} case(s) from {args.demo_dir}")
    failures = []
    successes = 0
    for case_dir in tqdm(cases, desc="Floor estimation"):
        output_dir = output_root / case_dir.name / "floor"
        output_json = output_dir / "floor_alignment.json"
        if output_is_current(output_dir, args.floor_placeholder_size) and not args.force:
            remove_legacy_outputs(output_dir)
            print(f"  {case_dir.name}: skip (existing {output_json})")
            successes += 1
            continue
        try:
            success, message = process_case(args, case_dir)
        except Exception as error:
            success, message = False, f"{type(error).__name__}: {error}"
        if success:
            successes += 1
            print(f"  {case_dir.name}: {message}")
        else:
            failures.append((case_dir.name, message))
            print(f"  {case_dir.name}: FAILED ({message})")

    print(f"\nDone. {successes}/{len(cases)} cases succeeded.")
    if failures:
        print(f"Failed cases ({len(failures)}):")
        for case_name, message in failures:
            print(f"  {case_name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main, "04_floor.log", primary_root_flags=("--output_dir",),
               fallback_root_flags=("--demo_dir",))
