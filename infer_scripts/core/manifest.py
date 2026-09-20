"""Per-case stage state, signatures, and downstream invalidation."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json, sha256_file, stable_hash

STAGE_ORDER = ["segmentation", "depth", "ccm", "mesh", "floor", "scene", "environment"]
STAGE_DEPENDENTS = {
    "segmentation": {"ccm", "floor", "environment"},
    "depth": {"floor", "scene"},
    "ccm": {"mesh", "scene"},
    "mesh": {"scene"},
    "floor": {"scene"},
    "scene": set(),
    "environment": set(),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(case_dir: Path) -> dict[str, Any]:
    return read_json(case_dir / "stage_manifest.json", {"schema": "mira_stage_manifest_v1", "stages": {}})


def stage_descendants(stage: str) -> set[str]:
    """Return the transitive downstream closure for one stage."""
    descendants: set[str] = set()
    pending = list(STAGE_DEPENDENTS[stage])
    while pending:
        candidate = pending.pop()
        if candidate in descendants:
            continue
        descendants.add(candidate)
        pending.extend(STAGE_DEPENDENTS[candidate])
    return descendants


def planned_stages(from_stage: str | None, to_stage: str) -> list[str]:
    """Select stages in execution order, using the DAG for explicit updates.

    ``to_stage`` remains an execution-order upper bound.  When ``from_stage``
    is explicit, branches that do not depend on it are excluded even if they
    fall between the two stages in :data:`STAGE_ORDER`.
    """
    end = STAGE_ORDER.index(to_stage)
    if from_stage is None:
        return STAGE_ORDER[: end + 1]
    start = STAGE_ORDER.index(from_stage)
    if start > end:
        raise ValueError("--from-stage must precede --to-stage")
    selected = {from_stage, *stage_descendants(from_stage)}
    return [stage for stage in STAGE_ORDER[start : end + 1] if stage in selected]


def _files(case_dir: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in case_dir.glob(pattern) if path.is_file())
    return sorted(paths)


def stage_input_paths(
    case_dir: Path,
    stage: str,
    depth_method: str,
    mesh_backend: str,
) -> list[Path]:
    """Return the concrete inputs consumed by a stage."""
    patterns = {
        "segmentation": ["input/source.png", "input/scene.png"],
        "depth": ["input/scene.png"],
        "ccm": ["input/scene.png", "input/mask_[0-9][0-9][0-9].png"],
        "mesh": [
            "CCM/rgb_mask.png", "CCM/masks.npy", "CCM/voxel_coords_*.npy",
        ],
        "floor": [
            "input/scene.png", "input/floor_mask.png",
            f"depth/{depth_method}/camera_pts_map.npy",
            f"depth/{depth_method}/valid_mask.npy",
        ],
        "scene": [
            "scene_graph.json", "CCM/rgb_mask.png", "CCM/canonical_coord_map*.npy",
            "CCM/canonical_pcd_*.ply",
            f"mesh/{mesh_backend}/[0-9][0-9][0-9].glb",
            "floor/floor_alignment.json", "floor/floor_texture.png",
            f"depth/{depth_method}/depth.npy",
            f"depth/{depth_method}/camera_pts_map.npy",
            f"depth/{depth_method}/valid_mask.npy",
            f"depth/{depth_method}/intrinsics.npy",
            f"depth/{depth_method}/fov_x_rad.txt",
        ],
        "environment": [
            "input/scene.png", "input/scene_fg.png", "input/floor_mask.png",
        ],
    }[stage]
    if stage == "mesh" and mesh_backend == "trellis2":
        # TRELLIS.2 uses the scene graph/annotation as a caption source and
        # prefers masks from review/ when present.
        patterns += [
            "input/scene.png", "input/mask_[0-9][0-9][0-9].png",
            "scene_graph.json", "review/annotation.json",
            "review/mask_[0-9][0-9][0-9].png",
        ]
    return _files(case_dir, patterns)


def signature(
    case_dir: Path,
    stage: str,
    config_fragment: Any,
    implementation: Path,
    *,
    depth_method: str = "ppd",
    mesh_backend: str = "sam3d",
) -> str:
    dependencies = [
        [str(path.relative_to(case_dir)), sha256_file(path)]
        for path in stage_input_paths(case_dir, stage, depth_method, mesh_backend)
    ]
    if implementation.is_dir():
        implementation_digest = stable_hash([
            [str(path.relative_to(implementation)), sha256_file(path)]
            for path in sorted(implementation.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        ])
    else:
        implementation_digest = sha256_file(implementation)
    return stable_hash({"stage": stage, "inputs": dependencies, "config": config_fragment,
                        "implementation": implementation_digest})


def reusable(case_dir: Path, stage: str, sig: str, complete: bool) -> bool:
    record = load_manifest(case_dir).get("stages", {}).get(stage, {})
    return bool(complete and record.get("status") == "complete" and record.get("signature") == sig)


def update_stage(case_dir: Path, stage: str, status: str, sig: str, **details: Any) -> None:
    manifest = load_manifest(case_dir)
    stages = manifest.setdefault("stages", {})
    previous = stages.get(stage, {})
    if status != "stale":
        previous = {
            key: value
            for key, value in previous.items()
            if key not in {"invalidated_by", "invalidated_at"}
        }
    stages[stage] = {**previous, "status": status, "signature": sig,
                     "updated_at": now(), "interpreter": details.pop("interpreter", sys.executable), **details}
    atomic_write_json(case_dir / "stage_manifest.json", manifest)


def invalidate_downstream(
    case_dir: Path,
    stage: str,
    mesh_backend: str | None = None,
) -> None:
    """Mark true DAG descendants stale without deleting their artifacts."""
    manifest = load_manifest(case_dir)
    stages = manifest.setdefault("stages", {})
    descendants = stage_descendants(stage)
    changed = False
    for key, record in stages.items():
        base = key.split("_", 1)[0]
        if base not in descendants:
            continue
        if (
            stage == "mesh"
            and mesh_backend
            and base == "scene"
            and "_" in key
            and key.split("_", 1)[1] != mesh_backend
        ):
            continue
        record.update({
            "status": "stale",
            "invalidated_by": stage,
            "invalidated_at": now(),
        })
        changed = True
    if changed:
        atomic_write_json(case_dir / "stage_manifest.json", manifest)


def clear_stage_outputs(
    case_dir: Path,
    stage: str,
    depth_method: str = "ppd",
    mesh_backend: str = "sam3d",
) -> None:
    """Remove only generated outputs for one stage before a forced rerun."""
    targets = {
        # Segmentation publishes atomically and archives the current review as
        # a new revision itself; do not erase human history here.
        "segmentation": [],
        "depth": [case_dir / "depth" / depth_method],
        "ccm": [case_dir / "CCM"],
        "mesh": [
            case_dir / "mesh" / mesh_backend,
            case_dir / "redraw" / mesh_backend,
        ],
        "floor": [case_dir / "floor"],
        "scene": [case_dir / "scene" / mesh_backend / f"{depth_method}_depth"],
        "environment": [case_dir / "environment"],
    }[stage]
    for target in targets:
        if target.is_dir(): shutil.rmtree(target)
        elif target.exists(): target.unlink()
