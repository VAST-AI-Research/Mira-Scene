"""Stage 0 automatic segmentation orchestration and artifact publication."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from infer_scripts.core.io import atomic_write_json
from .automatic import SOURCE_COMMIT
from .interactive.review import ReviewStore


@dataclass(frozen=True)
class Stage0Settings:
    object_profile: str = "major_v6"
    room_prompt: str = "list_objects_major_v5.txt"
    tabletop_prompt: str = "list_objects_tabletop_v1.txt"
    sam3_confidence: float = 0.5
    recycle: bool = True
    recycle_verifier_mode: str = "identity_upgrade"
    missing_object_critic_rounds: int = 0
    missing_object_critic_prompt: str = "missing_objects_major_v2.txt"
    missing_object_critic_max_overlap: float = 0.20
    save_debug: bool = False


def _binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.mode == "RGBA":
            return np.asarray(opened.getchannel("A")) > 0
        return np.asarray(opened.convert("L")) > 0


def _mask_records(stage1: Path) -> list[dict[str, Any]]:
    manifest_path = stage1 / "mask_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    by_file = {str(item.get("file")): item for item in manifest.get("masks", [])}
    records = []
    for path in sorted((stage1 / "segemented_obj").glob("*.png")):
        if path.stem == "the_floor" or path.stem.startswith("the_floor_"):
            continue
        item = by_file.get(path.name, {})
        caption = str(item.get("final_caption") or item.get("source_target") or path.stem).strip()
        records.append({
            "mask": _binary_mask(path), "name": caption, "caption": caption,
            "source": "automatic_segmentation", "source_id": path.stem,
            "source_file": path.name, "audit": item.get("audit", {}),
        })
    if not records:
        raise RuntimeError("automatic segmentation produced no foreground masks")
    return records


def _publish_scene_graph(case_dir: Path, records: list[dict[str, Any]], stage1: Path) -> None:
    source = json.loads((stage1 / "scene_tree.json").read_text(encoding="utf-8"))
    id_by_source = {record["source_id"]: f"object_{index:03d}" for index, record in enumerate(records)}
    nodes = [
        {"id": "floor", "kind": "static_environment", "name": "floor", "motion": "fixed", "fixed": True, "dynamic": False},
        {"id": "world_anchor", "kind": "virtual_static_anchor", "name": "wall_or_ceiling", "motion": "fixed", "fixed": True, "dynamic": False},
    ]
    edge_by_child = {str(edge.get("child")): edge for edge in source.get("edges", [])}
    edges = []
    for index, record in enumerate(records):
        object_id = f"object_{index:03d}"
        source_edge = edge_by_child.get(record["source_id"], {})
        raw_parent = source_edge.get("parent")
        if raw_parent == "floor":
            parent, relation = "floor", "rests_on"
        elif raw_parent in {"wall", "ceiling", "floor-wall"}:
            parent = "world_anchor"
            relation = "hangs_from" if raw_parent == "ceiling" else "fixed_to"
        else:
            parent = id_by_source.get(str(raw_parent))
            relation = "rests_on" if source_edge.get("relation") == "on" else str(source_edge.get("relation") or "unknown")
        motion = "fixed" if str(source_edge.get("type", "movable")) == "fixed" or raw_parent in {"wall", "ceiling", "floor-wall"} else "dynamic"
        nodes.append({
            "id": object_id, "kind": "object", "mask_index": index,
            "mask_file": f"input/mask_{index:03d}.png", "name": record["name"],
            "caption": record["caption"], "caption_source": "stage1_caption_audit",
            "name_locked": False, "caption_locked": False, "motion": motion,
            "fixed": motion == "fixed", "dynamic": motion == "dynamic",
            # A hanging object is suspended from above and should retain a
            # world-up canonical orientation unless the user later changes it
            # in the scene-graph editor.
            "upright_mode": "force" if relation == "hangs_from" else "auto", "confidence": 1.0,
            "automatic_source_id": record["source_id"],
        })
        edges.append({
            "child": object_id, "parent": parent, "relation": relation,
            "supporter_raw": raw_parent, "confidence": 1.0,
            "operational": parent is not None,
        })
    atomic_write_json(case_dir / "scene_graph.json", {
        "schema": "mira_scene_graph_v1", "case": case_dir.name,
        "edge_direction": "child_to_direct_supporter", "nodes": nodes, "edges": edges,
        "automatic_audit": {"source_commit": SOURCE_COMMIT, "source_file": "review/automatic_audit/scene_tree.json"},
    })


class SegmentationEngine:
    def __init__(self, sam3: Any, vlm: Any, settings: Stage0Settings | None = None):
        self.sam3, self.vlm = sam3, vlm
        self.settings = settings or Stage0Settings()

    def _argv(self, image: Path, output: Path, case_name: str) -> list[str]:
        settings = self.settings
        image_list = output / "images.txt"
        image_list.write_text(f"{image}, {case_name}\n", encoding="utf-8")
        argv = [
            "--image_list", str(image_list), "--output_folder", str(output),
            "--vlm_backend", "gpt", "--vlm_prompt_file", settings.room_prompt,
            "--object_profile", settings.object_profile,
            "--tabletop_vlm_prompt_file", settings.tabletop_prompt,
            "--sam3_confidence", str(settings.sam3_confidence),
            "--recycle_verifier_mode", settings.recycle_verifier_mode,
            "--missing_object_critic_rounds", str(settings.missing_object_critic_rounds),
            "--missing_object_critic_prompt_file", settings.missing_object_critic_prompt,
            "--missing_object_critic_max_overlap", str(settings.missing_object_critic_max_overlap),
        ]
        if not settings.recycle:
            argv.append("--no_recycle")
        if settings.save_debug:
            argv.append("--save_debug")
        return argv

    @staticmethod
    def _install_audit(case_dir: Path, stage1: Path) -> Path:
        review = case_dir / "review"
        review.mkdir(parents=True, exist_ok=True)
        target = review / "automatic_audit"
        if target.exists():
            revision = 0
            annotation = review / "annotation.json"
            if annotation.is_file():
                revision = int(json.loads(annotation.read_text(encoding="utf-8")).get("revision", 0))
            archive = review / "history" / f"revision_{revision:04d}" / "automatic_audit"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                shutil.rmtree(archive)
            target.replace(archive)
        shutil.move(str(stage1), str(target))
        return target

    def run(self, case_dir: Path, *, generate_graph: bool = True) -> dict[str, Any]:
        case_dir = case_dir.resolve()
        image = case_dir / "input" / "scene.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        work_parent = case_dir / "review"
        work_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".automatic-stage0-", dir=work_parent) as temporary:
            # The frozen standalone runner installs a process-wide exception
            # hook at import time so its CLI captures fatal tracebacks.  Mira
            # calls it in-process (including from the web server), therefore
            # restore the host hook after this invocation.
            previous_excepthook = __import__("sys").excepthook
            from .automatic.runner import main as run_automatic
            output = Path(temporary)
            try:
                run_automatic(self._argv(image, output, case_dir.name),
                                  sam3_backend=self.sam3, vlm_client=self.vlm)
            finally:
                __import__("sys").excepthook = previous_excepthook
            stage1 = output / case_dir.name / "stage1"
            records = _mask_records(stage1)
            floor_path = stage1 / "segemented_obj" / "the_floor.png"
            if not floor_path.is_file():
                raise RuntimeError("automatic segmentation produced no floor mask")
            floor = _binary_mask(floor_path)
            audit = self._install_audit(case_dir, stage1)
        mode_path = audit / "scene_mode_manifest.json"
        mode = json.loads(mode_path.read_text(encoding="utf-8")) if mode_path.is_file() else {"scene_mode": "ROOM"}
        annotation = ReviewStore(case_dir).initialize(records, floor, {
            "engine": "automatic_segmentation", "source_commit": SOURCE_COMMIT,
            "scene_mode": mode.get("scene_mode", "ROOM"), "automatic_audit": "review/automatic_audit",
        })
        if generate_graph:
            _publish_scene_graph(case_dir, records, audit)
        elif (case_dir / "scene_graph.json").exists():
            (case_dir / "scene_graph.json").unlink()
        return annotation
