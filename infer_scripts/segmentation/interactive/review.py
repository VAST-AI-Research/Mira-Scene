"""Versioned review storage and atomic publication to canonical case inputs."""

from __future__ import annotations

import base64
import io
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image

from infer_scripts.core.io import atomic_write_json
from infer_scripts.core.manifest import invalidate_downstream

MIRA_SIZE = 518
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def short_name(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    text = re.split(r"[,;:.!?。；：！？]", text, maxsplit=1)[0].strip()
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text[:80].strip() or "object"


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def mask_b64(mask: np.ndarray) -> str:
    out = io.BytesIO(); Image.fromarray(mask.astype(np.uint8)*255, "L").save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


def decode_mask(value: str) -> np.ndarray:
    if value.startswith("data:"): value = value.split(",",1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(value))).convert("L")
    if image.size != (MIRA_SIZE, MIRA_SIZE): raise ValueError("mask must be 518x518")
    return np.asarray(image) > 127


class ReviewStore:
    _lock = RLock()
    def __init__(self, case_dir: Path): self.case_dir = case_dir.resolve(); self.review = self.case_dir / "review"

    def _publish(self, annotation: dict[str, Any], masks: list[np.ndarray], floor: np.ndarray) -> None:
        input_dir = self.case_dir / "input"; input_dir.mkdir(exist_ok=True)
        temporary = self.case_dir / f".input-publish-{uuid.uuid4().hex}"; temporary.mkdir()
        try:
            for index, mask in enumerate(masks): Image.fromarray(mask.astype(np.uint8)*255, "L").save(temporary/f"mask_{index:03d}.png")
            Image.fromarray(floor.astype(np.uint8)*255, "L").save(temporary/"floor_mask.png")
            scene = np.asarray(Image.open(input_dir/"scene.png").convert("RGB")); union = np.logical_or.reduce(masks) if masks else np.zeros(scene.shape[:2],bool)
            fg = np.zeros_like(scene); fg[union]=scene[union]; Image.fromarray(fg).save(temporary/"scene_fg.png")
            for path in temporary.iterdir(): path.replace(input_dir/path.name)
            expected={f"mask_{index:03d}.png" for index in range(len(masks))}
            for old in input_dir.glob("mask_*.png"):
                if old.name not in expected: old.unlink()
            # review/ mirrors the editable layers while input/ is the canonical
            # contract consumed by downstream stages.
            for old in self.review.glob("mask_*.png"): old.unlink()
            for index, mask in enumerate(masks):
                Image.fromarray(mask.astype(np.uint8)*255, "L").save(self.review/f"mask_{index:03d}.png")
            Image.fromarray(floor.astype(np.uint8)*255, "L").save(self.review/"floor_mask.png")
            atomic_write_json(self.review/"annotation.json", annotation)
        finally: shutil.rmtree(temporary, ignore_errors=True)

    def initialize(self, objects: list[dict[str, Any]], floor: np.ndarray, metadata: dict[str,Any]) -> dict[str,Any]:
        entries=[]; masks=[]
        for index, obj in enumerate(objects):
            masks.append(np.asarray(obj["mask"],bool)); entries.append({"id":f"object_{index:03d}","name":short_name(obj.get("short_name") or obj["name"]),"short_name":short_name(obj.get("short_name") or obj["name"]),
                "caption":obj.get("caption",obj["name"]),"name_locked":False,"caption_locked":False,"source":obj.get("source","automatic")})
        annotation={"schema":"mira_segmentation_review_v1","case":self.case_dir.name,"revision":0,"created_at":_now(),
                    "updated_at":_now(),"objects":entries,"metadata":metadata}
        if self.review.exists() and (self.review/"annotation.json").is_file():
            old=json.loads((self.review/"annotation.json").read_text())
            history=self.review/"history"/f"revision_{int(old.get('revision',0)):04d}";history.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.review/"annotation.json",history/"annotation.json")
            for path in self.review.glob("mask_*.png"):shutil.copy2(path,history/path.name)
            if (self.review/"floor_mask.png").is_file():shutil.copy2(self.review/"floor_mask.png",history/"floor_mask.png")
            annotation["revision"]=int(old.get("revision",0))+1
        self.review.mkdir(parents=True,exist_ok=True); self._publish(annotation,masks,floor)
        return annotation

    def adopt_existing(self) -> dict[str, Any]:
        """Build a review from canonical input masks and an optional scene graph.

        Pipeline exports often have ``input/mask_*.png`` and ``scene_graph.json``
        without ``review/annotation.json``.  The web UI needs the annotation to
        show masks; this reconstructs it without rewriting ``input/``.
        """
        annotation_path = self.review / "annotation.json"
        if annotation_path.is_file():
            return json.loads(annotation_path.read_text(encoding="utf-8"))
        input_dir = self.case_dir / "input"
        mask_paths = sorted(input_dir.glob("mask_[0-9][0-9][0-9].png"))
        if not mask_paths:
            raise FileNotFoundError(f"{self.case_dir.name}: no existing masks")
        indices = [int(path.stem.split("_")[-1]) for path in mask_paths]
        if indices != list(range(len(indices))):
            raise ValueError(f"{self.case_dir.name}: existing masks must be consecutive mask_000.png …")
        graph_nodes: dict[int, dict[str, Any]] = {}
        graph_path = self.case_dir / "scene_graph.json"
        if graph_path.is_file():
            try:
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                graph = {}
            for node in graph.get("nodes", []) if isinstance(graph, dict) else []:
                if not isinstance(node, dict) or node.get("kind") != "object":
                    continue
                index = node.get("mask_index")
                if isinstance(index, int):
                    graph_nodes[index] = node
        entries = []
        for index, path in enumerate(mask_paths):
            node = graph_nodes.get(index, {})
            name = short_name(node.get("short_name") or node.get("name") or f"object_{index:03d}")
            entries.append({
                "id": f"object_{index:03d}", "name": name, "short_name": name,
                "caption": str(node.get("caption") or name),
                "name_locked": False, "caption_locked": False, "source": "existing_segmentation",
            })
        floor_path = input_dir / "floor_mask.png"
        if not floor_path.is_file():
            with Image.open(input_dir / "scene.png") as opened:
                Image.fromarray(np.zeros((opened.height, opened.width), np.uint8)).save(floor_path)
        annotation = {
            "schema": "mira_segmentation_review_v1", "case": self.case_dir.name,
            "revision": 0, "created_at": _now(), "updated_at": _now(),
            "objects": entries,
            "metadata": {"engine": "existing_segmentation", "source": "adopt_existing", "adopted_at": _now()},
        }
        self.review.mkdir(parents=True, exist_ok=True)
        for path in mask_paths:
            shutil.copy2(path, self.review / path.name)
        shutil.copy2(floor_path, self.review / "floor_mask.png")
        atomic_write_json(annotation_path, annotation)
        return annotation

    def read(self, include_masks: bool=True) -> dict[str,Any]:
        annotation=json.loads((self.review/"annotation.json").read_text())
        if include_masks:
            for index,obj in enumerate(annotation["objects"]): obj["mask_b64"]=base64.b64encode((self.case_dir/"input"/f"mask_{index:03d}.png").read_bytes()).decode()
            annotation["floor_mask_b64"]=base64.b64encode((self.case_dir/"input/floor_mask.png").read_bytes()).decode()
        return annotation

    def save(self, payload: dict[str,Any]) -> dict[str,Any]:
        with self._lock:
            old=self.read(False); revision=int(old.get("revision",0))+1
            history=self.review/"history"/f"revision_{int(old.get('revision',0)):04d}"; history.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.review/"annotation.json",history/"annotation.json")
            for path in (self.case_dir/"input").glob("mask_*.png"): shutil.copy2(path,history/path.name)
            shutil.copy2(self.case_dir/"input/floor_mask.png",history/"floor_mask.png")
            entries=[]; masks=[]
            for index,obj in enumerate(payload.get("objects",[])):
                oid=str(obj.get("id") or f"object_{index:03d}")
                if not SAFE_ID.fullmatch(oid): raise ValueError(f"invalid object id: {oid}")
                name=str(obj.get("name","")).strip()
                if not name: raise ValueError("object name is required")
                mask=decode_mask(str(obj.get("mask_b64", "")))
                if not mask.any(): raise ValueError("delete empty masks instead of saving")
                display_name=short_name(obj.get("short_name") or name)
                masks.append(mask); entries.append({"id":f"object_{index:03d}","name":display_name,"short_name":display_name,"caption":str(obj.get("caption") or name),
                    "name_locked":bool(obj.get("name_locked",True)),"caption_locked":bool(obj.get("caption_locked",True)),"source":obj.get("source","human")})
            floor=decode_mask(str(payload.get("floor_mask_b64", "")))
            annotation={**old,"revision":revision,"updated_at":_now(),"objects":entries}
            self._publish(annotation,masks,floor); invalidate_downstream(self.case_dir,"segmentation")
            return self.read()
