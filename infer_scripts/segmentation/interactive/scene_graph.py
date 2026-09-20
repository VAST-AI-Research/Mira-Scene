"""Captioned, backward-compatible scene graph generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter
import numpy as np

from infer_scripts.core.io import atomic_write_json
from ..backends.vlm import VLMClient, parse_json


def _short_name(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    text = text.split(",", 1)[0].split(";", 1)[0].strip()
    return text[:80] or "object"


def _highlight(scene: Image.Image, mask_path: Path, output: Path) -> None:
    rgb = np.asarray(scene, np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    shown = rgb * .3; shown[mask] = rgb[mask]
    edge = (np.asarray(Image.fromarray(mask.astype("uint8") * 255).filter(ImageFilter.MaxFilter(5))) > 127) & ~mask
    shown[edge] = (255, 0, 0)
    Image.fromarray(np.uint8(np.clip(shown, 0, 255))).save(output)


def generate_scene_graph(case_dir: Path, objects: list[dict[str, Any]], client: VLMClient,
                         preserve_existing: bool = True) -> dict[str, Any]:
    audit = case_dir / "review/scene_graph_input"; audit.mkdir(parents=True, exist_ok=True)
    scene_path = case_dir / "input/scene.png"; scene = Image.open(scene_path).convert("RGB")
    views = []
    for index, obj in enumerate(objects):
        path = audit / f"object_{index:03d}.png"; _highlight(scene, case_dir / "input" / f"mask_{index:03d}.png", path); views.append(path)
    prompt = """The first image is a scene and following images highlight objects in order. Return a JSON array, one record per object with: index, name (short singular category), caption (detailed 1-2 sentences covering appearance, material, shape and pose, useful for image redraw/3D generation), direct_supporter (floor, wall, ceiling, another zero-based object index, or unknown), relation (rests_on, fixed_to, hangs_from, or unknown), motion (fixed or dynamic), confidence (0..1). Direct support only. Wall/ceiling attachment is fixed."""
    parsed = parse_json(client.complete(prompt, [scene_path] + views))
    if isinstance(parsed, dict): parsed = parsed.get("objects", [])
    by_index = {int(x.get("index")): x for x in parsed if isinstance(x, dict) and str(x.get("index", "")).isdigit()}
    existing = {}
    old_path = case_dir / "scene_graph.json"
    if preserve_existing and old_path.is_file():
        old = json.loads(old_path.read_text())
        existing = {n.get("id"): n for n in old.get("nodes", []) if n.get("kind") == "object"}
    nodes = [
        {"id":"floor","kind":"static_environment","name":"floor","motion":"fixed","fixed":True,"dynamic":False},
        {"id":"world_anchor","kind":"virtual_static_anchor","name":"wall_or_ceiling","motion":"fixed","fixed":True,"dynamic":False},
    ]; edges = []
    for index, annotation in enumerate(objects):
        inferred = by_index.get(index, {}); oid = f"object_{index:03d}"; old = existing.get(oid, {})
        name_locked = bool(annotation.get("name_locked") or old.get("name_locked"))
        caption_locked = bool(annotation.get("caption_locked") or old.get("caption_locked"))
        name = (annotation.get("name") if name_locked else inferred.get("name")) or annotation.get("name") or "unknown"
        caption = (annotation.get("caption") if caption_locked else inferred.get("caption")) or annotation.get("caption") or str(name)
        motion = str(inferred.get("motion", "dynamic")).lower(); motion = motion if motion in {"fixed","dynamic"} else "dynamic"
        supporter = inferred.get("direct_supporter", "unknown"); relation = str(inferred.get("relation", "unknown"))
        parent = None
        if str(supporter).lower() in {"floor", "ground"}: parent, relation = "floor", "rests_on"
        elif str(supporter).lower() in {"wall", "ceiling"}: parent = "world_anchor"; motion = "fixed"; relation = "fixed_to" if str(supporter).lower()=="wall" else "hangs_from"
        elif str(supporter).isdigit() and 0 <= int(supporter) < len(objects) and int(supporter) != index: parent = f"object_{int(supporter):03d}"
        confidence = float(np.clip(float(inferred.get("confidence", .5)), 0, 1))
        # Unlocked names are VLM-owned.  In particular, adopted mask-only
        # cases initially use object_NNN placeholders; those placeholders
        # must not hide the semantic name inferred alongside the caption.
        short_name = _short_name(
            annotation.get("short_name")
            if name_locked
            else inferred.get("name") or annotation.get("short_name") or name
        )
        nodes.append({"id":oid,"kind":"object","mask_index":index,"mask_file":f"input/mask_{index:03d}.png",
            "name":short_name,"short_name":short_name,"caption":str(caption),"caption_source":"human" if caption_locked else "vlm",
            "name_locked":name_locked,"caption_locked":caption_locked,"motion":motion,"fixed":motion=="fixed",
            "dynamic":motion=="dynamic","upright_mode":old.get("upright_mode", "auto"),"confidence":confidence})
        edges.append({"child":oid,"parent":parent,"relation":relation,"supporter_raw":supporter,
                      "confidence":confidence,"operational":parent is not None})
    graph = {"schema":"mira_scene_graph_v1","case":case_dir.name,"edge_direction":"child_to_direct_supporter",
             "nodes":nodes,"edges":edges,"vlm_inference":{"model":client.model}}
    atomic_write_json(old_path, graph)
    return graph
