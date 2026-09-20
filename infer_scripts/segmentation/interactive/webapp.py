"""Standalone FastAPI segmentation review service."""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from infer_scripts.core.cases import prepare_case
from infer_scripts.core.manifest import signature,update_stage
from infer_scripts.core.io import atomic_write_json
from .review import ReviewStore, mask_b64, decode_mask
from ..backends.sam3 import Sam3Backend
from .scene_graph import generate_scene_graph
from ..backends.vlm import VLMClient


def create_app(output_root: Path, config: dict[str,Any], engine_override: Any | None = None):
    try:
        from fastapi import FastAPI, File, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc: raise RuntimeError("FastAPI, python-multipart and uvicorn are required for --web") from exc
    from infer_scripts.core.config import get,path_value
    sam3 = engine_override.sam3 if engine_override is not None else Sam3Backend(
        path_value(config,"external.sam3.repo",required=True),
        path_value(config,"external.sam3.checkpoint",required=True),
        str(get(config,"segmentation.device","cuda")),
        float(get(config,"segmentation.sam3_confidence",.35)))
    vlm = engine_override.vlm if engine_override is not None else VLMClient(
        str(get(config,"segmentation.vlm_model","gemini-2.5-pro")),
        str(get(config,"api.base_url","https://lumina.tripo3d.com/v1")),
        float(get(config,"api.timeout",300)))
    app=FastAPI(title="Mira-Scene Segmentation",version="1")
    root=output_root.resolve(); root.mkdir(parents=True,exist_ok=True)
    static=Path(__file__).parent/"web"
    def graph_status_path(target: Path) -> Path:
        return target / "review/scene_graph_status.json"

    def write_graph_status(target: Path, status: str, error: str | None = None) -> None:
        record: dict[str, Any] = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
        if error:
            record["error"] = error[:2000]
        atomic_write_json(graph_status_path(target), record)

    def read_graph_status(target: Path) -> dict[str, Any]:
        path = graph_status_path(target)
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and isinstance(value.get("status"), str):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        return {"status": "ready" if (target / "scene_graph.json").is_file() else "unavailable"}
    @app.exception_handler(ValueError)
    async def invalid_payload(_request,exc): return JSONResponse(status_code=422,content={"detail":str(exc)})
    def case(name:str)->Path:
        target=(root/name).resolve()
        try: target.relative_to(root)
        except ValueError: raise HTTPException(400,"invalid case")
        if not (target/"case.json").is_file(): raise HTTPException(404,"case not found")
        return target
    @app.get("/api/health")
    def health(): return {"status":"ok","cases":sum(1 for p in root.iterdir() if (p/"case.json").is_file())}
    @app.get("/api/capabilities")
    def capabilities():
        automatic_available = engine_override is not None and hasattr(engine_override, "run")
        if not automatic_available:
            from importlib.util import find_spec
            try:
                automatic_available = (find_spec("infer_scripts.segmentation.engine") is not None and
                                       find_spec("infer_scripts.segmentation.automatic.runner") is not None)
            except (ImportError, ModuleNotFoundError):
                automatic_available = False
        return {"interactive_segmentation": True,
                "automatic_segmentation": automatic_available,
                "scene_graph": True}
    @app.get("/api/cases")
    def cases():
        result = []
        for path in sorted(root.iterdir()):
            if not (path / "case.json").is_file():
                continue
            has_review = (path / "review/annotation.json").is_file()
            has_masks = has_review or any((path / "input").glob("mask_[0-9][0-9][0-9].png"))
            result.append({"id": path.name, "ready": has_masks,
                           "image_url": f"/api/cases/{path.name}/image"})
        return result
    @app.post("/api/cases")
    async def upload(file:Any=File(...)):
        suffix=Path(file.filename or "upload.png").suffix.lower()
        if suffix not in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}: raise HTTPException(400,"unsupported image")
        with tempfile.TemporaryDirectory(prefix="mira-upload-") as directory:
            source=Path(directory)/Path(file.filename or f"upload{suffix}").name
            source.write_bytes(await file.read())
            # Do not persist the TemporaryDirectory path: it is deleted before
            # this request returns and is useless in a reproducible manifest.
            target=prepare_case(source,root,source_reference=file.filename or source.name)
        # Uploading is intentionally preparation-only.  The user chooses
        # between interactive editing and the explicit Auto-Segment action.
        return {"id":target.name,"prepared":True,"ready":False}
    @app.post("/api/cases/{name}/automatic")
    async def automatic(name:str):
        target=case(name)
        try:
            if engine_override is not None and hasattr(engine_override, "run"):
                engine = engine_override
            else:
                from ..engine import SegmentationEngine, Stage0Settings
            # Keep the endpoint lazy: interactive Web does not require the
            # automatic package to be importable at startup.
                settings = Stage0Settings(
                    object_profile=str(get(config,"segmentation.object_profile","major_v6")),
                    room_prompt=str(get(config,"segmentation.room_prompt","list_objects_major_v5.txt")),
                    tabletop_prompt=str(get(config,"segmentation.tabletop_prompt","list_objects_tabletop_v1.txt")),
                    sam3_confidence=float(get(config,"segmentation.sam3_confidence",.5)),
                    recycle=bool(get(config,"segmentation.recycle",True)),
                    recycle_verifier_mode=str(get(config,"segmentation.recycle_verifier_mode","identity_upgrade")),
                    missing_object_critic_rounds=int(get(config,"segmentation.missing_object_critic_rounds",0)),
                    missing_object_critic_prompt=str(get(config,"segmentation.missing_object_critic_prompt","missing_objects_major_v2.txt")),
                    missing_object_critic_max_overlap=float(get(config,"segmentation.missing_object_critic_max_overlap",.2)),
                    save_debug=bool(get(config,"segmentation.save_debug",False)))
                engine = SegmentationEngine(sam3, vlm, settings)
            await asyncio.to_thread(engine.run,target)
        except ImportError as exc:
            raise HTTPException(501,"automatic segmentation is not installed") from exc
        except Exception as exc: raise HTTPException(503,f"segmentation runtime unavailable: {exc}") from exc
        sig=signature(target,"segmentation",config,Path(__file__).parent)
        update_stage(target,"segmentation","complete",sig,source="web_automatic")
        write_graph_status(target, "ready" if (target / "scene_graph.json").is_file() else "unavailable")
        return {"ok":True}
    @app.get("/api/cases/{name}/review")
    def review(name:str):
        target=case(name)
        annotation_path = target / "review/annotation.json"
        if not annotation_path.is_file():
            # Pipeline exports often have input masks and scene_graph.json
            # without a web review annotation.  Adopt those masks so the UI
            # can show them.  A freshly prepared image has neither and gets
            # an empty editable review instead.
            if any((target / "input").glob("mask_[0-9][0-9][0-9].png")):
                try:
                    ReviewStore(target).adopt_existing()
                except (OSError, ValueError) as exc:
                    raise HTTPException(404, f"case has no editable masks: {exc}") from exc
            else:
                scene = target / "input/scene.png"
                if not scene.is_file():
                    raise HTTPException(404, "case has no scene image")
                with Image.open(scene) as opened:
                    floor = np.zeros((opened.height, opened.width), dtype=bool)
                ReviewStore(target).initialize([], floor, {"engine": "interactive", "source": "web_prepare"})
        payload=ReviewStore(target).read(); payload["image"]={"width":518,"height":518}
        payload["image_url"]=f"/api/cases/{name}/image"; return payload
    @app.get("/api/cases/{name}/scene-graph")
    def scene_graph(name: str):
        target = case(name)
        path = target / "scene_graph.json"
        if not path.is_file():
            raise HTTPException(404, "scene graph is not available yet")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["generation_status"] = read_graph_status(target).get("status", "ready")
            return payload
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(503, "scene graph is unavailable") from exc

    @app.get("/api/cases/{name}/scene-graph/status")
    def scene_graph_status(name: str):
        target = case(name)
        return read_graph_status(target)

    @app.put("/api/cases/{name}/scene-graph")
    async def save_scene_graph(name: str, payload: dict[str, Any]):
        target = case(name)
        path = target / "scene_graph.json"
        if not path.is_file():
            raise HTTPException(404, "scene graph is not available yet")
        if not isinstance(payload, dict):
            raise HTTPException(422, "scene graph must be an object")
        nodes, edges = payload.get("nodes"), payload.get("edges")
        if not isinstance(nodes, list) or not nodes:
            raise HTTPException(422, "scene graph must contain nodes")
        if not isinstance(edges, list):
            raise HTTPException(422, "scene graph edges must be a list")
        node_ids: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                raise HTTPException(422, "scene graph nodes must be objects")
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id or node_id in node_ids:
                raise HTTPException(422, "scene graph contains invalid or duplicate node ids")
            if len(node_id) > 120 or len(str(node.get("name") or "")) > 160:
                raise HTTPException(422, "scene graph node field is too long")
            if node.get("kind") == "object" and node.get("upright_mode", "auto") not in {"auto", "force", "free"}:
                raise HTTPException(422, "object upright_mode must be auto, force, or free")
            node_ids.add(node_id)
        parents: dict[str, str] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                raise HTTPException(422, "scene graph edges must be objects")
            child, parent = edge.get("child"), edge.get("parent")
            if child not in node_ids or (parent is not None and parent not in node_ids):
                raise HTTPException(422, "scene graph edge references an unknown node")
            if parent is None:
                continue
            if child == parent or child in parents:
                raise HTTPException(422, "each node must have at most one distinct direct supporter")
            parents[child] = parent
        for start in parents:
            seen: set[str] = set()
            cursor = start
            while cursor in parents:
                if cursor in seen:
                    raise HTTPException(422, "support relationships must not contain a cycle")
                seen.add(cursor)
                cursor = parents[cursor]
        current = json.loads(path.read_text(encoding="utf-8"))
        # generation_status is transport metadata, not part of the editable
        # graph document persisted by this endpoint.
        saved = {key: value for key, value in payload.items() if key != "generation_status"}
        # Human edits replace VLM uncertainty.  Scene construction later
        # drops edges below --support_confidence (default 0.90), so a
        # manually chosen parent/relation must not keep a low VLM score.
        previous = {edge.get("child"): edge for edge in current.get("edges", []) if isinstance(edge, dict)}
        for edge in saved.get("edges", []):
            if not isinstance(edge, dict):
                continue
            old = previous.get(edge.get("child"))
            changed = old is None or any(
                old.get(field) != edge.get(field) for field in ("parent", "relation")
            ) or bool(old.get("operational")) != bool(edge.get("operational"))
            if changed:
                edge["confidence"] = 1.0
        saved["schema"] = current.get("schema", "mira_scene_graph_v1")
        saved["case"] = current.get("case", name)
        saved["edge_direction"] = "child_to_direct_supporter"
        saved["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        write_graph_status(target, "ready")
        saved["generation_status"] = "ready"
        return saved

    @app.put("/api/cases/{name}/scene_graph")
    async def save_scene_graph_legacy(name: str, payload: dict[str, Any]):
        return await save_scene_graph(name, payload)
    @app.put("/api/cases/{name}/review")
    async def save(name:str,payload:dict[str,Any]):
        target=case(name)
        if not (target/"review/annotation.json").is_file(): raise HTTPException(409,"create masks with interactive tools or run automatic segmentation before saving a review")
        estimate_scene_graph = payload.get("estimate_scene_graph") is True
        review_payload = {key: value for key, value in payload.items() if key != "estimate_scene_graph"}
        result=ReviewStore(target).save(review_payload)
        sig=signature(target,"segmentation",config,Path(__file__).parent)
        update_stage(target,"segmentation","complete",sig,source="web_review",human_revision=result["revision"])
        if estimate_scene_graph:
            write_graph_status(target, "estimating")
            # Publication is complete before graph regeneration; a graph
            # failure never loses masks or the previous graph.
            async def graph():
                try:
                    await asyncio.to_thread(generate_scene_graph,target,result["objects"],vlm,True)
                    write_graph_status(target, "ready")
                except Exception as exc:
                    write_graph_status(target, "failed", repr(exc))
                    (target/"logs").mkdir(exist_ok=True)
                    (target/"logs/scene_graph_async_error.log").write_text(repr(exc))
            asyncio.create_task(graph())
        elif (target / "scene_graph.json").is_file():
            write_graph_status(target, "stale")
        else:
            write_graph_status(target, "unavailable")
        result["image"]={"width":518,"height":518}
        result["image_url"]=f"/api/cases/{name}/image"; return result
    @app.post("/api/cases/{name}/predict")
    async def predict(name:str,payload:dict[str,Any]):
        mask_input=decode_mask(payload["mask_b64"]) if payload.get("mask_b64") else None
        try:
            masks,scores=await asyncio.to_thread(sam3.interactive,case(name)/"input/scene.png",prompt=str(payload.get("prompt", "")),
                points=payload.get("points"),labels=payload.get("labels"),box=payload.get("box"),mask_input=mask_input)
        except Exception as exc: raise HTTPException(503,f"segmentation runtime unavailable: {exc}") from exc
        return {"masks":[mask_b64(x) for x in masks],"scores":scores,"best_idx":int(np.argmax(scores)) if scores else 0}
    @app.get("/api/cases/{name}/image")
    def image(name:str): return FileResponse(case(name)/"input/scene.png",media_type="image/png")
    app.mount("/",StaticFiles(directory=static,html=True),name="static")
    return app


def run_web(output_root:Path,config:dict[str,Any],host:str,port:int):
    import uvicorn
    uvicorn.run(create_app(output_root,config),host=host,port=port)
