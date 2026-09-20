#!/usr/bin/env python3
"""Gemini redraw followed by TRELLIS.2 mesh generation.

For each object in a Mira-Scene case this script builds a scene/mask cutout,
asks Gemini to complete the object (``recon_full``), and feeds the result plus
the CCM voxel grid to TRELLIS.2. Results are written as
``mesh/trellis2/NNN.glb`` and intermediate redraws as
``redraw/trellis2/gemini/NNN.png``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

# TRELLIS.2 reads these at import time.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image

INFER_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFER_DIR.parent
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

# Keep these values in sync with 6_generate_environment_map.py.
LUMINA_DEFAULT_BASE_URL = "https://lumina.tripo3d.com/v1"
LUMINA_API_KEY_ENVS = ("CODEX_API_KEY", "LUMINA_API_KEY")
DEFAULT_ENV_FILE = None
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_TIMEOUT = 300
VOXEL_RES = 64
SS_RES = 32
NVDIFFRAST_FACE_LIMIT = 16777216
_hub_candidates = [Path(os.environ["TRELLIS_HUB"])] if os.environ.get("TRELLIS_HUB") else [
    Path("/mnt/share/pretrained_model/huggingface/hub"),
    Path("/mnt/pfs/share/pretrained_model/.cache/huggingface/hub"),
]
_hub = next((p for p in _hub_candidates if p.is_dir()), _hub_candidates[0])
DEFAULT_T2 = _hub / "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d"
DEFAULT_T2_SRC = Path(os.environ.get("TRELLIS2_SRC", "/mnt/pfs/users/huangzehuan/workspace/trellis-group/TRELLIS.2"))


def resolve_snapshot(path: Path, model_repo: str) -> Path:
    """Use an explicitly supplied snapshot, or discover a hash in the hub."""
    if path.is_dir() and (path / "pipeline.json").is_file():
        return path
    if path.parent.name == "snapshots":
        parent = path.parent
    else:
        parent = path / f"models--{model_repo}" / "snapshots"
    candidates = sorted(parent.glob("*"))
    valid = [p for p in candidates if (p / "pipeline.json").is_file()]
    if valid:
        return valid[-1]
    return path


def resolve_any_snapshot(path: Path) -> Path:
    """Recover from a stale hard-coded snapshot hash within the same repo."""
    if path.is_dir():
        return path
    if path.parent.name == "snapshots":
        candidates = sorted(p for p in path.parent.iterdir() if p.is_dir()) if path.parent.is_dir() else []
        if candidates:
            return candidates[-1]
    return path

ROLE = "Image 1 is the edit target: a segmented object cutout."
CAPTION_LINE = ('The target object caption is: "{caption}". Use this short phrase as a '
                "semantic target selector, not as permission to invent a generic replacement.")
RECONSTRUCT_BODY = """Image 1 is a segmentation cutout and may contain disconnected visible fragments of that same target where another scene object occluded it. It may also contain stray pixels or thin colored fringes along the outer edges from imprecise segmentation. Keep all observed pixels that semantically belong to the captioned target as appearance evidence, but remove anything that does not belong to that target, including edge fringes and halos.
Reconstruct the target as one complete, physically connected object. Connect target fragments by filling the genuinely occluded or truncated regions; do not leave floating fragments, holes shaped like occluders, or cropped ends. Preserve the target's visible color palette, local texture and pattern, material finish, wear, construction details, proportions, silhouette, pose, camera viewpoint, and perspective. Continue contours, materials, and repeated patterns conservatively. Do not redesign, restyle, recolor, clean, modernize, symmetrize, replace materials, or add decorative or functional parts. If completion evidence is ambiguous, use the simplest physically plausible continuation.
Increase resolution and detail: render crisp, sharp, well-defined surfaces and recover fine texture and material grain. Remove blur, pixelation, noise, and compression artifacts without smoothing away real surface wear.
Output exactly the captioned target object, centered on a flat uniform neutral light-gray background. Keep the original viewpoint. No scene floor, unrelated supporting tabletop, separate ground plane, platform, display stand, cast shadow that looks like geometry, neighboring objects, people, text, watermark, border, or collage. A tabletop, pedestal base, handle, spout, seat, cushion, or similar part must remain when it belongs to the captioned target."""
SILHOUETTE_GUARD = "Match the observed silhouette exactly where it is visible. Keep straight edges straight, keep corners as corners, and keep the outline profile, edge count, and part shapes of the visible regions unchanged. Only the genuinely missing regions may be newly drawn, and they must continue the visible outline rather than re-shape it. Do not round off, taper, bulge, smooth, or otherwise reinterpret any contour that is already visible."
FRAME_GUARD = "Frame the object so it lies entirely inside the canvas with clear empty background on all four sides. No part of it may touch or extend past any image edge. Scale it down to fit rather than cropping any part of it. Do not zoom or crop into the object."


def build_prompt(caption: str) -> str:
    return "\n".join((ROLE, CAPTION_LINE.format(caption=caption.strip()),
                       RECONSTRUCT_BODY, SILHOUETTE_GUARD, FRAME_GUARD))


def load_codex_api_key(env_file: str | None) -> bool:
    """Credentials must already be exported in the process environment."""
    return bool(os.environ.get("CODEX_API_KEY") or os.environ.get("LUMINA_API_KEY"))


def ensure_lumina_no_proxy(base_url: str) -> None:
    from urllib.parse import urlparse
    host = urlparse(base_url).hostname
    if not host:
        return
    values = []
    for key in ("NO_PROXY", "no_proxy"):
        values.extend(x.strip() for x in os.environ.get(key, "").split(",") if x.strip())
    if host.lower() not in {x.lower() for x in values}:
        values.append(host)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(dict.fromkeys(values))


def gemini_edit(image: Path, caption: str, args) -> tuple[bytes, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Gemini redraw requires the openai Python package") from exc
    load_codex_api_key(args.env_file)
    key = next((os.environ.get(k) for k in LUMINA_API_KEY_ENVS if os.environ.get(k)), None)
    if not key:
        raise RuntimeError("set CODEX_API_KEY or LUMINA_API_KEY")
    base = args.gemini_base_url or os.environ.get("LUMINA_API_BASE_URL", LUMINA_DEFAULT_BASE_URL)
    ensure_lumina_no_proxy(base)
    payload = io.BytesIO(image.read_bytes()); payload.name = image.name; payload.seek(0)
    prompt = build_prompt(caption)
    client = OpenAI(api_key=key, base_url=base.rstrip("/"), max_retries=0,
                    timeout=args.gemini_timeout)
    last = None
    for attempt in range(1, args.gemini_retries + 1):
        try:
            payload.seek(0)
            response = client.images.edit(model=args.gemini_model, image=payload,
                prompt=prompt, n=1, size="1024x1024", output_format="png",
                response_format="b64_json")
            encoded = response.data[0].b64_json
            if not encoded:
                raise RuntimeError("Lumina returned no inline image bytes")
            raw = base64.b64decode(encoded, validate=True)
            if not raw.startswith((b"\x89PNG", b"\xff\xd8")):
                raise RuntimeError("decoded Gemini bytes are not PNG/JPEG")
            return raw, prompt
        except Exception as exc:  # transient gateway failures are common
            last = exc
            print(f"    Gemini attempt {attempt}/{args.gemini_retries} failed: {exc}", flush=True)
            if attempt < args.gemini_retries:
                time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"Gemini redraw failed: {last}")


def preprocess_image(img: Image.Image) -> Image.Image:
    """Mirror TRELLIS.2 image preprocessing; input must contain alpha."""
    assert img.mode == "RGBA"
    scale = min(1.0, 1024 / max(img.size))
    if scale < 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    alpha = np.asarray(img)[..., 3]
    fg = np.argwhere(alpha > 0.8 * 255)
    if not fg.size:
        raise ValueError("alpha channel is empty")
    x0, y0, x1, y1 = fg[:, 1].min(), fg[:, 0].min(), fg[:, 1].max(), fg[:, 0].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(1, int(max(x1 - x0, y1 - y0)))
    img = img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    arr = np.asarray(img).astype(np.float32) / 255
    return Image.fromarray((arr[..., :3] * arr[..., 3:4] * 255).astype(np.uint8))


def matte(image: Path, model_dir: Path) -> Image.Image:
    from trellis2.pipelines.rembg import BiRefNet
    if not model_dir.is_dir():
        raise FileNotFoundError(f"RMBG snapshot not found: {model_dir}")
    import torch
    model = BiRefNet(str(model_dir)); model.cuda()
    result = model(Image.open(image).convert("RGB"))
    del model; torch.cuda.empty_cache()
    return result


def build_pipeline(t2: Path, dino: Path, low_vram: bool, source_dir: Path | None = None):
    import torch
    # TRELLIS.2 is distributed as a source tree.  Add it lazily so this script
    # can still be imported for configuration/help on machines without the
    # heavy dependency installed.
    source_dir = source_dir or DEFAULT_T2_SRC
    if str(source_dir) not in sys.path:
        sys.path.append(str(source_dir))
    import trellis2.models as models
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers
    cfg = json.loads((t2 / "pipeline.json").read_text())["args"]
    names = ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder",
             "tex_slat_flow_model_1024", "tex_slat_decoder"]
    loaded = {name: models.from_pretrained(str(t2 / cfg["models"][name])) for name in names}
    enc_args = dict(cfg["image_cond_model"]["args"])
    if dino.is_dir(): enc_args["model_name"] = str(dino)
    enc = getattr(image_feature_extractor, cfg["image_cond_model"]["name"])(**enc_args)
    pipe = Trellis2ImageTo3DPipeline(models=loaded,
        shape_slat_sampler=getattr(samplers, cfg["shape_slat_sampler"]["name"])(**cfg["shape_slat_sampler"]["args"]),
        tex_slat_sampler=getattr(samplers, cfg["tex_slat_sampler"]["name"])(**cfg["tex_slat_sampler"]["args"]),
        shape_slat_sampler_params=cfg["shape_slat_sampler"]["params"], tex_slat_sampler_params=cfg["tex_slat_sampler"]["params"],
        shape_slat_normalization=cfg["shape_slat_normalization"], tex_slat_normalization=cfg["tex_slat_normalization"],
        image_cond_model=enc, rembg_model=None, low_vram=low_vram)
    pipe._device = torch.device("cuda")
    if low_vram: enc.cuda()
    else: pipe.cuda()
    return pipe


def load_coords(path: Path, voxel_res: int = VOXEL_RES):
    import torch
    raw = np.load(path)
    if raw.ndim != 2 or raw.shape[1] != 3: raise ValueError(f"invalid voxel shape {raw.shape}")
    if raw.max() >= voxel_res: raise ValueError(f"voxel index exceeds {voxel_res}")
    coords = np.unique(raw // (voxel_res // SS_RES), axis=0) if voxel_res != SS_RES else raw
    t = torch.from_numpy(coords.astype(np.int32)).int().cuda()
    return torch.cat([torch.zeros_like(t[:, :1]), t], 1), len(raw), len(coords)


def generate_glb(image_path: Path, voxel_path: Path, output: Path, args, pipe=None):
    import torch
    import o_voxel
    src = Image.open(image_path)
    if src.mode != "RGBA": src = matte(image_path, args.rmbg)
    cond = preprocess_image(src)
    if args.save_condition:
        args.save_condition.parent.mkdir(parents=True, exist_ok=True); cond.save(args.save_condition)
    if pipe is None: pipe = build_pipeline(args.trellis2, args.dino, not args.no_low_vram)
    coords, n_in, n_tokens = load_coords(voxel_path, args.voxel_res)
    torch.manual_seed(args.seed)
    c512, c1024 = pipe.get_cond([cond], 512), pipe.get_cond([cond], 1024)
    shape, res = pipe.sample_shape_slat_cascade(c512, c1024, pipe.models["shape_slat_flow_model_512"], pipe.models["shape_slat_flow_model_1024"], 512, 1024, coords)
    tex = pipe.sample_tex_slat(c1024, pipe.models["tex_slat_flow_model_1024"], shape)
    mesh = pipe.decode_latent(shape, tex, res)[0]; mesh.simplify(NVDIFFRAST_FACE_LIMIT)
    glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=args.decimation,
        texture_size=args.texture_size, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.stem}.tmp{output.suffix}")
    glb.export(tmp, extension_webp=True); os.replace(tmp, output)
    return {"voxels_in": n_in, "ss_tokens": n_tokens, "shape_res": int(res),
            "mesh_verts": len(mesh.vertices), "mesh_faces": len(mesh.faces)}


def make_cutout(scene: Path, mask: Path, output: Path) -> None:
    rgb = Image.open(scene).convert("RGB")
    alpha = Image.open(mask).convert("L")
    if alpha.size != rgb.size: alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)
    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    output.parent.mkdir(parents=True, exist_ok=True); rgba.save(output)


def caption_for(case: Path, index: int) -> str:
    graph = case / "scene_graph.json"
    if graph.is_file():
        try:
            for n in json.loads(graph.read_text()).get("nodes", []):
                if n.get("kind") == "object" and n.get("mask_index") == index and n.get("name"):
                    return str(n["name"])
        except Exception: pass
    ann = case / "review" / "annotation.json"
    if ann.is_file():
        try:
            objs = json.loads(ann.read_text()).get("objects", [])
            for n in objs:
                if Path(str(n.get("mask_file", ""))).stem == f"mask_{index:03d}" and n.get("name"):
                    return str(n["name"])
        except Exception: pass
    return f"object_{index:03d}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--demo_dir", default=str(REPO_ROOT / "Mira_Scene_Demo" / "data"))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--case", action="append"); p.add_argument("--max_cases", type=int, default=-1)
    p.add_argument("--num_shards", type=int, default=1); p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gemini-base-url", dest="gemini_base_url", default=None)
    p.add_argument("--gemini-model", dest="gemini_model", default=DEFAULT_MODEL)
    p.add_argument("--gemini-timeout", dest="gemini_timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--gemini-retries", dest="gemini_retries", type=int, default=3)
    p.add_argument("--env-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--trellis2", type=Path, default=DEFAULT_T2,
                   help="TRELLIS.2 model snapshot containing pipeline.json")
    p.add_argument("--trellis2-src", type=Path, default=DEFAULT_T2_SRC,
                   help="TRELLIS.2 source tree (used when package is not installed)")
    p.add_argument("--rmbg", type=Path, default=_hub / "models--briaai--RMBG-2.0/snapshots/5df4c9c76d8170882c34f6986e848ee07fd0ba43")
    p.add_argument("--dino", type=Path, default=_hub / "models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/main")
    p.add_argument("--voxel-res", type=int, default=VOXEL_RES); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--texture-size", type=int, default=2048); p.add_argument("--decimation", type=int, default=300000)
    p.add_argument("--no-low-vram", action="store_true"); p.add_argument("--save-condition", type=Path, default=None)
    a = p.parse_args()
    if a.num_shards < 1 or not 0 <= a.shard_id < a.num_shards: p.error("invalid shard arguments")
    if a.gemini_retries < 1: p.error("--gemini-retries must be >= 1")
    if a.voxel_res < SS_RES or a.voxel_res % SS_RES: p.error("--voxel-res must be a multiple of 32")
    return a


def main():
    args = parse_args(); demo = Path(args.demo_dir).expanduser().resolve(); out = Path(args.output_dir or demo).expanduser().resolve()
    args.trellis2 = resolve_snapshot(args.trellis2, "microsoft--TRELLIS.2-4B")
    args.rmbg = resolve_any_snapshot(args.rmbg)
    args.dino = resolve_any_snapshot(args.dino)
    if not demo.is_dir(): raise FileNotFoundError(demo)
    cases = list(dict.fromkeys(args.case)) if args.case else sorted(x.name for x in demo.iterdir() if (x / "CCM").is_dir())
    if args.max_cases > 0: cases = cases[:args.max_cases]
    cases = cases[args.shard_id::args.num_shards]
    print(f"Processing {len(cases)} case(s) on shard {args.shard_id}/{args.num_shards}")
    pipe = None; failures = []
    for case_name in cases:
        case = demo / case_name; ccm = case / "CCM"; scene = case / "input" / "scene.png"
        voxels = sorted(ccm.glob("voxel_coords_*.npy"))
        if not scene.is_file() or not voxels: print(f"{case_name}: missing scene/voxels; skipped"); continue
        redraw_dir = out / case_name / "redraw" / "trellis2" / "gemini"; mesh_dir = out / case_name / "mesh" / "trellis2"; redraw_dir.mkdir(parents=True, exist_ok=True)
        for vp in voxels:
            try:
                idx = int(vp.stem.split("_")[-1]); glb_path = mesh_dir / f"{idx:03d}.glb"; input_rgba = redraw_dir / f"{idx:03d}_input_rgba.png"; redraw = redraw_dir / f"{idx:03d}.png"
                if glb_path.exists() and not args.overwrite: print(f"{case_name}/{idx:03d}: mesh exists; skipped"); continue
                mask = case / "review" / f"mask_{idx:03d}.png"
                if not mask.is_file(): mask = case / "input" / f"mask_{idx:03d}.png"
                if not mask.is_file(): raise FileNotFoundError(f"mask not found for {idx:03d}")
                if not input_rgba.exists() or args.overwrite: make_cutout(scene, mask, input_rgba)
                prompt = build_prompt(caption_for(case, idx))
                if not redraw.exists() or args.overwrite:
                    raw, prompt = gemini_edit(input_rgba, caption_for(case, idx), args); redraw.write_bytes(raw)
                    redraw.with_suffix(".json").write_text(json.dumps({"setting":"recon_full", "model":args.gemini_model, "caption":caption_for(case, idx), "prompt":prompt}, indent=2), encoding="utf-8")
                if pipe is None: pipe = build_pipeline(args.trellis2, args.dino, not args.no_low_vram, args.trellis2_src)
                stats = generate_glb(redraw, vp, glb_path, args, pipe)
                glb_path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(f"{case_name}/{idx:03d}: -> {glb_path}")
            except Exception as exc:
                failures.append((case_name, vp.name, repr(exc))); print(f"{case_name}/{vp.name}: FAILED: {exc}"); traceback.print_exc()
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception: pass
    print(f"Done. {len(failures)} failure(s).")
    for item in failures: print("  ", item)


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main,"03_trellis2_mesh.log",primary_root_flags=("--output_dir",),
               fallback_root_flags=("--demo_dir",))
