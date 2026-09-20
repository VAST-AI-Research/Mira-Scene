"""YAML configuration loading, validation, and path expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to read --config") from exc
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    data["_config_path"] = str(path)
    return data


def get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def path_value(config: dict[str, Any], dotted: str, *, required: bool = False) -> Path | None:
    raw = get(config, dotted)
    if raw in (None, ""):
        if required:
            raise ConfigError(f"missing required config value: {dotted}")
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(raw)))
    result = Path(expanded).resolve()
    if required and not result.exists():
        raise ConfigError(f"configured path does not exist ({dotted}): {result}")
    return result


def stage_python(config: dict[str, Any], stage: str) -> str:
    import sys
    return str(get(config, f"environments.{stage}.python", sys.executable))


def validate_for_stages(config: dict[str, Any], stages: set[str], mesh_backend: str) -> None:
    needs: dict[str, list[str]] = {
        "segmentation": ["external.sam3.repo", "external.sam3.checkpoint"],
        "depth": (["external.ppd.repo", "checkpoints.ppd", "checkpoints.ppd_moge", "checkpoints.ppd_da2"]
                  if str(get(config, "depth.method", "ppd")) == "ppd" else
                  ["external.moge.repo"]),
        "ccm": ["external.mira_ccm.repo", "external.unidataset.repo", "checkpoints.ccm"],
        "mesh": (["external.sam3d.repo", "checkpoints.sam3d"] if mesh_backend == "sam3d" else
                 ["external.trellis2.repo", "checkpoints.trellis2", "checkpoints.rmbg", "checkpoints.dinov3"]),
    }
    missing = []
    for stage in stages:
        for key in needs.get(stage, []):
            try:
                path_value(config, key, required=True)
            except ConfigError as exc:
                missing.append(str(exc))
    api_stages = stages & {"segmentation", "environment"}
    if "floor" in stages and bool(get(config, "floor.generate_texture", True)):
        api_stages.add("floor")
    if api_stages and not (os.environ.get("CODEX_API_KEY") or os.environ.get("LUMINA_API_KEY")):
        missing.append("CODEX_API_KEY or LUMINA_API_KEY is required by VLM/image generation stages")
    if missing:
        raise ConfigError("preflight failed:\n- " + "\n- ".join(dict.fromkeys(missing)))


def validate_interpreters(config: dict[str, Any], stages: set[str], mesh_backend: str) -> None:
    checks = set(stages)
    if "mesh" in checks and mesh_backend == "trellis2":
        checks.remove("mesh"); checks.add("trellis2")
    missing = []
    for stage in checks:
        value = Path(stage_python(config, stage)).expanduser()
        if not value.is_file(): missing.append(f"configured Python does not exist ({stage}): {value}")
    probes = {
        "segmentation": ["torch", "fastapi", "yaml", "cv2", "pycocotools", "matplotlib", "iopath"],
        "depth": ["torch", "numpy", "trimesh"],
        "ccm": ["torch", "diffusers"],
        "mesh": ["torch", "pytorch3d", "hydra", "diff_gaussian_rasterization"],
        "trellis2": ["torch", "openai", "trimesh", "flash_attn"],
        "floor": ["numpy", "trimesh"],
        "scene": ["torch", "trimesh"],
        "environment": ["numpy", "PIL"],
    }
    import subprocess
    for stage in checks:
        interpreter = Path(stage_python(config, stage)).expanduser()
        if not interpreter.is_file(): continue
        modules = probes.get(stage, [])
        statements = [f"import {m}" for m in modules]
        # All inference stages in this pipeline run on the GPU.  Importing
        # torch alone is insufficient because a mirrored conda channel may
        # silently provide a CPU-only build.
        if stage in {"segmentation", "depth", "ccm", "mesh", "trellis2"}:
            statements += [
                "import torch",
                "assert torch.version.cuda is not None, "
                "'PyTorch is CPU-only (torch.version.cuda is None)'",
                "assert torch.cuda.is_available(), "
                "'CUDA is unavailable in the configured environment'",
            ]
        if stage == "segmentation":
            repo = path_value(config, "external.sam3.repo", required=True)
            statements += [f"import sys;sys.path.insert(0,{str(repo)!r})",
                           "from sam3.model.sam3_image_processor import Sam3Processor",
                           "from sam3.model_builder import build_sam3_image_model"]
        elif stage == "depth" and str(get(config, "depth.method", "ppd")) == "ppd":
            repo = path_value(config, "external.ppd.repo", required=True)
            statements += [f"import sys;sys.path.insert(0,{str(repo)!r})",
                           "from ppd.models.ppd import PixelPerfectDepth"]
        elif stage == "ccm":
            diff = path_value(config, "external.mira_ccm.repo", required=True)
            uni = path_value(config, "external.unidataset.repo", required=True)
            statements += [f"import sys;sys.path[:0]=[{str(diff / 'src')!r},{str(uni / 'src')!r}]",
                           "from miraccm.systems.shape_synthesis.data_processor.ccm_voxel import DataProcessor",
                           "from miraccm.pipelines.shape_synthesis.pipeline_ccm_voxel import CCMVoxelPipeline"]
        elif stage == "mesh":
            repo = path_value(config, "external.sam3d.repo", required=True)
            statements += ["import os;os.environ['LIDRA_SKIP_INIT']='true'",
                           "from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings",
                           # SAM3D's upstream module assumes a CUDA device even
                           # during import. Preflight must remain runnable on a
                           # login/CPU node, so stub only these two queries;
                           # real inference still runs in the configured GPU
                           # environment unchanged.
                           "import torch;torch.cuda.is_available=lambda:True;torch.cuda.get_device_name=lambda *_:'RTX 4090'",
                           f"import sys;sys.path.insert(0,{str(repo)!r})",
                           "from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap"]
        elif stage == "trellis2":
            repo = path_value(config, "external.trellis2.repo", required=True)
            statements += ["import os;os.environ.setdefault('ATTN_BACKEND','sdpa')",
                           "import transformers;assert int(transformers.__version__.split('.')[0]) < 5, 'RMBG-2.0 requires transformers<5'",
                           f"import sys;sys.path.insert(0,{str(repo)!r})",
                           "import o_voxel",
                           "from trellis2.pipelines import Trellis2ImageTo3DPipeline"]
        result = subprocess.run([str(interpreter), "-c", ";".join(statements)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            missing.append(f"Python dependency check failed ({stage}): {result.stderr.strip()[-1000:]}")
    if missing: raise ConfigError("preflight failed:\n- " + "\n- ".join(missing))
