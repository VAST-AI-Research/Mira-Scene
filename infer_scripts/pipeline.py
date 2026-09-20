#!/usr/bin/env python3
"""Reproducible stage-wise Mira-Scene inference pipeline."""

from __future__ import annotations

import argparse
import codecs
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from infer_scripts.core.cases import discover_images, prepare_cases
from infer_scripts.core.config import (
    ConfigError,
    get,
    load_config,
    path_value,
    stage_python,
    validate_for_stages,
    validate_interpreters,
)
from infer_scripts.core.io import atomic_write_json, files_complete
from infer_scripts.core.manifest import (
    STAGE_ORDER,
    clear_stage_outputs,
    invalidate_downstream,
    load_manifest,
    planned_stages,
    reusable,
    signature,
    update_stage,
)

STAGES = {
    "segmentation": {
        "number": "00",
        "script": "0_segmentation.py",
        "patterns": [
            "review/annotation.json",
            "input/mask_*.png",
            "input/floor_mask.png",
            "input/scene_fg.png",
            "scene_graph.json",
        ],
    },
    "depth": {
        "number": "01",
        "script": "1_estimation_depth.py",
        "patterns": [
            "depth/ppd/depth.npy",
            "depth/ppd/camera_pts_map.npy",
            "depth/ppd/valid_mask.npy",
        ],
    },
    "ccm": {
        "number": "02",
        "script": "2_inference_CCM.py",
        "patterns": ["CCM/masks.npy", "CCM/voxel_coords_*.npy"],
    },
    "mesh": {
        "number": "03",
        "script": "3_inference_mesh.py",
        "patterns": ["mesh/{backend}/[0-9][0-9][0-9].glb"],
    },
    "floor": {
        "number": "04",
        "script": "4_estimate_floor.py",
        "patterns": [
            "floor/floor_alignment.json",
            "floor/floor_plane.glb",
            "floor/floor_texture.png",
        ],
    },
    "scene": {
        "number": "05",
        "script": "5_construct_scene.py",
        "patterns": [
            "scene/{backend}/ppd_depth/scene.glb",
            "scene/{backend}/ppd_depth/scene_with_floor.glb",
            "scene/{backend}/ppd_depth/scene_optimization.json",
        ],
    },
    "environment": {
        "number": "06",
        "script": "6_generate_environment_map.py",
        "patterns": [
            "environment/environment_equirect.png",
            "environment/environment_metadata.json",
        ],
    },
}
LOG_NAMES = {
    "segmentation": "00_segmentation.log",
    "depth": "01_depth.log",
    "ccm": "02_ccm.log",
    "floor": "04_floor.log",
    "scene": "05_scene.log",
    "environment": "06_environment.log",
}
PARALLEL_STAGES = {"ccm", "mesh"}


def parse_gpu_ids(value: str) -> list[int]:
    """Parse an explicit, ordered list of physical GPU identifiers."""
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be a comma-separated list such as 0,1,2,3"
        )
    try:
        gpu_ids = [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpu-ids values must be integers") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids values must be non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids values must not be repeated")
    return gpu_ids


@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    num_shards: int
    gpu_id: int
    command: list[str]
    cases: list[Path]


@dataclass(frozen=True)
class ShardResult:
    spec: ShardSpec
    process: subprocess.CompletedProcess[str]


def stage_log_name(stage: str, mesh_backend: str) -> str:
    if stage == "mesh":
        return f"03_{mesh_backend}_mesh.log"
    if stage == "scene":
        return f"05_{mesh_backend}_scene.log"
    return LOG_NAMES[stage]


def run_streamed(
    cmd: list[str],
    *,
    env: dict[str, str],
    logs: list[tuple[Path, str, str]],
) -> subprocess.CompletedProcess[str]:
    """Run a stage while teeing its combined output to the terminal and logs.

    ``subprocess.run(..., stdout=PIPE)`` does not return captured output until
    the child exits.  Heavy model stages can therefore appear frozen for many
    minutes.  Read the pipe incrementally instead, while retaining the output
    for manifest tracebacks and per-case failure handling.

    Each log tuple is ``(path, mode, header)``.  Keeping log opening here means
    partial output is flushed even if the user interrupts the pipeline.
    """
    child_env = env.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    captured: list[str] = []

    with ExitStack() as stack:
        handles: list[TextIO] = []
        for path, mode, header in logs:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open(mode, encoding="utf-8"))
            handle.write(header)
            handle.flush()
            handles.append(handle)

        process = subprocess.Popen(
            cmd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Binary mode plus read1-like raw reads makes carriage-return based
            # progress updates visible without waiting for a newline.
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                value = decoder.decode(chunk)
                if not value:
                    continue
                captured.append(value)
                sys.stdout.write(value)
                sys.stdout.flush()
                for handle in handles:
                    handle.write(value)
                    handle.flush()

            tail = decoder.decode(b"", final=True)
            if tail:
                captured.append(tail)
                sys.stdout.write(tail)
                sys.stdout.flush()
                for handle in handles:
                    handle.write(tail)
                    handle.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            # Do not leave a GPU-heavy stage running after its pipeline parent
            # has been interrupted. Partial output is already safely flushed.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()

    return subprocess.CompletedProcess(cmd, returncode, "".join(captured))


def build_shard_specs(
    config: dict[str, Any],
    stage: str,
    cases: list[Path],
    root: Path,
    backend: str,
    depth: str,
    force: bool,
    gpu_ids: list[int],
) -> list[ShardSpec]:
    """Build one case-sharded command per GPU without starting any process."""
    if stage not in PARALLEL_STAGES:
        raise ValueError(f"stage {stage!r} does not support GPU sharding")
    num_shards = min(len(cases), len(gpu_ids))
    if num_shards == 0:
        return []
    specs = []
    for shard_id, gpu_id in enumerate(gpu_ids[:num_shards]):
        cmd = command(config, stage, cases, root, backend, depth, force)
        cmd += ["--num_shards", str(num_shards), "--shard_id", str(shard_id)]
        specs.append(
            ShardSpec(
                shard_id=shard_id,
                num_shards=num_shards,
                gpu_id=gpu_id,
                command=cmd,
                cases=cases[shard_id::num_shards],
            )
        )
    return specs


def run_sharded_streamed(
    specs: list[ShardSpec],
    *,
    env: dict[str, str],
    pipeline_log: Path,
    stage: str,
    mesh_backend: str,
) -> list[ShardResult]:
    """Run independent GPU workers and stream their output safely.

    Root pipeline output is serialized through one lock. Per-case handles are
    only owned by the worker responsible for that case, so workers never write
    the same case log or manifest concurrently.
    """
    if not specs:
        return []
    output_lock = threading.Lock()
    processes: dict[int, subprocess.Popen[bytes]] = {}
    captured: dict[int, list[str]] = {spec.shard_id: [] for spec in specs}
    returncodes: dict[int, int] = {}
    thread_errors: list[BaseException] = []

    pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        pipeline_handle = stack.enter_context(pipeline_log.open("a", encoding="utf-8"))
        case_handles: dict[int, list[TextIO]] = {}
        for spec in specs:
            command_line = " ".join(spec.command)
            header = (
                f"\n## {stage} shard {spec.shard_id + 1}/{spec.num_shards} "
                f"gpu={spec.gpu_id}\n$ {command_line}\n"
            )
            pipeline_handle.write(header)
            handles = []
            for case in spec.cases:
                path = case / "logs" / stage_log_name(stage, mesh_backend)
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open("w", encoding="utf-8"))
                handle.write(header)
                handle.flush()
                handles.append(handle)
            case_handles[spec.shard_id] = handles
        pipeline_handle.flush()

        try:
            for spec in specs:
                child_env = env.copy()
                child_env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
                child_env.setdefault("PYTHONUNBUFFERED", "1")
                processes[spec.shard_id] = subprocess.Popen(
                    spec.command,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

        def consume(spec: ShardSpec) -> None:
            process = processes[spec.shard_id]
            assert process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            prefix = (
                f"[{stage} shard {spec.shard_id + 1}/{spec.num_shards} "
                f"gpu={spec.gpu_id}] "
            )
            try:
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    value = decoder.decode(chunk)
                    if not value:
                        continue
                    captured[spec.shard_id].append(value)
                    display = prefix + value.replace("\n", "\n" + prefix)
                    if display.endswith(prefix):
                        display = display[: -len(prefix)]
                    with output_lock:
                        sys.stdout.write(display)
                        sys.stdout.flush()
                        pipeline_handle.write(display)
                        pipeline_handle.flush()
                    for handle in case_handles[spec.shard_id]:
                        handle.write(value)
                        handle.flush()
                tail = decoder.decode(b"", final=True)
                if tail:
                    captured[spec.shard_id].append(tail)
                    with output_lock:
                        sys.stdout.write(prefix + tail)
                        sys.stdout.flush()
                        pipeline_handle.write(prefix + tail)
                        pipeline_handle.flush()
                    for handle in case_handles[spec.shard_id]:
                        handle.write(tail)
                        handle.flush()
                returncodes[spec.shard_id] = process.wait()
            except BaseException as exc:
                thread_errors.append(exc)
                if process.poll() is None:
                    process.terminate()
                returncodes[spec.shard_id] = process.wait()
            finally:
                process.stdout.close()

        threads = [
            threading.Thread(target=consume, args=(spec,), daemon=True)
            for spec in specs
        ]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=0.1)
        except KeyboardInterrupt:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 10
            for process in processes.values():
                if process.poll() is None:
                    try:
                        process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for thread in threads:
                thread.join(timeout=2)
            raise

    if thread_errors:
        raise RuntimeError("failed while reading sharded stage output") from thread_errors[0]
    return [
        ShardResult(
            spec,
            subprocess.CompletedProcess(
                spec.command,
                returncodes[spec.shard_id],
                "".join(captured[spec.shard_id]),
            ),
        )
        for spec in specs
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, required=True, help="one image or a flat directory"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--ccm-ckpt-dir",
        type=Path,
        default=None,
        help="override checkpoints.ccm from the YAML configuration",
    )
    p.add_argument("--mesh-backend", choices=["sam3d", "trellis2"], default=None)
    p.add_argument("--force-stage", action="append", choices=STAGE_ORDER, default=[])
    p.add_argument(
        "--from-stage",
        choices=STAGE_ORDER,
        default=None,
        help=(
            "update this stage and its dependency-graph descendants; omitted "
            "runs the complete pipeline"
        ),
    )
    p.add_argument("--to-stage", choices=STAGE_ORDER, default="environment")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=None,
        metavar="IDS",
        help=(
            "comma-separated physical GPU IDs for case-sharded CCM/mesh workers; "
            "omitting this option preserves single-process execution"
        ),
    )
    return p.parse_args()


def complete(case: Path, stage: str, depth_method: str, mesh_backend: str = "sam3d") -> bool:
    patterns = list(STAGES[stage]["patterns"])
    if stage == "mesh":
        patterns = [x.replace("{backend}", mesh_backend) for x in patterns]
    if stage == "depth":
        patterns = [x.replace("/ppd/", f"/{depth_method}/") for x in patterns]
    if stage == "scene":
        patterns = [
            x.replace("{backend}", mesh_backend).replace(
                "/ppd_depth/", f"/{depth_method}_depth/"
            )
            for x in patterns
        ]
    if not files_complete(case, patterns):
        return False
    if stage in {"ccm", "mesh"}:
        masks = list((case / "input").glob("mask_[0-9][0-9][0-9].png"))
        outputs = list(
            (case / ("CCM" if stage == "ccm" else f"mesh/{mesh_backend}")).glob(
                "voxel_coords_[0-9][0-9][0-9].npy"
                if stage == "ccm"
                else "[0-9][0-9][0-9].glb"
            )
        )
        return bool(masks) and len(outputs) == len(masks)
    return True


def migrate_legacy_meshes(case: Path) -> bool:
    """Move pre-backend-layout artifacts into the recorded backend directory.

    The old layout stored either backend directly under ``mesh/``. Infer the
    backend from the historical command when possible; old records without
    provenance predate TRELLIS.2 support and are treated as SAM3D.
    """
    mesh_root = case / "mesh"
    if not mesh_root.is_dir():
        return False
    legacy = sorted(mesh_root.glob("[0-9][0-9][0-9].glb"))
    legacy += sorted(mesh_root.glob("[0-9][0-9][0-9]_overlay.glb"))
    legacy += sorted(mesh_root.glob("[0-9][0-9][0-9].json"))
    if not legacy:
        return False
    manifest = load_manifest(case)
    stages = manifest.setdefault("stages", {})
    old_record = stages.get("mesh", {})
    old_args = " ".join(str(value) for value in old_record.get("args", []))
    backend = "trellis2" if "3_trellis2_mesh.py" in old_args else "sam3d"
    target = mesh_root / backend
    target.mkdir(parents=True, exist_ok=True)
    for source in legacy:
        destination = target / source.name
        if destination.exists():
            source.unlink()
        else:
            shutil.move(str(source), str(destination))
    record_key = f"mesh_{backend}"
    if record_key not in stages and "mesh" in stages:
        stages[record_key] = stages.pop("mesh")
    atomic_write_json(case / "stage_manifest.json", manifest)
    return True


def stage_record_key(stage: str, mesh_backend: str) -> str:
    """Use separate records for backend-specific mesh and scene artifacts."""
    return f"{stage}_{mesh_backend}" if stage in {"mesh", "scene"} else stage


def repo_commit(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def common_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpaths = [str(HERE)]
    for key, suffix in (
        ("external.mira_ccm.repo", "src"),
        ("external.unidataset.repo", "src"),
    ):
        path = path_value(config, key)
        if path:
            pythonpaths.append(str(path / suffix))
    env["PYTHONPATH"] = os.pathsep.join(
        pythonpaths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    ppd = path_value(config, "external.ppd.repo")
    moge = path_value(config, "external.moge.repo")
    if ppd:
        env["MIRA_PPD_ROOT"] = str(ppd)
    if moge:
        env["MIRA_MOGE2_ROOT"] = str(moge)
    ck = path_value(config, "checkpoints.ppd")
    if ck:
        env["MIRA_PPD_CHECKPOINT"] = str(ck)
    for key, name in (
        ("checkpoints.ppd_moge", "MIRA_PPD_MOGE_CHECKPOINT"),
        ("checkpoints.ppd_da2", "MIRA_PPD_DA2_CHECKPOINT"),
    ):
        path = path_value(config, key)
        if path:
            env[name] = str(path)
    sam3d = path_value(config, "external.sam3d.repo")
    if sam3d:
        env["MIRA_SAM3D_ROOT"] = str(sam3d)
    return env


def stage_config_fragment(
    config: dict[str, Any],
    stage: str,
    backend: str,
    depth: str,
) -> dict[str, Any]:
    """Return only configuration that can affect one stage's outputs."""
    keys = {
        "segmentation": ["segmentation", "external.sam3", "environments.segmentation"],
        "depth": [
            "depth", "external.ppd", "external.moge", "checkpoints.ppd",
            "checkpoints.ppd_moge", "checkpoints.ppd_da2", "environments.depth",
        ],
        "ccm": [
            "external.mira_ccm", "external.unidataset", "checkpoints.ccm",
            "environments.ccm",
        ],
        "mesh": (
            [
                "mesh", "external.trellis2", "checkpoints.trellis2",
                "checkpoints.rmbg", "checkpoints.dinov3", "environments.trellis2",
            ]
            if backend == "trellis2"
            else ["mesh", "external.sam3d", "checkpoints.sam3d", "environments.mesh"]
        ),
        "floor": ["floor", "api", "environments.floor"],
        "scene": ["scene", "environments.scene"],
        "environment": ["environment", "api", "environments.environment"],
    }[stage]
    return {
        "stage": stage,
        "mesh_backend": backend if stage in {"mesh", "scene"} else None,
        "depth_method": depth if stage in {"depth", "floor", "scene"} else None,
        "values": {key: get(config, key) for key in keys},
    }


def command(
    config: dict[str, Any],
    stage: str,
    cases: list[Path],
    root: Path,
    backend: str,
    depth: str,
    force: bool,
) -> list[str]:
    py = stage_python(config, stage)
    script = HERE / STAGES[stage]["script"]
    base = [py, str(script)]
    selected = sum((["--case", c.name] for c in cases), [])
    if stage == "segmentation":
        return (
            base
            + [
                "--output",
                str(root),
                "--config",
                str(config["_config_path"]),
                *selected,
            ]
            + (["--force"] if force else [])
        )
    if stage == "depth":
        cmd = base + ["--data_dir", str(root), "--method", depth, *selected]
        if depth == "ppd":
            cmd += [
                "--ppd-checkpoint",
                str(path_value(config, "checkpoints.ppd", required=True)),
                "--ppd-moge-checkpoint",
                str(path_value(config, "checkpoints.ppd_moge", required=True)),
                "--ppd-da2-checkpoint",
                str(path_value(config, "checkpoints.ppd_da2", required=True)),
            ]
        if bool(get(config, "depth.enable_metric", False)):
            cmd.append("--enable_metric")
        if force:
            cmd.append("--force")
        return cmd
    if stage == "ccm":
        return base + [
            "--demo_dir",
            str(root),
            "--output_dir",
            str(root),
            "--ckpt_dir",
            str(path_value(config, "checkpoints.ccm", required=True)),
            "--use_cropped_condition",
            *selected,
        ]
    if stage == "mesh":
        if backend == "trellis2":
            base = [stage_python(config, "trellis2"), str(HERE / "3_trellis2_mesh.py")]
            cmd = base + [
                "--demo_dir",
                str(root),
                "--output_dir",
                str(root),
                "--trellis2",
                str(path_value(config, "checkpoints.trellis2", required=True)),
                "--trellis2-src",
                str(path_value(config, "external.trellis2.repo", required=True)),
                "--rmbg",
                str(path_value(config, "checkpoints.rmbg", required=True)),
                "--dino",
                str(path_value(config, "checkpoints.dinov3", required=True)),
                *selected,
            ]
            if force:
                cmd.append("--overwrite")
            return cmd
        return base + [
            "--demo_dir",
            str(root),
            "--output_dir",
            str(root),
            "--ckpt_dir",
            str(path_value(config, "checkpoints.sam3d", required=True)),
            *selected,
        ] + (["--with_texture_baking"]
              if bool(get(config, "mesh.bake_texture", True)) else [])
    if stage == "floor":
        cmd = base + [
            "--demo_dir",
            str(root),
            "--output_dir",
            str(root),
            "--depth_method",
            depth,
            *selected,
        ]
        if bool(get(config, "floor.generate_texture", True)):
            cmd.append("--generate_floor_texture")
        if bool(get(config, "floor.require_generated", False)):
            cmd.append("--require_generated_floor_texture")
        if force:
            cmd.append("--force")
        return cmd
    if stage == "scene":
        cmd = base + [
            "--data_dir",
            str(root),
            "--output_dir",
            str(root),
            "--depth_method",
            depth,
            "--solve_method",
            str(get(config, "scene.solve_method", "gravity_joint")),
            "--mesh-backend",
            backend,
            *selected,
        ]
        if force:
            cmd.append("--force")
        return cmd
    cmd = base + ["--data_dir", str(root), "--output_dir", str(root), *selected]
    cmd += [
        "--alignment_fov_degrees",
        str(get(config, "environment.alignment_fov_degrees", 60.0)),
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.ccm_ckpt_dir is not None:
        ccm_ckpt_dir = args.ccm_ckpt_dir.expanduser().resolve()
        if not ccm_ckpt_dir.is_dir():
            raise FileNotFoundError(f"CCM checkpoint directory does not exist: {ccm_ckpt_dir}")
        checkpoints = config.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            raise ConfigError("configuration value checkpoints must be a mapping")
        checkpoints["ccm"] = str(ccm_ckpt_dir)
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = planned_stages(args.from_stage, args.to_stage)
    excluded_forced = sorted(set(args.force_stage) - set(stages))
    if excluded_forced:
        raise ValueError(
            "--force-stage is outside the selected dependency plan: "
            + ", ".join(excluded_forced)
        )
    backend = args.mesh_backend or str(get(config, "mesh.backend", "sam3d"))
    depth = str(get(config, "depth.method", "ppd"))
    validate_for_stages(config, set(stages), backend)
    validate_interpreters(config, set(stages), backend)
    images = discover_images(args.input)
    cases = prepare_cases(images, root)
    commits = {
        key: repo_commit(path_value(config, f"external.{key}.repo"))
        for key in (
            "sam3",
            "sam3d",
            "trellis2",
            "ppd",
            "moge",
            "mira_ccm",
            "unidataset",
        )
    }
    pipeline = {
        "schema": "mira_pipeline_manifest_v1",
        "config": str(args.config.resolve()),
        "input": str(args.input.resolve()),
        "cases": [c.name for c in cases],
        "stages": stages,
        "mesh_backend": backend,
        "ccm_checkpoint": str(path_value(config, "checkpoints.ccm"))
        if "ccm" in stages
        else None,
        "gpu_ids": args.gpu_ids,
        "stage_shards": {},
        "external_commits": commits,
        "status": "preflight_ok",
    }
    atomic_write_json(root / "pipeline_manifest.json", pipeline)
    if args.preflight_only:
        return
    active = list(cases)
    any_failure = False
    pipeline_log = root / "pipeline.log"
    if not args.dry_run:
        for case in active:
            migrate_legacy_meshes(case)
    for stage in stages:
        record_key = stage_record_key(stage, backend)
        forced = stage in args.force_stage
        if forced and not args.dry_run:
            for case in active:
                invalidate_downstream(case, stage, backend)
        fragment = stage_config_fragment(config, stage, backend, depth)
        implementation = (
            HERE / "3_trellis2_mesh.py"
            if stage == "mesh" and backend == "trellis2"
            else HERE / STAGES[stage]["script"]
        )
        sigs = {
            c: signature(
                c,
                stage,
                fragment,
                implementation,
                depth_method=depth,
                mesh_backend=backend,
            )
            for c in active
        }
        pending = [
            c
            for c in active
            if forced
            or not reusable(
                c,
                record_key,
                sigs[c],
                complete(c, stage, depth, backend),
            )
        ]
        if not pending:
            continue
        shard_specs = (
            build_shard_specs(
                config,
                stage,
                pending,
                root,
                backend,
                depth,
                True,
                args.gpu_ids,
            )
            if args.gpu_ids and stage in PARALLEL_STAGES
            else []
        )
        cmd = command(config, stage, pending, root, backend, depth, True)
        if shard_specs:
            pipeline["stage_shards"][stage] = len(shard_specs)
            for spec in shard_specs:
                print(
                    f"RUN shard={spec.shard_id + 1}/{spec.num_shards} "
                    f"gpu={spec.gpu_id}",
                    " ".join(spec.command),
                )
        else:
            pipeline["stage_shards"][stage] = 1
            print("RUN", " ".join(cmd))
        if args.dry_run:
            continue
        for case in pending:
            invalidate_downstream(case, stage, backend)
            clear_stage_outputs(case, stage, depth, backend)
        started = time.time()
        stage_env = common_env(config)
        if args.gpu_ids:
            stage_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        if shard_specs:
            shard_results = run_sharded_streamed(
                shard_specs,
                env=stage_env,
                pipeline_log=pipeline_log,
                stage=stage,
                mesh_backend=backend,
            )
            case_results = {
                case: result.process
                for result in shard_results
                for case in result.spec.cases
            }
            case_gpu_ids = {
                case: result.spec.gpu_id
                for result in shard_results
                for case in result.spec.cases
            }
        else:
            command_line = " ".join(cmd)
            result = run_streamed(
                cmd,
                env=stage_env,
                logs=[
                    (pipeline_log, "a", f"\n## {stage}\n$ {command_line}\n"),
                    *[
                        (
                            case / "logs" / stage_log_name(stage, backend),
                            "w",
                            f"$ {command_line}\n",
                        )
                        for case in pending
                    ],
                ],
            )
            case_results = {case: result for case in pending}
            case_gpu_ids = {
                case: args.gpu_ids[0]
                for case in pending
                if args.gpu_ids
            }
        failed = []
        completed = []
        for case in pending:
            case_result = case_results[case]
            output = case_result.stdout or ""
            ok = complete(case, stage, depth, backend)
            update_stage(
                case,
                record_key,
                "complete" if ok else "failed",
                sigs[case],
                args=case_result.args,
                interpreter=case_result.args[0],
                started_at_epoch=started,
                completed_at_epoch=time.time(),
                duration_seconds=time.time() - started,
                dependency_versions=commits,
                checkpoint_identifiers=get(config, "checkpoints", {}),
                mesh_backend=backend if stage in {"mesh", "scene"} else None,
                gpu_id=case_gpu_ids.get(case),
                num_shards=len(shard_specs) if shard_specs else 1,
                traceback=None if ok else output[-8000:],
                returncode=case_result.returncode,
            )
            if not ok:
                failed.append(case)
            else:
                completed.append(case)
        # A stage-wide process can abort before reaching later cases. Retry the
        # incomplete cases individually so one failure does not poison the batch.
        if failed and len(pending) > 1:
            retry_failed = []
            for case in failed:
                clear_stage_outputs(case, stage, depth, backend)
                retry_cmd = command(config, stage, [case], root, backend, depth, True)
                retry_line = " ".join(retry_cmd)
                retry_env = common_env(config)
                if args.gpu_ids:
                    retry_env["CUDA_VISIBLE_DEVICES"] = str(
                        case_gpu_ids.get(case, args.gpu_ids[0])
                    )
                retry = run_streamed(
                    retry_cmd,
                    env=retry_env,
                    logs=[
                        (
                            pipeline_log,
                            "a",
                            f"\n## {stage} isolated retry\n$ {retry_line}\n",
                        ),
                        (
                            case / "logs" / stage_log_name(stage, backend),
                            "a",
                            f"\n## isolated retry\n$ {retry_line}\n",
                        ),
                    ],
                )
                retry_output = retry.stdout or ""
                ok = complete(case, stage, depth, backend)
                update_stage(
                    case,
                    record_key,
                    "complete" if ok else "failed",
                    sigs[case],
                    args=retry_cmd,
                    interpreter=retry_cmd[0],
                    dependency_versions=commits,
                    checkpoint_identifiers=get(config, "checkpoints", {}),
                    mesh_backend=backend if stage in {"mesh", "scene"} else None,
                    gpu_id=case_gpu_ids.get(case),
                    num_shards=1,
                    traceback=None if ok else retry_output[-8000:],
                    returncode=retry.returncode,
                )
                if not ok:
                    retry_failed.append(case)
            failed = retry_failed
        if failed:
            any_failure = True
            active = [c for c in active if c not in failed]
        if not active:
            break
    pipeline["status"] = "failed" if any_failure else "complete"
    pipeline["failed_cases"] = [c.name for c in cases if c not in active]
    atomic_write_json(root / "pipeline_manifest.json", pipeline)
    if any_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
