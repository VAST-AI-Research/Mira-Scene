#!/usr/bin/env python3
"""Run the fixed BlendSwap SAM3D evaluation with GT depth.

The evaluation dataset already provides ordered instance masks, GT depth, and
GT scenes. This driver skips segmentation/depth inference, runs CCM and SAM3D
through the normal resumable pipeline, constructs a graph-free scene with GT
depth, and finally runs the scene metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
INFER_ROOT = REPO_ROOT / "infer_scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_scripts.core.config import (
    ConfigError,
    load_config,
    path_value,
    stage_python,
)
from infer_scripts.core.io import atomic_write_json, sha256_file
from infer_scripts.pipeline import parse_gpu_ids


DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_output" / "blendswap_eval_sam3d_gt"
DEPTH_FILES = (
    "depth.npy",
    "camera_pts_map.npy",
    "valid_mask.npy",
    "intrinsics.npy",
    "fov_x_rad.txt",
)
FORCE_STAGES = ("ccm", "mesh", "scene")


class EvaluationError(RuntimeError):
    """Actionable evaluation preparation or execution failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="BlendSwap evaluation root containing input/, depth/gt/, and gt/ per case",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="override checkpoints.ccm from the YAML configuration",
    )
    parser.add_argument("--case", action="append", help="Exact BlendSwap case ID; repeatable")
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=FORCE_STAGES,
        default=[],
        help="Force CCM, SAM3D mesh, or graph-free scene reconstruction; repeatable",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Evaluate existing scene/sam3d/gt_depth/scene.glb files",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-2d-iou", action="store_true")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=None,
        metavar="IDS",
        help="comma-separated physical GPU IDs used to shard CCM and SAM3D cases",
    )
    args = parser.parse_args()
    if args.preflight_only and args.skip_inference:
        parser.error("--preflight-only cannot be combined with --skip-inference")
    if args.skip_inference and args.skip_eval:
        parser.error("--skip-inference and --skip-eval cannot be combined")
    return args


def resolve_ccm_checkpoint(
    config: dict,
    override: Path | None,
) -> Path:
    """Resolve the CCM checkpoint with CLI-over-YAML precedence."""
    selected = override if override is not None else path_value(config, "checkpoints.ccm")
    if selected is None:
        raise EvaluationError(
            "CCM checkpoint is not configured; pass --ckpt-dir or set checkpoints.ccm"
        )
    checkpoint = Path(selected).expanduser().resolve()
    if not checkpoint.is_dir():
        source = "--ckpt-dir" if override is not None else "checkpoints.ccm"
        raise EvaluationError(
            f"CCM checkpoint directory from {source} does not exist: {checkpoint}"
        )
    return checkpoint


def selected_cases(data_dir: Path, requested: Iterable[str] | None) -> list[Path]:
    if not data_dir.is_dir():
        raise EvaluationError(f"BlendSwap data directory does not exist: {data_dir}")
    if requested:
        names = list(dict.fromkeys(requested))
        cases = [data_dir / name for name in names]
        missing = [path.name for path in cases if not path.is_dir()]
        if missing:
            raise EvaluationError("unknown BlendSwap case(s): " + ", ".join(missing))
        return cases
    cases = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not cases:
        raise EvaluationError(f"no cases found under {data_dir}")
    return cases


def require_files(case_dir: Path, paths: Iterable[Path]) -> None:
    missing = [str(path.relative_to(case_dir)) for path in paths if not path.is_file()]
    if missing:
        raise EvaluationError(f"{case_dir.name}: missing required file(s): {', '.join(missing)}")


def mask_sources(case_dir: Path) -> list[Path]:
    paths = sorted((case_dir / "input").glob("mask_[0-9][0-9][0-9].png"))
    expected = [f"mask_{index:03d}.png" for index in range(len(paths))]
    if not paths:
        raise EvaluationError(f"{case_dir.name}: no indexed evaluation masks found")
    if [path.name for path in paths] != expected:
        raise EvaluationError(
            f"{case_dir.name}: masks must be contiguous from mask_000.png; "
            f"found {[path.name for path in paths]}"
        )
    return paths


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.eval-prepare.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def prepare_eval_cases(
    data_dir: Path,
    output_dir: Path,
    requested: Iterable[str] | None = None,
) -> tuple[list[str], Path]:
    """Validate BlendSwap inputs and prepare only the CCM/SAM3D case inputs."""
    cases = selected_cases(data_dir, requested)
    staging = output_dir / ".eval_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    selected_names = {case.name for case in cases}
    for old in staging.iterdir():
        if (old.is_symlink() or old.is_file()) and old.stem not in selected_names:
            old.unlink()

    prepared: list[str] = []
    for source_case in cases:
        scene = source_case / "input" / "scene.png"
        depth = source_case / "depth" / "gt"
        require_files(
            source_case,
            [
                scene,
                source_case / "gt" / "scene_camera.glb",
                source_case / "gt" / "camera.json",
                *(depth / name for name in DEPTH_FILES),
            ],
        )
        masks = mask_sources(source_case)

        target_case = output_dir / source_case.name
        target_input = target_case / "input"
        target_input.mkdir(parents=True, exist_ok=True)
        for stale in target_input.glob("mask_[0-9][0-9][0-9].png"):
            stale.unlink()
        for stale_name in ("mask.png", "floor_mask.png", "scene_fg.png"):
            (target_input / stale_name).unlink(missing_ok=True)
        for source in [scene, *masks]:
            copy_file(source, target_input / source.name)

        # Evaluation intentionally has no synthetic graph. Remove one left by
        # an older run so neither signatures nor later stages can consume it.
        (target_case / "scene_graph.json").unlink(missing_ok=True)
        atomic_write_json(
            target_case / "case.json",
            {
                "schema": "mira_case_v1",
                "case_id": source_case.name,
                "source": str(scene.resolve()),
                "source_filename": scene.name,
                "source_sha256": sha256_file(scene),
                "evaluation_data_root": str(data_dir),
                "evaluation_depth": "gt",
            },
        )
        staged_image = staging / f"{source_case.name}.png"
        if staged_image.is_symlink() or staged_image.exists():
            staged_image.unlink()
        staged_image.symlink_to(scene.resolve())
        prepared.append(source_case.name)
    return prepared, staging


def pipeline_command(args: argparse.Namespace, staging: Path) -> list[str]:
    command = [
        sys.executable,
        str(INFER_ROOT / "pipeline.py"),
        "--input",
        str(staging),
        "--output",
        str(args.output_dir),
        "--config",
        str(args.config),
        "--mesh-backend",
        "sam3d",
        "--from-stage",
        "ccm",
        "--to-stage",
        "mesh",
    ]
    if args.ckpt_dir is not None:
        command += ["--ccm-ckpt-dir", str(args.ckpt_dir)]
    for stage in dict.fromkeys(args.force_stage):
        if stage in {"ccm", "mesh"}:
            command += ["--force-stage", stage]
    if getattr(args, "gpu_ids", None):
        command += ["--gpu-ids", ",".join(map(str, args.gpu_ids))]
    if args.preflight_only:
        command.append("--preflight-only")
    return command


def scene_command(
    args: argparse.Namespace,
    scene_python: Path,
    cases: Iterable[str],
) -> list[str]:
    command = [
        str(scene_python),
        str(EVAL_ROOT / "construct_scene_eval.py"),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(args.output_dir),
    ]
    for case in cases:
        command += ["--case", case]
    if "scene" in args.force_stage:
        command.append("--force")
    if args.preflight_only:
        command.append("--preflight-only")
    return command


def evaluation_command(
    args: argparse.Namespace,
    evaluation_python: Path,
    evaluation_dir: Path,
    cases: Iterable[str],
) -> list[str]:
    command = [
        str(evaluation_python),
        str(EVAL_ROOT / "eval_scene.py"),
        "--pred_dir",
        str(args.output_dir),
        "--gt_dir",
        str(args.data_dir),
        "--pred_scene_file",
        "scene/sam3d/gt_depth/scene.glb",
        "--device",
        args.eval_device,
        "--output_csv",
        str(evaluation_dir / "eval_scene_results.csv"),
        "--per_object_csv",
        str(evaluation_dir / "eval_scene_results_per_object.csv"),
    ]
    if args.no_2d_iou:
        command.append("--no_2d_iou")
    for case in cases:
        command += ["--scene", case]
    return command


def run_logged(
    command: list[str],
    log_path: Path,
    gpu_ids: list[int] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        child_env = os.environ.copy()
        if gpu_ids:
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise EvaluationError(
            f"command failed with exit code {returncode}: {shlex.join(command)}"
        )


def validate_scenes(output_dir: Path, cases: Iterable[str]) -> None:
    invalid = []
    for case in cases:
        directory = output_dir / case / "scene" / "sam3d" / "gt_depth"
        scene_path = directory / "scene.glb"
        metadata_path = directory / "scene_optimization.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            metadata = {}
        if (
            not scene_path.is_file()
            or scene_path.stat().st_size == 0
            or metadata.get("schema") != "mira_evaluation_scene_v1"
            or metadata.get("mode") != "graph_free_evaluation"
            or metadata.get("mesh_backend") != "sam3d"
            or metadata.get("depth_method") != "gt"
            or metadata.get("solve_method") != "joint"
        ):
            invalid.append(case)
    if invalid:
        raise EvaluationError(
            "missing or incompatible graph-free GT scene(s): " + ", ".join(invalid)
        )


def validate_evaluation_csv(path: Path) -> int:
    if not path.is_file():
        raise EvaluationError(f"evaluation produced no result CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("scene_name") != "AVERAGE"]
    if not rows:
        raise EvaluationError(f"evaluation produced no valid scene rows: {path}")
    return len(rows)


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if not args.data_dir.is_dir():
        raise EvaluationError(f"BlendSwap data directory does not exist: {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.skip_inference:
        selected_checkpoint = args.ckpt_dir or path_value(config, "checkpoints.ccm")
        args.ckpt_dir = (
            Path(selected_checkpoint).expanduser().resolve()
            if selected_checkpoint is not None
            else None
        )
    else:
        args.ckpt_dir = resolve_ccm_checkpoint(config, args.ckpt_dir)
    cases, staging = prepare_eval_cases(args.data_dir, args.output_dir, args.case)

    scene_python = Path(stage_python(config, "scene")).expanduser().resolve()
    evaluation_python = Path(stage_python(config, "ccm")).expanduser().resolve()
    pipeline = pipeline_command(args, staging)
    scene = scene_command(args, scene_python, cases)
    evaluation_dir = args.output_dir / "evaluation" / "sam3d_gt"
    evaluation = evaluation_command(args, evaluation_python, evaluation_dir, cases)
    metadata = {
        "schema": "mira_evaluation_run_v2",
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "ccm_checkpoint": str(args.ckpt_dir) if args.ckpt_dir is not None else None,
        "mesh_backend": "sam3d",
        "depth_method": "gt",
        "gpu_ids": args.gpu_ids,
        "scene_mode": "graph_free_legacy_joint",
        "cases": cases,
        "pipeline_command": pipeline,
        "scene_command": scene,
        "evaluation_command": evaluation,
        "status": "prepared",
    }
    metadata_path = evaluation_dir / "run.json"
    atomic_write_json(metadata_path, metadata)

    print(f"Prepared {len(cases)} BlendSwap evaluation case(s)")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    if args.dry_run:
        print("PIPELINE", shlex.join(pipeline))
        print("SCENE", shlex.join(scene))
        print("EVALUATION", shlex.join(evaluation))
        return 0

    if not args.skip_inference:
        run_logged(pipeline, evaluation_dir / "pipeline_driver.log", args.gpu_ids)
        if not scene_python.is_file():
            raise EvaluationError(f"scene Python does not exist: {scene_python}")
        run_logged(scene, evaluation_dir / "construct_scene.log", args.gpu_ids)
        if args.preflight_only:
            metadata["status"] = "preflight_ok"
            atomic_write_json(metadata_path, metadata)
            return 0
        validate_scenes(args.output_dir, cases)

    if args.skip_eval:
        metadata["status"] = "scene_complete" if not args.skip_inference else "prepared"
        atomic_write_json(metadata_path, metadata)
        return 0
    validate_scenes(args.output_dir, cases)
    if not evaluation_python.is_file():
        raise EvaluationError(f"evaluation Python does not exist: {evaluation_python}")
    for name in ("eval_scene_results.csv", "eval_scene_results_per_object.csv"):
        (evaluation_dir / name).unlink(missing_ok=True)
    run_logged(evaluation, evaluation_dir / "eval_scene.log", args.gpu_ids)
    count = validate_evaluation_csv(evaluation_dir / "eval_scene_results.csv")
    metadata["status"] = "complete"
    metadata["evaluated_scenes"] = count
    atomic_write_json(metadata_path, metadata)
    print(f"Evaluation complete: {count} scene(s), results in {evaluation_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, EvaluationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
