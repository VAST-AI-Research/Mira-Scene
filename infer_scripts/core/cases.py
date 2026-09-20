"""Input discovery and deterministic Mira case preparation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from .io import atomic_write_json, sha256_file

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MIRA_SIZE = 518


def safe_stem(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip()).strip("._-")
    return (value or "case")[:96]


def discover_images(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported input image: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"no supported images found in {source}")
    return images


def _case_id(image: Path, digest: str, output_root: Path) -> str:
    base = safe_stem(image.stem)
    existing = output_root / base / "case.json"
    if not existing.is_file():
        return base
    import json
    try:
        record = json.loads(existing.read_text(encoding="utf-8"))
    except Exception:
        record = {}
    if record.get("source_sha256") == digest:
        return base
    return f"{base}-{digest[:8]}"


def prepare_case(
    image_path: Path,
    output_root: Path,
    *,
    source_reference: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(image_path)
    case_id = _case_id(image_path, digest, output_root)
    case_dir = output_root / case_id
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "logs").mkdir(exist_ok=True)

    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = original.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    normalized = original.crop((left, top, left + side, top + side)).resize(
        (MIRA_SIZE, MIRA_SIZE), Image.Resampling.LANCZOS
    )
    # source.png is the EXIF-corrected original; scene.png is the model input.
    original.save(input_dir / "source.png", format="PNG")
    normalized.save(input_dir / "scene.png", format="PNG")
    source = source_reference or str(image_path.resolve())
    preprocessing = {
        "schema": "mira_preprocessing_v1", "source": source,
        "source_filename": image_path.name if source_reference is None else source_reference,
        "source_sha256": digest, "original_size": [width, height],
        "crop_xywh": [left, top, side, side], "output_size": [MIRA_SIZE, MIRA_SIZE],
        "exif_transposed": True,
    }
    atomic_write_json(input_dir / "preprocessing.json", preprocessing)
    atomic_write_json(case_dir / "case.json", {
        "schema": "mira_case_v1", "case_id": case_id, "source": source,
        "source_filename": image_path.name if source_reference is None else source_reference,
        "source_sha256": digest,
    })
    return case_dir


def prepare_cases(images: Iterable[Path], output_root: Path) -> list[Path]:
    return [prepare_case(image, output_root) for image in images]
