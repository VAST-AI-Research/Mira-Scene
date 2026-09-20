"""Utilities for the opt-in masked-image missing-object critic pass."""

import json
import os
import re
from glob import glob

import numpy as np
from PIL import Image

from .vlm_policy import filter_major_object_targets


COVER_COLOR = (0, 255, 255)

_WALL_RELATION_RE = re.compile(
    r"\s+(?:against|along)\s+(?:the\s+)?(?:rear|far|left|right|side)?\s*wall\b",
    re.IGNORECASE,
)
_LOW_VALUE_STYLE_RE = re.compile(
    r"\b(?:slatted|ribbed|fluted|paneled|panelled|tufted|ornate|decorative)\s+",
    re.IGNORECASE,
)


def load_delivered_foreground_union(seg_obj_dir, image_size=None):
    """Load the union of delivered RGBA masks, excluding all floor masks."""
    union = None
    for path in sorted(glob(os.path.join(seg_obj_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "the_floor" or name.startswith("the_floor_"):
            continue
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        mask = rgba[..., 3] > 0
        if union is None:
            union = np.zeros(mask.shape, dtype=bool)
        if mask.shape != union.shape:
            raise ValueError(
                f"mask shape mismatch in {seg_obj_dir}: {mask.shape} != {union.shape}"
            )
        union |= mask

    if union is not None:
        return union
    if image_size is None:
        raise ValueError("image_size is required when no foreground masks exist")
    width, height = image_size
    return np.zeros((height, width), dtype=bool)


def make_covered_image(image, delivered_union, color=COVER_COLOR):
    """Replace delivered foreground pixels with a solid, auditable color."""
    rgb = np.asarray(image.convert("RGB")).copy()
    if delivered_union.shape != rgb.shape[:2]:
        raise ValueError(
            f"union/image shape mismatch: {delivered_union.shape} != {rgb.shape[:2]}"
        )
    rgb[delivered_union] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def parse_missing_object_response(response):
    """Parse the critic's plain one-target-per-line response."""
    text = str(response or "").strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    if not text or text.casefold() in {"none", "no missing objects", "no missing object"}:
        return []

    targets = []
    seen = set()
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        line = line.strip("` ")
        if not line or line.casefold() in {
            "none", "no missing objects", "no missing object"
        }:
            continue
        if len(line) >= 100:
            continue
        key = line.casefold()
        if key not in seen:
            targets.append(line)
            seen.add(key)
    return targets


def candidate_covered_fraction(candidate_mask, delivered_union):
    """Return intersection(candidate, delivered) / area(candidate)."""
    candidate = np.asarray(candidate_mask) > 0.5
    if candidate.shape != delivered_union.shape:
        raise ValueError(
            f"candidate/union shape mismatch: {candidate.shape} != {delivered_union.shape}"
        )
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return float(np.logical_and(candidate, delivered_union).sum() / area)


def normalize_missing_target_sam3_prompt(target):
    """Reduce critic prose to a SAM-friendly phrase without changing identity."""
    text = re.sub(r"\s+", " ", str(target or "")).strip()
    text = _WALL_RELATION_RE.sub("", text)
    text = _LOW_VALUE_STYLE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def mask_box_cxcywh(mask):
    """Return SAM3's normalized center-x/center-y/width/height box format."""
    binary = np.asarray(mask) > 0.5
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return [0.5, 0.5, 0.0, 0.0]
    height, width = binary.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [
        (x0 + x1) / (2.0 * width),
        (y0 + y1) / (2.0 * height),
        (x1 - x0) / float(width),
        (y1 - y0) / float(height),
    ]


def build_critic_prompt(prompt_template, existing_targets):
    names = [str(name).strip() for name in existing_targets if str(name).strip()]
    target_block = "\n".join(f"- {name}" for name in names) or "- NONE"
    return prompt_template.rstrip() + "\n\nAlready delivered target names:\n" + target_block


def discover_missing_targets(
    raw_image,
    covered_image,
    existing_targets,
    generate_vlm_response,
    prompt_file,
    object_profile="major_v4",
):
    """Ask the critic for missing targets and apply the existing profile gate."""
    with open(prompt_file, encoding="utf-8") as handle:
        template = handle.read().strip()
    prompt = build_critic_prompt(template, existing_targets)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "IMAGE 1 - original scene:"},
                {"type": "image", "image": raw_image.convert("RGB")},
                {
                    "type": "text",
                    "text": "IMAGE 2 - the same scene with delivered foreground masks covered in cyan:",
                },
                {"type": "image", "image": covered_image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = generate_vlm_response(messages)
    parsed = parse_missing_object_response(response)
    if object_profile == "standard":
        targets, removed = parsed, []
    else:
        targets, removed = filter_major_object_targets(parsed, object_profile)
    return {
        "prompt": prompt,
        "response": response,
        "parsed_targets": parsed,
        "targets": targets,
        "profile_removed": removed,
        "message_layout": [
            "IMAGE 1: original scene",
            "IMAGE 2: same scene with delivered foreground masks covered in cyan",
            "TEXT: missing-object critic prompt and already delivered target names",
        ],
    }


def save_candidate_evidence(raw_image, candidate_mask, output_path):
    """Save a deterministic context view for one candidate decision."""
    rgb = np.asarray(raw_image.convert("RGB")).copy()
    mask = np.asarray(candidate_mask) > 0.5
    tint = np.asarray((255, 64, 160), dtype=np.float32)
    rgb_float = rgb.astype(np.float32)
    rgb_float[mask] = 0.45 * rgb_float[mask] + 0.55 * tint
    Image.fromarray(np.clip(rgb_float, 0, 255).astype(np.uint8)).save(output_path)


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
