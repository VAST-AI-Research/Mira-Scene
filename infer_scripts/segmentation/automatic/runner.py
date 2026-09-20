import cv2
import argparse
import json
import os
import sys
import traceback
from datetime import datetime
import numpy as np
import pycocotools.mask as mask_utils
import re
from PIL import Image
from glob import glob
from functools import partial
from ..utils.rendering.vis import visualize_masks_on_image_cv2, save_seg_obj
from .policies.vlm_policy import (
    set_vlm_backend,
    set_vlm_client,
    analyze_scene_object_lists,
    generate_vlm_response,
    normalize_sam3_prompt,
    plan_scene_mode,
)
from .policies.missing_object_critic import (
    candidate_covered_fraction,
    discover_missing_targets,
    load_delivered_foreground_union,
    make_covered_image,
    mask_box_cxcywh,
    normalize_missing_target_sam3_prompt,
    save_candidate_evidence,
    write_manifest,
)
from ..utils.image_io import load_image
from .policies.target_cardinality import (
    classify_target_cardinality,
    is_plural_group_target,
)
from .agent.sam3_adapter import (
    call_backend_service,
)
from .agent.session import run_single_image_inference
from .agent.selection import is_plural_target as is_plural_target_legacy
from ..utils.logging import get_logger, attach_file_handler

logger = get_logger("stage1")


def classify_target_for_profile(target, object_profile):
    """Return the versioned cardinality decision used by every stage1 branch."""
    if object_profile in {"major_v5", "major_v6"}:
        return classify_target_cardinality(target)
    plural = is_plural_target_legacy(target)
    return {
        "cardinality": "plural_group" if plural else "single",
        "reason": "legacy plural-head heuristic" if plural else "legacy singular fallback",
    }


def target_is_plural_group(target, object_profile):
    if object_profile in {"major_v5", "major_v6"}:
        return is_plural_group_target(target)
    return is_plural_target_legacy(target)


def _excepthook(exc_type, exc_value, exc_tb):
    """Route uncaught exceptions through the logger so the per-image FileHandler
    captures the traceback into stage1_log_*.txt before the script exits."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Uncaught exception:\n" + tb_text)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook


def build_agent_components(args, sam3_backend=None):
    """Build the segmentation agent around the shared SAM3 backend."""
    if sam3_backend is None:
        raise RuntimeError("automatic segmentation requires an injected Sam3Backend")

    # llm_config is used only to name output files
    llm_config = {"name": args.vlm_backend}

    # Replace the vLLM-server send_generate_request with our VLM backend
    def send_generate_request(messages, _max_retries=3):
        def _convert(msgs):
            converted = []
            for msg in msgs:
                content = msg["content"]
                if isinstance(content, str):
                    converted.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
                    continue
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), str):
                        new_content.append({"type": "image", "image": Image.open(item["image"]).convert("RGB")})
                    else:
                        new_content.append(item)
                converted.append({"role": msg["role"], "content": new_content})
            return converted

        def _clean_think(text):
            """Strip <think> blocks (closed or truncated/unclosed)."""
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            if '<think>' in text:
                text = text[:text.index('<think>')]
            return text.strip()

        FORMAT_HINT = (
            "IMPORTANT: Keep <think> under 100 words. "
            "Multiple masks may be selected only after checking that every mask belongs to "
            "the original target. Never union a neighbouring, independently nameable object "
            "just because it touches the target or was returned by a broad fallback prompt. "
            "Disambiguation rule for 'X on/in Y' queries: "
            "(1) If Y is a physical container that holds X (vase, bowl, jar, cup, basket, tray, pot), "
            "use Y as the text_prompt to capture the whole unit (e.g. 'flowers in glass vase' → text_prompt='glass vase'). "
            "(2) If Y is furniture or a surface that X merely rests on (table, shelf, bookcase, desk, counter, floor), "
            "X is the actual grounding target — use X as the text_prompt (e.g. 'books on coffee table' → text_prompt='books'). "
            "You MUST end with a <tool> JSON call. Example: "
            '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'
        )

        # Detect if this is the examine_each_mask (Accept/Reject) phase by checking system prompt content
        _is_checking_phase = any(
            isinstance(m.get("content"), str) and "detail-oriented visual understanding" in m["content"]
            for m in messages if m.get("role") == "system"
        )
        _is_verifier_phase = any(
            isinstance(m.get("content"), str)
            and (
                "strict segmentation selection verifier" in m["content"]
                or "strict plural-target segmentation membership verifier" in m["content"]
            )
            for m in messages if m.get("role") == "system"
        )
        _is_plural_verifier_phase = any(
            isinstance(m.get("content"), str)
            and "strict plural-target segmentation membership verifier" in m["content"]
            for m in messages if m.get("role") == "system"
        )

        # Only inject FORMAT_HINT on Round 1 (no prior segment_phrase assistant message yet)
        _is_first_round = not any(
            m.get("role") == "assistant" and
            any(isinstance(c, dict) and "segment_phrase" in c.get("text", "") for c in (m.get("content") if isinstance(m.get("content"), list) else []))
            for m in messages
        )

        prev_response = None
        for attempt in range(_max_retries):
            converted = _convert(messages)
            if prev_response is None:
                # First attempt: inject FORMAT_HINT only in Round 1 (non-checking phase)
                if not _is_checking_phase and not _is_verifier_phase and _is_first_round:
                    converted.append({"role": "user", "content": [{"type": "text", "text": FORMAT_HINT}]})
            else:
                # Retry: feed the truncated response back to the model
                converted.append({"role": "assistant", "content": [{"type": "text", "text": prev_response}]})
                if _is_checking_phase:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off. Output ONLY your verdict now: "
                        "<verdict>Accept</verdict> or <verdict>Reject</verdict>."}]})
                elif _is_verifier_phase:
                    if _is_plural_verifier_phase:
                        retry_text = (
                            "Your previous answer did not satisfy the required plural verifier schema. "
                            "Do not output selected_candidate_ids, Markdown, or prose. Re-evaluate EVERY "
                            "proposed candidate and output ONLY one JSON object with exactly these top-level "
                            "keys: verdict, keep_ids, candidates, reason. Each candidates item MUST include "
                            "id, group_membership, instance_status, visibility, border_status, duplicate_of, "
                            "identity, confidence, and reason. Write duplicate_of:null explicitly when there "
                            "is no duplicate. Example: {\"verdict\":\"repair\",\"keep_ids\":[1],"
                            "\"candidates\":[{\"id\":1,\"group_membership\":\"yes\","
                            "\"instance_status\":\"single_instance\",\"visibility\":\"fully_visible\","
                            "\"border_status\":\"not_cropped\",\"duplicate_of\":null,"
                            "\"identity\":\"chair\",\"confidence\":\"high\","
                            "\"reason\":\"one complete chair\"}],\"reason\":\"...\"}"
                        )
                    else:
                        retry_text = "Output ONLY the requested selection-verifier JSON object now."
                    converted.append({"role": "user", "content": [{"type": "text", "text": retry_text}]})
                else:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off before the <tool> call. "
                        "Output ONLY the <tool> JSON call now, no thinking. Example: "
                        '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'}]})
            response = generate_vlm_response(converted)
            clean = _clean_think(response)
            if _is_checking_phase:
                # checking phase expects <verdict>...</verdict>; do not check <tool>
                return clean
            if _is_verifier_phase:
                # Both selection verifiers have a strict JSON contract.  GPT can
                # occasionally answer with a human-readable Markdown audit even
                # though it understood the candidates correctly; feed that answer
                # back once more instead of allowing the defensive parser to turn a
                # valid plural selection into an empty/uncertain result.
                if "{" in clean and "keep_ids" in clean and "candidates" in clean:
                    return clean
                prev_response = response
                if attempt < _max_retries - 1:
                    logger.info(
                        "    ⚠️ verifier response missing JSON fields "
                        "(attempt %s), retrying...",
                        attempt + 1,
                    )
                continue
            if "<tool>" in clean and "</tool>" in clean:
                inner = clean.split("<tool>", 1)[1].split("</tool>", 1)[0].strip()
                # Some gateway/model formats put a harmless python-tag marker
                # immediately before the JSON tool object.
                inner = re.sub(r"^(?:<\|python_tag\|>\s*)+", "", inner).strip()
                try:
                    value, _ = json.JSONDecoder().raw_decode(inner)
                except (json.JSONDecodeError, TypeError):
                    value = None
                if isinstance(value, dict) and isinstance(value.get("name"), str):
                    return clean
                prev_response = response
                if attempt < _max_retries - 1:
                    logger.info(
                        "    ⚠️ VLM tool tags did not contain a JSON object "
                        "(attempt %s), retrying...",
                        attempt + 1,
                    )
                continue
            # A few model variants omit the outer <tool> wrapper and emit the
            # JSON object after a python_tag marker.  Accept only a real object
            # with a known tool name; malformed XML remains on the retry path.
            bare = re.sub(r"^<\|python_tag\|>\s*", "", clean).strip()
            try:
                bare_value, _ = json.JSONDecoder().raw_decode(bare)
            except (json.JSONDecodeError, TypeError):
                bare_value = None
            if isinstance(bare_value, dict) and isinstance(bare_value.get("name"), str):
                return f"<tool>{bare}</tool>"
            prev_response = response
            if attempt < _max_retries - 1:
                logger.info(f"    ⚠️ VLM response missing <tool> tags (attempt {attempt+1}), retrying...")
        return clean

    call_sam_service = partial(call_backend_service, sam3_backend=sam3_backend)

    return llm_config, send_generate_request, call_sam_service


def decode_agent_masks(output_json_path):
    """Decode the RLE mask from the agent's JSON output into a numpy array."""
    with open(output_json_path, 'r') as f:
        pred = json.load(f)

    h = pred["orig_img_h"]
    w = pred["orig_img_w"]
    rle_masks = pred.get("pred_masks", [])
    scores = pred.get("pred_scores", [])

    masks = []
    valid_scores = []
    for i, rle_str in enumerate(rle_masks):
        rle = {"counts": rle_str, "size": [h, w]}
        binary_mask = mask_utils.decode(rle).astype(np.float32)
        masks.append(binary_mask)
        if i < len(scores):
            valid_scores.append(scores[i])

    if masks:
        masks = np.stack(masks)
    else:
        masks = np.zeros((0, h, w), dtype=np.float32)

    return masks, np.array(valid_scores)


CAPTION_AUDIT_PROMPT = """You are the final mask-caption consistency auditor for an image segmentation pipeline.

The original image and a mask-only image are provided. In the mask-only image, pixels outside
the candidate mask are black; judge the candidate from the visible pixels inside the mask, not
from the bounding box. The requested caption is only a hypothesis and may be too broad.

Scene mode: {scene_mode}
Requested caption: {requested_caption}

For ROOM mode, a compound such as "sofa with pillows", "bed with bedding", or "pot with plant"
is valid only when the mask visibly contains the host and the assigned component. Do not include
a neighboring independently nameable object merely because it touches the host.
For TABLETOP mode, one mask must describe one independently nameable physical object. Never keep
a container-plus-contents "with" caption, and never union neighboring tabletop objects. A location
phrase such as "inside the pen holder" or "on the desk" identifies the object but is not included
in its pixels.

Return JSON only, with no Markdown:
{{"caption_status":"exact|partial|overmerged|wrong", "final_caption":"short noun phrase", "included_components":["..."], "missing_components":["..."], "extra_instances":["..."], "action":"keep|rescue_components|downgrade_caption|split_or_drop"}}
"""


def _caption_safe(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip("`\"'")
    return text[:96].strip(" ,.;:") or "unknown object"


def _tabletop_head_caption(text):
    """Conservatively remove an accidental compound from a TABLETOP caption."""
    value = re.split(
        r"\b(?:with|and|inside|in|on|near|beside|behind|under|above)\b",
        str(text or ""), maxsplit=1, flags=re.IGNORECASE,
    )[0].strip(" ,.;:")
    return _caption_safe(value)


def _caption_audit_parse(response, requested, scene_mode):
    raw = str(response or "").strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                   flags=re.IGNORECASE | re.DOTALL).strip()
    parsed = None
    try:
        parsed = json.loads(clean)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return {
            "caption_status": "audit_error",
            "final_caption": _tabletop_head_caption(requested) if scene_mode == "TABLETOP" else _caption_safe(requested),
            "included_components": [], "missing_components": [], "extra_instances": [],
            "action": "keep", "raw_response": raw,
        }
    status = str(parsed.get("caption_status", "partial")).strip().lower()
    if status not in {"exact", "partial", "overmerged", "wrong"}:
        status = "partial"
    final_caption = _caption_safe(parsed.get("final_caption") or requested)
    if scene_mode == "TABLETOP":
        # The tabletop contract is stronger than a model's free-form caption.
        final_caption = _tabletop_head_caption(final_caption)
    action = str(parsed.get("action", "keep")).strip().lower()
    if action not in {"keep", "rescue_components", "downgrade_caption", "split_or_drop"}:
        action = "keep"
    if scene_mode == "TABLETOP" and action == "rescue_components":
        action = "downgrade_caption"
    return {
        "caption_status": status,
        "final_caption": final_caption,
        "included_components": parsed.get("included_components", []) if isinstance(parsed.get("included_components", []), list) else [],
        "missing_components": parsed.get("missing_components", []) if isinstance(parsed.get("missing_components", []), list) else [],
        "extra_instances": parsed.get("extra_instances", []) if isinstance(parsed.get("extra_instances", []), list) else [],
        "action": action,
        "raw_response": raw,
    }


def _run_caption_audit_for_mask(image, mask, requested_caption, scene_mode, audit_dir):
    """Run and persist one final caption audit for a saved mask."""
    os.makedirs(audit_dir, exist_ok=True)
    image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    binary = np.asarray(mask) > 0.5
    raw_path = os.path.join(audit_dir, "raw.png")
    mask_path = os.path.join(audit_dir, "mask_only.png")
    image.save(raw_path)
    rgb = np.asarray(image).copy()
    rgb[~binary] = 0
    Image.fromarray(rgb).save(mask_path)
    prompt = CAPTION_AUDIT_PROMPT.format(
        scene_mode=scene_mode,
        requested_caption=requested_caption,
    )
    with open(os.path.join(audit_dir, "prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(prompt)
    messages = [{
        "role": "system",
        "content": "Return only the JSON object requested by the user. Do not describe your process.",
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt + "\n\nIMAGE 1: original scene."},
            {"type": "image", "image": image},
            {"type": "text", "text": "IMAGE 2: mask-only evidence."},
            {"type": "image", "image": Image.open(mask_path).convert("RGB")},
        ],
    }]
    try:
        response = generate_vlm_response(messages, audit_dir, "response")
        result = _caption_audit_parse(response, requested_caption, scene_mode)
    except Exception as exc:
        response = f"{type(exc).__name__}: {exc}"
        result = _caption_audit_parse("", requested_caption, scene_mode)
        result["caption_status"] = "audit_error"
        result["error"] = response
    result.update({
        "requested_caption": requested_caption,
        "scene_mode": scene_mode,
        "raw_image": os.path.relpath(raw_path, audit_dir),
        "mask_only_image": os.path.relpath(mask_path, audit_dir),
    })
    with open(os.path.join(audit_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_caption_audit_pass(image_path, img_output_dir, seg_obj_dir, agent_output_dir,
                           image_stem, llm_name, scene_mode):
    """Audit every final foreground mask and rename it to a mask-faithful caption."""
    manifest_path = os.path.join(img_output_dir, "mask_manifest.json")
    if os.path.exists(manifest_path):
        try:
            existing = json.load(open(manifest_path, encoding="utf-8"))
            if existing.get("scene_mode") == scene_mode and existing.get("version") == "v6_caption_audit_v1":
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    image = Image.open(image_path).convert("RGB")
    audit_root = os.path.join(img_output_dir, "caption_audit")
    records = []
    used_names = set()
    # Recycle masks have an internal ``recycled_`` filename prefix, but their
    # canonical requested caption is persisted in recycle_manifest.json.  Use
    # that mapping so audit records and later missing-object accounting never
    # expose the bookkeeping prefix as part of the object name.
    recycle_requested = {}
    recycle_manifest_path = os.path.join(img_output_dir, "recycle_manifest.json")
    if os.path.exists(recycle_manifest_path):
        try:
            recycle_manifest = json.load(open(recycle_manifest_path, encoding="utf-8"))
            for item in recycle_manifest.get("applied", []):
                name = str(item.get("name", "")).strip()
                phrase = str(item.get("phrase", "")).strip()
                if name and phrase:
                    recycle_requested[name] = phrase
        except (OSError, json.JSONDecodeError, TypeError):
            recycle_requested = {}
    for path in sorted(glob(os.path.join(seg_obj_dir, "*.png"))):
        original_name = os.path.basename(path)
        if original_name == "the_floor.png" or original_name.startswith("the_floor_"):
            continue
        stem = os.path.splitext(original_name)[0]
        match = re.match(r"^(.*)_(\d{3})$", stem)
        if not match:
            continue
        prompt_safe, mask_index_text = match.groups()
        mask_index = int(mask_index_text)
        pred_path = os.path.join(
            agent_output_dir,
            f"{image_stem}_{prompt_safe}_agent_{llm_name}_pred.json",
        )
        requested = prompt_safe.replace("_", " ")
        requested = recycle_requested.get(stem, requested)
        source_json = None
        if os.path.exists(pred_path):
            try:
                pred = json.load(open(pred_path, encoding="utf-8"))
                requested = pred.get("canonical_target") or pred.get("text_prompt") or requested
                source_json = os.path.basename(pred_path)
            except (OSError, json.JSONDecodeError):
                pass
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        mask = rgba[..., 3] > 0
        audit_name = f"{len(records):04d}_{_caption_safe(requested).replace(' ', '_')[:64]}"
        result = _run_caption_audit_for_mask(
            image, mask, requested, scene_mode,
            os.path.join(audit_root, audit_name),
        )
        final_caption = _caption_safe(result.get("final_caption") or requested)
        final_safe = final_caption.replace("/", "_").replace(" ", "_")
        candidate_name = f"{final_safe}_{mask_index:03d}.png"
        suffix = 1
        while candidate_name in used_names or os.path.exists(os.path.join(seg_obj_dir, candidate_name)) and candidate_name != original_name:
            candidate_name = f"{final_safe}_{mask_index:03d}_v{suffix}.png"
            suffix += 1
        if candidate_name != original_name:
            os.replace(path, os.path.join(seg_obj_dir, candidate_name))
        used_names.add(candidate_name)
        records.append({
            "file": candidate_name,
            "source_file": original_name,
            "source_target": requested,
            "final_caption": final_caption,
            "source_json": source_json,
            "source_mask_index": mask_index,
            "audit": result,
        })
    manifest = {
        "version": "v6_caption_audit_v1",
        "scene_mode": scene_mode,
        "masks": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def analyze_scene_tree(image_path, seg_obj_dir, agent_output_dir, image_stem, llm_name, save_dir, save_debug=False):
    """
    After segmentation, build the scene tree by calling the VLM per object on SAM3 viz overlays.
    The object list is taken strictly from filenames under segemented_obj/ (per mask id), skipping the_floor_*.

    Args:
        image_path: path to the source image
        seg_obj_dir: segemented_obj/ directory defining the full object list
        agent_output_dir: directory of agent outputs (contains *_pred.json)
        image_stem: image filename without extension
        llm_name: VLM backend name (used in output filenames)
        save_dir: directory in which to save the scene tree
        save_debug: if True, persist debug_scene_tree/ on disk (per-object overlays
            and the per-message VLM input images). When False, overlays stay
            in-memory only and no debug folder is created.
    """
    debug_dir = os.path.join(save_dir, "debug_scene_tree") if save_debug else None
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    # Step 1: scan segemented_obj/ for the full per-id object list, skipping the_floor_*
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    obj_ids = []
    for f in seg_files:
        obj_id = os.path.splitext(os.path.basename(f))[0]
        if obj_id.startswith("the_floor"):
            continue
        obj_ids.append(obj_id)

    logger.info(f"\n{'='*40}")
    logger.info(f"Constructing scene tree for {len(obj_ids)} objects (from segemented_obj)...")

    # Step 2: use the final RGBA files as the source of truth. This also supports v6
    # caption-audit renames and recycled masks that do not have a matching agent JSON.
    obj_data = {}
    scene_img = Image.open(image_path).convert("RGB")
    scene_rgb = np.asarray(scene_img).copy()
    for obj_id in obj_ids:
        obj_path = os.path.join(seg_obj_dir, f"{obj_id}.png")
        if not os.path.exists(obj_path):
            logger.info(f"  Scene tree: missing final mask for '{obj_id}', skipping")
            continue
        rgba = np.asarray(Image.open(obj_path).convert("RGBA"))
        binary = rgba[..., 3] > 0
        if not binary.any():
            logger.info(f"  Scene tree: empty final mask for '{obj_id}', skipping")
            continue
        overlay_path = (
            os.path.join(debug_dir, f"overlay_{obj_id}.png")
            if debug_dir is not None else None
        )
        overlay = None
        if overlay_path is not None and os.path.exists(overlay_path):
            try:
                overlay = Image.open(overlay_path).convert("RGB")
            except Exception:
                os.remove(overlay_path)
        if overlay is None:
            tinted = scene_rgb.copy()
            color = np.asarray((255, 180, 0), dtype=np.float32)
            tinted[binary] = (0.45 * tinted[binary] + 0.55 * color).astype(np.uint8)
            overlay = Image.fromarray(tinted, mode="RGB")
            if overlay_path is not None:
                overlay.save(overlay_path)
        ys, xs = np.where(binary)
        center = (float(xs.mean()), float(ys.mean())) if len(xs) else (scene_img.width / 2.0, scene_img.height / 2.0)
        obj_data[obj_id] = {"overlay": overlay, "center": center}

    # Step 3: query the VLM per obj_id to determine its parent
    available_parents = ["floor", "wall", "ceiling", "floor-wall"] + obj_ids
    edges = []
    result_lines = []

    # Preload the_floor mask to detect the floor-copy case (and avoid self-loops)
    floor_mask = None
    floor_png = os.path.join(seg_obj_dir, "the_floor.png")
    if os.path.exists(floor_png):
        fm = np.array(Image.open(floor_png))
        floor_mask = fm[:, :, 3] > 0 if fm.ndim == 4 else fm.any(axis=-1) if fm.ndim == 3 else fm > 0

    def _iou_with_floor(obj_id):
        if floor_mask is None or obj_id not in obj_data:
            return 0.0
        obj_png = os.path.join(seg_obj_dir, f"{obj_id}.png")
        if not os.path.exists(obj_png):
            return 0.0
        om = np.array(Image.open(obj_png))
        obj_mask = om[:, :, 3] > 0 if om.ndim == 4 else om.any(axis=-1) if om.ndim == 3 else om > 0
        inter = np.logical_and(floor_mask, obj_mask).sum()
        union = np.logical_or(floor_mask, obj_mask).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    for obj_id in obj_ids:
        if obj_id not in obj_data:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on"})
            result_lines.append(f"{obj_id} -> floor | on")
            continue

        # Find nearby objects by distance
        cx, cy = obj_data[obj_id]["center"]
        distances = []
        for other in obj_ids:
            if other == obj_id or other not in obj_data:
                continue
            ox, oy = obj_data[other]["center"]
            dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
            distances.append((dist, other))
        distances.sort()
        nearby = [n for _, n in distances[:5]]

        # Build VLM message: original image + current obj overlay + nearby obj overlays
        scene_img = Image.open(image_path).convert("RGB")
        content = [
            {"type": "image", "image": scene_img},
            {"type": "text", "text": "Full scene image above.\n\n"},
            {"type": "image", "image": obj_data[obj_id]["overlay"]},
            {"type": "text", "text": f'Current object: "{obj_id}" (colored mask overlay above)\n\nNearby objects:\n'},
        ]
        for nearby_id in nearby:
            content.append({"type": "image", "image": obj_data[nearby_id]["overlay"]})
            content.append({"type": "text", "text": f'Nearby object: "{nearby_id}"\n'})

        parents_str = ", ".join(f'"{p}"' for p in available_parents if p != obj_id)
        content.append({"type": "text", "text": (
            f'\nDetermine what supports or holds "{obj_id}" in this scene.\n'
            f"The parent MUST be one of: {parents_str}\n\n"
            "Rules:\n"
            '- "on": object rests on the TOPMOST surface of the parent — the parent does not extend above the object\n'
            '- "inside": object rests on an INTERMEDIATE horizontal surface of the parent — the parent\'s structure\n'
            "  extends above the object (e.g. item on a countertop that is part of a merged cabinet system which also has upper cabinets; item stored inside a basket, box, or drawer unit).\n"
            "  Judge by looking at the WHOLE parent mask as a single object, not individual parts.\n"
            '- "attach": mounted/fixed to a surface — use "wall" or "ceiling" as parent\n'
            "  - wall attach: picture frame, window, wall shelf, wall-mounted TV\n"
            "  - ceiling attach: hanging lamp, ceiling fan\n"
            '- "hang": object is draped/hung from a rod, rail, or hook — use the rod/rail as parent\n'
            "  - e.g. curtains hang from curtain rod, coats hang from hook\n"
            "- If object sits on another object, parent is that object, not floor\n"
            "- If object rests on floor AND is fixed against/to a wall (radiator, large cabinet, built-in unit):\n"
            '  use parent "floor-wall", relation "on-attach"\n\n'
            "Type rules:\n"
            '- "fixed": immovable furniture (cabinets, shelves, radiators, built-in units, bookcases) or anything attached to wall/ceiling\n'
            '- "movable": can be picked up or pushed (chairs, cups, books, toys, etc.)\n\n'
            f"Output ONLY one line:\n"
            f"{obj_id} -> parent_name | relation | type\n"
        )})

        if debug_dir is not None:
            img_idx = 0
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    item["image"].save(os.path.join(debug_dir, f"msg_{obj_id}_{img_idx}.png"))
                    img_idx += 1

        # Self-loop guard: if the mask is identical to the_floor, force parent=floor
        if _iou_with_floor(obj_id) > 0.99:
            logger.info(f"  ⚠️  '{obj_id}' mask is identical to the_floor (floor-copy fallback) — forcing parent=floor")
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable  [floor-copy forced]")
            continue

        messages = [{"role": "user", "content": content}]
        logger.info(f"  Querying parent for: '{obj_id}'...")
        response = generate_vlm_response(messages)
        logger.info(f"    Response: {response.strip()}")

        # Parse format: obj_id -> parent | relation | type
        found = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if "->" not in line:
                continue
            parts = line.split("->")
            if len(parts) != 2:
                continue
            rest = parts[1].strip()
            fields = [f.strip() for f in rest.split("|")]
            if len(fields) >= 3:
                parent, relation, obj_type = fields[0], fields[1], fields[2]
            elif len(fields) == 2:
                parent, relation = fields[0], fields[1]
                obj_type = "movable"
            else:
                parent = fields[0]
                relation = "on"
                obj_type = "movable"
            # normalize type
            obj_type = obj_type.lower().strip()
            if obj_type not in ("fixed", "movable"):
                obj_type = "movable"
            parent = parent.strip()
            # Case-insensitive match
            parent_lower = parent.lower()
            matched = next((p for p in available_parents if p.lower() == parent_lower), None)
            if matched is None:
                logger.info(f"    ⚠️ Invalid parent '{parent}' (not in segemented_obj or roots), fallback to floor")
                parent = "floor"
                relation = "on"
                obj_type = "movable"
            else:
                parent = matched  # use canonical casing
            edges.append({"child": obj_id, "parent": parent, "relation": relation, "type": obj_type})
            result_lines.append(f"{obj_id} -> {parent} | {relation} | {obj_type}")
            found = True
            break
        if not found:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable")

    # Save aggregated scene tree as JSON
    os.makedirs(save_dir, exist_ok=True)
    scene_tree = {
        "roots": ["floor", "wall", "ceiling", "floor-wall"],
        "nodes": obj_ids,
        "edges": edges,
    }
    tree_path = os.path.join(save_dir, "scene_tree.json")
    with open(tree_path, 'w') as f:
        json.dump(scene_tree, f, indent=2, ensure_ascii=False)
    logger.info(f"Scene tree saved to {tree_path} ({len(edges)} edges)")

    return scene_tree


def _pick_most_floor_like(seg_obj_dir, image_rgb):
    """
    Called when the floor agent fails after 10 rounds. Picks the most floor-like mask
    from the existing segemented_obj/ masks, scored as x_span * bottom_pos * bbox_area_frac.
    The selected mask is COPIED as the_floor (the original PNG is kept).
    Note: the_floor.png will be identical to one object mask; dedup and scene-tree logic handle this case.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    if not seg_files:
        return None

    if hasattr(image_rgb, 'shape'):
        img_h, img_w = image_rgb.shape[:2]
    else:
        img_w, img_h = image_rgb.size

    best_score, best_name, best_mask = -1.0, None, None
    for fpath in seg_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        if name == "the_floor" or name.startswith("the_floor_"):
            continue
        m = np.array(Image.open(fpath).convert("L")) > 0
        if m.sum() == 0:
            continue
        # Use bounding box rather than pixel count: an occluded floor has sparse pixels but a wide bbox
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        x_span = (x_max - x_min + 1) / img_w       # horizontal extent: a floor should span the image
        bottom_pos = y_max / img_h                  # bbox-bottom location (lower in image = better)
        bbox_area_frac = ((x_max - x_min + 1) * (y_max - y_min + 1)) / (img_w * img_h)
        score = x_span * bottom_pos * bbox_area_frac
        if score > best_score:
            best_score, best_name, best_mask = score, name, m

    if best_name is None:
        return None

    logger.info(f"    ⚠️  floor fallback (copy): using '{best_name}' as floor (score={best_score:.2f}), original PNG kept")

    # Encode best_mask as RLE and write a synthetic pred.json
    img_h, img_w = best_mask.shape
    rle = mask_utils.encode(np.asfortranarray(best_mask.astype(np.uint8)))
    result_json = {
        "orig_img_h": img_h,
        "orig_img_w": img_w,
        "pred_boxes": [],
        "pred_masks": [rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]],
        "pred_scores": [1.0],
    }
    return result_json


def dedup_seg_masks(seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name,
                    iou_thresh=0.3, overlap_thresh=0.8):
    """
    Pairwise-deduplicate masks inside segemented_obj/:
    - if one side is the_floor, keep the_floor and drop the other;
    - otherwise keep the alphabetically-earlier filename.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in seg_files]

    def is_floor(n):
        return n == "the_floor" or n.startswith("the_floor_")

    # Load every mask
    masks = {}
    for name, fpath in zip(names, seg_files):
        m = np.array(Image.open(fpath).convert("L")) > 0
        if m.sum() > 0:
            masks[name] = m

    removed = set()
    name_list = list(masks.keys())
    for i in range(len(name_list)):
        a = name_list[i]
        if a in removed:
            continue
        for j in range(i + 1, len(name_list)):
            b = name_list[j]
            if b in removed:
                continue
            ma, mb = masks[a], masks[b]
            intersection = np.logical_and(ma, mb).sum()
            if intersection == 0:
                continue
            union = np.logical_or(ma, mb).sum()
            iou = intersection / union
            overlap_a = intersection / ma.sum()
            overlap_b = intersection / mb.sum()
            if iou > iou_thresh or max(overlap_a, overlap_b) > overlap_thresh:
                # If one is the floor, it is likely the copy-fallback case: warn but do not delete
                if is_floor(a) != is_floor(b):
                    floor_name = a if is_floor(a) else b
                    obj_name = b if is_floor(a) else a
                    logger.info(f"  ⚠️  dedup WARNING: '{obj_name}' and '{floor_name}' have high overlap "
                          f"(IoU={iou:.2f}) — floor is likely a copy fallback, keeping both")
                    continue
                # Choose which to drop: two floor variants -> later one; otherwise the alphabetically later name
                to_remove = b if not is_floor(b) else a
                if is_floor(a) and is_floor(b):
                    to_remove = b
                logger.info(f"  dedup: removing '{to_remove}' (IoU={iou:.2f}, "
                      f"overlap_a={overlap_a:.2f}, overlap_b={overlap_b:.2f})")
                removed.add(to_remove)

    if not removed:
        return

    # Delete files
    for seg_name in removed:
        seg_png = os.path.join(seg_obj_dir, f"{seg_name}.png")
        if os.path.exists(seg_png):
            os.remove(seg_png)
        for extra in [
            os.path.join(agent_output_dir, f"{image_stem}_{seg_name}_agent_{llm_name}_pred.json"),
            os.path.join(img_output_dir, "overlay", f"{seg_name}.jpg"),
        ]:
            if os.path.exists(extra):
                os.remove(extra)
    logger.info(f"  dedup removed {removed}")


def load_cached_results(img_output_dir, agent_output_dir, image_stem, llm_name):
    """
    Try to load existing outputs to skip re-running the agent.

    Returns:
        objects: list[str] or None (None means no cache hit)
        cached_objects: set of object names that already have _pred.json
    """
    # Load the existing object list
    obj_list_path = os.path.join(img_output_dir, "scene_object_lists.txt")
    if not os.path.exists(obj_list_path):
        return None, set()

    with open(obj_list_path, "r") as f:
        objects = [line.strip() for line in f if line.strip()]
    if not objects:
        return None, set()

    # Always append "the floor" in memory (not written to disk)
    if "the floor" not in objects:
        objects.append("the floor")

    # Find objects that already have *_pred.json AND a matching PNG in segemented_obj/
    seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
    cached = set()
    for name in objects:
        prompt_safe = name.replace("/", "_").replace(" ", "_")
        base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
        json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
        if not os.path.exists(json_path):
            continue
        # Also require at least one matching PNG inside segemented_obj/
        has_png = any(True for _ in glob(os.path.join(seg_obj_dir, f"{prompt_safe}_*.png")))
        if has_png:
            cached.add(name)

    return objects, cached


def _delivered_target_names(seg_obj_dir):
    names = []
    seen = set()
    for path in sorted(glob(os.path.join(seg_obj_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "the_floor" or name.startswith("the_floor_"):
            continue
        name = re.sub(r"_\d{3}$", "", name).replace("_", " ")
        if name.casefold() not in seen:
            names.append(name)
            seen.add(name.casefold())
    return names


def _rewrite_agent_prediction_masks(json_path, masks, scores, critic_metadata):
    """Make pred.json indices match the critic masks written to segemented_obj/."""
    with open(json_path, encoding="utf-8") as handle:
        prediction = json.load(handle)
    encoded = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray((np.asarray(mask) > 0.5).astype(np.uint8)))
        counts = rle["counts"]
        encoded.append(counts.decode("utf-8") if isinstance(counts, bytes) else counts)
    prediction["pred_masks"] = encoded
    prediction["pred_scores"] = [float(score) for score in scores]
    prediction["pred_boxes"] = [mask_box_cxcywh(mask) for mask in masks]
    prediction["missing_object_critic"] = critic_metadata
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(prediction, handle, indent=4, ensure_ascii=False)


def _repair_critic_prediction_boxes(manifest, agent_output_dir, image_stem, llm_name):
    """Repair early critic-v2 outputs whose selected masks lacked matching boxes."""
    repaired = 0
    for round_record in manifest.get("rounds", []):
        for target_record in round_record.get("targets", []):
            if not target_record.get("written_masks"):
                continue
            target = target_record.get("target", "")
            prompt_safe = target.replace("/", "_").replace(" ", "_")
            json_path = os.path.join(
                agent_output_dir,
                f"{image_stem}_{prompt_safe}_agent_{llm_name}_pred.json",
            )
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                continue
            try:
                with open(json_path, encoding="utf-8") as handle:
                    prediction = json.load(handle)
            except (json.JSONDecodeError, OSError):
                continue
            if len(prediction.get("pred_boxes", [])) == len(prediction.get("pred_masks", [])):
                continue
            masks, scores = decode_agent_masks(json_path)
            _rewrite_agent_prediction_masks(
                json_path,
                masks,
                scores,
                prediction.get("missing_object_critic", {
                    "version": manifest.get("version"),
                    "round": round_record.get("round"),
                    "canonical_target": target,
                }),
            )
            repaired += 1
    return repaired


def run_missing_object_critic_pass(
    image_path,
    img_output_dir,
    agent_output_dir,
    seg_obj_dir,
    image_stem,
    llm_name,
    llm_config,
    send_generate_request,
    call_sam_service,
    rounds,
    prompt_file,
    object_profile,
    max_candidate_overlap=0.20,
    save_debug=False,
):
    """Discover and segment major objects absent from the delivered mask union."""
    from . import PROMPTS_DIR

    resolved_prompt_file = prompt_file
    if not os.path.isabs(resolved_prompt_file):
        resolved_prompt_file = os.path.join(PROMPTS_DIR, resolved_prompt_file)
    manifest_path = os.path.join(img_output_dir, "critic_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "version": "masked_missing_object_critic_v2",
            "config": {
                "prompt_file": resolved_prompt_file,
                "object_profile": object_profile,
                "max_candidate_covered_fraction": max_candidate_overlap,
                "cover_color_rgb": [0, 255, 255],
            },
            "rounds": [],
        }

    repaired_predictions = _repair_critic_prediction_boxes(
        manifest, agent_output_dir, image_stem, llm_name
    )
    if repaired_predictions:
        logger.info(
            "  missing-object critic: repaired %d prediction box contract(s)",
            repaired_predictions,
        )

    completed_rounds = len(manifest.get("rounds", []))
    if completed_rounds >= rounds:
        logger.info(
            "  missing-object critic: %d requested round(s) already recorded; skipping",
            rounds,
        )
        return manifest

    raw_image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(raw_image)
    critic_root = os.path.join(img_output_dir, "missing_object_critic")
    os.makedirs(critic_root, exist_ok=True)

    for round_index in range(completed_rounds + 1, rounds + 1):
        round_dir = os.path.join(critic_root, f"round_{round_index:02d}")
        os.makedirs(round_dir, exist_ok=True)
        delivered_before = _delivered_target_names(seg_obj_dir)
        delivered_union = load_delivered_foreground_union(seg_obj_dir, raw_image.size)
        covered_image = make_covered_image(raw_image, delivered_union)
        raw_path = os.path.join(round_dir, "raw_image.png")
        covered_path = os.path.join(round_dir, "covered_image.png")
        raw_image.save(raw_path)
        covered_image.save(covered_path)

        discovery = discover_missing_targets(
            raw_image,
            covered_image,
            delivered_before,
            generate_vlm_response,
            resolved_prompt_file,
            object_profile=object_profile,
        )
        prompt_path = os.path.join(round_dir, "prompt.txt")
        response_path = os.path.join(round_dir, "response.txt")
        with open(prompt_path, "w", encoding="utf-8") as handle:
            handle.write(discovery["prompt"])
        with open(response_path, "w", encoding="utf-8") as handle:
            handle.write(discovery["response"])

        delivered_keys = {name.casefold() for name in delivered_before}
        round_record = {
            "round": round_index,
            "delivered_before": delivered_before,
            "artifacts": {
                "raw_image": os.path.relpath(raw_path, img_output_dir),
                "covered_image": os.path.relpath(covered_path, img_output_dir),
                "prompt": os.path.relpath(prompt_path, img_output_dir),
                "response": os.path.relpath(response_path, img_output_dir),
            },
            "message_layout": discovery["message_layout"],
            "prompt": discovery["prompt"],
            "response": discovery["response"],
            "parsed_targets": discovery["parsed_targets"],
            "profile_removed": discovery["profile_removed"],
            "targets": [],
        }

        accepted_this_round = 0
        for target_index, target in enumerate(discovery["targets"]):
            target_record = {
                "target": target,
                "cardinality": classify_target_for_profile(target, object_profile),
                "candidates": [],
            }
            round_record["targets"].append(target_record)
            if target.casefold() in delivered_keys:
                target_record["status"] = "already_delivered_exact_name"
                continue

            prompt_safe = target.replace("/", "_").replace(" ", "_")
            base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
            output_json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
            target_dir = os.path.join(
                round_dir, "targets", f"{target_index:02d}_{prompt_safe}"
            )
            os.makedirs(target_dir, exist_ok=True)
            try:
                critic_sam3_prompt = normalize_missing_target_sam3_prompt(target)
                sam3_prompt = normalize_sam3_prompt(critic_sam3_prompt)
                target_record["sam3_initial_prompt"] = sam3_prompt
                run_single_image_inference(
                    image_path,
                    target,
                    llm_config,
                    send_generate_request,
                    call_sam_service,
                    output_dir=agent_output_dir,
                    debug=save_debug,
                    verify_multi_selection=not target_is_plural_group(
                        target, object_profile
                    ),
                    plural_membership_verifier=(
                        object_profile in {"major_v5", "major_v6"}
                        and target_is_plural_group(target, object_profile)
                    ),
                    sam3_search_prompt=sam3_prompt,
                )
            except Exception as exc:
                target_record["status"] = "agent_error"
                target_record["error"] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "  missing-object critic agent failed for %r: %s: %s",
                    target, type(exc).__name__, exc,
                )
                continue

            if not os.path.exists(output_json_path):
                target_record["status"] = "agent_gave_up"
                continue
            if os.path.getsize(output_json_path) == 0:
                target_record["status"] = "agent_output_empty"
                logger.warning("  missing-object critic: empty pred JSON for %r; skipping", target)
                continue
            try:
                masks, scores = decode_agent_masks(output_json_path)
            except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
                target_record["status"] = "agent_output_invalid"
                target_record["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "  missing-object critic: invalid pred JSON for %r; skipping: %s",
                    target, exc,
                )
                continue
            if len(masks) == 0:
                target_record["status"] = "agent_returned_no_masks"
                continue

            if len(masks) > 1 and not target_is_plural_group(target, object_profile):
                candidate_masks = [np.any(masks > 0.5, axis=0)]
                candidate_scores = [float(scores.max()) if len(scores) else 0.0]
                source_ids = [list(range(len(masks)))]
            else:
                candidate_masks = [mask > 0.5 for mask in masks]
                candidate_scores = [
                    float(scores[index]) if index < len(scores) else 0.0
                    for index in range(len(candidate_masks))
                ]
                source_ids = [[index] for index in range(len(candidate_masks))]

            accepted_masks = []
            accepted_scores = []
            for candidate_index, (mask, score, ids) in enumerate(zip(
                candidate_masks, candidate_scores, source_ids
            )):
                overlap = candidate_covered_fraction(mask, delivered_union)
                accepted = overlap <= max_candidate_overlap
                evidence_path = os.path.join(
                    target_dir, f"candidate_{candidate_index:03d}_overlay.png"
                )
                cutout_path = os.path.join(
                    target_dir, f"candidate_{candidate_index:03d}_cutout.png"
                )
                save_candidate_evidence(raw_image, mask, evidence_path)
                save_seg_obj(image_rgb, mask, out_path=cutout_path)
                target_record["candidates"].append({
                    "candidate": candidate_index,
                    "source_mask_ids": ids,
                    "sam3_score": score,
                    "covered_fraction": overlap,
                    "threshold": max_candidate_overlap,
                    "accepted": accepted,
                    "reason": (
                        "candidate is mostly outside delivered foreground"
                        if accepted else
                        "candidate overlaps too much with delivered foreground"
                    ),
                    "overlay": os.path.relpath(evidence_path, img_output_dir),
                    "cutout": os.path.relpath(cutout_path, img_output_dir),
                })
                if accepted:
                    accepted_masks.append(mask)
                    accepted_scores.append(score)

            if not accepted_masks:
                target_record["status"] = "all_candidates_overlap_delivered"
                continue

            _rewrite_agent_prediction_masks(
                output_json_path,
                accepted_masks,
                accepted_scores,
                {
                    "version": "masked_missing_object_critic_v2",
                    "round": round_index,
                    "canonical_target": target,
                },
            )
            written = []
            for accepted_index, mask in enumerate(accepted_masks):
                output_path = os.path.join(
                    seg_obj_dir, f"{prompt_safe}_{accepted_index:03d}.png"
                )
                save_seg_obj(image_rgb, mask, out_path=output_path)
                written.append(os.path.relpath(output_path, img_output_dir))
            target_record["status"] = "accepted_before_dedup"
            target_record["written_masks"] = written
            accepted_this_round += len(accepted_masks)
            delivered_keys.add(target.casefold())

        if accepted_this_round:
            dedup_seg_masks(
                seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name
            )
        survived_this_round = 0
        for target_record in round_record["targets"]:
            written = target_record.get("written_masks", [])
            if written:
                survived = [
                    path for path in written
                    if os.path.exists(os.path.join(img_output_dir, path))
                ]
                survived_this_round += len(survived)
                target_record["survived_masks"] = survived
                target_record["status"] = (
                    "accepted" if len(survived) == len(written) else "partly_or_fully_deduped"
                )

        delivered_after_union = load_delivered_foreground_union(seg_obj_dir, raw_image.size)
        after_image = make_covered_image(raw_image, delivered_after_union)
        after_path = os.path.join(round_dir, "covered_image_after.png")
        after_image.save(after_path)
        round_record["artifacts"]["covered_image_after"] = os.path.relpath(
            after_path, img_output_dir
        )
        round_record["delivered_after"] = _delivered_target_names(seg_obj_dir)
        round_record["accepted_mask_count"] = accepted_this_round
        round_record["survived_mask_count"] = survived_this_round
        manifest.setdefault("rounds", []).append(round_record)
        write_manifest(manifest_path, manifest)
        logger.info(
            "  missing-object critic round %d: %d target(s), %d accepted mask(s)",
            round_index, len(discovery["targets"]), accepted_this_round,
        )
        if not discovery["targets"] or survived_this_round == 0:
            break

    return manifest


def main(argv=None, *, sam3_backend=None, vlm_client=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to a single image file, or a folder of images"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Path to a txt file where each line is either an image path or a folder path"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Root output directory. Stage-1 output lands at "
             "{output_folder}/{image_stem}/stage1/ (default: output)"
    )
    parser.add_argument(
        "--vlm_backend",
        type=str,
        default=os.getenv("REST3D_VLM_BACKEND", "gpt"),
        choices=["gpt", "gpt4o", "gemini", "anthropic"],
        help="VLM backend to use: gpt (OpenAI-compatible API), gemini (Google API) "
             "or anthropic (Claude Messages API)"
    )
    parser.add_argument(
        "--vlm_prompt_file",
        type=str,
        default="list_objects.txt",
        help="VLM prompt file. Relative paths resolve against the bundled prompts/; "
             "absolute paths are used as-is. (default: list_objects.txt)"
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="If set, persist the debug_scene_tree/ folder (per-object overlays + "
             "VLM input snapshots). Off by default — overlays stay in memory."
    )
    parser.add_argument(
        "--sam3_confidence",
        type=float,
        default=0.5,
        help="SAM3 mask confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no_recycle",
        action="store_true",
        help="Disable unclaimed-candidate recycle (only for an ablation).",
    )
    parser.add_argument(
        "--recycle_verifier_mode",
        default=os.getenv("REST3D_RECYCLE_VERIFIER_MODE", "identity_upgrade"),
        choices=["legacy_joint", "identity_geometry", "identity_upgrade", "shadow_compare"],
        help="Recycle VLM policy. identity_upgrade actively runs mask upgrade then identity; "
             "legacy_joint remains the rollback policy.",
    )
    parser.add_argument(
        "--object_profile",
        default=os.getenv("REST3D_OBJECT_PROFILE", "standard"),
        choices=["standard", "major", "major_v3", "major_v4", "major_v5", "major_v6"],
        help="Object granularity profile. 'major' is the strict v2 policy; 'major_v3' "
             "also admits foreground lights and substantial rocking/ride-on play objects; "
             "'major_v4' returns to an original-REST3D-style prompt with a broad "
             "importance/physical-scale granularity rule; 'major_v5' keeps that list "
             "policy and fixes plural-group routing; 'major_v6' routes each image through "
             "a scene planner and uses a separate tabletop object-list prompt.",
    )
    parser.add_argument(
        "--tabletop_vlm_prompt_file",
        default="list_objects_tabletop_v1.txt",
        help="Object-list prompt used by major_v6 TABLETOP scenes.",
    )
    parser.add_argument(
        "--missing_object_critic_rounds",
        type=int,
        default=0,
        help="Opt-in masked-image missing-object critic rounds (default: 0/off).",
    )
    parser.add_argument(
        "--missing_object_critic_prompt_file",
        default="missing_objects_major_v2.txt",
        help="Critic prompt file; relative paths resolve against the bundled prompts/.",
    )
    parser.add_argument(
        "--missing_object_critic_max_overlap",
        type=float,
        default=0.20,
        help="Maximum fraction of a critic candidate already covered by delivered foreground.",
    )
    args = parser.parse_args(argv)

    # List of (abs_image_path, output_name) tuples
    images_list = []

    if args.image_list:
        with open(args.image_list, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                logger.info(f"Warning: skipping malformed line (expected 'path, name'): {line}")
                continue
            img_path, output_name = parts
            if not os.path.isfile(img_path):
                logger.info(f"Warning: image not found, skipping: {img_path}")
                continue
            images_list.append((img_path, output_name))
    elif args.image_folder:
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        if os.path.isfile(args.image_folder):
            found = [args.image_folder]
        else:
            found = []
            for ext in image_extensions:
                found.extend(glob(os.path.join(args.image_folder, f"*{ext}")))
                found.extend(glob(os.path.join(args.image_folder, f"*{ext.upper()}")))
        for img_path in sorted(found):
            output_name = os.path.splitext(os.path.basename(img_path))[0]
            images_list.append((img_path, output_name))
    else:
        raise ValueError("Must provide either --image_folder or --image_list")

    output_root = args.output_folder
    os.makedirs(output_root, exist_ok=True)

    set_vlm_backend(args.vlm_backend)
    if vlm_client is not None:
        set_vlm_client(vlm_client)

    # Build agent components (SAM 3 segmentor + VLM backend)
    llm_config, send_generate_request, call_sam_service = build_agent_components(
        args, sam3_backend=sam3_backend
    )
    llm_name = llm_config["name"]

    for img_path, output_name in images_list:
        abs_img_path = os.path.abspath(img_path)
        image_stem = os.path.splitext(os.path.basename(img_path))[0]
        # Per-image, per-stage output: {output_root}/{image_stem}/stage1/
        img_output_dir = os.path.join(output_root, output_name, "stage1")
        agent_output_dir = os.path.join(img_output_dir, "segment_agent_out")
        seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
        os.makedirs(seg_obj_dir, exist_ok=True)

        # Capture this image's log into a timestamped stage1_log_<YYYYMMDD_HHMMSS>.txt
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_fh = attach_file_handler(logger, os.path.join(img_output_dir, f"stage1_log_{_ts}.txt"))

        logger.info(f"\n{'='*60}")
        logger.info(f"Running stage1: scene tree construction, Processing image: {img_path}")
        logger.info(f"VLM backend set to: {args.vlm_backend}")
        logger.info(f"Object profile: {args.object_profile}")

        scene_mode = "ROOM"
        list_prompt_file = args.vlm_prompt_file
        list_object_profile = args.object_profile
        if args.object_profile == "major_v6":
            mode_manifest_path = os.path.join(img_output_dir, "scene_mode_manifest.json")
            mode_manifest = None
            if os.path.exists(mode_manifest_path):
                try:
                    mode_manifest = json.load(open(mode_manifest_path, encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    mode_manifest = None
            if not mode_manifest:
                mode_manifest = plan_scene_mode(abs_img_path, save_dir=img_output_dir)
            scene_mode = str(mode_manifest.get("scene_mode", "ROOM")).upper()
            if scene_mode not in {"ROOM", "TABLETOP"}:
                scene_mode = "ROOM"
            if scene_mode == "TABLETOP":
                list_prompt_file = args.tabletop_vlm_prompt_file
                list_object_profile = "tabletop"
            else:
                list_prompt_file = "list_objects_major_v5.txt"
                list_object_profile = "major_v5"
            mode_manifest["selected_prompt_file"] = list_prompt_file
            mode_manifest["list_object_profile"] = list_object_profile
            mode_manifest["scene_output_name"] = output_name
            with open(mode_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(mode_manifest, handle, indent=2, ensure_ascii=False)

        _prompt_display = list_prompt_file if os.path.isabs(list_prompt_file) \
            else os.path.join("bundled prompts", list_prompt_file)
        logger.info(f"Scene mode: {scene_mode}")
        logger.info(f"Stage1 prompt: {_prompt_display}")
        logger.info(f"{'='*60}")

        # Lazy load: check for cached results first
        cached_objects, cached_set = load_cached_results(
            img_output_dir, agent_output_dir, image_stem, llm_name
        )

        if cached_objects and len(cached_set) == len(cached_objects):
            # All objects already have agent output; skip to scene-tree analysis
            objects = cached_objects
            logger.info(f"Loaded {len(objects)} cached objects, skipping agent segmentation")
        else:
            # Step 1: VLM generates the object list
            if cached_objects:
                objects = cached_objects
                logger.info(f"Loaded {len(objects)} objects from cache, {len(cached_set)}/{len(objects)} already segmented")
            else:
                objects = analyze_scene_object_lists(
                    abs_img_path,
                    save_dir=img_output_dir,
                    vlm_prompt_file=list_prompt_file,
                    object_profile=list_object_profile,
                )
                if not objects:
                    logger.info(f"VLM identified no objects; skipping {img_path}")
                    continue
                logger.info(f"Identified {len(objects)} objects: {objects}")

            if args.object_profile in {"major_v5", "major_v6"}:
                cardinality_records = []
                for target in objects:
                    if target == "the floor":
                        continue
                    decision = classify_target_for_profile(target, args.object_profile)
                    cardinality_records.append({"target": target, **decision})
                    logger.info(
                        "  cardinality: %r -> %s (%s)",
                        target, decision["cardinality"], decision["reason"],
                    )
                with open(
                    os.path.join(img_output_dir, "target_cardinality.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "version": "major_v5_plural_groups_v1",
                            "object_profile": args.object_profile,
                            "targets": cardinality_records,
                        },
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )

            # Load the image so we can save masks
            image = load_image(img_path, backend="cv2", image_format="bgr")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 2: run_single_image_inference for each object
            for obj_prompt in objects:
                if obj_prompt in cached_set:
                    logger.info(f"\n  Skipping '{obj_prompt}' (cached)")
                    continue
                logger.info(f"\n  Agent segmenting: '{obj_prompt}'")

                if obj_prompt == "the floor":
                    # Same VLM+SAM agent path as other objects (max 10 rounds)
                    prompt_safe_floor = "the_floor"
                    base_floor = f"{image_stem}_{prompt_safe_floor}_agent_{llm_name}"
                    floor_json_path = os.path.join(agent_output_dir, f"{base_floor}_pred.json")
                    floor_png_path = os.path.join(seg_obj_dir, "the_floor.png")

                    try:
                        run_single_image_inference(
                            abs_img_path, obj_prompt, llm_config,
                            send_generate_request, call_sam_service,
                            output_dir=agent_output_dir, debug=args.save_debug,
                        )
                    except Exception as exc:
                        # Floor has its own SAM3-synonym/copy fallback below.  A
                        # malformed VLM tool call must not abort the whole shard and
                        # prevent later scenes from running.
                        logger.error(
                            "    floor agent crashed: %s: %s; continuing with floor fallbacks",
                            type(exc).__name__, exc,
                        )

                    # Check whether the agent succeeded
                    agent_succeeded = False
                    if os.path.exists(floor_json_path):
                        masks, _ = decode_agent_masks(floor_json_path)
                        if len(masks) > 0:
                            save_seg_obj(image, masks[0], out_path=floor_png_path)
                            logger.info(f"    floor agent succeeded: saved to the_floor.png")
                            agent_succeeded = True

                    if not agent_succeeded:
                        # Floor agent failed in 10 rounds: try SAM3 synonyms first, then fall back to copy
                        FLOOR_SYNONYMS = [
                            "carpet", "rug", "floor mat", "flooring",
                            "hardwood floor", "tile floor", "ground",
                        ]
                        synonym_succeeded = False
                        for synonym in FLOOR_SYNONYMS:
                            logger.info(f"    🔄 floor synonym retry: '{synonym}'")
                            # call_sam_service builds the save path internally; reconstruct it for the check
                            prompt_safe_syn = synonym.replace("/", "_").replace(" ", "_")
                            # output_folder_path subdir name is image_path.replace("/", "-")
                            syn_sub = abs_img_path.replace("/", "-")
                            syn_json = os.path.join(agent_output_dir, "sam_synonym",
                                                    syn_sub, f"{prompt_safe_syn}.json")
                            os.makedirs(os.path.dirname(syn_json), exist_ok=True)
                            call_sam_service(
                                image_path=abs_img_path, text_prompt=synonym,
                                output_folder_path=os.path.join(agent_output_dir, "sam_synonym"),
                            )
                            if os.path.exists(syn_json):
                                syn_masks, _ = decode_agent_masks(syn_json)
                                if len(syn_masks) > 0:
                                    # Write the synonym result into floor_json_path (keep format consistent)
                                    with open(syn_json) as f:
                                        result_syn = json.load(f)
                                    result_syn["text_prompt"] = "the floor"
                                    result_syn["image_path"] = abs_img_path
                                    json.dump(result_syn, open(floor_json_path, "w"), indent=4)
                                    save_seg_obj(image, syn_masks[0], out_path=floor_png_path)
                                    logger.info(f"    ✅ floor synonym '{synonym}' succeeded: saved to the_floor.png")
                                    synonym_succeeded = True
                                    break

                        if not synonym_succeeded:
                            # All synonyms failed -> copy the most floor-like object mask
                            result_json = _pick_most_floor_like(seg_obj_dir, image)
                            if result_json is not None:
                                result_json["text_prompt"] = "the floor"
                                result_json["image_path"] = abs_img_path
                                json.dump(result_json, open(floor_json_path, "w"), indent=4)
                                masks, _ = decode_agent_masks(floor_json_path)
                                if len(masks) > 0:
                                    save_seg_obj(image, masks[0], out_path=floor_png_path)
                            else:
                                logger.info(f"    ⚠️  floor: agent+synonyms failed and no object masks available, skipping")
                    continue  # Skip Step 3: floor already handled above
                else:
                    try:
                        sam3_prompt = normalize_sam3_prompt(obj_prompt)
                        if sam3_prompt != obj_prompt:
                            logger.info(
                                f"    SAM3 initial prompt: '{sam3_prompt}' "
                                f"(canonical target: '{obj_prompt}')"
                            )
                        run_single_image_inference(
                            abs_img_path, obj_prompt, llm_config,
                            send_generate_request, call_sam_service,
                            output_dir=agent_output_dir, debug=args.save_debug,
                            verify_multi_selection=not target_is_plural_group(
                                obj_prompt, args.object_profile
                            ),
                            plural_membership_verifier=(
                                args.object_profile in {"major_v5", "major_v6"}
                                and target_is_plural_group(obj_prompt, args.object_profile)
                            ),
                            sam3_search_prompt=sam3_prompt,
                        )
                    except Exception as exc:
                        logger.error(
                            "    agent crashed on %r: %s: %s; skipping this object",
                            obj_prompt, type(exc).__name__, exc,
                        )

                # Step 3: decode the RLE mask from the agent JSON and save it
                prompt_for_filename = obj_prompt.replace("/", "_").replace(" ", "_")
                base_filename = f"{image_stem}_{prompt_for_filename}_agent_{llm_name}"
                output_json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")

                if not os.path.exists(output_json_path):
                    # Agent failed to pick a mask within 10 rounds; skip without saving a wrong result
                    logger.info(f"    Warning: agent failed to segment '{obj_prompt}'; skipping (no fallback mask written)")
                    continue

                if os.path.getsize(output_json_path) == 0:
                    logger.warning("    empty pred JSON for '%s'; skipping object", obj_prompt)
                    continue
                try:
                    masks, _ = decode_agent_masks(output_json_path)
                except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
                    logger.warning(
                        "    invalid pred JSON for '%s'; skipping object: %s",
                        obj_prompt, exc,
                    )
                    continue
                logger.info(f"    Found {len(masks)} masks")

                # A plural target represents separate instances. A singular target is
                # unioned only after the selection verifier says its masks are one object.
                if len(masks) > 1:
                    if target_is_plural_group(obj_prompt, args.object_profile):
                        logger.info("    Keeping %d separate instances for plural target", len(masks))
                    else:
                        combined = np.zeros_like(masks[0], dtype=bool)
                        for mask in masks:
                            combined |= mask > 0.5
                        masks = [combined.astype(np.float32)]
                        logger.info("    Merged verified same-target masks into 1 object")

                prompt_safe = prompt_for_filename

                vis_overlay_masks_path = os.path.join(img_output_dir, "overlay", f"{prompt_safe}.jpg")
                os.makedirs(os.path.dirname(vis_overlay_masks_path), exist_ok=True)
                visualize_masks_on_image_cv2(image, masks, out_path=vis_overlay_masks_path)

                vis_each_seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
                os.makedirs(vis_each_seg_obj_dir, exist_ok=True)
                for mask_idx, mask in enumerate(masks):
                    vis_each_seg_obj_path = os.path.join(
                        vis_each_seg_obj_dir,
                        f"{prompt_safe}_{mask_idx:03d}.png"
                    )
                    save_seg_obj(image, mask, out_path=vis_each_seg_obj_path)

        # Step 4: pairwise-deduplicate masks (floor takes priority; otherwise keep either)
        dedup_seg_masks(seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name)

        # Step 5: recover valid unclaimed SAM3 candidates before building the scene tree.
        if not args.no_recycle:
            try:
                from .recycle import apply_unclaimed_recycle
                image_rgb = np.array(Image.open(abs_img_path).convert("RGB"))
                # major_v6 is a routing profile: ROOM deliberately reuses the
                # conservative major_v5 inventory/recycle granularity, while
                # TABLETOP has its own independent-object policy.  Pass the
                # effective scene profile to recycle instead of the outer
                # router name, which has no standalone recycle policy.
                recycle_object_profile = list_object_profile
                recycled = apply_unclaimed_recycle(
                    img_output_dir,
                    image_rgb,
                    image_rgb,
                    generate_vlm_response,
                    logger=logger,
                    verifier_mode=args.recycle_verifier_mode,
                    object_profile=recycle_object_profile,
                )
                logger.info("  recycle: wrote %d recovered instances", len(recycled))
                if recycled:
                    dedup_seg_masks(
                        seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name
                    )
                    from .recycle import finalize_recycle_manifest
                    final_recycled = finalize_recycle_manifest(img_output_dir)
                    logger.info(
                        "  recycle: %d/%d recovered instances survived final dedup",
                        len(final_recycled), len(recycled),
                    )
            except Exception as exc:
                logger.error("  recycle failed: %s: %s", type(exc).__name__, exc)

        # Step 6 (opt-in): discover major foreground still absent from the delivered union.
        if args.missing_object_critic_rounds > 0:
            try:
                run_missing_object_critic_pass(
                    abs_img_path,
                    img_output_dir,
                    agent_output_dir,
                    seg_obj_dir,
                    image_stem,
                    llm_name,
                    llm_config,
                    send_generate_request,
                    call_sam_service,
                    rounds=args.missing_object_critic_rounds,
                    prompt_file=args.missing_object_critic_prompt_file,
                    object_profile=args.object_profile,
                    max_candidate_overlap=args.missing_object_critic_max_overlap,
                    save_debug=args.save_debug,
                )
            except Exception as exc:
                logger.error(
                    "  missing-object critic failed: %s: %s",
                    type(exc).__name__, exc,
                )

        # v6 audits every final foreground mask in both ROOM and TABLETOP mode. The
        # audit runs after recycle/critic so the caption contract covers the complete
        # delivered set, including masks recovered outside the first object list.
        if args.object_profile == "major_v6":
            try:
                manifest = run_caption_audit_pass(
                    abs_img_path,
                    img_output_dir,
                    seg_obj_dir,
                    agent_output_dir,
                    image_stem,
                    llm_name,
                    scene_mode,
                )
                logger.info(
                    "  caption audit: %d final foreground masks (%s)",
                    len(manifest.get("masks", [])), scene_mode,
                )
            except Exception as exc:
                logger.error(
                    "  caption audit failed: %s: %s",
                    type(exc).__name__, exc,
                )

        # Step 7: analyze the scene tree (object list comes from segemented_obj/)
        logger.info(f"\n{'='*80}")
        logger.info(f"Scene tree construction: {output_name}")
        logger.info(f"{'='*80}")
        analyze_scene_tree(abs_img_path, seg_obj_dir, agent_output_dir, image_stem, llm_name, img_output_dir, save_debug=args.save_debug)

        logger.info(f"\n{'='*40}")
        logger.info(f"Saving stage1 object masks + scene tree to {img_output_dir}")
        logger.info(f"{'='*40}")

        logger.removeHandler(_log_fh)
        _log_fh.close()


if __name__ == "__main__":
    main()
