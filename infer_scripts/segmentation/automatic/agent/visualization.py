# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from ...utils.rendering.visualizer import Visualizer
from ...utils.rendering.zoom_in import render_zoom_in


def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """
    Unified visualization function.

    If zoom_in_index is None:
        - Render all masks in input_json (equivalent to visualize_masks_from_result_json).
        - Returns: PIL.Image

    If zoom_in_index is provided:
        - Returns two PIL.Images:
            1) Output identical to zoom_in_and_visualize(input_json, index).
            2) The same instance rendered via the general overlay using the color
               returned by (1), equivalent to calling visualize_masks_from_result_json
               on a single-mask json_i with color=color_hex.
    """
    # Common fields
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json["original_image_path"]

    # ---------- Mode A: Full-scene render ----------
    if zoom_in_index is None:
        boxes = np.array(input_json["pred_boxes"])
        rle_masks = [
            {"size": (orig_h, orig_w), "counts": rle}
            for rle in input_json["pred_masks"]
        ]
        binary_masks = [mask_utils.decode(rle) for rle in rle_masks]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        viz = Visualizer(
            img_rgb,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz.overlay_instances(
            boxes=boxes,
            masks=rle_masks,
            binary_masks=binary_masks,
            assigned_colors=None,
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_all_masks = Image.fromarray(viz.output.get_image())
        return pil_all_masks

    # ---------- Mode B: Zoom-in pair ----------
    else:
        idx = int(zoom_in_index)
        num_masks = len(input_json.get("pred_masks", []))
        if idx < 0 or idx >= num_masks:
            raise ValueError(
                f"zoom_in_index {idx} is out of range (0..{num_masks - 1})."
            )

        # (1) Replicate zoom_in_and_visualize
        object_data = {
            "labels": [{"noun_phrase": f"mask_{idx}"}],
            "segmentation": {
                "counts": input_json["pred_masks"][idx],
                "size": [orig_h, orig_w],
            },
        }
        pil_img = Image.open(img_path)
        pil_mask_i_zoomed, color_hex = render_zoom_in(
            object_data, pil_img, mask_alpha=mask_alpha
        )

        # (2) Single-instance render with the same color
        boxes_i = np.array([input_json["pred_boxes"][idx]])
        rle_i = {"size": (orig_h, orig_w), "counts": input_json["pred_masks"][idx]}
        bin_i = mask_utils.decode(rle_i)

        img_bgr_i = cv2.imread(img_path)
        if img_bgr_i is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb_i = cv2.cvtColor(img_bgr_i, cv2.COLOR_BGR2RGB)

        viz_i = Visualizer(
            img_rgb_i,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz_i.overlay_instances(
            boxes=boxes_i,
            masks=[rle_i],
            binary_masks=[bin_i],
            assigned_colors=[color_hex],
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_mask_i = Image.fromarray(viz_i.output.get_image())

        return pil_mask_i, pil_mask_i_zoomed


def visualize_selection(input_json: dict, selected_indices: list[int]):
    """Render a proposed 1-based selection as one union outline.

    This intentionally uses a single union rather than re-numbering the candidates: the
    verifier receives the usual numbered candidate board separately and this image answers
    the different question, namely whether their *combined* pixels remain one object.
    """
    if not selected_indices:
        raise ValueError("cannot visualize an empty selection")
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    valid = sorted({int(i) for i in selected_indices
                    if 1 <= int(i) <= len(input_json.get("pred_masks", []))})
    if not valid:
        raise ValueError("selection has no valid mask indices")
    rles = [{"size": (orig_h, orig_w), "counts": input_json["pred_masks"][i - 1]}
            for i in valid]
    union = np.zeros((orig_h, orig_w), dtype=bool)
    for rle in rles:
        union |= mask_utils.decode(rle).astype(bool)
    ys, xs = np.nonzero(union)
    if not len(xs):
        raise ValueError("selection decodes to an empty union")
    box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]])
    img_bgr = cv2.imread(input_json["original_image_path"])
    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {input_json['original_image_path']}"
        )
    viz = Visualizer(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    union_rle = mask_utils.encode(np.asfortranarray(union.astype(np.uint8)))
    viz.overlay_instances(
        boxes=box,
        masks=[union_rle],
        binary_masks=[union],
        assigned_colors=["#00d2e6"],
        alpha=0.22,
        label_mode="none",
    )
    return Image.fromarray(viz.output.get_image())


def visualize_mask_evidence(input_json: dict, one_based_index: int, padding: int = 12):
    """Show only a candidate's RGB pixels beside its binary silhouette.

    Candidate zooms retain background pixels inside the bounding box.  Those pixels can
    belong to a different object and are a particularly strong source of verifier errors
    in occluded indoor scenes.  This view makes mask membership unambiguous.
    """
    idx = int(one_based_index)
    if not 1 <= idx <= len(input_json.get("pred_masks", [])):
        raise ValueError(f"invalid one-based mask index: {idx}")
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    rle = {
        "size": (orig_h, orig_w),
        "counts": input_json["pred_masks"][idx - 1],
    }
    mask = mask_utils.decode(rle).astype(bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"candidate {idx} decodes to an empty mask")

    img_bgr = cv2.imread(input_json["original_image_path"])
    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {input_json['original_image_path']}"
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(orig_w, int(xs.max()) + padding + 1)
    y1 = min(orig_h, int(ys.max()) + padding + 1)
    mask_crop = mask[y0:y1, x0:x1]
    rgb_crop = img_rgb[y0:y1, x0:x1]

    cutout = np.full_like(rgb_crop, 245)
    cutout[mask_crop] = rgb_crop[mask_crop]
    silhouette = np.full_like(rgb_crop, 245)
    silhouette[mask_crop] = np.array([0, 190, 210], dtype=np.uint8)
    gap = np.full((cutout.shape[0], 8, 3), 220, dtype=np.uint8)
    return Image.fromarray(np.concatenate([cutout, gap, silhouette], axis=1))
