# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from ...utils.geometry.mask_overlap_removal import remove_overlapping_masks
from .visualization import visualize


def call_backend_service(sam3_backend, image_path: str, text_prompt: str,
                         output_folder_path: str = "sam3_output"):
    """Expose Mira's external-SAM3 adapter through the frozen agent JSON contract."""
    masks, scores = sam3_backend.text(Path(image_path), text_prompt, limit=100)
    with Image.open(image_path) as opened:
        width, height = opened.size
    boxes, encoded, valid_scores = [], [], []
    for mask, score in zip(masks, scores):
        binary = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(binary)
        if not len(xs):
            continue
        boxes.append([
            float(xs.min() / width), float(ys.min() / height),
            float((xs.max() - xs.min() + 1) / width),
            float((ys.max() - ys.min() + 1) / height),
        ])
        rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
        encoded.append(rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"])
        valid_scores.append(float(score))
    sample = remove_overlapping_masks({
        "orig_img_h": height, "orig_img_w": width, "pred_boxes": boxes,
        "pred_masks": encoded, "pred_scores": valid_scores,
    })
    order = sorted(range(len(sample["pred_scores"])), key=lambda i: sample["pred_scores"][i], reverse=True)
    for key in ("pred_boxes", "pred_masks", "pred_scores"):
        sample[key] = [sample[key][i] for i in order]
    safe = text_prompt.replace("/", "_")
    subdir = os.path.join(output_folder_path, image_path.replace("/", "-"))
    os.makedirs(subdir, exist_ok=True)
    output_json_path = os.path.join(subdir, f"{safe}.json")
    output_image_path = os.path.join(subdir, f"{safe}.png")
    payload = {"original_image_path": image_path, "output_image_path": output_image_path, **sample}
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    visualize(payload).save(output_image_path)
    return output_json_path
