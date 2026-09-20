#!/usr/bin/env python3
"""Run CCM and voxel inference for prepared Mira-Scene demo cases.

Input layout::

    demo_data/
    └── <case>/
        └── input/
            ├── scene.png                 # RGB scene image
            ├── mask.png                  # optional integer instance-id map
            ├── mask_000.png              # alternative: binary instance masks
            ├── mask_001.png
            └── floor_mask.png            # ignored by CCM inference

The image loader also accepts ``input/original_image.png`` and legacy images
at the case root. Masks can be supplied as one integer id-map ``input/mask.png``
or as multiple ``input/mask*.png`` files.

Output layout (default)::

    demo_data/
    └── <case>/
        └── CCM/
            ├── canonical_coord_map.npy
            ├── canonical_coord_map_restored.npy   # with cropped condition
            ├── canonical_coord_map_restored.png   # with cropped condition
            ├── canonical_pcd_000.ply
            ├── canonical_pcd_000_overlay.ply
            ├── voxel_coords_000.npy
            ├── masks.npy
            ├── rgb_mask.png
            └── rgb_ccm_cropped.png

Pass ``--output_dir ROOT`` to write to ``ROOT/<case>/CCM`` instead.

Single-GPU example::

    DATA_DIR=demo_data
    CKPT_DIR=/path/to/ccm_pipeline
    CUDA_VISIBLE_DEVICES=0 python infer_scripts/2_inference_CCM.py \
        --demo_dir "$DATA_DIR" \
        --ckpt_dir "$CKPT_DIR" \
        --use_cropped_condition

Multi-GPU example (one independent scene shard per GPU)::

    DATA_DIR=demo_data
    CKPT_DIR=/path/to/ccm_pipeline
    NUM_GPUS=8
    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES="$SHARD_ID" python infer_scripts/2_inference_CCM.py \
            --demo_dir "$DATA_DIR" \
            --ckpt_dir "$CKPT_DIR" \
            --use_cropped_condition \
            --num_shards "$NUM_GPUS" \
            --shard_id "$SHARD_ID" &
    done
    wait

``--num_shards`` splits the globally sorted case list. Each process loads its
own model on the single GPU exposed through ``CUDA_VISIBLE_DEVICES`` and writes
to disjoint case directories, so no distributed launcher is required.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

INFER_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFER_DIR.parent
for import_path in (INFER_DIR,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.save_outputs import save_ccm_outputs


DEFAULT_DEMO_DIR = REPO_ROOT / "Mira_Scene_Demo" / "data"


def load_mask_from_dir(mask_input):
    """Load masks from an id-map PNG or a directory of mask*.png files."""
    masks = []
    if os.path.isfile(mask_input):
        id_map = np.array(Image.open(mask_input))
        unique_ids = np.unique(id_map)
        for obj_id in unique_ids[unique_ids != 0]:
            masks.append((id_map == obj_id).astype(np.float32))
    elif os.path.isdir(mask_input):
        mask_paths = sorted(
            os.path.join(mask_input, filename)
            for filename in os.listdir(mask_input)
            if filename != "floor_mask.png"
            and ((filename.startswith("mask") and filename.endswith(".png"))
                 or filename.endswith("mask.png"))
        )
        masks = [np.array(Image.open(path).convert("L")) / 255.0 for path in mask_paths]
    return masks


def discover_scene_names(demo_dir, scene_filter=None, max_cases=-1, requested=None):
    if requested:
        scene_names = list(dict.fromkeys(requested))
        missing = [
            name for name in scene_names
            if not os.path.isdir(os.path.join(demo_dir, name))
        ]
        if missing:
            raise FileNotFoundError("missing case directories: " + ", ".join(missing))
    else:
        scene_names = sorted(
            name for name in os.listdir(demo_dir)
            if os.path.isdir(os.path.join(demo_dir, name))
        )
    if scene_filter:
        scene_names = [name for name in scene_names if scene_filter in name]
    if max_cases > 0:
        scene_names = scene_names[:max_cases]
    return scene_names


def load_scenes(demo_dir, scene_names):
    """Load images and instance masks from prepared demo case folders."""
    images, masks, valid_names = [], [], []
    for name in scene_names:
        scene_dir = os.path.join(demo_dir, name)
        image_path = next(
            (os.path.join(scene_dir, candidate) for candidate in (
                "original_image.png", "scene.jpg", "scene.png",
                "input/scene.png", "input/original_image.png",
            ) if os.path.exists(os.path.join(scene_dir, candidate))),
            None,
        )
        if image_path is None:
            continue
        image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        segmentation_path = os.path.join(scene_dir, "segmentation.png")
        mask_path = os.path.join(scene_dir, "input", "mask.png")
        mask_dir = os.path.join(scene_dir, "input")
        try:
            if os.path.exists(segmentation_path):
                scene_masks = load_mask_from_dir(segmentation_path)
            elif os.path.exists(mask_path):
                scene_masks = load_mask_from_dir(mask_path)
            elif os.path.isdir(mask_dir) and any(
                filename.startswith("mask") and filename.endswith(".png")
                for filename in os.listdir(mask_dir)
            ):
                scene_masks = load_mask_from_dir(mask_dir)
            else:
                scene_masks = load_mask_from_dir(scene_dir)
        except ValueError:
            continue
        if not scene_masks:
            continue
        images.append(image)
        masks.append(scene_masks)
        valid_names.append(name)
    return images, masks, valid_names


def parse_args():
    parser = argparse.ArgumentParser(description="CCM inference for demo_data")
    parser.add_argument("--demo_dir", type=str, default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to --demo_dir, writing <case>/CCM.")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--use_cropped_condition", action="store_true")
    parser.add_argument(
        "--layout_pred_mode",
        choices=["velocity", "x0_to_v", "x0"],
        default="velocity",
        help="Layout prediction mode supported by the repository's original CCM pipeline.",
    )
    parser.add_argument("--layout_t_eps", type=float, default=1e-5)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--scene_filter", type=str, default=None)
    parser.add_argument("--case", action="append",
                        help="Exact case name; repeat as needed")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    # This is intentionally a demo-only entry point.  Reject old/wild-data
    # arguments instead of silently falling back to the default demo folder.
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards")
    return args


def main():
    args = parse_args()
    # The launcher injects configured Mira-CCM/UniDataset sources into
    # PYTHONPATH; importing here keeps --help usable without those projects.
    from miraccm.systems.shape_synthesis.data_processor.ccm_voxel import DataProcessor
    from miraccm.pipelines.shape_synthesis.pipeline_ccm_voxel import CCMVoxelPipeline
    demo_dir = os.path.abspath(os.path.expanduser(args.demo_dir))
    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(f"demo directory does not exist: {demo_dir}")
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"checkpoint directory does not exist: {args.ckpt_dir}")
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or demo_dir))
    os.makedirs(output_dir, exist_ok=True)

    all_scene_names = discover_scene_names(
        demo_dir, args.scene_filter, args.max_cases, requested=args.case
    )
    records = list(enumerate(all_scene_names))[args.shard_id::args.num_shards]
    scene_names = [name for _, name in records]
    global_scene_indices = {name: index for index, name in records}
    images, masks, valid_names = load_scenes(demo_dir, scene_names)
    print(f"Loaded {len(images)}/{len(scene_names)} valid cases from {demo_dir} "
          f"(shard {args.shard_id + 1}/{args.num_shards})")
    if not valid_names:
        print("No valid cases assigned to this shard; nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    processor = DataProcessor(use_cropped_condition=args.use_cropped_condition)
    print(f"Loading CCM pipeline from {args.ckpt_dir}")
    pipe = CCMVoxelPipeline.from_pretrained(
        args.ckpt_dir, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    layout_res = pipe.transformer.config.latent_config["layout"]["pos_embedder"]["resolution"]
    print(f"Layout resolution: {layout_res}")

    for scene_idx, (image, scene_masks, scene_name) in enumerate(
        tqdm(zip(images, masks, valid_names), total=len(images), desc="Inference")
    ):
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + global_scene_indices[scene_name]
        )
        inp = processor.prepare_inference_input(image, scene_masks, device=device)
        model_inp = {key: value.to(dtype) if isinstance(value, torch.Tensor) else value
                     for key, value in inp.items()}
        mask_src = inp["mask_cropped_1ch"] if args.use_cropped_condition else inp["mask_1ch"]
        layout_mask = F.interpolate(mask_src.float(), size=(layout_res, layout_res), mode="nearest").to(dtype)
        image_src = model_inp["image_cropped"] if args.use_cropped_condition else model_inp["image"]
        layout_image = F.interpolate(image_src.float(), size=(layout_res, layout_res), mode="bilinear", align_corners=False).to(dtype)
        output = pipe(
            image=model_inp["image"], mask=model_inp["mask"],
            image_cropped=model_inp["image_cropped"],
            mask_cropped=model_inp["mask_cropped"],
            num_inference_steps=args.num_inference_steps, resolution=args.resolution,
            guidance_scale=args.guidance_scale, keep_layout_condition_in_uncond=True,
            layout_mask_for_concat=layout_mask, layout_image_for_conv=layout_image,
            use_cropped_condition=args.use_cropped_condition,
            layout_pred_mode=args.layout_pred_mode, layout_t_eps=args.layout_t_eps,
            generator=generator,
        )
        ccm_pred = output.latent_voxel_cam_pts
        ccm_h, ccm_w = ccm_pred.shape[-2:]
        ccm_pred = ccm_pred * (F.interpolate(mask_src.float(), size=(ccm_h, ccm_w), mode="nearest") > 0.5).to(ccm_pred.dtype)
        height, width = inp["ori_image"].shape[-2:]
        ccm_upsampled = F.interpolate(ccm_pred.float(), size=(height, width), mode="bilinear", align_corners=False).clamp(-0.5, 0.5)
        ccm_upsampled = ccm_upsampled * (F.interpolate(mask_src.float(), size=(height, width), mode="nearest") > 0.5).float()
        save_ccm_outputs(
            save_dir=os.path.join(output_dir, scene_name, "CCM"), inp=inp,
            ccm_pred_masked=ccm_pred, ccm_upsampled=ccm_upsampled,
            canonical_pcds=output.pcds, voxel_coords=output.coords,
            use_cropped_condition=args.use_cropped_condition, data_processor=processor,
        )
        print(f"  [{scene_idx:03d}] {scene_name} -> {output_dir}/{scene_name}/CCM")
    print(f"\nDone. {len(valid_names)} cases processed.")


if __name__ == "__main__":
    from core.stage_logging import run_logged
    run_logged(main, "02_ccm.log", primary_root_flags=("--output_dir",),
               fallback_root_flags=("--demo_dir",))
