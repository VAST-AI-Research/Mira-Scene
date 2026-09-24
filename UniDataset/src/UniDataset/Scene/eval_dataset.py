"""
EvalDataset for reading evaluation data in the format defined by eval_data/README.md.

Returns raw image/mask (numpy) + GT data. Preprocessing is handled by
DataProcessor.prepare_inference_input() in validation_step, ensuring
consistency with inference_CCM.py.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    """
    Evaluation dataset that reads from the standardized eval_data format.

    Expected structure per case:
        {eval_dir}/{scene_id}/input/scene.png
        {eval_dir}/{scene_id}/input/mask.png
        {eval_dir}/{scene_id}/gt/canonical_coord_map.npy  (optional)
        {eval_dir}/{scene_id}/gt/voxel.npy               (optional)

    Returns raw numpy image/mask + GT tensors. The caller (validation_step)
    is responsible for calling DataProcessor.prepare_inference_input() to
    build pipeline inputs.
    """

    def __init__(
        self,
        eval_dir: str,
        height: int = 518,
        width: int = 518,
        voxel_res: int = 64,
        split: str = "test",
        **kwargs,
    ):
        super().__init__()
        self.eval_dir = eval_dir
        self.height = height
        self.width = width
        self.voxel_res = voxel_res

        # Discover cases: directories that contain input/scene.png
        self.cases = sorted([
            d for d in os.listdir(eval_dir)
            if os.path.isfile(os.path.join(eval_dir, d, "input", "scene.png"))
        ])
        print(f"EvalDataset: {len(self.cases)} cases from {eval_dir}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene_id = self.cases[index]
        input_dir = os.path.join(self.eval_dir, scene_id, "input")
        gt_dir = os.path.join(self.eval_dir, scene_id, "gt")

        # --- Load input (raw numpy, same as inference_CCM.py) ---
        scene_img = np.array(
            Image.open(os.path.join(input_dir, "scene.png")).convert("RGB")
        ).astype(np.float32) / 255.0  # [H, W, 3]

        # --- Load masks (single or multi-instance) ---
        mask_path = os.path.join(input_dir, "mask.png")
        mask_files = sorted(glob.glob(os.path.join(input_dir, "mask_*.png")))

        if mask_files:
            # Multi-instance: mask_000.png, mask_001.png, ...
            masks = [
                (np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
                for p in mask_files
            ]
        elif os.path.exists(mask_path):
            # Single instance: mask.png
            mask_img = np.array(
                Image.open(mask_path).convert("L")
            ).astype(np.float32) / 255.0
            masks = [(mask_img > 0.5).astype(np.float32)]
        else:
            masks = []

        NI = len(masks)

        # --- Load GT (optional) ---
        gt = {}

        # GT CCM: try per-instance first, then single
        ccm_per_instance = sorted(glob.glob(os.path.join(gt_dir, "canonical_coord_map_[0-9]*.npy")))
        ccm_single = os.path.join(gt_dir, "canonical_coord_map.npy")
        if ccm_per_instance:
            gt["canonical_coord_map"] = torch.from_numpy(
                np.stack([np.load(p) for p in ccm_per_instance])
            ).float()  # [NI, 3, H, W]
        elif os.path.exists(ccm_single):
            ccm_data = np.load(ccm_single)
            if ccm_data.ndim == 3:
                ccm_data = ccm_data[None]  # [1, 3, H, W]
            gt["canonical_coord_map"] = torch.from_numpy(ccm_data).float()
        else:
            gt["canonical_coord_map"] = torch.zeros(max(NI, 1), 3, self.height, self.width)

        # GT Voxel: try per-instance first, then single
        voxel_per_instance = sorted(glob.glob(os.path.join(gt_dir, "voxel_[0-9]*.npy")))
        voxel_single = os.path.join(gt_dir, "voxel.npy")
        if voxel_per_instance:
            gt["voxel"] = torch.from_numpy(
                np.stack([np.load(p) for p in voxel_per_instance])
            ).long()  # [NI, R, R, R]
        elif os.path.exists(voxel_single):
            voxel_data = np.load(voxel_single)
            if voxel_data.ndim == 3:
                voxel_data = voxel_data[None]  # [1, R, R, R]
            gt["voxel"] = torch.from_numpy(voxel_data).long()
        else:
            gt["voxel"] = torch.zeros(
                max(NI, 1), self.voxel_res, self.voxel_res, self.voxel_res, dtype=torch.long
            )

        return {
            "id": scene_id,
            "image": scene_img,         # [H, W, 3] float32 [0,1]
            "masks": masks,             # list of [H, W] float32 (length NI)
            "gt": gt,
        }

    def collate(self, batch):
        """Custom collate: keep raw numpy structure, batch as list."""
        return batch  # list of dicts, processed individually in validation_step
