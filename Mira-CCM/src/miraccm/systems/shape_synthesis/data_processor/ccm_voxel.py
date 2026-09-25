"""
DataProcessor that uses canonical_coord_map as the layout representation.

Layout representation:
  - prepare_condition_info: layout condition is canonical_coord_map [B, NI, 3, H, W]
    from the dataloader.

Coordinate conventions:
  - Canonical space: Z-up, coordinates in roughly [-0.5, 0.5]
  - Camera space:    OpenGL (+X right, +Y up, -Z forward)

All outputs from prepare_condition_info are shaped [B*NI, ...], so the system
code can call the same visualization helpers.
"""

from torchvision import transforms
from einops import rearrange
import os
import PIL
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from UniDataset.utils.pcd_utils import (
    voxels_to_pcd,
    compute_similarity_transform,
    generate_uniform_voxel_centers,
    sample_points_from_bbox,
    compute_similarity_transform_from_bbox,
)
from miraccm.utils.image_utils.segment import masks2idmap
from miraccm.utils.system_utils.logging import debug, info, warn
import trimesh
import cv2

from UniDataset.utils.img_and_mask_transforms import crop_around_mask_with_padding

from .utils import transform_pcd, downsample_cube_pcd, transform_pcd_simple


class DataProcessor:
    def __init__(
        self,
        voxel_res: int = 64,
        use_cropped_condition: bool = False,
    ):
        self.image_transforms = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(518),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(518),
        ])
        self.color_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.voxel_res = voxel_res

        self.box_size_factor = 1.2
        self.cropped_img_transform = transforms.Compose([
            transforms.Resize((518, 518), antialias=True),
        ])

        # Color palette for visualization (24 labels)
        self.COLORS = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 255, 255],
            [0, 255, 255, 255],
            [128, 0, 0, 255],
            [0, 128, 0, 255],
            [0, 0, 128, 255],
            [128, 128, 0, 255],
            [128, 0, 128, 255],
            [0, 128, 128, 255],
            [64, 0, 0, 255],
            [0, 64, 0, 255],
            [0, 0, 64, 255],
            [64, 64, 0, 255],
            [64, 0, 64, 255],
            [0, 64, 64, 255],
            [192, 192, 192, 255],
            [128, 128, 128, 255],
            [255, 165, 0, 255],
            [75, 0, 130, 255],
            [238, 130, 238, 255],
        ]

        self.use_cropped_condition = use_cropped_condition

    # ------------------------------------------------------------------
    # Unified inference input preparation
    # ------------------------------------------------------------------

    def prepare_inference_input(self, image, masks, device="cuda"):
        """Build pipeline input dict from raw image + masks.

        This is the single source of truth for preprocessing. Both
        inference_CCM.py and validation_step should call this method.

        Args:
            image: [H, W, 3] numpy float32 in [0, 1]
            masks: list of [H, W] numpy float32 (one per instance, 1=fg)
            device: target device string

        Returns:
            dict ready to pass to pipeline / system.inference():
                image, ori_image, image_cropped, ori_image_cropped,
                mask, mask_1ch, mask_cropped, mask_cropped_1ch,
                crop_params, NI
        """
        from UniDataset.utils.img_and_mask_transforms import crop_around_mask

        NI = len(masks)
        mask_tensor = torch.from_numpy(np.stack(masks))[:, None]  # [NI, 1, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None].expand(NI, -1, -1, -1).clone()

        image_tensor = self.image_transforms(image_tensor)
        mask_tensor = self.mask_transforms(mask_tensor)
        part_image = image_tensor * mask_tensor

        cropped_rgb_list, cropped_mask_list, crop_params = [], [], []
        for i in range(NI):
            cr, cm, cp = crop_around_mask(
                part_image[i], mask_tensor[i],
                box_size_factor=self.box_size_factor,
                target_h=518, target_w=518,
            )
            cropped_rgb_list.append(cr)
            cropped_mask_list.append(cm)
            crop_params.append(cp)

        cropped_rgb = torch.stack(cropped_rgb_list).to(device)
        cropped_mask = torch.stack(cropped_mask_list).to(device)
        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        ori_image = image_tensor
        image_norm = self.color_transforms(image_tensor)
        mask_3ch = self.mask_transforms(mask_tensor.expand(-1, 3, -1, -1))
        image_crop_norm = self.color_transforms(cropped_rgb)
        mask_crop_3ch = self.mask_transforms(cropped_mask.expand(-1, 3, -1, -1))

        return {
            "image": image_norm,
            "ori_image": ori_image,
            "image_cropped": image_crop_norm,
            "ori_image_cropped": cropped_rgb,
            "mask": mask_3ch,
            "mask_1ch": mask_3ch[:, :1],
            "mask_cropped": mask_crop_3ch,
            "mask_cropped_1ch": mask_crop_3ch[:, :1],
            "crop_params": crop_params,
            "NI": NI,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_crop_params(
        crop_params,
        batch_size: int,
        num_instances: int,
        device: torch.device,
        default_spatial_size=None,
    ):
        """Normalize crop params into a flat tensor dict on the target device."""
        if crop_params is None:
            return None

        expected = batch_size * num_instances
        default_h = default_w = None
        if default_spatial_size is not None:
            default_h = int(default_spatial_size[0])
            default_w = int(default_spatial_size[1])

        required_keys = ("top", "left", "bottom", "right")
        entries = []

        def _normalize_entry(param):
            if not isinstance(param, dict):
                raise TypeError(f"crop_params entry must be a dict, got {type(param)}")

            missing_keys = [key for key in required_keys if key not in param]
            if missing_keys:
                raise KeyError(f"crop_params is missing required keys: {missing_keys}")

            orig_h = int(param.get("orig_h", default_h))
            orig_w = int(param.get("orig_w", default_w))
            if orig_h is None or orig_w is None:
                raise KeyError(
                    "crop_params must include 'orig_h'/'orig_w', or a default "
                    "spatial size must be provided."
                )

            return {
                "top": int(param["top"]),
                "left": int(param["left"]),
                "bottom": int(param["bottom"]),
                "right": int(param["right"]),
                "orig_h": orig_h,
                "orig_w": orig_w,
                "pad_h": int(param.get("pad_h", 0)),
                "pad_w": int(param.get("pad_w", 0)),
                "pad_h_extra": int(param.get("pad_h_extra", 0)),
                "pad_w_extra": int(param.get("pad_w_extra", 0)),
            }

        if isinstance(crop_params, dict):
            if all(torch.is_tensor(value) for value in crop_params.values()):
                count = int(crop_params["top"].numel())
                for idx in range(count):
                    entries.append(
                        _normalize_entry(
                            {
                                key: (
                                    crop_params[key].reshape(-1)[idx].item()
                                    if key in crop_params
                                    else None
                                )
                                for key in ("top", "left", "bottom", "right", "orig_h", "orig_w", "pad_h", "pad_w", "pad_h_extra", "pad_w_extra")
                            }
                        )
                    )
            else:
                entries.append(_normalize_entry(crop_params))
        elif isinstance(crop_params, (list, tuple)):
            for item in crop_params:
                if isinstance(item, dict):
                    entries.append(_normalize_entry(item))
                elif isinstance(item, (list, tuple)):
                    for sub_item in item:
                        entries.append(_normalize_entry(sub_item))
                else:
                    raise TypeError(f"Unsupported crop_params entry type: {type(item)}")
        else:
            raise TypeError(f"Unsupported crop_params container type: {type(crop_params)}")

        if len(entries) == batch_size and num_instances > 1:
            entries = [entry.copy() for entry in entries for _ in range(num_instances)]

        if len(entries) != expected:
            raise ValueError(
                f"crop_params count mismatch: got {len(entries)}, expected {expected} "
                f"(batch_size={batch_size}, num_instances={num_instances})"
            )

        return {
            key: torch.tensor(
                [entry[key] for entry in entries], dtype=torch.long, device=device
            )
            for key in ("top", "left", "bottom", "right", "orig_h", "orig_w", "pad_h", "pad_w", "pad_h_extra", "pad_w_extra")
        }

    @staticmethod
    def restore_canonical_coord_map(
        canonical_coord_map_cropped: torch.Tensor,
        crop_params,
    ) -> torch.Tensor:
        """Restore cropped canonical_coord_map tensors to the full image canvas.

        Args:
            canonical_coord_map_cropped: [N, 3, Hc, Wc] cropped-and-resized CCM.
            crop_params: crop metadata. Prefer the normalized tensor dict from
                ``processed_batch["crop_params"]``. Raw dict/list inputs are also
                accepted if they include ``orig_h`` and ``orig_w``.

        Returns:
            canonical_coord_map: [N, 3, H, W] with zeros outside the crop box.
        """
        if canonical_coord_map_cropped is None:
            return None

        if canonical_coord_map_cropped.ndim != 4:
            raise ValueError(
                "canonical_coord_map_cropped must have shape [N, 3, H, W], "
                f"got {tuple(canonical_coord_map_cropped.shape)}"
            )

        num_maps = canonical_coord_map_cropped.shape[0]
        if not (
            isinstance(crop_params, dict)
            and all(torch.is_tensor(value) for value in crop_params.values())
        ):
            crop_params = DataProcessor._prepare_crop_params(
                crop_params=crop_params,
                batch_size=num_maps,
                num_instances=1,
                device=canonical_coord_map_cropped.device,
            )

        required_keys = ("top", "left", "bottom", "right", "orig_h", "orig_w")
        missing_keys = [key for key in required_keys if key not in crop_params]
        if missing_keys:
            raise KeyError(
                f"crop_params is missing required keys for CCM restoration: {missing_keys}"
            )
        for key in required_keys:
            if crop_params[key].numel() != num_maps:
                raise ValueError(
                    f"crop_params['{key}'] contains {crop_params[key].numel()} "
                    f"entries, but {num_maps} CCM maps were provided."
                )

        output_sizes = {
            (int(crop_params["orig_h"][idx].item()), int(crop_params["orig_w"][idx].item()))
            for idx in range(num_maps)
        }
        if len(output_sizes) != 1:
            raise ValueError(
                "All CCMs in one tensor must restore to the same full-image size; "
                f"got {sorted(output_sizes)}."
            )

        restored = []
        for idx in range(num_maps):
            top = int(crop_params["top"][idx].item())
            left = int(crop_params["left"][idx].item())
            bottom = int(crop_params["bottom"][idx].item())
            right = int(crop_params["right"][idx].item())
            orig_h = int(crop_params["orig_h"][idx].item())
            orig_w = int(crop_params["orig_w"][idx].item())

            if not (0 <= top < bottom <= orig_h and 0 <= left < right <= orig_w):
                raise ValueError(
                    "Invalid crop box for CCM restoration: "
                    f"(top={top}, left={left}, bottom={bottom}, right={right}) "
                    f"for full image ({orig_h}, {orig_w})."
                )

            crop_h = max(bottom - top, 1)
            crop_w = max(right - left, 1)

            # Read pad-to-square metadata (defaults to 0 for backward compat)
            ph   = int(crop_params["pad_h"][idx].item())       if "pad_h"       in crop_params else 0
            pw   = int(crop_params["pad_w"][idx].item())       if "pad_w"       in crop_params else 0
            ph_e = int(crop_params["pad_h_extra"][idx].item()) if "pad_h_extra" in crop_params else 0
            pw_e = int(crop_params["pad_w_extra"][idx].item()) if "pad_w_extra" in crop_params else 0
            if min(ph, pw, ph_e, pw_e) < 0:
                raise ValueError("CCM crop padding values must be non-negative.")

            restored_map = canonical_coord_map_cropped.new_zeros(
                canonical_coord_map_cropped.shape[1], orig_h, orig_w
            )

            has_padding = (ph + pw + ph_e + pw_e) > 0
            if has_padding:
                # Reverse: resize to padded square -> strip padding -> paste
                sq = crop_h + ph + ph_e  # == crop_w + pw + pw_e
                if sq != crop_w + pw + pw_e:
                    raise ValueError(
                        "Inconsistent pad-to-square crop metadata: "
                        f"height gives {sq}, width gives {crop_w + pw + pw_e}."
                    )
                resized_sq = F.interpolate(
                    canonical_coord_map_cropped[idx : idx + 1],
                    size=(sq, sq),
                    mode="nearest",
                )[0]
                resized_crop = resized_sq[
                    :, ph : sq - ph_e, pw : sq - pw_e
                ]
            else:
                resized_crop = F.interpolate(
                    canonical_coord_map_cropped[idx : idx + 1],
                    size=(crop_h, crop_w),
                    mode="nearest",
                )[0]

            restored_map[:, top:bottom, left:right] = resized_crop
            restored.append(restored_map)

        return torch.stack(restored, dim=0)

    # ------------------------------------------------------------------
    # prepare_condition_info
    # ------------------------------------------------------------------

    def prepare_condition_info(
        self,
        batch,
        stage: str = "train",
        down_factor: int = 1,
    ):
        """Prepare condition information for the batch.

                Unified implementation for default/cropped modes:
                    - ``image`` / ``ori_image`` / ``mask`` follow the default (scene) path.
                    - ``crop_params`` is always normalized and recorded.
                    - Only ``canonical_coord_map`` source changes with
                        ``self.use_cropped_condition``:
                            * False: use ``canonical_coord_map``
                            * True : use ``canonical_coord_map_cropped``

        Expected batch keys:
            rgb_scene          : [B, NI, 3, H, W]
            rgb_cropped        : [B, NI, 3, H, W]
            mask               : [B, NI, 1, H, W]
            mask_cropped       : [B, NI, 1, H, W]
            voxel              : [B, NI, Res, Res, Res]
            canonical_coord_map: [B, NI, 3, H, W]  (layout condition from dataloader)
            fov                : [B,]  horizontal FOV in radians (optional)
            canonical_coord_map_cropped : [B, NI, 3, H, W]  (only when use_cropped_condition)
            crop_params        : crop metadata for restoring cropped CCMs (optional)

        All outputs are shaped [B*NI, ...].
        processed_batch also contains 'batch_size' (B) and 'NI' for later use.
        """
        return self._prepare_condition_info_default(batch, stage, down_factor)

    def _prepare_condition_info_default(
        self,
        batch,
        stage: str = "train",
        down_factor: int = 1,
    ):
        B, NI = batch["rgb_scene"].shape[0], batch["rgb_scene"].shape[1]

        processed_batch = {}
        processed_batch["batch_size"] = B
        processed_batch["NI"] = NI
        processed_batch["crop_params"] = self._prepare_crop_params(
            crop_params=batch.get("crop_params"),
            batch_size=B,
            num_instances=NI,
            device=batch["voxel"].device,
            default_spatial_size=batch["rgb_scene"].shape[-2:],
        )

        # Scene image: [B, NI, 3, H, W] -> [B*NI, 3, H, W]
        rgb_scene_flat = batch["rgb_scene"].reshape(B * NI, *batch["rgb_scene"].shape[2:])
        resized_img = self.image_transforms(rgb_scene_flat)
        processed_batch["image"]     = self.color_transforms(resized_img)
        processed_batch["ori_image"] = resized_img

        # Cropped image: [B, NI, 3, H, W] -> [B*NI, 3, H, W]
        rgb_cropped_flat = batch["rgb_cropped"].reshape(B * NI, *batch["rgb_cropped"].shape[2:])
        resized_img_cropped = self.image_transforms(rgb_cropped_flat)
        processed_batch["image_cropped"] = self.color_transforms(resized_img_cropped)
        processed_batch["ori_image_cropped"] = resized_img_cropped

        # Mask: [B, NI, 1, H, W] -> [B*NI, 3, H, W]
        mask_flat = batch["mask"].reshape(B * NI, *batch["mask"].shape[2:])
        processed_batch["mask"] = self.mask_transforms(mask_flat.repeat(1, 3, 1, 1))

        mask_cropped_flat = batch["mask_cropped"].reshape(B * NI, *batch["mask_cropped"].shape[2:])
        processed_batch["mask_cropped"] = self.mask_transforms(mask_cropped_flat.repeat(1, 3, 1, 1))

        # canonical_coord_map: [B, NI, 3, H, W] -> [B*NI, 3, H, W]
        # Unified behavior: only canonical_coord_map source is switched by
        # self.use_cropped_condition; image/mask stay on default path.
        ccm_key = "canonical_coord_map_cropped" if self.use_cropped_condition else "canonical_coord_map"
        ccm_mask = mask_cropped_flat if self.use_cropped_condition else mask_flat
        if ccm_key in batch and batch[ccm_key] is not None:
            ccm = batch[ccm_key].reshape(B * NI, *batch[ccm_key].shape[2:])
            ccm = ccm * ccm_mask  # zero out background; mask shape: [B*NI, 1, H, W]
            processed_batch["canonical_coord_map"] = ccm.clamp(min=-0.5, max=0.5)
        else:
            processed_batch["canonical_coord_map"] = None

        # Voxel: [B, NI, Res, Res, Res] -> [B*NI, 1, Res, Res, Res]
        voxel_flat = batch["voxel"].reshape(B * NI, *batch["voxel"].shape[2:])
        processed_batch["voxel"] = voxel_flat[:, None]

        # Context: simplified -- always None
        processed_batch["context_voxel"] = None
        processed_batch["context_pcd"]   = None

        device = batch["voxel"].device
        processed_batch["select_indices"] = (
            torch.arange(NI, device=device).unsqueeze(0).expand(B, NI).contiguous()
        )

        return processed_batch

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualization_train(
        self, batch, save_dir="./debug_inference", voxel_res=64, rank=None
    ):
        """Save a compact visual audit of all image/layout conditions.

        The processed training batch is flattened to ``[B*NI, ...]``.  Each
        row in ``train_vis.png`` therefore corresponds to one object
        instance and contains, from left to right:

        ``image | mask | image_cropped | mask_cropped |
        image_cropped_masked | canonical_coord_map``.

        ``ori_*`` tensors are used so ImageNet normalization is not visible in
        the debug image.  The optional ``condition_mask*`` keys are supported
        for processors that augment the masks; the current processor falls
        back to its regular masks.
        """
        os.makedirs(save_dir, exist_ok=True)
        rank_suffix = f"_rank{int(rank)}" if rank is not None else ""

        # ---- Voxel (first item only) ----
        voxel = batch.get("voxel")
        if voxel is not None and voxel.shape[0] > 0:
            save_path = f"{save_dir}/voxel_0{rank_suffix}.ply"
            coords_np = voxels_to_pcd(
                voxel[0, 0], voxel_res=voxel_res
            ).cpu().numpy()
            trimesh.PointCloud(coords_np).export(save_path, file_type="ply")

        # ---- Build per-instance condition rows ----
        if "ori_image" not in batch or batch["ori_image"].shape[0] == 0:
            return

        def _rgb(value, index):
            image = value[index].detach().float().cpu().numpy()
            image = image.transpose(1, 2, 0).clip(0, 1)
            return (image * 255.0).astype(np.uint8)

        def _mask(value, index):
            mask = value[index, 0].detach().float().cpu().numpy()
            mask = (mask.clip(0, 1) * 255.0).astype(np.uint8)
            return np.stack([mask] * 3, axis=-1)

        def _resize_rgb(image, target_h, target_w):
            if image.shape[:2] == (target_h, target_w):
                return image
            return np.asarray(
                PIL.Image.fromarray(image).resize(
                    (target_w, target_h), PIL.Image.BILINEAR
                )
            )

        B = batch["ori_image"].shape[0]
        rows = []
        for b in range(B):
            image = _rgb(batch["ori_image"], b)
            mask = _mask(batch.get("condition_mask", batch["mask"]), b)
            image_cropped = _rgb(batch["ori_image_cropped"], b)
            mask_cropped = _mask(
                batch.get("condition_mask_cropped", batch["mask_cropped"]), b
            )

            # This is the exact object-focused image condition when the
            # current processor has no separate augmented mask key.
            cropped_mask_1ch = batch.get(
                "condition_mask_cropped", batch["mask_cropped"]
            )[b, :1].detach().float()
            image_cropped_masked = (
                batch["ori_image_cropped"][b].detach().float()
                * cropped_mask_1ch
            ).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            image_cropped_masked = (image_cropped_masked * 255.0).astype(np.uint8)

            parts = [image, mask, image_cropped, mask_cropped, image_cropped_masked]
            target_h, target_w = image.shape[:2]

            # CCM false-color: canonical coordinates [-0.5, 0.5] -> RGB.
            ccm = batch.get("canonical_coord_map")
            if ccm is not None:
                ccm_hw = ccm[b].detach().float().cpu().permute(1, 2, 0).numpy()
                ccm_vis = ((ccm_hw.clip(-0.5, 0.5) + 0.5) * 255.0).astype(np.uint8)
                parts.append(_resize_rgb(ccm_vis, target_h, target_w))

            # Every tile must have the same height before horizontal concat.
            parts = [_resize_rgb(part, target_h, target_w) for part in parts]
            rows.append(np.concatenate(parts, axis=1))

        max_w = max(row.shape[1] for row in rows)
        padded = []
        for row in rows:
            if row.shape[1] < max_w:
                pad = np.zeros(
                    (row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8
                )
                row = np.concatenate([row, pad], axis=1)
            padded.append(row)

        grid = np.concatenate(padded, axis=0)
        PIL.Image.fromarray(grid).save(f"{save_dir}/train_vis{rank_suffix}.png")
