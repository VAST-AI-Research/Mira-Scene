"""
save_outputs.py

Shared save logic for CCM inference outputs.
Used by both inference_CCM.py and system_ccm_voxel.py (validation_step).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image


def save_ccm_outputs(
    save_dir,
    inp,
    ccm_pred_masked,
    ccm_upsampled,
    canonical_pcds,
    voxel_coords,
    use_cropped_condition,
    data_processor,
):
    """Save inference outputs in the standard CCM format.

    Args:
        save_dir: output directory
        inp: preprocessed input dict (from data_processor.prepare_inference_input)
        ccm_pred_masked: [NI, 3, H', W'] masked CCM in cropped space
        ccm_upsampled: [NI, 3, H, W] upsampled CCM (masked, clipped)
        canonical_pcds: list of [N, 3] numpy arrays
        voxel_coords: list of [K, 3] tensors
        use_cropped_condition: bool
        data_processor: DataProcessor instance (for restore_canonical_coord_map)

    Output files:
        rgb_mask.png                    — scene image + colored id-map mask (side by side)
        masks.npy                       — [NI, H, W] uint8 per-instance binary masks
        canonical_coord_map.npy         — [NI, 3, H', W'] all instances CCM (cropped space)
        canonical_coord_map_restored.npy— [NI, 3, H, W] all instances CCM (scene space, if use_cropped)
        canonical_coord_map_restored.png— all instances CCM false-color stacked vertically
        rgb_ccm_cropped.png             — cropped RGB + CCM pairs stacked vertically
        voxel_coords_{i:03d}.npy        — [K_i, 3] per-instance voxel coords
        canonical_pcd_{i:03d}.ply       — per-instance canonical PCD
        canonical_pcd_{i:03d}_overlay.ply — per-instance overlay (PCD red + CCM green)
    """
    os.makedirs(save_dir, exist_ok=True)

    NI = ccm_pred_masked.shape[0]
    H, W = inp["ori_image"].shape[-2:]
    ccm_h, ccm_w = ccm_pred_masked.shape[-2:]

    # --- rgb_mask.png (scene image + colored id-map) ---
    ori_np = (inp["ori_image"][0].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    from miraccm.utils.image_utils.segment import masks2idmap
    mask_idmap = masks2idmap(inp["mask_1ch"][:, 0])  # [NI, H, W] -> palette image
    mask_vis_3ch = np.array(mask_idmap.convert("RGB"))
    Image.fromarray(np.concatenate([ori_np, mask_vis_3ch], axis=1)).save(
        os.path.join(save_dir, "rgb_mask.png")
    )

    # --- masks.npy (per-instance binary masks) ---
    masks_np = (inp["mask_1ch"][:, 0].float().cpu().numpy() > 0.5).astype(np.uint8)  # [NI, H, W]
    np.save(os.path.join(save_dir, "masks.npy"), masks_np)

    # --- CCM npy (cropped space, all instances) ---
    np.save(os.path.join(save_dir, "canonical_coord_map.npy"),
            ccm_pred_masked.float().cpu().numpy())  # [NI, 3, H', W']

    # --- CCM restored (scene space) ---
    if use_cropped_condition:
        ccm_restored = data_processor.restore_canonical_coord_map(
            canonical_coord_map_cropped=ccm_upsampled,
            crop_params=inp["crop_params"],
        )
        mask_full = F.interpolate(inp["mask"][:, :1].float(), size=(H, W), mode="nearest")

        np.save(os.path.join(save_dir, "canonical_coord_map_restored.npy"),
                ccm_restored.float().cpu().numpy())  # [NI, 3, H, W]

        # False-color PNG (all instances stacked vertically)
        vis_rows = []
        for i in range(NI):
            ccm_i_vis = ((ccm_restored[i].float().clamp(-0.5, 0.5) + 0.5) * 255).byte()
            ccm_i_vis = ccm_i_vis * (mask_full[i] > 0.5).byte()
            vis_rows.append(ccm_i_vis.permute(1, 2, 0).cpu().numpy())
        Image.fromarray(np.concatenate(vis_rows, axis=0)).save(
            os.path.join(save_dir, "canonical_coord_map_restored.png")
        )

    # --- rgb_ccm_cropped.png (all instances stacked vertically) ---
    _mask_crop = inp["mask_cropped_1ch"].float()
    ccm_cropped_vis = ((ccm_pred_masked.float().clamp(-0.5, 0.5) + 0.5) * 255).byte()
    ccm_cropped_vis = ccm_cropped_vis * (F.interpolate(
        _mask_crop, size=(ccm_h, ccm_w), mode="nearest",
    ) > 0.5).byte()

    rgb_cropped_np = (
        inp["ori_image_cropped"].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)
    concat_rows = []
    for i in range(NI):
        rgb_i = rgb_cropped_np[i]
        ccm_i = ccm_cropped_vis[i].permute(1, 2, 0).cpu().numpy()
        if ccm_i.shape[0] != rgb_i.shape[0]:
            ccm_i = np.array(Image.fromarray(ccm_i).resize(
                (rgb_i.shape[1], rgb_i.shape[0]), Image.NEAREST
            ))
        concat_rows.append(np.concatenate([rgb_i, ccm_i], axis=1))
    Image.fromarray(np.concatenate(concat_rows, axis=0)).save(
        os.path.join(save_dir, "rgb_ccm_cropped.png")
    )

    # --- Canonical PCDs (per-instance, PLY) ---
    _mask_src = inp["mask_cropped_1ch"] if use_cropped_condition else inp["mask_1ch"]
    ccm_for_overlay = ccm_upsampled * (
        F.interpolate(_mask_src.float(), size=(H, W), mode="nearest") > 0.5
    ).float()
    for i, pcd in enumerate(canonical_pcds):
        if len(pcd) == 0:
            continue
        # Pure PCD
        trimesh.PointCloud(pcd).export(
            os.path.join(save_dir, f"canonical_pcd_{i:03d}.ply")
        )
        # Overlay: PCD (red) + CCM points (green)
        ccm_pts = ccm_for_overlay[i].permute(1, 2, 0).float().cpu().numpy()  # [H, W, 3]
        ccm_flat = ccm_pts.reshape(-1, 3)
        ccm_valid = np.abs(ccm_flat).sum(axis=-1) > 1e-6
        ccm_pts_valid = ccm_flat[ccm_valid]
        colors_pcd = np.tile([255, 0, 0, 255], (len(pcd), 1)).astype(np.uint8)
        colors_ccm = np.tile([0, 255, 0, 255], (len(ccm_pts_valid), 1)).astype(np.uint8)
        combined_pts = np.concatenate([pcd, ccm_pts_valid], axis=0)
        combined_colors = np.concatenate([colors_pcd, colors_ccm], axis=0)
        trimesh.PointCloud(combined_pts, colors=combined_colors).export(
            os.path.join(save_dir, f"canonical_pcd_{i:03d}_overlay.ply")
        )

    # --- Voxel coords (per-instance) ---
    for i, coords in enumerate(voxel_coords):
        c = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
        np.save(os.path.join(save_dir, f"voxel_coords_{i:03d}.npy"), c)
