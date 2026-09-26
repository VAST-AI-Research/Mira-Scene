"""
Training/validation system for image-to-sparse-structure using a 2-D canonical
coordinate map (CCM) as the layout representation.
"""

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from einops import rearrange
from omegaconf import OmegaConf
from transformers import BitImageProcessor, Dinov2WithRegistersModel

from ...models.autoencoders.autoencoder_kl_voxel import AutoencoderKLVoxel
from ...pipelines.shape_synthesis.pipeline_ccm_voxel import (
    CCMVoxelPipeline,
    CCMVoxelDiTModel,
)
from ...schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from ...utils.system_utils.optimizer import parse_scheduler
from ...utils.system_utils.logging import debug, info, warn
from ...utils.system_utils.misc import get_rank
from ...utils.typing import *
from ..base import BaseSystem

from ...utils.image_utils.segment import masks2idmap
import trimesh
from UniDataset.utils.pcd_utils import (
    compute_similarity_transform,
    transform_pcd_simple,
    voxels_to_pcd,
)

from miraccm.utils.metrics import (
    chamfer_distance_numpy,
    ccm_mse_masked_batch,
    ccm_l1_masked_batch,
    one_way_chamfer_distance_numpy,
)
from miraccm.utils.save_outputs import save_ccm_outputs


class CCMVoxelDiTSystem(BaseSystem):
    """
    MMDiT system that uses a 2-D canonical coordinate map as the layout stream.

    Simplified variant: bidirectional attention, conv patchify, concat img+mask,
    offset 3D position embed for layout – all hardcoded ON.
    """

    @dataclass
    class Config(BaseSystem.Config):
        # Model
        pretrained_model_name_or_path: str = ""

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        gradient_checkpointing: bool = False
        weighting_scheme: str = "logit_normal"
        # Weighted training objective:
        # total_loss = loss_weight_shape * loss_shape + loss_weight_layout * loss_layout
        loss_weight_shape: float = 1.0
        loss_weight_layout: float = 1.0

        # Layout loss type: "mse" or "l1"
        layout_loss_type: str = "mse"
        # Layout loss masking: when True, only compute loss within foreground mask
        layout_masked_loss: bool = True

        # Optional branch-wise learning rates for optimizer parameter groups.
        # When None, falls back to optimizer.args.lr.
        optimizer_lr_shape: Optional[float] = None
        optimizer_lr_layout: Optional[float] = None

        resolution: int = 64

        # Condition
        text_drop_prob: float = 0.1
        # When True, whole-condition dropout keeps layout-specific conditions
        # (layout image/mask tokens, layout concat inputs)
        # active so the layout branch stays conditioned during training.
        keep_layout_condition_in_uncond: bool = True

        # Evaluation
        eval_seed: int = 42
        eval_num_inference_steps: int = 50
        eval_guidance_scale: float = 6.0
        # Deterministically cap the dense CCM correspondences used by Sim(3)
        # fitting so validation memory/time does not scale with image area.
        metric_alignment_max_points: int = 20000

        # Best model tracking: save pipeline_best when this metric improves
        # Options: "val/ccm_to_gt_voxel_cd", "val/chamfer_distance",
        # "val/ccm_mse", "val/ccm_l1"
        best_metric: str = "val/ccm_to_gt_voxel_cd"
        best_metric_mode: str = "min"  # "min" or "max"

        latent_config: Dict[str, Any] = field(default_factory=dict)

        # Canonical coord map resolution (H' = W' after downsampling)
        # This is the resolution the CCM is interpolated to before being used
        # as the layout latent.  Must match latent_config.layout.pos_embedder.resolution.
        layout_ccm_resolution: int = 518

        # When True, enable the cascaded layout stream architecture:
        #   1. Input-side concat of image + mask onto CCM tokens
        #   2. First-half blocks process at given patch_size (coarse)
        #   3. Midpoint: proj_fusion + pixel-shuffle 2× upsample
        #   4. Second-half blocks at fine resolution
        #   5. FinalLayerLayout (adaLN) + unpatchify with patch_size // 2
        # Requires layout patch_size >= 2.
        use_cascaded_layout: bool = True

        # When True, DataProcessor.prepare_condition_info uses cropped images
        # (rgb_cropped, mask_cropped, canonical_coord_map_cropped) in place of
        # the scene-level fields (image, mask, canonical_coord_map) and sets
        # fov to None.
        use_cropped_condition: bool = True

        # When True, initialize layout branch from pretrained shape weights
        # (xavier new modules + copy shape → layout). Set to False when resuming
        # from a checkpoint that already has initialized layout weights.
        init_layout_from_shape: bool = True

    cfg: Config

    # ------------------------------------------------------------------
    # configure
    # ------------------------------------------------------------------

    def configure(self):
        # The current layout implementation and its validation coordinate
        # conventions are defined for the cascaded, crop-aligned path.  Fail
        # early when an incompatible configuration is supplied instead of
        # silently training/evaluating a different conditioning graph.
        assert self.cfg.keep_layout_condition_in_uncond is True, (
            "CCMVoxelDiTSystem requires keep_layout_condition_in_uncond=True."
        )
        assert self.cfg.use_cascaded_layout is True, (
            "CCMVoxelDiTSystem requires use_cascaded_layout=True."
        )
        assert self.cfg.use_cropped_condition is True, (
            "CCMVoxelDiTSystem requires use_cropped_condition=True."
        )
        super().configure()
        # Best metric tracking
        self._best_metric_value = float("inf") if self.cfg.best_metric_mode == "min" else float("-inf")

        from .data_processor.ccm_voxel import DataProcessor

        latent_config = self.cfg.latent_config
        if OmegaConf.is_config(latent_config):
            latent_config = OmegaConf.to_container(latent_config, resolve=True)

        transformer: CCMVoxelDiTModel = (
            CCMVoxelDiTModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="transformer",
                low_cpu_mem_usage=False,
                latent_config=latent_config,
                use_cascaded_layout=self.cfg.use_cascaded_layout,
            )
        )

        pipeline: CCMVoxelPipeline = (
            CCMVoxelPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                transformer=transformer,
            )
        )

        self.pipeline: CCMVoxelPipeline = pipeline
        self.vae: AutoencoderKLVoxel = self.pipeline.vae
        self.transformer: CCMVoxelDiTModel = self.pipeline.transformer
        self._set_shape_layout_self_attn_mode()
        self.feature_extractor: BitImageProcessor = self.pipeline.feature_extractor
        self.image_encoder: Dinov2WithRegistersModel = self.pipeline.image_encoder
        self.noise_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(
            pipeline.scheduler
        )

        # Weight initialisation
        if self.cfg.init_layout_from_shape:
            self.transformer.init_layout_from_shape()

        # Trainable modules
        trainable_modules = self.cfg.trainable_modules
        if trainable_modules and len(trainable_modules) > 0:
            self.transformer.requires_grad_(False)
            for name, module in self.transformer.named_modules():
                for tm in trainable_modules:
                    if tm in name:
                        module.requires_grad_(True)
        else:
            self.transformer.requires_grad_(True)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        self.transformer.train()
        self.vae.eval()
        self.image_encoder.eval()

        if self.cfg.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.data_processor = DataProcessor(
            use_cropped_condition=self.cfg.use_cropped_condition,
        )

    def _set_shape_layout_self_attn_mode(self) -> None:
        """Hardcoded to bidirectional mode."""
        mode = "bidirectional"

        # Set mode explicitly on each DiT block's attn1 module.
        count = 0
        for block in getattr(self.transformer, "blocks", []):
            attn1 = getattr(block, "attn1", None)
            if attn1 is None:
                continue
            setattr(attn1, "shape_layout_self_attn_mode", mode)
            count += 1

        if count == 0:
            raise RuntimeError(
                "No attn1 modules found in transformer.blocks; "
                "shape_layout_self_attn_mode was not applied."
            )

        # Verify assignment took effect.
        verify_count = 0
        for block in getattr(self.transformer, "blocks", []):
            attn1 = getattr(block, "attn1", None)
            if attn1 is None:
                continue
            if getattr(attn1, "shape_layout_self_attn_mode", None) == mode:
                verify_count += 1

        if verify_count != count:
            raise RuntimeError(
                f"Failed to set shape_layout_self_attn_mode for all attn1 modules: "
                f"{verify_count}/{count}"
            )

        info(
            f"Set shape-layout self-attention mode to '{mode}' for "
            f"{verify_count} attn1 modules"
        )

    def on_fit_start(self):
        pass

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        voxel = batch["voxel"]

        # ---- VAE encode ----
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            shape_latents = self.vae.encode(voxel.to(self.vae.dtype)).latent_dist.sample()

        # ---- CCM → 2-D layout latent (bilinear downsample) ----
        ccm = batch['canonical_coord_map']  # [B*NI, 3, H, W]
        if ccm is None:
            raise ValueError(
                "'canonical_coord_map' not found in batch. "
                "Ensure the DataProcessor provides it."
            )

        H_layout = self.cfg.layout_ccm_resolution
        layout_latent = F.interpolate(
            ccm.float(),
            size=(H_layout, H_layout),
            mode='bilinear',
            align_corners=False,
        ).to(dtype=shape_latents.dtype)

        latent_dict = {
            'shape': shape_latents,
            'layout': layout_latent,
        }

        batch_size = shape_latents.shape[0]

        # ---- Prepare pixel-aligned mask for layout-token concat (always enabled) ----
        layout_mask_for_concat = F.interpolate(
            (batch['mask_cropped'] if self.cfg.use_cropped_condition else batch['mask'])[:, :1].float(),
            size=(H_layout, H_layout),
            mode='nearest',
        ).to(dtype=shape_latents.dtype)

        # ---- Prepare downsampled RGB image for spatial conv fusion (always enabled) ----
        layout_image_for_conv = F.interpolate(
            (batch['image_cropped'] if self.cfg.use_cropped_condition else batch['image']).float(),
            size=(H_layout, H_layout),
            mode='bilinear',
            align_corners=False,
        ).to(dtype=shape_latents.dtype)

        # ---- Flow-matching noise sampling ----
        sigmas = compute_density_for_timestep_sampling(
            self.cfg.weighting_scheme,
            batch_size,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(shape_latents.device)
        sigmas = self.noise_scheduler._time_shift_constant(sigmas)
        timesteps = self.noise_scheduler._sigma_to_t(sigmas)

        noise_dict = {k: torch.randn_like(v) for k, v in latent_dict.items()}
        noisy_latents_dict = {
            k: self.noise_scheduler.scale_noise(v, timesteps, noise_dict[k], sigma=sigmas)
            for k, v in latent_dict.items()
        }

        # ---- Condition encoding ----
        with torch.no_grad():
            image_embeds, _ = self.pipeline.encode_image(
                batch["image"], shape_latents.device, num_images_per_prompt=1
            )
            mask_embeds, _ = self.pipeline.encode_image(
                batch["mask"], shape_latents.device, num_images_per_prompt=1
            )
            image_cropped_embeds, _ = self.pipeline.encode_image(
                batch["image_cropped"], shape_latents.device, num_images_per_prompt=1
            )
            mask_cropped_embeds, _ = self.pipeline.encode_image(
                batch["mask_cropped"], shape_latents.device, num_images_per_prompt=1
            )

        shape_hs_dict = {
            "image": image_embeds,
            "mask": mask_embeds,
            "image_cropped": image_cropped_embeds,
            "mask_cropped": mask_cropped_embeds,
        }
        # When use_cropped_condition, the layout stream is conditioned on the
        # cropped-image embeddings so it sees the same crop-aligned view as the CCM.
        if self.cfg.use_cropped_condition:
            layout_hs_dict = {
                "image_cropped": image_cropped_embeds.clone(),
                "mask_cropped": mask_cropped_embeds.clone(),
            }
        else:
            layout_hs_dict = {
                # Keep layout tensors independent from shape_hs_dict so optional
                # training-time dropout can zero only shape-side conditions.
                "image": image_embeds.clone(),
                "mask": mask_embeds.clone(),
            }

        # Whole-condition dropout
        for idx in range(batch_size):
            if random.random() < self.cfg.text_drop_prob:
                for key in shape_hs_dict:
                    if shape_hs_dict[key] is not None:
                        shape_hs_dict[key][idx] = torch.zeros_like(shape_hs_dict[key][idx])
                for key in layout_hs_dict:
                    if (
                        layout_hs_dict[key] is not None
                        and not self.cfg.keep_layout_condition_in_uncond
                    ):
                        layout_hs_dict[key][idx] = torch.zeros_like(layout_hs_dict[key][idx])
                if layout_mask_for_concat is not None:
                    if not self.cfg.keep_layout_condition_in_uncond:
                        layout_mask_for_concat[idx] = torch.zeros_like(layout_mask_for_concat[idx])
                if layout_image_for_conv is not None:
                    if not self.cfg.keep_layout_condition_in_uncond:
                        layout_image_for_conv[idx] = torch.zeros_like(layout_image_for_conv[idx])

        # ---- Model prediction ----
        model_pred = self.transformer(
            noisy_latents_dict,
            timesteps,
            encoder_hidden_states=shape_hs_dict,
            layout_encoder_hidden_states=layout_hs_dict,
            layout_mask_for_concat=layout_mask_for_concat,
            layout_image_for_conv=layout_image_for_conv,
            return_dict=False,
        )

        # ---- Flow-matching loss ----
        target_dict = {k: noise_dict[k] - latent_dict[k] for k in latent_dict}

        weighting = compute_loss_weighting_for_sd3(self.cfg.weighting_scheme, sigmas)
        # 2-D layout: weighting must broadcast to [B, C, H', W']
        weighting_4d = weighting.view(-1, 1, 1, 1)
        weighting_5d = weighting.view(-1, 1, 1, 1, 1)

        loss_dict = {}

        # Shape: standard MSE over all voxels
        loss_dict["shape"] = torch.mean(
            (weighting_5d * (model_pred["shape"] - target_dict["shape"]) ** 2).reshape(batch_size, -1),
            dim=1,
        )

        # Layout loss
        layout_diff = model_pred["layout"] - target_dict["layout"]
        if self.cfg.layout_loss_type == "l1":
            layout_pixel_loss = weighting_4d * layout_diff.abs()
        else:
            layout_pixel_loss = weighting_4d * layout_diff ** 2

        if self.cfg.layout_masked_loss:
            layout_mask = layout_mask_for_concat  # [B, 1, H, H], 0/1
            masked_loss = (layout_pixel_loss * layout_mask).sum(dim=(1, 2, 3))  # [B]
            mask_count = layout_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * layout_diff.shape[1]  # [B], × channels
            loss_dict["layout"] = masked_loss / mask_count
        else:
            loss_dict["layout"] = torch.mean(
                layout_pixel_loss.reshape(batch_size, -1), dim=1,
            )

        return loss_dict

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        processed_batch = self.data_processor.prepare_condition_info(
            batch, stage="train",
        )

        if self.trainer is not None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            for idx, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", f"group_{idx}")
                self.log(
                    f"train/lr_{group_name}",
                    float(group["lr"]),
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )

        loss_dict = self(processed_batch)

        loss_shape = loss_dict["shape"].mean()
        loss_layout = loss_dict["layout"].mean()
        total_loss = (
            self.cfg.loss_weight_shape * loss_shape
            + self.cfg.loss_weight_layout * loss_layout
        )

        self.log("train/loss_shape", loss_shape, prog_bar=True)
        self.log("train/loss_layout", loss_layout, prog_bar=True)
        self.log("train/loss_total", total_loss, prog_bar=True)

        self.check_train(processed_batch)

        return {"loss": total_loss}

    def configure_optimizers(self):
        optimizer_cfg = self.cfg.optimizer

        if not hasattr(optimizer_cfg, "args"):
            raise ValueError("optimizer config must contain `args`.")

        optimizer_args = dict(optimizer_cfg.args)
        base_lr = optimizer_args.pop("lr", None)

        if base_lr is None and (
            self.cfg.optimizer_lr_shape is None or self.cfg.optimizer_lr_layout is None
        ):
            raise ValueError(
                "Please set optimizer.args.lr or both optimizer_lr_shape/optimizer_lr_layout."
            )

        lr_shape = (
            float(self.cfg.optimizer_lr_shape)
            if self.cfg.optimizer_lr_shape is not None
            else float(base_lr)
        )
        lr_layout = (
            float(self.cfg.optimizer_lr_layout)
            if self.cfg.optimizer_lr_layout is not None
            else float(base_lr)
        )

        # Store trainable parameter statistics for debugging/inspection.
        trainable_param_stats = []
        layout_params = []
        shape_params = []
        layout_param_numel = 0
        shape_param_numel = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            if "layout" in name or "add_" in name:
                param_type = "layout"
                layout_params.append(parameter)
                layout_param_numel += int(parameter.numel())
            else:
                param_type = "shape"
                shape_params.append(parameter)
                shape_param_numel += int(parameter.numel())

            trainable_param_stats.append(
                {
                    "name": name,
                    "numel": int(parameter.numel()),
                    "type": param_type,
                }
            )

        self.trainable_param_stats = trainable_param_stats
        self.trainable_param_total = int(sum(item["numel"] for item in trainable_param_stats))
        self.trainable_param_group_stats = {
            "shape": {
                "tensor_count": len(shape_params),
                "numel": shape_param_numel,
            },
            "layout": {
                "tensor_count": len(layout_params),
                "numel": layout_param_numel,
            },
        }

        # Persist trainable-parameter stats to an inspectable experiment path
        if get_rank() == 0:
            save_path_prefix = f"it{self.true_global_step}-trainable-params-rank0"
            save_dir = self.get_save_path(f"{save_path_prefix}")
            os.makedirs(save_dir, exist_ok=True)
            stats_file = os.path.join(save_dir, "trainable_param_stats.txt")
            with open(stats_file, "w", encoding="utf-8") as fout:
                fout.write(f"total_trainable_params: {self.trainable_param_total}\n")
                fout.write(
                    f"shape_param_tensors: {self.trainable_param_group_stats['shape']['tensor_count']}\n"
                )
                fout.write(
                    f"shape_param_numel: {self.trainable_param_group_stats['shape']['numel']}\n"
                )
                fout.write(
                    f"layout_param_tensors: {self.trainable_param_group_stats['layout']['tensor_count']}\n"
                )
                fout.write(
                    f"layout_param_numel: {self.trainable_param_group_stats['layout']['numel']}\n"
                )
                fout.write("type\tname\tnumel\n")
                for item in self.trainable_param_stats:
                    fout.write(f"{item['type']}\t{item['name']}\t{item['numel']}\n")
            info(f"Saved trainable parameter stats to {stats_file}")

        param_groups = []
        if len(shape_params) > 0:
            param_groups.append({"params": shape_params, "lr": lr_shape, "name": "shape"})
        if len(layout_params) > 0:
            param_groups.append({"params": layout_params, "lr": lr_layout, "name": "layout"})

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found for optimizer.")

        if optimizer_cfg.name in ["FusedAdam"]:
            import apex

            optim = getattr(apex.optimizers, optimizer_cfg.name)(param_groups, **optimizer_args)
        elif optimizer_cfg.name in ["Adam8bit", "AdamW8bit"]:
            import bitsandbytes as bnb

            optim = bnb.optim.Adam8bit(param_groups, **optimizer_args)
        else:
            optim = getattr(torch.optim, optimizer_cfg.name)(param_groups, **optimizer_args)

        info(
            f"Optimizer param groups -> shape: {len(shape_params)} params @ lr={lr_shape}, "
            f"layout: {len(layout_params)} params @ lr={lr_layout}"
        )

        ret = {"optimizer": optim}
        if self.cfg.scheduler is not None:
            ret.update({"lr_scheduler": parse_scheduler(self.cfg.scheduler, optim)})
        return ret

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def on_check_train(self, batch):
        save_path_prefix = f"it{self.true_global_step}-train-{self.global_rank}"
        self.data_processor.visualization_train(
            batch=batch,
            save_dir=self.get_save_path(f"{save_path_prefix}"),
        )

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "pipeline")
            os.makedirs(save_dir, exist_ok=True)
            self.pipeline.save_pretrained(save_dir)

    def on_validation_epoch_end(self):
        """Check if current val metric is best, save pipeline_best if so."""
        # Sanity validation is only a smoke test; it must not replace a real
        # best checkpoint before training starts.
        if self.trainer.sanity_checking:
            return

        metric_key = self.cfg.best_metric
        # All ranks read the synchronized epoch metric before rank zero is
        # allowed to write.  This keeps the hook order identical under DDP.
        callback_metrics = self.trainer.callback_metrics
        if metric_key not in callback_metrics:
            return

        current_value = float(callback_metrics[metric_key])
        if np.isnan(current_value):
            return

        is_better = (
            (self.cfg.best_metric_mode == "min" and current_value < self._best_metric_value)
            or (self.cfg.best_metric_mode == "max" and current_value > self._best_metric_value)
        )

        if is_better:
            # Keep the comparison state identical on every rank.  Only the
            # actual filesystem write is restricted to rank zero.
            self._best_metric_value = current_value
            if self.global_rank == 0:
                save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "pipeline_best")
                os.makedirs(save_dir, exist_ok=True)
                self.pipeline.save_pretrained(save_dir)
                info(
                    f"New best {metric_key}={current_value:.6f} at step {self.true_global_step}. "
                    f"Saved pipeline_best."
                )
            # Do not let the other ranks enter the next train epoch while rank
            # zero is still serializing the pipeline.
            self.trainer.strategy.barrier("pipeline_best_save")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, batch):
        """Run pipeline inference, return CCM prediction + canonical PCDs."""
        num_instances = batch["image"].shape[0]

        # Prepare pipeline inputs
        layout_mask_for_concat = F.interpolate(
            (batch["mask_cropped"] if self.cfg.use_cropped_condition else batch["mask"])[:, :1].float(),
            size=(self.cfg.layout_ccm_resolution, self.cfg.layout_ccm_resolution),
            mode='nearest',
        )
        layout_image_for_conv = F.interpolate(
            (batch["image_cropped"] if self.cfg.use_cropped_condition else batch["image"]).float(),
            size=(self.cfg.layout_ccm_resolution, self.cfg.layout_ccm_resolution),
            mode='bilinear',
            align_corners=False,
        )

        output = self.pipeline(
            image=batch["image"],
            mask=batch["mask"],
            image_cropped=batch["image_cropped"],
            mask_cropped=batch["mask_cropped"],
            num_inference_steps=self.cfg.eval_num_inference_steps,
            resolution=self.cfg.resolution,
            guidance_scale=self.cfg.eval_guidance_scale,
            keep_layout_condition_in_uncond=self.cfg.keep_layout_condition_in_uncond,
            layout_mask_for_concat=layout_mask_for_concat,
            layout_image_for_conv=layout_image_for_conv,
            use_cropped_condition=self.cfg.use_cropped_condition,
        )

        canonical_coord_map_pred = output.latent_voxel_cam_pts  # [B, 3, H', W']
        canonical_pcds = [output.pcds[i] for i in range(num_instances)]
        voxel_pred = output.samples  # [B, 1, R, R, R]

        # Downsample pred/GT to layout resolution for MSE loss
        ccm_gt = batch["canonical_coord_map"]
        H_layout = self.cfg.layout_ccm_resolution
        ccm_gt_small = F.interpolate(
            ccm_gt.float(), size=(H_layout, H_layout), mode='bilinear', align_corners=False
        )
        ccm_pred_small = F.interpolate(
            canonical_coord_map_pred.float(), size=(H_layout, H_layout),
            mode='bilinear', align_corners=False,
        )

        return {
            "canonical_coord_map_pred": canonical_coord_map_pred,
            "canonical_pcds": canonical_pcds,
            "voxel_pred": voxel_pred,
            "voxel_coords": output.coords,
            "ccm_pred_small": ccm_pred_small,
            "ccm_gt_small": ccm_gt_small,
        }

    # ------------------------------------------------------------------
    # validation_step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        """Validation using EvalDataset: same preprocessing as inference_CCM.py + metrics + save."""
        device = self.device

        # Keep one case-level value per metric and log each key exactly once
        # after the loop.  This is important in DDP: every rank must enter the
        # same sync_dist collectives in the same order.
        step_metrics: Dict[str, List[float]] = {
            "val/ccm_mse": [],
            "val/ccm_l1": [],
            "val/chamfer_distance": [],
            "val/ccm_to_gt_voxel_cd": [],
        }

        for case_idx, case in enumerate(batch):
            image = case["image"]       # [H, W, 3] numpy float32
            masks = case["masks"]       # list of [H, W] numpy float32
            gt = case["gt"]
            scene_id = case.get("id", f"case_{batch_idx}_{case_idx}")

            # Preprocess (same path as inference_CCM.py)
            inp = self.data_processor.prepare_inference_input(image, masks, device=device)
            NI = inp["NI"]

            # Attach GT CCM for inference() to compute ccm_pred_small / ccm_gt_small
            # Support both [NI, 3, H, W] (multi-instance) and [3, H, W] (single-instance)
            gt_ccm = gt["canonical_coord_map"].to(device)
            if gt_ccm.dim() == 3:
                gt_ccm = gt_ccm.unsqueeze(0)  # [1, 3, H, W]
            # If GT has fewer instances than NI (e.g. single GT for multi-mask), repeat
            if gt_ccm.shape[0] < NI:
                gt_ccm = gt_ccm[:1].expand(NI, -1, -1, -1)
            inp["canonical_coord_map"] = gt_ccm  # [NI, 3, H, W]

            output = self.inference(inp)

            # --- Metrics ---
            crop_mask = inp["mask_cropped_1ch"]
            H, W = inp['ori_image'].shape[-2:]

            ccm_pred = output["canonical_coord_map_pred"]  # [NI, 3, H', W']
            ccm_h, ccm_w = ccm_pred.shape[-2:]

            # The layout branch predicts on the cropped canvas.  Keep these
            # crop-space tensors for the existing output visualization path.
            mask_for_ccm = F.interpolate(
                crop_mask.float(), size=(ccm_h, ccm_w), mode="nearest"
            )
            ccm_pred_masked = ccm_pred * (mask_for_ccm > 0.5).to(ccm_pred.dtype)

            # Upsample only within the crop canvas.  The next step explicitly
            # inverts crop/pad/resize to obtain a scene-level CCM.
            ccm_crop_full_res = F.interpolate(
                ccm_pred_masked.float(), size=(H, W), mode="bilinear", align_corners=False,
            )
            ccm_crop_full_res = ccm_crop_full_res * (
                F.interpolate(crop_mask.float(), size=(H, W), mode="nearest") > 0.5
            ).float()
            # Keep clipping as an output-visualization convention only.  A
            # clamp before Sim(3) fitting would destroy a valid global scale
            # discrepancy and bias both the fitted transform and metrics.
            ccm_upsampled = ccm_crop_full_res.clamp(-0.5, 0.5)

            # Compare prediction and GT only after both have been expressed in
            # full-image pixel coordinates.  This fixes the previous crop CCM
            # versus full-image GT mismatch.
            ccm_pred_full = self.data_processor.restore_canonical_coord_map(
                canonical_coord_map_cropped=ccm_crop_full_res,
                crop_params=inp["crop_params"],
            )
            ccm_gt_full = gt_ccm.float()
            if ccm_gt_full.shape[-2:] != (H, W):
                ccm_gt_full = F.interpolate(
                    ccm_gt_full, size=(H, W), mode="nearest",
                )
            mask_full = inp["mask_1ch"]
            if mask_full.shape[-2:] != (H, W):
                mask_full = F.interpolate(mask_full.float(), size=(H, W), mode="nearest")

            # Use corresponding valid CCM pixels to estimate an independent
            # prediction -> GT similarity transform for every instance.
            alignments: List[Optional[Dict[str, Any]]] = []
            ccm_pred_aligned = ccm_pred_full.clone()
            for i in range(NI):
                valid = mask_full[i, 0] > 0.5
                valid = valid & torch.isfinite(ccm_gt_full[i]).all(dim=0)
                valid = valid & torch.isfinite(ccm_pred_full[i]).all(dim=0)
                # Zero is the background/invalid-depth sentinel in these CCMs.
                valid = valid & (ccm_gt_full[i].abs().sum(dim=0) > 1e-6)
                valid = valid & (ccm_pred_full[i].abs().sum(dim=0) > 1e-6)
                pred_points = ccm_pred_full[i].permute(1, 2, 0)[valid]
                gt_points = ccm_gt_full[i].permute(1, 2, 0)[valid]
                transform = None
                if pred_points.shape[0] >= 3:
                    # Directly use UniDataset's shared implementation. It
                    # estimates the prediction -> GT Sim(3) and returns R/t/s.
                    max_points = max(int(self.cfg.metric_alignment_max_points), 3)
                    if pred_points.shape[0] > max_points:
                        indices = torch.linspace(
                            0, pred_points.shape[0] - 1, max_points,
                            device=pred_points.device,
                        ).long()
                        pred_points, gt_points = pred_points[indices], gt_points[indices]
                    finite = torch.isfinite(pred_points).all(dim=1) & torch.isfinite(gt_points).all(dim=1)
                    pred_points, gt_points = pred_points[finite], gt_points[finite]
                    if pred_points.shape[0] >= 3:
                        pred_centered = pred_points - pred_points.mean(dim=0, keepdim=True)
                        gt_centered = gt_points - gt_points.mean(dim=0, keepdim=True)
                        if (
                            torch.linalg.vector_norm(pred_centered) > torch.finfo(torch.float32).eps
                            and torch.linalg.vector_norm(gt_centered) > torch.finfo(torch.float32).eps
                            and torch.linalg.matrix_rank(pred_centered) >= 2
                            and torch.linalg.matrix_rank(gt_centered) >= 2
                        ):
                            try:
                                transform = compute_similarity_transform(
                                    pred_points.detach().float().cpu(),
                                    gt_points.detach().float().cpu(),
                                )
                                if not all(torch.isfinite(torch.as_tensor(transform[k])).all() for k in ("R", "t", "s")):
                                    transform = None
                                elif float(torch.as_tensor(transform["s"]).item()) <= 0:
                                    transform = None
                            except (RuntimeError, ValueError):
                                transform = None
                alignments.append(transform)
                if transform is not None:
                    ccm_hwc = ccm_pred_full[i].permute(1, 2, 0)
                    ccm_valid = torch.isfinite(ccm_hwc).all(dim=-1) & (
                        ccm_hwc.abs().sum(dim=-1) > 1e-6
                    )
                    ccm_points = ccm_hwc[ccm_valid].detach().float().cpu().numpy()
                    ccm_points = transform_pcd_simple(transform, ccm_points)
                    aligned_hwc = ccm_pred_aligned[i].permute(1, 2, 0).clone()
                    aligned_hwc[ccm_valid] = torch.as_tensor(
                        ccm_points, device=aligned_hwc.device, dtype=aligned_hwc.dtype
                    )
                    ccm_pred_aligned[i] = aligned_hwc.permute(2, 0, 1)

            # If a particular Sim(3) cannot be estimated, its restored
            # full-image CCM remains in the CCM metric rather than comparing a
            # crop canvas to GT.  Its voxel CD is skipped below.
            ccm_mse = ccm_mse_masked_batch(
                ccm_pred_aligned, ccm_gt_full, mask_full
            )
            if np.isfinite(ccm_mse):
                step_metrics["val/ccm_mse"].append(float(ccm_mse))
            ccm_l1 = ccm_l1_masked_batch(
                ccm_pred_aligned, ccm_gt_full, mask_full
            )
            if np.isfinite(ccm_l1):
                step_metrics["val/ccm_l1"].append(float(ccm_l1))

            # GT occupancy is converted to voxel-center points for CD.  Raw
            # voxel IoU is deliberately removed because the two grids may be
            # related by the CCM-derived Sim(3), not index-aligned.
            gt_voxel = gt["voxel"].to(device)
            if gt_voxel.dim() == 3:
                gt_voxel = gt_voxel.unsqueeze(0)  # [1, R, R, R]
            gt_voxel = gt_voxel.unsqueeze(1)  # [NI, 1, R, R, R]
            if gt_voxel.shape[0] < NI:
                gt_voxel = gt_voxel[:1].expand(NI, -1, -1, -1, -1)

            # PCD Chamfer Distance after applying exactly the Sim(3) estimated
            # from the corresponding CCM.  Never fall back to an unaligned CD.
            voxel_res = gt_voxel.shape[-1]
            cd_list = []
            ccm_to_gt_voxel_cd_list = []
            for i in range(NI):
                gt_pcd_i = voxels_to_pcd(
                    gt_voxel[i, 0], voxel_res=voxel_res, min_bound=-0.5, max_bound=0.5,
                ).cpu().numpy()

                # One-way visible-surface consistency: aligned predicted CCM
                # points -> complete GT voxel-center cloud. This intentionally
                # does not penalize invisible GT surfaces absent from CCM.
                ccm_valid_i = (mask_full[i, 0] > 0.5) & (
                    ccm_pred_full[i].abs().sum(dim=0) > 1e-6
                )
                ccm_pcd_i = (
                    ccm_pred_aligned[i].permute(1, 2, 0)[ccm_valid_i]
                    .detach().float().cpu().numpy()
                )
                if ccm_pcd_i.shape[0] > int(self.cfg.metric_alignment_max_points):
                    ccm_sel = np.linspace(
                        0, ccm_pcd_i.shape[0] - 1,
                        int(self.cfg.metric_alignment_max_points),
                        dtype=np.int64,
                    )
                    ccm_pcd_i = ccm_pcd_i[ccm_sel]
                ccm_to_voxel_i = one_way_chamfer_distance_numpy(
                    ccm_pcd_i, gt_pcd_i
                )
                if np.isfinite(ccm_to_voxel_i):
                    ccm_to_gt_voxel_cd_list.append(float(ccm_to_voxel_i))

                if alignments[i] is None:
                    continue
                pred_pcd_i = output["canonical_pcds"][i] if i < len(output["canonical_pcds"]) else np.zeros((0, 3))
                pred_pcd_i = transform_pcd_simple(alignments[i], pred_pcd_i)
                cd_i = chamfer_distance_numpy(pred_pcd_i, gt_pcd_i)
                if np.isfinite(cd_i):
                    cd_list.append(float(cd_i))
            if cd_list:
                step_metrics["val/chamfer_distance"].append(float(np.mean(cd_list)))
            if ccm_to_gt_voxel_cd_list:
                step_metrics["val/ccm_to_gt_voxel_cd"].append(
                    float(np.mean(ccm_to_gt_voxel_cd_list))
                )

            # --- Save results (same function as inference_CCM.py) ---
            save_dir = self.get_save_path(
                f"it{self.true_global_step}-validation/{scene_id}/CCM"
            )
            save_ccm_outputs(
                save_dir=save_dir,
                inp=inp,
                ccm_pred_masked=ccm_pred_masked,
                ccm_upsampled=ccm_upsampled,
                canonical_pcds=output["canonical_pcds"],
                voxel_coords=output["voxel_coords"],
                use_cropped_condition=self.cfg.use_cropped_condition,
                data_processor=self.data_processor,
            )

        # Every rank registers every metric exactly once.  The local mean is
        # weighted by its number of valid cases, so sync_dist computes the
        # global case-level mean even when some ranks have no valid CD.
        for metric_name in (
            "val/ccm_mse",
            "val/ccm_l1",
            "val/chamfer_distance",
            "val/ccm_to_gt_voxel_cd",
        ):
            values = step_metrics[metric_name]
            local_count = len(values)
            local_mean = float(np.mean(values)) if values else 0.0
            self.log(
                metric_name,
                local_mean,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=local_count,
            )

    def test_step(self, batch, batch_idx):
        pass

    def on_test_end(self):
        pass
