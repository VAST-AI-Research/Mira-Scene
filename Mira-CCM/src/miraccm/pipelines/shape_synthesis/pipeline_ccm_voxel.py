"""
Pipeline for sparse-structure generation using a 2-D canonical coordinate map
(CCM) as the layout latent representation.
"""

import inspect
import math
import os
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import BitImageProcessor, Dinov2WithRegistersModel

from ...loaders.lora_pipeline import ImgToShapeLoraLoaderMixin
from ...models.autoencoders.autoencoder_kl_sparse_structure import (
    SparseStructureVAEModel as AutoencoderKLVoxel,
)
from ...models.transformers.dit_ccm_voxel import (
    CCMVoxelDiTModel,
)
from ...schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from ..pipeline_utils import TransformerDiffusionMixin
from .pipeline_shapediff_output import SparseStructurePipelineOutput

from einops import rearrange

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: retrieve timesteps
# ---------------------------------------------------------------------------

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not "
                f"support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not "
                f"support custom sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CCMVoxelPipeline(
    DiffusionPipeline, TransformerDiffusionMixin, ImgToShapeLoraLoaderMixin
):
    """
    Pipeline for sparse structure generation with canonical coordinate map layout.

    The layout stream now operates on a 2-D canonical coordinate map (CCM) latent
    of shape [B, 3, H', W'] instead of a 3-D volumetric layout.

    Simplified: no pointmap, moge_feature, or context_scene conditioning.
    """

    def __init__(
        self,
        vae: AutoencoderKLVoxel,
        transformer: CCMVoxelDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder: Dinov2WithRegistersModel,
        feature_extractor: BitImageProcessor,
    ):
        super().__init__()

        # Register core modules
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.decoder_channels) - 1)

    def build_latent_mapping(self):
        pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def get_branch_guidance_scale(self, branch_name: str) -> float:
        if branch_name == "layout":
            return 1.0
        return self.guidance_scale

    # ------------------------------------------------------------------
    # Condition encoding
    # ------------------------------------------------------------------

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(
            image, output_hidden_states=True
        ).hidden_states[-1]
        image_embeds = F.layer_norm(image_embeds, image_embeds.shape[-1:])
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    # ------------------------------------------------------------------
    # Latent preparation  (2-D layout)
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        resolution,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Build initial noise latents.

        Shape stream:  [B, in_channels, res_3d, res_3d, res_3d]  (3-D, unchanged)
        Layout stream: [B, 3,           res_2d, res_2d]           (2-D, new)
        """
        latent_shape_dict = {}
        for k, v in self.transformer.config.latent_config.items():
            if k == 'layout':
                # 2-D CCM layout
                res_2d = v['pos_embedder']['resolution']
                latent_shape_dict[k] = (batch_size, v['in_channels'], res_2d, res_2d)
            else:
                # 3-D shape latent (unchanged)
                latent_shape_dict[k] = (
                    (batch_size,) + (v['in_channels'],) + (v['pos_embedder']['resolution'],) * 3
                )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested "
                f"an effective batch size of {batch_size}."
            )

        latents_dict = {
            k: randn_tensor(v, generator=generator, device=device, dtype=dtype)
            for k, v in latent_shape_dict.items()
        }

        return latents_dict

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        mask: PipelineImageInput = None,
        image_cropped: PipelineImageInput = None,
        mask_cropped: PipelineImageInput = None,
        num_inference_steps: int = 50,
        resolution: int = 64,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents_dict"],
        return_dict: bool = True,
        keep_layout_condition_in_uncond: bool = False,
        # Optional pixel-aligned single-channel mask for layout-token concat.
        # Shape: [B, 1, layout_resolution, layout_resolution].
        # When None and the transformer has concat_img_mask_to_layout=True,
        # the mask branch will not be concatenated.
        layout_mask_for_concat: Optional[torch.FloatTensor] = None,
        # Optional downsampled RGB image for spatial conv fusion.
        # Shape: [B, 3, layout_resolution, layout_resolution].
        # Required when the transformer has use_conv_patchify_for_layout=True and
        # concat_img_mask_to_layout=True.
        layout_image_for_conv: Optional[torch.FloatTensor] = None,
        # When True, the layout stream cross-attention uses cropped embeddings
        # (image_cropped / mask_cropped) instead of the scene-level ones.
        use_cropped_condition: bool = False,
        # Layout prediction mode: "velocity" (default) or "x0_to_v" (JiT).
        # When "x0_to_v", the model output for layout is treated as x0_pred
        # and converted to velocity before the scheduler step.
        layout_pred_mode: str = "velocity",
        layout_t_eps: float = 1e-5,
    ):
        # ------------------------------------------------------------------
        # 1. Call parameters
        # ------------------------------------------------------------------
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self.transformer.device

        # ------------------------------------------------------------------
        # 2. Encode condition images
        # ------------------------------------------------------------------
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )
        if mask is not None:
            mask_embeds, negative_mask_embeds = self.encode_image(
                mask, device, num_shapes_per_prompt
            )
            assert image_cropped is not None and mask_cropped is not None, (
                "Both image_cropped and mask_cropped should be provided when mask is provided."
            )
            image_cropped_embeds, _ = self.encode_image(
                image_cropped, device, num_shapes_per_prompt
            )
            mask_cropped_embeds, _ = self.encode_image(
                mask_cropped, device, num_shapes_per_prompt
            )
        else:
            mask_embeds = negative_mask_embeds = None
            image_cropped_embeds = mask_cropped_embeds = None

        # ------------------------------------------------------------------
        # 3. Build encoder hidden-state dicts
        # ------------------------------------------------------------------
        shape_hs_dict: Dict[str, Optional[torch.Tensor]] = {
            "image": image_embeds,
            "mask": mask_embeds,
            "image_cropped": image_cropped_embeds,
            "mask_cropped": mask_cropped_embeds,
        }
        if use_cropped_condition:
            layout_hs_dict: Dict[str, Optional[torch.Tensor]] = {
                "image_cropped": image_cropped_embeds,
                "mask_cropped": mask_cropped_embeds,
            }
        else:
            layout_hs_dict: Dict[str, Optional[torch.Tensor]] = {
                "image": image_embeds,
                "mask": mask_embeds,
            }

        # Classifier-free guidance: prepend zero (uncond) embeddings
        if self.do_classifier_free_guidance:
            shape_hs_dict = {
                k: torch.cat([torch.zeros_like(v), v], dim=0) if v is not None else None
                for k, v in shape_hs_dict.items()
            }
            if keep_layout_condition_in_uncond:
                layout_hs_dict = {
                    k: torch.cat([v, v], dim=0) if v is not None else None
                    for k, v in layout_hs_dict.items()
                }
            else:
                layout_hs_dict = {
                    k: torch.cat([torch.zeros_like(v), v], dim=0) if v is not None else None
                    for k, v in layout_hs_dict.items()
                }

        # layout_mask_for_concat: CFG prepends a zero (uncond) copy.
        if layout_mask_for_concat is not None:
            layout_mask_for_concat = layout_mask_for_concat.to(
                device=device, dtype=image_embeds.dtype
            )
        if self.do_classifier_free_guidance and layout_mask_for_concat is not None:
            if keep_layout_condition_in_uncond:
                layout_mask_for_concat = torch.cat(
                    [layout_mask_for_concat, layout_mask_for_concat], dim=0
                )
            else:
                layout_mask_for_concat = torch.cat(
                    [torch.zeros_like(layout_mask_for_concat), layout_mask_for_concat], dim=0
                )

        # layout_image_for_conv: CFG prepends a zero (uncond) copy.
        if layout_image_for_conv is not None:
            layout_image_for_conv = layout_image_for_conv.to(
                device=device, dtype=image_embeds.dtype
            )
        if self.do_classifier_free_guidance and layout_image_for_conv is not None:
            if keep_layout_condition_in_uncond:
                layout_image_for_conv = torch.cat(
                    [layout_image_for_conv, layout_image_for_conv], dim=0
                )
            else:
                layout_image_for_conv = torch.cat(
                    [torch.zeros_like(layout_image_for_conv), layout_image_for_conv], dim=0
                )

        # ------------------------------------------------------------------
        # 4. Prepare timesteps
        # ------------------------------------------------------------------
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # ------------------------------------------------------------------
        # 5. Prepare latents (2-D layout latent)
        # ------------------------------------------------------------------
        num_channels_latents = self.transformer.config.in_channels
        latents_dict = self.prepare_latents(
            batch_size * num_shapes_per_prompt,
            num_channels_latents,
            resolution,
            image_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        # ------------------------------------------------------------------
        # 6. Denoising loop
        # ------------------------------------------------------------------
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input_dict = (
                    {k: torch.cat([v] * 2) for k, v in latents_dict.items()}
                    if self.do_classifier_free_guidance
                    else {k: v for k, v in latents_dict.items()}
                )

                timestep = t.expand(latent_model_input_dict['shape'].shape[0])
                timestep = timestep.to(dtype=image_embeds.dtype, device=image_embeds.device)

                noise_pred_dict = self.transformer(
                    latent_model_input_dict,
                    timestep,
                    encoder_hidden_states=shape_hs_dict,
                    layout_encoder_hidden_states=layout_hs_dict,
                    attention_kwargs=attention_kwargs,
                    layout_mask_for_concat=layout_mask_for_concat,
                    layout_image_for_conv=layout_image_for_conv,
                    return_dict=False,
                )

                # Classifier-free guidance
                if self.do_classifier_free_guidance:
                    def apply_cfg(noise_pred, branch_name):
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        guidance_scale = self.get_branch_guidance_scale(branch_name)
                        return noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    noise_pred_dict = {
                        k: apply_cfg(v, k) for k, v in noise_pred_dict.items()
                    }

                # Scheduler step
                # For layout branch in x0 prediction modes, convert x0_pred → velocity
                # before passing to the scheduler (which expects velocity).
                new_latents_dict = {}
                for k, v in latents_dict.items():
                    pred = noise_pred_dict[k]
                    if k == "layout" and layout_pred_mode in ("x0_to_v", "x0"):
                        sigma = self.scheduler.sigmas[i]
                        sigma_clamped = sigma.clamp_min(layout_t_eps)
                        pred = (v - pred) / sigma_clamped
                    new_latents_dict[k] = self.scheduler.step(
                        pred, t, v, return_dict=False,
                        advance_step_index=False,
                    )[0]
                latents_dict = new_latents_dict
                self.scheduler.advance_step_index()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for cb_k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[cb_k] = locals()[cb_k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents_dict = callback_outputs.pop("latents_dict", latents_dict)
                    shape_hs_dict = callback_outputs.pop("shape_hs_dict", shape_hs_dict)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # ------------------------------------------------------------------
        # 7. Extract layout prediction  [B, 3, H', W']
        # ------------------------------------------------------------------
        canonical_coord_map_pred = latents_dict['layout']  # [B, 3, H', W']
        latents = latents_dict['shape']

        # ------------------------------------------------------------------
        # 8. Decode sparse structure
        # ------------------------------------------------------------------
        hidden_states = self.vae.decode(latents).sample

        pcds, coords = [], []
        for idx in range(hidden_states.shape[0]):
            coords_idx = torch.argwhere(hidden_states[idx] > 0)[:, [1, 2, 3]].int()
            coords.append(coords_idx)
            from UniDataset.utils.pcd_utils import voxels_to_pcd
            coords_tensor = voxels_to_pcd(
                hidden_states[idx][0], voxel_res=resolution,
                min_bound=-0.5, max_bound=0.5,
            )
            pcds.append(coords_tensor.cpu().numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (hidden_states, pcds, coords, canonical_coord_map_pred)

        # Store canonical_coord_map_pred in latent_voxel_cam_pts field for compatibility
        return SparseStructurePipelineOutput(
            samples=hidden_states,
            pcds=pcds,
            coords=coords,
            latent_voxel_cam_pts=canonical_coord_map_pred,
        )
