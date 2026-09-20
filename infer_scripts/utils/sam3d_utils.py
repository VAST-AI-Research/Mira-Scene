import os

os.environ["LIDRA_SKIP_INIT"] = "true"
import sys
from pathlib import Path

SAM3D_DIR = os.environ.get(
    "MIRA_SAM3D_ROOT", "/mnt/pfs/users/sunyangtian/Scene/sam-3d-objects"
)
sys.path.append(SAM3D_DIR)

from loguru import logger
from PIL import Image

import numpy as np
import torch

from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

from copy import deepcopy
from typing import Union, Optional

from omegaconf import OmegaConf
from hydra.utils import instantiate


def merge_mask_to_rgba(image, mask):
    mask = mask.astype(np.uint8) * 255
    mask = mask[..., None]
    # embed mask in alpha channel
    rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
    return rgba_image


def make_scene_mesh(outputs, glb_list, in_place=False):
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    import trimesh
    scene = trimesh.Scene()
    for glb, output in zip(glb_list, outputs):

        # align SLAT mesh (y up, [-0.5,0.5] coord) to pcd (z up, [-0.5,0.5] coord)
        R_align = trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,
            direction=[1, 0, 0],
            point=[0, 0, 0],
        )
        glb.apply_transform(R_align)
        # glb.vertices are now in canonical space (z up, [-0.5, 0.5])

        if isinstance(output['R'], torch.Tensor):
            R = output['R'].cpu().numpy()
            t = output['t'].cpu().numpy()
            s = float(output['s'].cpu().numpy())
        else:
            R = np.asarray(output['R'])
            t = np.asarray(output['t'])
            s = float(output['s'])

        # Build 4x4 canonical -> world transform:  v_world = s * R @ v_canonical + t
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3,  3] = t

        scene.add_geometry(glb, transform=T)

    return scene



def _sam3d_config_and_workspace(config_file):
    """Return the config path and its logical checkpoint workspace.

    Hugging Face snapshots store ``pipeline.yaml`` as a symlink to a content
    addressed file under ``blobs/``.  The pipeline config refers to sibling
    YAML/checkpoint files by relative name, so using the resolved blob parent
    as ``workspace_dir`` breaks those references.  ``path_value`` may already
    have resolved the symlink before this function is called; in that case,
    recover the snapshot entry that points to the blob.
    """
    supplied = Path(config_file).expanduser().absolute()
    if not supplied.is_file():
        raise FileNotFoundError(f"SAM3D pipeline config does not exist: {supplied}")

    logical_config = supplied
    if supplied.parent.name == "blobs":
        model_cache = supplied.parent.parent
        candidates = list(
            model_cache.glob("snapshots/*/checkpoints/pipeline.yaml")
        )
        matching = [
            path
            for path in candidates
            if path.is_file() and path.resolve() == supplied.resolve()
        ]
        if matching:
            logical_config = matching[0]

    workspace = logical_config.parent
    config = OmegaConf.load(logical_config)
    relative_dependencies = [
        value
        for key, value in config.items()
        if (key.endswith("_config_path") or key.endswith("_ckpt_path"))
        and value
        and not os.path.isabs(str(value))
    ]
    missing = [
        str(value)
        for value in relative_dependencies
        if not (workspace / str(value)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"SAM3D checkpoint workspace is incomplete: {workspace}; "
            f"missing relative dependencies: {', '.join(missing)}"
        )
    return logical_config, workspace, config


def load_sam3d_pipeline(config_file):
    config_path, workspace, config = _sam3d_config_and_workspace(config_file)
    logger.info(f"Loading SAM3D config {config_path} with workspace {workspace}")
    config.workspace_dir = str(workspace)
    config.compile_model = False
    config._target_ = f"{__name__}.CustomInferencePipelinePointMap"
    pipeline = instantiate(config)
    return pipeline




class CustomInferencePipelinePointMap(InferencePipelinePointMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
    ) -> dict:
        image = self.merge_image_and_mask(image, mask)
        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            if estimate_plane:
                return self.estimate_plane(pointmap_dict, image)

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )

            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            # We could probably use the decoder from the models themselves
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                }
                # return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)

            try:
                if (
                    with_layout_postprocess
                    and self.layout_post_optimization_method is not None
                ):
                    assert glb is not None, "require mesh to run postprocessing"
                    logger.info("Running layout post optimization method...")
                    postprocessed_pose = self.run_post_optimization(
                        deepcopy(glb),
                        pointmap_dict["intrinsics"],
                        ss_return_dict,
                        ss_input_dict,
                    )
                    ss_return_dict.update(postprocessed_pose)
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            # glb.export("sample.glb")
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
                "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
            }
        

    def run_stage2(
        self,
        image: Union[None, Image.Image, np.ndarray],
        coords: torch.Tensor,
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
    ) -> dict:
        image = self.merge_image_and_mask(image, mask)
        with self.device: 

            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)

            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)

            try:
                if (
                    with_layout_postprocess
                    and self.layout_post_optimization_method is not None
                ):
                    logger.info("Not Implemented ... ")
                    pass
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            # glb.export("sample.glb")
            logger.info("Finished!")

            return {
                **outputs,
            }
