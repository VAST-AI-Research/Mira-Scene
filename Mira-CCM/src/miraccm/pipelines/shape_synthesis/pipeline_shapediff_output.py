from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from diffusers.utils import BaseOutput


@dataclass
class SparseStructurePipelineOutput(BaseOutput):
    r"""
    Output class for sparse structure pipelines.
    """

    samples: torch.Tensor
    pcds: List[np.ndarray]
    coords: List[torch.Tensor]
    latent_voxel_cam_pts: Optional[torch.Tensor] = None
