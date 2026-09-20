"""
depth_estimation.py

Unified interface for monocular depth / camera-space point map estimation.

Supported methods:
  - "moge":  MoGe (Ruicheng/moge-vitl)
  - "moge2": MoGe-2 metric depth (Ruicheng/moge-2-vitl)
  - "ppd":   Pixel-Perfect Depth

All methods return a unified DepthEstimationResult:

    camera_pts_map : np.ndarray [H, W, 3]   — camera-space 3D points (OpenGL: +X right, +Y up, -Z forward)
    valid_mask     : np.ndarray [H, W]       — bool, True = valid pixel
    depth          : np.ndarray [H, W]       — positive depth values
    intrinsics     : np.ndarray [3, 3]       — camera intrinsic matrix (pixel units)
    fov_x_rad      : float                   — horizontal FOV in radians

Usage:
    from depth_estimation import create_depth_estimator

    estimator = create_depth_estimator("moge")
    result = estimator.estimate(image_np)  # image_np: [H, W, 3] float32 in [0, 1]

    result.camera_pts_map   # [H, W, 3] OpenGL
    result.valid_mask       # [H, W] bool
    result.depth            # [H, W] float
    result.fov_x_rad        # float
"""

from dataclasses import dataclass

import numpy as np
import torch
import os
import sys


# ---------------------------------------------------------------------------
# Unified output format
# ---------------------------------------------------------------------------

@dataclass
class DepthEstimationResult:
    """Unified output from any depth estimation method."""
    camera_pts_map: np.ndarray    # [H, W, 3] camera-space points (OpenGL convention)
    valid_mask: np.ndarray        # [H, W] bool
    depth: np.ndarray             # [H, W] positive depth
    intrinsics: np.ndarray        # [3, 3] pixel-unit intrinsic matrix
    fov_x_rad: float              # horizontal FOV in radians


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DepthEstimator:
    """Base class for depth estimation methods."""

    def estimate(self, image: np.ndarray) -> DepthEstimationResult:
        """Estimate depth from a single image.

        Args:
            image: [H, W, 3] float32 in [0, 1], RGB format.

        Returns:
            DepthEstimationResult with all fields populated.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MoGe estimator
# ---------------------------------------------------------------------------

class MoGeEstimator(DepthEstimator):
    """Depth estimation via MoGe (monocular geometry)."""

    def __init__(self, pretrained: str = "Ruicheng/moge-vitl", device: str = "cuda"):
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(self.pretrained)
            self.model.eval().float().to(self.device)
            for p in self.model.parameters():
                p.requires_grad_(False)

    def estimate(self, image: np.ndarray) -> DepthEstimationResult:
        self._ensure_model()

        H, W = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            output = self.model.infer(image_tensor)

        # MoGe outputs in OpenCV convention (+X right, +Y down, +Z forward)
        # Convert to OpenGL (+X right, +Y up, -Z forward)
        points = output["points"].cpu().numpy()      # [H, W, 3] OpenCV
        depth = output["depth"].cpu().numpy()         # [H, W]
        mask = output["mask"].cpu().numpy()           # [H, W]

        # OpenCV → OpenGL
        camera_pts = points.copy()
        camera_pts[..., 1] *= -1   # Y down → Y up
        camera_pts[..., 2] *= -1   # Z forward → -Z forward

        # Intrinsics (MoGe returns normalized intrinsics)
        intrinsics_norm = output["intrinsics"].cpu().numpy()  # [3, 3]
        intrinsics = intrinsics_norm.copy()
        intrinsics[0, 0] *= W
        intrinsics[1, 1] *= H
        intrinsics[0, 2] *= W
        intrinsics[1, 2] *= H

        # FOV from intrinsics
        fx = float(intrinsics[0, 0])
        fov_x_rad = float(2.0 * np.arctan(W / 2.0 / fx))

        return DepthEstimationResult(
            camera_pts_map=camera_pts.astype(np.float32),
            valid_mask=mask.astype(bool),
            depth=depth.astype(np.float32),
            intrinsics=intrinsics.astype(np.float32),
            fov_x_rad=fov_x_rad,
        )


class MoGe2Estimator(DepthEstimator):
    """Depth estimation via MoGe-2 (metric point/depth prediction)."""

    DEFAULT_CHECKPOINT = os.environ.get(
        "MIRA_MOGE2_CHECKPOINT",
        "/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl",
    )
    DEFAULT_REPO = os.environ.get("MIRA_MOGE2_ROOT", "/mnt/pfs/users/sunyangtian/projectpp/MoGe")

    def __init__(
        self,
        pretrained: str = DEFAULT_CHECKPOINT,
        repo_root: str = DEFAULT_REPO,
        device: str = "cuda",
        resolution_level: int = 9,
        num_tokens: int = None,
        use_fp16: bool = True,
    ):
        self.pretrained = pretrained
        self.repo_root = repo_root
        self.device = torch.device(device)
        self.resolution_level = resolution_level
        self.num_tokens = num_tokens
        self.use_fp16 = use_fp16
        self.model = None

    @staticmethod
    def _resolve_checkpoint(path: str) -> str:
        """Accept a model.pt path, HF snapshot directory, or cache root."""
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            direct = os.path.join(path, "model.pt")
            if os.path.isfile(direct):
                return direct
            snapshots = os.path.join(path, "snapshots")
            candidates = []
            if os.path.isdir(snapshots):
                for name in sorted(os.listdir(snapshots), reverse=True):
                    candidate = os.path.join(snapshots, name, "model.pt")
                    if os.path.isfile(candidate):
                        candidates.append(candidate)
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"MoGe-2 checkpoint not found: {path} (expected model.pt or HF cache root)"
        )

    def _ensure_model(self):
        if self.model is not None:
            return
        try:
            import utils3d_moge  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "MoGe-2 requires the pinned utils3d_moge package; the unrelated "
                "utils3d package does not provide utils3d.pt. Install it with: "
                "python -m pip install --no-build-isolation --no-deps "
                "'utils3d_moge @ git+https://github.com/EasternJournalist/"
                "utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37'"
            ) from error
        if self.repo_root:
            # Put the reference implementation first so an unrelated v1
            # `moge` package installed in the environment cannot be imported.
            if self.repo_root in sys.path:
                sys.path.remove(self.repo_root)
            sys.path.insert(0, self.repo_root)
        from moge.model.v2 import MoGeModel

        checkpoint = self._resolve_checkpoint(self.pretrained)
        self.model = MoGeModel.from_pretrained(checkpoint).to(self.device).eval()

    def estimate(self, image: np.ndarray) -> DepthEstimationResult:
        self._ensure_model()
        H, W = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        use_fp16 = bool(self.use_fp16 and self.device.type == "cuda")
        with torch.inference_mode():
            output = self.model.infer(
                image_tensor,
                num_tokens=self.num_tokens,
                resolution_level=self.resolution_level,
                use_fp16=use_fp16,
                apply_mask=True,
            )

        # MoGe-2 returns metric points/depth in OpenCV convention
        # (+X right, +Y down, +Z forward), with normalized intrinsics.
        points = output["points"].detach().cpu().numpy()
        depth = output["depth"].detach().cpu().numpy()
        mask = output.get("mask")
        mask = mask.detach().cpu().numpy().astype(bool) if mask is not None else np.ones((H, W), dtype=bool)
        intrinsics_norm = output["intrinsics"].detach().cpu().numpy()
        if points.shape != (H, W, 3) or depth.shape != (H, W):
            raise RuntimeError(
                f"MoGe-2 returned points/depth shapes {points.shape}/{depth.shape}, expected {(H, W, 3)}/{(H, W)}"
            )

        K = intrinsics_norm.astype(np.float32, copy=True)
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
        finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
        valid = mask & finite

        camera_pts = points.astype(np.float32, copy=True)
        camera_pts[..., 1] *= -1.0  # OpenCV Y-down -> OpenGL Y-up
        camera_pts[..., 2] *= -1.0  # OpenCV Z-forward -> OpenGL -Z-forward
        depth_out = depth.astype(np.float32, copy=True)
        depth_out[~valid] = np.inf
        camera_pts[~valid] = np.inf
        fov_x_rad = float(2.0 * np.arctan(W / 2.0 / K[0, 0]))
        return DepthEstimationResult(
            camera_pts_map=camera_pts,
            valid_mask=valid,
            depth=depth_out,
            intrinsics=K,
            fov_x_rad=fov_x_rad,
        )


# ---------------------------------------------------------------------------
# Pixel-Perfect Depth estimator
# ---------------------------------------------------------------------------

class PPDEstimator(DepthEstimator):
    """Depth estimation via Pixel-Perfect Depth (PPD + MoGe metric alignment)."""

    def __init__(
        self,
        ppd_checkpoint: str = os.environ.get("MIRA_PPD_CHECKPOINT", "/mnt/pfs/users/sunyangtian/project/pixel-perfect-depth/checkpoints/ppd.pth"),
        moge_checkpoint: str = os.environ.get("MIRA_PPD_MOGE_CHECKPOINT", "/mnt/pfs/users/sunyangtian/project/pixel-perfect-depth/checkpoints/moge2.pt"),
        da2_checkpoint: str = os.environ.get("MIRA_PPD_DA2_CHECKPOINT", "/mnt/pfs/users/sunyangtian/project/pixel-perfect-depth/checkpoints/depth_anything_v2_vitl.pth"),
        device: str = "cuda",
    ):
        self.ppd_checkpoint = ppd_checkpoint
        self.moge_checkpoint = moge_checkpoint
        self.da2_checkpoint = da2_checkpoint
        self.device = torch.device(device)
        self.ppd_model = None
        self.moge_model = None

    def _ensure_models(self):
        if self.ppd_model is not None:
            return

        import sys
        ppd_root = os.environ.get("MIRA_PPD_ROOT", "/mnt/pfs/users/sunyangtian/project/pixel-perfect-depth")
        if ppd_root not in sys.path:
            sys.path.insert(0, ppd_root)

        from ppd.moge.model.v2 import MoGeModel
        from ppd.models.ppd import PixelPerfectDepth

        self.moge_model = MoGeModel.from_pretrained(self.moge_checkpoint)
        self.moge_model = self.moge_model.to(self.device).eval()

        self.ppd_model = PixelPerfectDepth(
            semantics_model="DA2",
            semantics_pth=self.da2_checkpoint,
            sampling_steps=20,
        )
        self.ppd_model.load_state_dict(
            torch.load(self.ppd_checkpoint, map_location="cpu"), strict=False
        )
        self.ppd_model = self.ppd_model.to(self.device).eval()

    def estimate(self, image: np.ndarray) -> DepthEstimationResult:
        self._ensure_models()

        import cv2
        from ppd.utils.align_depth_func import recover_metric_depth_ransac

        H, W = image.shape[:2]
        # PPD expects BGR uint8
        image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # PPD inference → relative depth
        depth_rel, resize_image = self.ppd_model.infer_image(image_bgr)
        depth_rel = depth_rel.squeeze().cpu().numpy()
        resize_H, resize_W = resize_image.shape[:2]

        # MoGe → metric depth + intrinsics (for scale alignment)
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_tensor = torch.tensor(
            moge_image / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        moge_depth, mask, intrinsic_norm = self.moge_model.infer(moge_tensor)

        moge_depth_np = moge_depth.cpu().numpy() if isinstance(moge_depth, torch.Tensor) else np.asarray(moge_depth)
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask, dtype=bool)
        intrinsic_np = intrinsic_norm.cpu().numpy() if isinstance(intrinsic_norm, torch.Tensor) else np.asarray(intrinsic_norm)
        moge_depth_np[~mask_np] = moge_depth_np[mask_np].max()

        # Denormalize intrinsics
        K = intrinsic_np.copy()
        K[0, 0] *= resize_W
        K[1, 1] *= resize_H
        K[0, 2] *= resize_W
        K[1, 2] *= resize_H

        # Align PPD depth to metric scale via RANSAC
        metric_depth = recover_metric_depth_ransac(depth_rel, moge_depth_np, mask_np)

        # Resize to original resolution
        metric_depth_orig = cv2.resize(metric_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_orig = cv2.resize(
            mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Scale intrinsics to original resolution
        K_orig = K.copy()
        K_orig[0, 0] *= W / resize_W
        K_orig[1, 1] *= H / resize_H
        K_orig[0, 2] *= W / resize_W
        K_orig[1, 2] *= H / resize_H

        # Unproject to point map (OpenCV convention)
        u, v = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        fx, fy = K_orig[0, 0], K_orig[1, 1]
        cx, cy = K_orig[0, 2], K_orig[1, 2]
        Z = metric_depth_orig
        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        point_map_cv = np.stack([X, Y, Z], axis=-1)  # OpenCV

        # OpenCV → OpenGL
        camera_pts = point_map_cv.copy()
        camera_pts[..., 1] *= -1
        camera_pts[..., 2] *= -1

        # Statistical outlier removal
        import open3d as o3d
        valid_indices = np.where(mask_orig.ravel())[0]
        pts_valid = point_map_cv.reshape(-1, 3)[valid_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_valid)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        # Update mask to exclude outliers
        filtered_mask = np.zeros(H * W, dtype=bool)
        filtered_mask[valid_indices[ind]] = True
        mask_orig = filtered_mask.reshape(H, W)

        # FOV
        fov_x_rad = float(2.0 * np.arctan(W / 2.0 / K_orig[0, 0]))

        return DepthEstimationResult(
            camera_pts_map=camera_pts.astype(np.float32),
            valid_mask=mask_orig,
            depth=metric_depth_orig.astype(np.float32),
            intrinsics=K_orig.astype(np.float32),
            fov_x_rad=fov_x_rad,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ESTIMATORS = {
    "moge": MoGeEstimator,
    "moge2": MoGe2Estimator,
    "ppd": PPDEstimator,
}


def create_depth_estimator(method: str = "moge", **kwargs) -> DepthEstimator:
    """Create a depth estimator by method name.

    Args:
        method: "moge", "moge2" or "ppd"
        **kwargs: passed to the estimator constructor

    Returns:
        DepthEstimator instance
    """
    if method not in _ESTIMATORS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(_ESTIMATORS.keys())}")
    return _ESTIMATORS[method](**kwargs)
