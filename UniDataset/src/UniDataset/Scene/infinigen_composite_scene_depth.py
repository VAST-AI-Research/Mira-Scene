"""
Infinigen composite scene dataset with depth-based canonical coordinate map.

Based on blenderproc_scene_lightweight.py, but:
  - Removes latent_voxel_centers / latent_voxel_cam_pts
  - Removes MoGe point-map loading
  - Adds view_samples_dir: loads depth.npy from
      {view_samples_dir}/{unique_id}/depth.npy
    where unique_id = preprocessed_view['id'] (e.g. "scene__floor_0__0")
  - For each selected object instance i, computes:
        canonical_coord_map[i] = depth_to_canonical_coord_map(
            depth, mask[i], fov,
            T_cam_to_canonical[i] = inv(world_to_cam @ mesh_to_world[i])
        )
  - Returns pack entries matching ThreeDFutureSceneDepthDataset /
    ObjaverseSceneDepthDataset

Transform chain per object:
    Z-up canonical  →  (mesh_to_world[i])
                    →  Y-up world space
                    →  (world_to_cam)
                    →  camera space (OpenGL: +X right, +Y up, -Z forward)

depth_to_canonical_coord_map unprojection (same as threedfuture_scene_depth):
    X_cam = (u - cx) * d / fx
    Y_cam = -(v - cy) * d / fx   (+Y up, image v is down)
    Z_cam = -d                   (-Z into scene)
    canonical = T_cam_to_can @ [X_cam, Y_cam, Z_cam, 1]
"""

import hashlib
import io
import json
import os
import random
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from .typing import *
from UniDataset.utils.config import parse_structured
from UniDataset.utils.img_and_mask_transforms import crop_around_mask

from ..utils.normalization_utils import normalize_object
from ..utils.voxel_utils import voxelize_trimesh_obj, gen_voxel_grid

# Enable OpenEXR reading before cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


# ---------------------------------------------------------------------------
# Depth loading (same logic as threedfuture_scene_depth.py)
# ---------------------------------------------------------------------------

def load_depth(view_dir: str) -> np.ndarray:
    """
    Load depth from view_dir.  Tries depth.npy first, then depth.exr.
    Returns (H, W) float32.  Raises FileNotFoundError if neither exists.
    """
    npy_path = os.path.join(view_dir, "depth.npy")
    exr_path = os.path.join(view_dir, "depth.exr")

    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)

    if os.path.exists(exr_path):
        try:
            import cv2 as _cv2
            arr = _cv2.imread(exr_path, _cv2.IMREAD_ANYCOLOR | _cv2.IMREAD_ANYDEPTH)
        except ImportError:
            arr = None
        if arr is None:
            raise FileNotFoundError(
                f"depth.exr found but could not be loaded in {view_dir} (cv2 unavailable or read error)"
            )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    raise FileNotFoundError(
        f"No depth file found in {view_dir} (tried depth.npy and depth.exr)"
    )


# ---------------------------------------------------------------------------
# Depth → canonical coordinate map
# ---------------------------------------------------------------------------

def depth_to_canonical_coord_map(
    depth: np.ndarray,               # (H, W) float32 – Z-depth, positive in front
    mask: np.ndarray,                # (H, W) bool/uint8 – 1 = valid foreground
    fov_rad: float,                  # horizontal FOV in radians
    T_cam_to_canonical: np.ndarray,  # (4, 4) float64 – camera OpenGL → canonical
) -> np.ndarray:                     # (H, W, 3) float32
    """
    Unproject depth to camera space (OpenGL convention), then transform to
    canonical space via T_cam_to_canonical.

    Camera OpenGL convention:
        X = (u - cx) * d / fx    (+X right)
        Y = -(v - cy) * d / fx   (+Y up, image v is down)
        Z = -d                   (-Z into scene)

    Background / invalid pixels are set to zero in the output.
    """
    H, W = depth.shape
    fx = 0.5 * W / np.tan(0.5 * fov_rad)
    cx, cy = W / 2.0, H / 2.0

    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth < 1e6)
    )

    canonical_map = np.zeros((H, W, 3), dtype=np.float32)
    if not np.any(valid):
        return canonical_map

    v_idx, u_idx = np.where(valid)
    d = depth[valid].astype(np.float64)

    X_cam = (u_idx.astype(np.float64) - cx) * d / fx
    Y_cam = -(v_idx.astype(np.float64) - cy) * d / fx
    Z_cam = -d

    ones  = np.ones(len(d), dtype=np.float64)
    cam_h = np.stack([X_cam, Y_cam, Z_cam, ones], axis=0)   # (4, N)
    can_h = T_cam_to_canonical.astype(np.float64) @ cam_h   # (4, N)
    canonical_map[v_idx, u_idx] = can_h[:3].T.astype(np.float32)
    return canonical_map


# ---------------------------------------------------------------------------
# Azimuth-based canonical rotation (same convention as objaverse_scene_depth_dataset_alpha)
# ---------------------------------------------------------------------------

_ROT_Z = {
    0:   np.eye(3, dtype=np.float32),
    90:  np.array([[ 0, -1, 0], [ 1,  0, 0], [0, 0, 1]], dtype=np.float32),
    -90: np.array([[ 0,  1, 0], [-1,  0, 0], [0, 0, 1]], dtype=np.float32),
    180: np.array([[-1,  0, 0], [ 0, -1, 0], [0, 0, 1]], dtype=np.float32),
}

_DIAGONAL_AZIMUTH_CENTERS = (-135.0, -45.0, 45.0, 135.0)


def _compute_azimuth_and_rotation(
    cam_pos_world: np.ndarray,   # (3,) camera position in Y-up world space
    mesh_to_world: np.ndarray,   # (4, 4) Z-up canonical → Y-up world
) -> Tuple[float, int]:
    """
    Compute camera azimuth in canonical space and the quantized Z-rotation
    (0/±90/180) needed to bring the canonical object's front toward the
    camera.

    Steps:
      1. Transform camera from Y-up world to Z-up canonical space.
      2. Compute azimuth = atan2(x, -y)  (Blender Z-up, 0° = -Y front).
      3. Quantize to nearest 90°.
    """
    cam_can = (np.linalg.inv(mesh_to_world) @ np.append(cam_pos_world, 1.0))[:3]
    az = float(np.degrees(np.arctan2(cam_can[0], -cam_can[1])))

    if -45 <= az <= 45:
        rot_deg = 0
    elif 45 < az <= 135:
        rot_deg = -90
    elif -135 <= az < -45:
        rot_deg = 90
    else:           # > 135 or < -135
        rot_deg = 180

    return az, rot_deg


def _wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _is_diagonal_azimuth(azimuth_deg: float, margin_deg: float) -> bool:
    """Return True when azimuth lies near a diagonal direction."""
    azimuth_deg = _wrap_angle_deg(azimuth_deg)
    for center_deg in _DIAGONAL_AZIMUTH_CENTERS:
        if abs(_wrap_angle_deg(azimuth_deg - center_deg)) <= margin_deg:
            return True
    return False


def _rotate_coord_map_z(coord_map: torch.Tensor, rot_deg: int) -> torch.Tensor:
    """Rotate coordinate values in [3, H, W] around the Z axis."""
    if rot_deg == 0:
        return coord_map
    R = torch.from_numpy(_ROT_Z[rot_deg])
    C, H, W = coord_map.shape
    return (R @ coord_map.reshape(3, -1)).reshape(C, H, W)


def _rotate_voxel_z(voxel: torch.Tensor, rot_deg: int) -> torch.Tensor:
    """Rotate a [R, R, R] occupancy grid around Z by 0/±90/180°."""
    if rot_deg == 0:
        return voxel
    if rot_deg == -90:      # (x,y,z) → (y, -x, z)
        return voxel.permute(1, 0, 2).flip(1)
    if rot_deg == 90:       # (x,y,z) → (-y, x, z)
        return voxel.permute(1, 0, 2).flip(0)
    return voxel.flip(0).flip(1)   # 180°: (x,y,z) → (-x, -y, z)


def _rotate_bbox_z(bbox: np.ndarray, rot_deg: int) -> np.ndarray:
    """Rotate a (2, 3) min/max bbox around Z; recompute min/max from 8 corners."""
    if rot_deg == 0:
        return bbox
    R = _ROT_Z[rot_deg]
    lo, hi = bbox[0], bbox[1]
    corners = np.array([
        [lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]],
        [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]],
        [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]],
    ], dtype=np.float32)
    rot = (R @ corners.T).T
    return np.stack([rot.min(axis=0), rot.max(axis=0)])


# ---------------------------------------------------------------------------
# Scale-aware voxel cache helpers
# ---------------------------------------------------------------------------

def _voxel_scale_cache_path(
    cache_dir: str,
    model_id: str,
    scale_vec: np.ndarray,
    voxel_res: int,
    geometry_sha256: str,
) -> str:
    """
    Build a cache path keyed on geometry content, scale aspect ratio and resolution.

    Only the aspect ratio of scale_vec matters because normalize_object()
    always rescales the mesh to [-0.5, 0.5].  Dividing by the max absolute
    component collapses all uniform scales to the same key.
    """
    if len(geometry_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in geometry_sha256.lower()
    ):
        raise ValueError("geometry_sha256 must be a 64-character hexadecimal digest")
    ratio = scale_vec / np.max(np.abs(scale_vec))
    ratio_str = "_".join(f"{v:.3f}" for v in np.round(ratio, 3))
    filename = (
        f"{model_id}__geom{geometry_sha256[:16]}__{ratio_str}__voxel{voxel_res}.pt"
    )
    return os.path.join(cache_dir, filename)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InfinigenCompositeSceneDepthDatasetConfig:
    renderings_root: str = ""
    poses_dir: str = ""
    model_data_dir: str = ""
    cam_K_path: str = ""
    output_dir: str = ""
    scene_id: str = ""
    height: int = 360
    width: int = 480
    data_indices: Optional[List[Any]] = field(default_factory=lambda: [0, -1])
    repeat: int = 1
    seed: int = 42
    debug: bool = False
    sort_scene_state_objects: bool = True
    cache_scene_geometry: bool = True
    dedup_objects: bool = True
    num_instances_per_batch: int = -1
    error_log_path: str = "error_scale_zero.log"
    preprocess_json_path: str = ""
    preprocessed_only_valid_views: bool = True

    # Optional runtime quality filters applied to the resized instance mask.
    # A value <= 0 disables the corresponding filter.  These are deliberately
    # runtime guards in addition to (and independent of) preprocess.json.
    min_mask_area_ratio: float = 0.0
    min_mask_bbox_short_px: int = 0

    # Depth source: {view_samples_dir}/{unique_id}/depth.npy
    # unique_id comes from preprocessed_view['id'] (e.g. "sceneid__floor_0__0")
    view_samples_dir: str = ""
    # Optional immutable ERP-nearest depth overlay; never falls back when set.
    depth_patch_root: str = ""
    depth_patch_require_ready: bool = True
    depth_resize_mode: str = "bilinear"

    with_mesh: bool = False
    use_bbox_layout: bool = False        # output canonical_bboxes when True
    num_samples_per_dim: int = 8         # unused (kept for config compat)
    skip_exists_check: bool = False      # skip os.path.exists per h5 (fast on slow NFS)
    canonicalize_azimuth: bool = False   # rotate canonical quantities so camera faces front
    voxel_cache_dir: str = ""            # disk cache keyed by geometry hash, scale and resolution

    # scale fields kept for config compat but NOT applied to canonical maps
    moge_scale_info_dir: str = ""
    scale_info_filename: str = "scale_info.json"
    use_global_scale: bool = False
    diagonal_azimuth_margin_deg: float = 15.0  # drop objects near diagonal azimuths
    include_transformation: bool = False       # include transformation & azimuth_estimate in pack


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InfinigenCompositeSceneDepthDataset(Dataset):
    """
    BlenderProc scene dataset that outputs canonical_coord_map instead of
    latent_voxel_centers / latent_voxel_cam_pts.

    For each selected object instance i:
      1. Load depth.npy from {view_samples_dir}/{unique_id}/depth.npy
      2. Filter by object mask[i]
      3. Unproject depth to camera OpenGL space using horizontal FOV
      4. Apply inv(world_to_cam @ mesh_to_world[i]) to get Z-up canonical coords

    Pack keys mirror ThreeDFutureSceneDepthDataset / ObjaverseSceneDepthDataset:
      id, num_instances, rgb, mask, masks, rgb_scene,
      rgb_cropped, mask_cropped, fov, height, width,
      voxel, voxel_res, canonical_coord_map, select_indices
    Optional:
      surface, scene_mesh  (with_mesh=True)
      canonical_bboxes     (use_bbox_layout=True)
    """

    def __init__(self, split: str = "train", **kwargs) -> None:
        super().__init__()
        self.cfg: InfinigenCompositeSceneDepthDatasetConfig = parse_structured(
            InfinigenCompositeSceneDepthDatasetConfig, kwargs
        )
        if self.cfg.min_mask_area_ratio < 0:
            raise ValueError("min_mask_area_ratio must be >= 0")
        if self.cfg.min_mask_bbox_short_px < 0:
            raise ValueError("min_mask_bbox_short_px must be >= 0")
        if not self.cfg.preprocess_json_path:
            raise ValueError(
                "InfinigenCompositeSceneDepthDataset requires preprocess_json_path."
            )
        if self.cfg.depth_resize_mode not in ("nearest", "bilinear"):
            raise ValueError("depth_resize_mode must be nearest or bilinear")
        if self.cfg.depth_patch_root:
            if self.cfg.depth_resize_mode != "nearest":
                raise ValueError("Reprojected depth requires nearest resize, aligned with masks")
            marker = ".READY.json" if self.cfg.depth_patch_require_ready else "BUILD_COMPLETE.json"
            marker_path = os.path.join(self.cfg.depth_patch_root, marker)
            if not os.path.isfile(marker_path):
                raise RuntimeError(f"Depth patch is not ready: {marker_path}")
            with open(marker_path) as stream:
                patch = json.load(stream)
            with open(self.cfg.preprocess_json_path, "rb") as stream:
                preprocess_hash = hashlib.sha256(stream.read()).hexdigest()
            if (patch.get("processing_version") != "erp-nearest-depth-v2"
                    or patch.get("preprocess_sha256") != preprocess_hash
                    or os.path.realpath(patch.get("source_root", "") + "/renderings")
                    != os.path.realpath(self.cfg.renderings_root)):
                raise RuntimeError("Depth patch does not match the configured dataset/index")
            if patch.get("failed_views", -1) != 0 or patch.get("completed_views") != patch.get("expected_views"):
                raise RuntimeError("Depth patch is incomplete")
        random.seed(self.cfg.seed)

        if not self.cfg.poses_dir:
            self.cfg.poses_dir = os.path.join(self.cfg.renderings_root, "poses")

        self._scene_state_cache: Dict[str, Dict[str, Any]] = {}
        self._scale_info_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._use_preprocessed_index: bool = False
        self._preprocessed_entries: List[Tuple[str, str, Dict[str, Any]]] = []
        self._preprocessed_view_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._preprocessed_scene_obj_keys: Dict[str, List[str]] = {}
        self._preprocessed_scene_obj_key_to_index: Dict[str, Dict[str, int]] = {}
        self._preprocessed_scene_required_keys: Dict[str, Set[str]] = {}
        self._load_preprocessed_index()

        self.all_items = self._collect_items()
        self.split_data(self.cfg.data_indices)
        self.all_items = [self.all_items[i] for i in self._allowed_indices]
        unique_scenes = len({scene_id for scene_id, _ in self.all_items})
        print(
            f"InfinigenCompositeSceneDepthDataset: {len(self.all_items)} entries "
            f"({unique_scenes} unique scenes, split={split})"
        )

        # Per-scene cam_K: try loading from each scene dir; fallback to global/config path
        self._scene_cam_K_cache: Dict[str, Optional[np.ndarray]] = {}
        self._global_cam_K: Optional[np.ndarray] = None
        cam_k_path = self.cfg.cam_K_path
        if not cam_k_path and self.cfg.renderings_root:
            cam_k_path = os.path.join(self.cfg.renderings_root, "cam_K.npy")
        if cam_k_path and os.path.exists(cam_k_path):
            self._global_cam_K = np.load(cam_k_path)

        self.voxel_res = 64
        self.box_size_factor = 1.2

        # Scale-aware voxel cache: disk path + in-memory LRU dict
        self.voxel_cache_dir = self.cfg.voxel_cache_dir or (
            os.path.join(self.cfg.model_data_dir, "..", "voxel_cache_blenderproc")
            if self.cfg.model_data_dir else ""
        )
        self._voxel_mem_cache: OrderedDict = OrderedDict()
        self._voxel_mem_cache_maxsize = 500
        self._model_geometry_sha256_cache: Dict[str, Tuple[int, int, str]] = {}

        self.moge_scale_info_dir = self.cfg.moge_scale_info_dir or (
            os.path.join(self.cfg.renderings_root, "moge_output")
            if self.cfg.renderings_root
            else ""
        )

    # ------------------------------------------------------------------
    # Internal helpers (copied / adapted from BlenderProcSceneLightweightDataset)
    # ------------------------------------------------------------------

    def _collect_items(self) -> List[Tuple[str, str]]:
        items = []
        for scene_id, view_relpath, rec in self._preprocessed_entries:
            if self.cfg.scene_id and scene_id != self.cfg.scene_id:
                continue
            if self.cfg.preprocessed_only_valid_views and not rec.get("valid", False):
                continue
            view_relpath_os = view_relpath.replace("/", os.sep)
            h5_path = os.path.join(
                self.cfg.renderings_root, scene_id, view_relpath_os
            )
            if self.cfg.skip_exists_check or os.path.exists(h5_path):
                items.append((scene_id, h5_path))
            elif self.cfg.debug:
                print(f"[debug] Missing HDF5 from preprocess index: {h5_path}")
        items.sort(key=lambda x: (x[0], x[1]))
        if len(items) == 0:
            raise ValueError(
                "No valid views found from preprocess_json_path after filtering."
            )
        return items

    @staticmethod
    def _normalize_relpath(relpath: str) -> str:
        return relpath.replace("\\", "/")

    def _load_preprocessed_index(self) -> None:
        path = self.cfg.preprocess_json_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"preprocess_json_path not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        scene_results = data.get("scene_results", [])
        for scene_rec in scene_results:
            scene_id = str(scene_rec.get("scene_id", ""))
            if not scene_id:
                continue
            obj_keys = scene_rec.get("obj_keys", [])
            if isinstance(obj_keys, list):
                self._preprocessed_scene_obj_keys[scene_id] = obj_keys
                self._preprocessed_scene_obj_key_to_index[scene_id] = {
                    key: idx for idx, key in enumerate(obj_keys)
                }
            self._preprocessed_scene_required_keys.setdefault(scene_id, set())

        results = data.get("results", [])
        for rec in results:
            scene_id = str(rec.get("scene_id", ""))
            relpath = rec.get("view_relpath", "")
            if not scene_id or not relpath:
                continue
            relpath_norm = self._normalize_relpath(relpath)
            self._preprocessed_entries.append((scene_id, relpath_norm, rec))
            self._preprocessed_view_by_key[(scene_id, relpath_norm)] = rec

            scene_obj_keys = self._preprocessed_scene_obj_keys.get(scene_id, [])
            req = self._preprocessed_scene_required_keys.setdefault(scene_id, set())
            for idx in rec.get("valid_object_indices", []):
                if isinstance(idx, int) and 0 <= idx < len(scene_obj_keys):
                    req.add(scene_obj_keys[idx])

        self._use_preprocessed_index = len(self._preprocessed_entries) > 0
        if not self._use_preprocessed_index:
            raise ValueError(
                f"No valid view records found in preprocess json: {path}"
            )
        if self.cfg.debug:
            print(
                f"[debug] Loaded preprocess index from {path}: "
                f"views={len(self._preprocessed_entries)}, "
                f"scenes={len(self._preprocessed_scene_obj_keys)}"
            )

    def _get_preprocessed_view_record(
        self, scene_id: str, h5_path: str
    ) -> Optional[Dict[str, Any]]:
        if not self._use_preprocessed_index:
            return None
        base_dir = os.path.join(self.cfg.renderings_root, scene_id)
        try:
            relpath = os.path.relpath(h5_path, base_dir)
        except ValueError:
            return None
        relpath_norm = self._normalize_relpath(relpath)
        return self._preprocessed_view_by_key.get((scene_id, relpath_norm))

    def _get_preprocessed_selected_obj_keys(
        self, scene_id: str, preprocessed_view: Optional[Dict[str, Any]]
    ) -> List[str]:
        if preprocessed_view is None:
            return []
        scene_obj_keys = self._preprocessed_scene_obj_keys.get(scene_id, [])
        selected_keys: List[str] = []
        for idx in preprocessed_view.get("valid_object_indices", []):
            if isinstance(idx, int) and 0 <= idx < len(scene_obj_keys):
                selected_keys.append(scene_obj_keys[idx])
        return selected_keys

    @staticmethod
    def _decompose_scale(matrix_world: np.ndarray) -> np.ndarray:
        A = matrix_world[:3, :3]
        scale = np.linalg.norm(A, axis=0)
        if np.linalg.det(A) < 0:
            scale[0] *= -1.0
        return scale

    def _get_cam_K(self, scene_id: str) -> Optional[np.ndarray]:
        """Load per-scene cam_K.npy, falling back to global cam_K."""
        if scene_id in self._scene_cam_K_cache:
            return self._scene_cam_K_cache[scene_id]
        if self.cfg.renderings_root:
            per_scene_path = os.path.join(self.cfg.renderings_root, scene_id, "cam_K.npy")
            if os.path.exists(per_scene_path):
                cam_K = np.load(per_scene_path)
                self._scene_cam_K_cache[scene_id] = cam_K
                return cam_K
        self._scene_cam_K_cache[scene_id] = self._global_cam_K
        return self._global_cam_K

    def _compute_fov(self, scene_id: str, orig_w: int, preprocessed_view: Optional[Dict[str, Any]] = None) -> float:
        """Compute FOV: prefer preprocess JSON, then per-scene cam_K, then fallback 60deg."""
        if preprocessed_view is not None:
            fov_from_json = preprocessed_view.get("fov")
            if fov_from_json is not None:
                return float(fov_from_json)
        cam_K = self._get_cam_K(scene_id)
        if cam_K is not None:
            fx = float(cam_K[0, 0]) * (self.cfg.width / orig_w)
            return float(2.0 * np.arctan(self.cfg.width / (2.0 * fx)))
        return float(np.deg2rad(60))

    def _resolve_error_log_path(self) -> str:
        path = self.cfg.error_log_path
        if not path:
            return ""
        return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)

    def _log_scale_error(
        self,
        scene_id: str,
        obj_key: str,
        matrix_world: np.ndarray,
        scale_vec: np.ndarray,
    ) -> None:
        log_path = self._resolve_error_log_path()
        if not log_path:
            return
        try:
            with open(log_path, "a") as f:
                f.write(
                    "scale_zero_or_too_small\t"
                    f"scene={scene_id}\t"
                    f"key={obj_key}\t"
                    f"scale={scale_vec.tolist()}\t"
                    f"matrix={matrix_world.tolist()}\n"
                )
        except Exception:
            pass

    def _load_scale_info(self, unique_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load per-view scale_info.json and cache it."""
        if not unique_id:
            return None
        if unique_id in self._scale_info_cache:
            return self._scale_info_cache[unique_id]
        if not self.moge_scale_info_dir:
            self._scale_info_cache[unique_id] = None
            return None
        scale_path = os.path.join(
            self.moge_scale_info_dir, unique_id, self.cfg.scale_info_filename
        )
        if not os.path.exists(scale_path):
            self._scale_info_cache[unique_id] = None
            return None
        try:
            with open(scale_path, "r") as f:
                scale_info = json.load(f)
            self._scale_info_cache[unique_id] = scale_info
            return scale_info
        except Exception:
            self._scale_info_cache[unique_id] = None
            return None

    def _voxel_mem_insert(self, key: str, voxel_dict: Any) -> None:
        """Insert into the LRU memory cache, evicting the oldest entry if full."""
        self._voxel_mem_cache[key] = voxel_dict
        self._voxel_mem_cache.move_to_end(key)
        while len(self._voxel_mem_cache) > self._voxel_mem_cache_maxsize:
            self._voxel_mem_cache.popitem(last=False)

    def _model_geometry_sha256(self, model_path: str) -> str:
        """Hash one OBJ once per dataset process and refresh after a file change."""
        stat = os.stat(model_path)
        cached = self._model_geometry_sha256_cache.get(model_path)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
        digest = hashlib.sha256()
        with open(model_path, "rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
        self._model_geometry_sha256_cache[model_path] = (
            stat.st_size,
            stat.st_mtime_ns,
            value,
        )
        return value

    @staticmethod
    def _valid_voxel_cache(
        voxel_dict: Any, geometry_sha256: str, voxel_res: int
    ) -> bool:
        return bool(
            isinstance(voxel_dict, dict)
            and voxel_dict.get("source_obj_sha256") == geometry_sha256
            and int(voxel_dict.get("voxel_res", -1)) == int(voxel_res)
        )

    def _load_depth(self, unique_id: str) -> Optional[np.ndarray]:
        """
        Load depth.npy from {view_samples_dir}/{unique_id}/depth.npy.
        Returns (H, W) float32 or None if not found.
        """
        if self.cfg.depth_patch_root:
            if not unique_id or os.path.basename(unique_id) != unique_id:
                raise RuntimeError(f"Invalid depth patch ID: {unique_id}")
            view_dir = os.path.join(self.cfg.depth_patch_root, "view_samples", unique_id)
            try:
                with open(os.path.join(view_dir, "metadata.json")) as stream:
                    metadata = json.load(stream)
                with open(os.path.join(view_dir, "depth.npy"), "rb") as stream:
                    payload = stream.read()
                if hashlib.sha256(payload).hexdigest() != metadata.get("depth_sha256"):
                    raise ValueError("Depth patch checksum mismatch")
                arr = np.load(io.BytesIO(payload), allow_pickle=False)
                if (metadata.get("id") != unique_id
                        or metadata.get("processing_version") != "erp-nearest-depth-v2"
                        or arr.dtype != np.float32 or arr.ndim != 2
                        or list(arr.shape) != metadata.get("shape")):
                    raise ValueError("Depth patch metadata/shape/dtype mismatch")
                return arr
            except (OSError, ValueError, KeyError, EOFError) as error:
                # RuntimeError is intentional: do not retry a different training
                # sample or silently fall back to the known-noisy old depth.
                raise RuntimeError(f"Required repaired depth failed for {unique_id}: {error}") from error
        if not self.cfg.view_samples_dir:
            return None
        view_dir = os.path.join(self.cfg.view_samples_dir, unique_id)
        try:
            return load_depth(view_dir)
        except FileNotFoundError:
            if self.cfg.debug:
                print(f"[debug] Depth not found for {unique_id} in {view_dir}")
            return None

    def _load_scene_state(self, scene_id: str) -> Dict[str, Any]:
        if scene_id in self._scene_state_cache:
            return self._scene_state_cache[scene_id]
        state_path = self._scene_state_path(scene_id)
        with open(state_path, "r") as f:
            data = json.load(f)
        self._scene_state_cache[scene_id] = data
        return data

    def _scene_state_path(self, scene_id: str) -> str:
        if self.cfg.poses_dir:
            state_path = os.path.join(
                self.cfg.poses_dir, f"{scene_id}_scene_state.json"
            )
            if os.path.exists(state_path):
                return state_path
        scene_dir = os.path.join(self.cfg.renderings_root, scene_id)
        state_path = os.path.join(scene_dir, f"{scene_id}_scene_state.json")
        if os.path.exists(state_path):
            return state_path
        raise FileNotFoundError(
            f"Scene state not found for {scene_id} in poses_dir or scene dir."
        )

    def _parse_object_key(self, key: str) -> Tuple[str, str, str]:
        if "||" in key:
            uid, rest = key.split("||", 1)
            inst_mark = rest.split("|", 1)[0]
            return uid, "", inst_mark
        parts = key.split("|")
        uid = parts[0] if len(parts) > 0 else ""
        jid = parts[1] if len(parts) > 1 else ""
        inst_mark = parts[2] if len(parts) > 2 else ""
        return uid, jid, inst_mark

    def split_data(self, data_indices):
        self._allowed_indices = list(range(len(self.all_items)))
        if len(data_indices) == 2:
            start_idx, end_idx = data_indices
            gap = 1
        elif len(data_indices) == 3:
            start_idx, end_idx, gap = data_indices
        else:
            start_idx, end_idx, gap = 0, None, 1
        if end_idx is not None and end_idx < 0:
            end_idx = len(self._allowed_indices)
        self._allowed_indices = (
            self._allowed_indices[start_idx:end_idx:gap] * self.cfg.repeat
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.all_items)

    def _should_retry_on_item_error(self, error: Exception) -> bool:
        if isinstance(error, (FileNotFoundError, IOError)):
            return True
        msg = str(error)
        retry_markers = (
            "No objects from preprocess selection are visible in segmap",
            "No objects remain after visibility / diagonal-azimuth filtering",
            "No valid objects after crop",
        )
        return any(marker in msg for marker in retry_markers)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        max_retries = min(max(len(self.all_items), 1), 20)
        cur_index = index
        for attempt in range(max_retries):
            try:
                return self._get_item(cur_index)
            except Exception as e:
                should_retry = self._should_retry_on_item_error(e)
                if not should_retry or len(self.all_items) <= 1:
                    print(f"Error processing index {cur_index}: {e}.")
                    traceback.print_exc()
                    raise
                cur_index = random.randint(0, len(self.all_items) - 1)
        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_retries} retries "
            f"starting from index {index}."
        )

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def _get_item(self, index: int) -> Dict[str, Any]:
        """
        Single-view processing pipeline:
          1) Read h5: colors / instance_segmaps / attrs / cam_Ts
          2) Select objects from preprocess valid_object_indices
          3) Build per-object transforms + voxels
          4) Load depth; compute canonical_coord_map per object
          5) Return pack with canonical_coord_map (no latent_voxel_*)
        """
        scene_id, h5_path = self.all_items[index % len(self.all_items)]
        preprocessed_view = self._get_preprocessed_view_record(scene_id, h5_path)
        if preprocessed_view is None:
            raise ValueError(
                f"View not found in preprocess index: scene={scene_id}, h5={h5_path}"
            )
        scene_state = self._load_scene_state(scene_id)

        # Determine unique_id (used for depth path and scale_info)
        unique_id = preprocessed_view.get("id") or preprocessed_view.get("unique_id")

        # ----------------------------------------------------------------
        # Load h5 data
        # ----------------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            colors = np.array(f["colors"])
            instance_segmaps = np.array(f["instance_segmaps"])
            attrs_raw = f["instance_attribute_maps"][()].decode("utf-8")
            cam_Ts = np.array(f["cam_Ts"], dtype=np.float32) if "cam_Ts" in f else None

        attrs = json.loads(attrs_raw)
        attrs_by_key: Dict[Tuple[str, str, str], List[int]] = {}
        for a in attrs:
            uid_a = a.get("uid", "")
            jid_a = a.get("jid", "")
            inst_mark_a = a.get("inst_mark", "")
            idx_a = a.get("idx")
            if inst_mark_a and idx_a is not None:
                attrs_by_key.setdefault((uid_a, jid_a, inst_mark_a), []).append(idx_a)

        orig_h, orig_w = colors.shape[:2]
        if (orig_h, orig_w) != (self.cfg.height, self.cfg.width):
            colors = cv2.resize(
                colors, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_LINEAR
            )
            instance_segmaps = cv2.resize(
                instance_segmaps,
                (self.cfg.width, self.cfg.height),
                interpolation=cv2.INTER_NEAREST,
            )

        cam_T: Optional[np.ndarray] = None
        if cam_Ts is not None:
            cam_T = cam_Ts[0] if cam_Ts.ndim == 3 else cam_Ts

        # ----------------------------------------------------------------
        # Y-up ↔ Z-up helpers
        # ----------------------------------------------------------------
        y_up2z_up     = trimesh.transformations.rotation_matrix(np.deg2rad( 90), [1, 0, 0])
        y_up2z_up_inv = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0])

        # ----------------------------------------------------------------
        # world_to_cam  (camera: OpenGL, +X right, +Y up, -Z forward)
        # ----------------------------------------------------------------
        if cam_T is not None:
            cam_T_np = cam_T.astype(np.float64)
            world_to_cam = np.linalg.inv(cam_T_np)   # (4, 4)
        else:
            world_to_cam = np.eye(4, dtype=np.float64)

        # ----------------------------------------------------------------
        # FOV
        # ----------------------------------------------------------------
        fov = self._compute_fov(scene_id, orig_w, preprocessed_view)

        # ----------------------------------------------------------------
        # Phase 1: Lightweight pre-filtering (no mesh loading)
        # Collect candidates that pass mask / matrix_world / model_path checks.
        # ----------------------------------------------------------------
        selected_obj_keys = set(
            self._get_preprocessed_selected_obj_keys(scene_id, preprocessed_view)
        )

        iter_objects = scene_state.get("objects", [])
        if self.cfg.sort_scene_state_objects:
            iter_objects = sorted(iter_objects, key=lambda o: o.get("key", ""))

        @dataclass
        class _Candidate:
            key: str
            uid: str
            jid: str
            inst_mark: str
            mask: np.ndarray
            matrix_world: np.ndarray
            model_path: str

        candidates: List[_Candidate] = []
        seen_obj_keys: set = set()
        for obj_entry in iter_objects:
            key = obj_entry.get("key", "")
            if key not in selected_obj_keys:
                continue
            uid, jid, inst_mark = self._parse_object_key(key)
            if not inst_mark or inst_mark.startswith("layout_"):
                continue
            if self.cfg.dedup_objects:
                dedup_key = (uid, jid, inst_mark)
                if dedup_key in seen_obj_keys:
                    continue
                seen_obj_keys.add(dedup_key)

            matrix_world = np.array(
                obj_entry.get("matrix_world", []), dtype=np.float32
            )
            if matrix_world.shape != (4, 4) or not np.all(np.isfinite(matrix_world)):
                continue

            inst_ids = attrs_by_key.get((uid, jid, inst_mark))
            if not inst_ids:
                continue
            mask = np.isin(instance_segmaps, inst_ids).astype(np.uint8)
            if int(mask.sum()) <= 0:
                continue

            # Apply optional quality filtering before random instance
            # subsampling and before any expensive mesh/voxel work.  The
            # thresholds are measured on the mask after the HDF5 image has
            # been resized to cfg.height x cfg.width.
            mask_area_ratio = float(mask.mean())
            if (
                self.cfg.min_mask_area_ratio > 0
                and mask_area_ratio < self.cfg.min_mask_area_ratio
            ):
                continue
            if self.cfg.min_mask_bbox_short_px > 0:
                ys, xs = np.nonzero(mask)
                bbox_short_px = min(
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                )
                if bbox_short_px < self.cfg.min_mask_bbox_short_px:
                    continue

            model_id = jid or uid
            if not model_id:
                continue
            model_path = os.path.join(
                self.cfg.model_data_dir, model_id, "raw_model.obj"
            )
            if not os.path.exists(model_path):
                continue

            candidates.append(_Candidate(
                key=key, uid=uid, jid=jid, inst_mark=inst_mark,
                mask=mask, matrix_world=matrix_world, model_path=model_path,
            ))

        if len(candidates) == 0:
            raise RuntimeError(
                f"No objects from preprocess selection are visible in segmap: "
                f"scene={scene_id}, h5={h5_path}"
            )

        # ----------------------------------------------------------------
        # Early sub-sampling: pick candidates BEFORE heavy mesh loading
        # ----------------------------------------------------------------
        num_target = self.cfg.num_instances_per_batch
        if num_target > 0 and len(candidates) > num_target:
            # Over-sample slightly to tolerate crop failures
            oversample = min(len(candidates), num_target + 2)
            candidates = random.sample(candidates, oversample)

        cam_pos_world: Optional[np.ndarray] = None
        if cam_T is not None:
            cam_pos_world = cam_T[:3, 3].astype(np.float64)

        # ----------------------------------------------------------------
        # Phase 2: Heavy per-object loading (mesh, transforms) – only
        # for the (sub-sampled) candidates
        # ----------------------------------------------------------------
        masks_list: List[np.ndarray] = []
        voxel_models_list: List[trimesh.Trimesh] = []
        mesh_to_world_list: List[np.ndarray] = []
        scale_vecs_list: List[np.ndarray] = []
        obj_keys_list: List[str] = []
        geometry_sha256_list: List[str] = []
        models: List[trimesh.Trimesh] = []   # for with_mesh
        trimesh_scene_obj = trimesh.Scene()
        canonical_bboxes_list: List[torch.Tensor] = []

        for cand in candidates:
            try:
                raw_model = trimesh.load(cand.model_path, force="mesh")
            except Exception:
                continue

            local2world = cand.matrix_world.astype(np.float64)

            # Bake non-uniform scale into mesh vertices before normalisation
            scale_vec = self._decompose_scale(local2world)
            if np.any(np.abs(scale_vec) <= 1e-8):
                self._log_scale_error(scene_id, cand.key, local2world, scale_vec)
                raise ValueError(
                    f"Invalid scale (<= 1e-8) in matrix_world for {scene_id}: {cand.key}"
                )
            scale_mat = np.eye(4, dtype=np.float64)
            scale_mat[:3, :3] = np.diag(scale_vec)
            scale_inv = np.eye(4, dtype=np.float64)
            scale_inv[:3, :3] = np.diag(1.0 / scale_vec)

            voxel_model = raw_model.copy()
            voxel_model.apply_transform(scale_mat)
            voxel_model, normalize_transform = normalize_object(voxel_model, margin=0.0)

            # Convert model from Y-up to Z-up (canonical space is Z-up)
            voxel_model.apply_transform(y_up2z_up)

            # mesh_to_world: Z-up canonical → Y-up world
            # chain: Z-up → (y_up2z_up_inv) → Y-up normalised → (inv(norm)) → Y-up scaled
            #        → (scale_inv) removed, local2world keeps only rotation+translation
            mesh_to_world = (
                local2world
                @ scale_inv
                @ np.linalg.inv(normalize_transform)
                @ y_up2z_up_inv
            )  # (4, 4) float64

            if (
                cam_pos_world is not None
                and self.cfg.diagonal_azimuth_margin_deg > 0
            ):
                azimuth_deg, _ = _compute_azimuth_and_rotation(
                    cam_pos_world, mesh_to_world
                )
                if _is_diagonal_azimuth(
                    azimuth_deg, self.cfg.diagonal_azimuth_margin_deg
                ):
                    continue

            masks_list.append(cand.mask)
            voxel_models_list.append(voxel_model)
            mesh_to_world_list.append(mesh_to_world)
            scale_vecs_list.append(scale_vec)
            obj_keys_list.append(cand.key)
            geometry_sha256_list.append(self._model_geometry_sha256(cand.model_path))
            models.append(raw_model)
            if self.cfg.use_bbox_layout:
                canonical_bboxes_list.append(
                    torch.from_numpy(voxel_model.bounds.copy()).float()  # [2, 3]
                )
            trimesh_scene_obj.add_geometry(raw_model, transform=local2world)

        if len(masks_list) == 0:
            raise RuntimeError(
                f"No objects remain after visibility / diagonal-azimuth filtering: "
                f"scene={scene_id}, h5={h5_path}"
            )

        # ----------------------------------------------------------------
        # Build image / mask tensors
        # ----------------------------------------------------------------
        masks_np = np.stack(masks_list, axis=0)         # (N, H, W) uint8
        scene_image = colors.astype(np.uint8)           # (H, W, 3)
        part_images_np = (
            scene_image[None] * masks_np[..., None]
        )                                               # (N, H, W, 3)
        num_instances = len(masks_list)

        part_images_t = (
            torch.from_numpy(part_images_np).float().permute(0, 3, 1, 2) / 255.0
        )                                               # (N, 3, H, W)
        masks_t = torch.from_numpy(masks_np).float()[:, None]  # (N, 1, H, W)
        scene_img_t = (
            torch.from_numpy(scene_image).float().permute(2, 0, 1) / 255.0
        )[None].repeat(num_instances, 1, 1, 1)          # (N, 3, H, W)

        # ----------------------------------------------------------------
        # Crop
        # ----------------------------------------------------------------
        valid_indices: List[int] = []
        cropped_rgb_list: List[torch.Tensor] = []
        cropped_mask_list: List[torch.Tensor] = []
        for inst_idx in range(num_instances):
            mask_i = masks_t[inst_idx]
            if torch.sum(mask_i > 0.5).item() <= 0:
                continue
            try:
                # Keep cropped RGB semantics consistent with the other scene
                # dataloaders and inference preprocessing: crop the masked
                # object image, while retaining the complete scene separately
                # in ``rgb_scene``.
                cropped_rgb_i, cropped_mask_i, _ = crop_around_mask(
                    part_images_t[inst_idx],
                    mask_i,
                    box_size_factor=self.box_size_factor,
                    target_h=self.cfg.height,
                    target_w=self.cfg.width,
                )
            except Exception:
                continue
            cropped_rgb_list.append(cropped_rgb_i)
            cropped_mask_list.append(cropped_mask_i)
            valid_indices.append(inst_idx)

        if len(valid_indices) == 0:
            raise ValueError(
                f"No valid objects after crop for scene={scene_id}, h5={h5_path}"
            )

        # Filter all per-instance lists to valid_indices
        masks_np         = masks_np[valid_indices]
        part_images_t    = part_images_t[valid_indices]
        masks_t          = masks_t[valid_indices]
        scene_img_t      = scene_img_t[valid_indices]
        voxel_models_list = [voxel_models_list[i] for i in valid_indices]
        mesh_to_world_list = [mesh_to_world_list[i] for i in valid_indices]
        scale_vecs_list   = [scale_vecs_list[i] for i in valid_indices]
        obj_keys_list     = [obj_keys_list[i] for i in valid_indices]
        geometry_sha256_list = [geometry_sha256_list[i] for i in valid_indices]
        models            = [models[i] for i in valid_indices]
        if canonical_bboxes_list:
            canonical_bboxes_list = [canonical_bboxes_list[i] for i in valid_indices]
        num_instances = len(valid_indices)

        cropped_rgb  = torch.stack(cropped_rgb_list,  dim=0)  # [N, 3, H, W]
        cropped_mask = torch.stack(cropped_mask_list, dim=0)  # [N, 1, H, W]

        # ----------------------------------------------------------------
        # Final sub-sampling (trim oversample buffer to exact target)
        # ----------------------------------------------------------------
        select_indices_for_output = list(range(num_instances))
        if num_target > 0:
            if num_instances >= num_target:
                selected = random.sample(range(num_instances), num_target)
            else:
                selected = random.choices(range(num_instances), k=num_target)
            select_indices_for_output = selected
            masks_np          = masks_np[selected]
            part_images_t     = part_images_t[selected]
            masks_t           = masks_t[selected]
            scene_img_t       = scene_img_t[selected]
            cropped_rgb       = cropped_rgb[selected]
            cropped_mask      = cropped_mask[selected]
            voxel_models_list  = [voxel_models_list[i] for i in selected]
            mesh_to_world_list = [mesh_to_world_list[i] for i in selected]
            scale_vecs_list    = [scale_vecs_list[i] for i in selected]
            obj_keys_list      = [obj_keys_list[i] for i in selected]
            geometry_sha256_list = [geometry_sha256_list[i] for i in selected]
            models             = [models[i] for i in selected]
            if canonical_bboxes_list:
                canonical_bboxes_list = [canonical_bboxes_list[i] for i in selected]
            num_instances = num_target

        # ----------------------------------------------------------------
        # Load depth
        # ----------------------------------------------------------------
        depth_raw = self._load_depth(unique_id)   # (H_raw, W_raw) or None
        H, W = self.cfg.height, self.cfg.width
        if depth_raw is not None:
            H_raw, W_raw = depth_raw.shape
            if (H_raw, W_raw) != (H, W):
                if self.cfg.depth_resize_mode == "nearest":
                    depth = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    depth_pil = Image.fromarray(depth_raw, mode="F")
                    depth = np.array(
                        depth_pil.resize((W, H), resample=Image.BILINEAR), dtype=np.float32
                    )
            else:
                depth = depth_raw
        else:
            depth = None

        # ----------------------------------------------------------------
        # Canonical coordinate map per instance
        # ----------------------------------------------------------------
        canonical_maps: List[torch.Tensor] = []
        for i in range(num_instances):
            obj_mask = masks_np[i]           # (H, W) uint8

            if depth is not None:
                # T_can_to_cam = world_to_cam @ mesh_to_world
                T_can_to_cam = world_to_cam @ mesh_to_world_list[i]   # (4, 4) float64
                T_cam_to_can = np.linalg.inv(T_can_to_cam)
                cmap_np = depth_to_canonical_coord_map(
                    depth, obj_mask, fov, T_cam_to_can
                )                                # (H, W, 3)
            else:
                cmap_np = np.zeros((H, W, 3), dtype=np.float32)

            canonical_maps.append(
                torch.from_numpy(cmap_np).permute(2, 0, 1).float()  # [3, H, W]
            )

        canonical_coord_map = torch.stack(canonical_maps, dim=0)  # [N, 3, H, W]

        # ----------------------------------------------------------------
        # Crop canonical coordinate map with the same crop logic as RGB/mask
        # ----------------------------------------------------------------
        canonical_coord_map_cropped_list: List[torch.Tensor] = []
        crop_params_list: List[Dict[str, int]] = []
        for i in range(num_instances):
            _, _, cropped_cmap_i, crop_params_i = crop_around_mask(
                part_images_t[i],
                masks_t[i],
                box_size_factor=self.box_size_factor,
                target_h=self.cfg.height,
                target_w=self.cfg.width,
                extra_maps=[canonical_coord_map[i]],
            )
            canonical_coord_map_cropped_list.append(cropped_cmap_i)
            crop_params_list.append(crop_params_i)

        # (canonical_coord_map_cropped is stacked after the optional azimuth rotation below)

        # ----------------------------------------------------------------
        # Voxelise  (scale-aware disk + memory cache)
        # ----------------------------------------------------------------
        voxel_list: List[torch.Tensor] = []
        for i in range(num_instances):
            model_id = (
                obj_keys_list[i].split("|")[1]
                or obj_keys_list[i].split("|")[0]
            )
            scale_vec = scale_vecs_list[i]
            geometry_sha256 = geometry_sha256_list[i]

            # Build cache key / path
            cache_path = (
                _voxel_scale_cache_path(
                    self.voxel_cache_dir, model_id, scale_vec, self.voxel_res,
                    geometry_sha256,
                )
                if self.voxel_cache_dir else ""
            )

            # ① memory cache hit
            voxel_dict = self._voxel_mem_cache.get(cache_path) if cache_path else None
            if voxel_dict is not None:
                if not self._valid_voxel_cache(
                    voxel_dict, geometry_sha256, self.voxel_res
                ):
                    raise RuntimeError(
                        f"invalid in-memory voxel cache metadata: {cache_path}"
                    )
                self._voxel_mem_cache.move_to_end(cache_path)

            # ② disk cache hit
            if voxel_dict is None and cache_path and os.path.exists(cache_path):
                try:
                    voxel_dict = torch.load(cache_path, map_location="cpu")
                    if self._valid_voxel_cache(
                        voxel_dict, geometry_sha256, self.voxel_res
                    ):
                        self._voxel_mem_insert(cache_path, voxel_dict)
                    else:
                        voxel_dict = None
                except Exception:
                    voxel_dict = None

            # ③ compute from scratch
            if voxel_dict is None:
                voxel_dict = voxelize_trimesh_obj(
                    voxel_models_list[i],
                    voxel_res=self.voxel_res,
                )
                voxel_dict["source_obj_sha256"] = geometry_sha256
                voxel_dict["source_model_id"] = model_id
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    try:
                        torch.save(voxel_dict, cache_path)
                    except Exception:
                        pass
                if cache_path:
                    self._voxel_mem_insert(cache_path, voxel_dict)

            voxel_list.append(gen_voxel_grid(voxel_dict))

        voxels = torch.stack(voxel_list, dim=0)   # [N, res, res, res]

        # ----------------------------------------------------------------
        # Optional: rotate per-instance canonical quantities based on azimuth
        # ----------------------------------------------------------------
        azimuth_estimates = [None] * num_instances
        azimuth_rotations = [0] * num_instances
        if cam_T is not None:
            cam_pos_world = cam_T[:3, 3].astype(np.float64)
            for i in range(num_instances):
                az_deg, rot_deg = _compute_azimuth_and_rotation(
                    cam_pos_world, mesh_to_world_list[i]
                )
                azimuth_estimates[i] = az_deg
                if not self.cfg.canonicalize_azimuth:
                    continue
                azimuth_rotations[i] = rot_deg
                if rot_deg == 0:
                    continue
                canonical_coord_map[i] = _rotate_coord_map_z(
                    canonical_coord_map[i], rot_deg
                )
                canonical_coord_map_cropped_list[i] = _rotate_coord_map_z(
                    canonical_coord_map_cropped_list[i], rot_deg
                )
                voxels[i] = _rotate_voxel_z(voxels[i], rot_deg)
                if canonical_bboxes_list:
                    bbox_np = canonical_bboxes_list[i].numpy()
                    canonical_bboxes_list[i] = torch.from_numpy(
                        _rotate_bbox_z(bbox_np, rot_deg)
                    ).float()

        canonical_coord_map_cropped = torch.stack(
            canonical_coord_map_cropped_list, dim=0
        )  # [N, 3, H, W]  (rebuild after possible per-instance rotation)

        # ----------------------------------------------------------------
        # Build data_id
        # ----------------------------------------------------------------
        if unique_id:
            data_id = unique_id
        else:
            view_id_stem = os.path.splitext(os.path.basename(h5_path))[0]
            data_id = f"{scene_id}_{view_id_stem}"

        select_indices_out = torch.tensor(
            select_indices_for_output, dtype=torch.long
        )

        # ----------------------------------------------------------------
        # Pack – mirrors ThreeDFutureSceneDepthDataset /
        #         ObjaverseSceneDepthDataset
        # ----------------------------------------------------------------
        pack: Dict[str, Any] = {
            "id":             data_id,
            "num_instances":  num_instances,
            "rgb":            part_images_t,       # [N, 3, H, W], not used!
            "mask":           masks_t,              # [N, 1, H, W]
            "masks":          masks_t,              # [N, 1, H, W]
            "rgb_scene":      scene_img_t,          # [N, 3, H, W]
            "rgb_cropped":    cropped_rgb,           # [N, 3, H, W]
            "mask_cropped":   cropped_mask,          # [N, 1, H, W]
            "fov":            fov,
            "height":         self.cfg.height,
            "width":          self.cfg.width,
            "voxel":          voxels,               # [N, res, res, res]
            "voxel_res":      self.voxel_res,
            # Depth-derived canonical coordinate map
            "canonical_coord_map": canonical_coord_map,  # [N, 3, H, W]
            "canonical_coord_map_cropped": canonical_coord_map_cropped,  # [N, 3, H, W]
            "crop_params":    crop_params_list,
            "select_indices": select_indices_out,
            "azimuth_rotation": azimuth_rotations,       # List[int], length N
        }

        if self.cfg.include_transformation:
            pack["transformation"] = torch.from_numpy(
                np.stack(mesh_to_world_list, axis=0)
            ).float()  # [N, 4, 4], Z-up canonical -> Y-up world
            pack["azimuth_estimate"] = azimuth_estimates  # List[Optional[float]], length N

        if self.cfg.use_bbox_layout and canonical_bboxes_list:
            pack["canonical_bboxes"] = torch.stack(canonical_bboxes_list).float()

        if self.cfg.with_mesh:
            pack["surface"]    = voxel_models_list   # normalized + y_up2z_up, same as lightweight
            pack["scene_mesh"] = trimesh_scene_obj

        return pack

    # ------------------------------------------------------------------

    def collate(self, batch):
        """DataLoader collate: mesh/list fields as Python list; rest default."""
        elem = batch[0]
        collated: Dict[str, Any] = {}
        for key in elem:
            if key in (
                "surface",
                "scene_mesh",
                "id",
                "scene_id",
                "obj_keys",
                "crop_params",
                "azimuth_estimate",
                "azimuth_rotation",
            ):
                collated[key] = [d[key] for d in batch]
            else:
                collated[key] = torch.utils.data.default_collate(
                    [d[key] for d in batch]
                )
        collated["num_instances_per_batch"] = self.cfg.num_instances_per_batch
        return collated
