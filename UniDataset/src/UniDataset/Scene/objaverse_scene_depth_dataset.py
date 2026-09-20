#!/usr/bin/env python3
"""
Objaverse single-object scene dataset with depth-based canonical coordinate map.

Produces a dense canonical_coord_map derived from depth.exr:

  1. Load depth.exr (or depth.npy) from valid_scenes/{obj_id}_{view}/
  2. Filter valid pixels via mask.png (foreground mask)
  3. Unproject depth to 3-D camera space using horizontal FOV
     (OpenGL convention: +X right, +Y up, -Z forward)
  4. Apply T_cam_to_canonical = inv(w2c @ T_st) to obtain canonical
     coordinates (Blender Z-up, normalised to [-0.5, 0.5])

Output tensor  canonical_coord_map : [1, 3, H, W]  (float32)
Background / invalid pixels are set to zero.
"""

import json
import os
import traceback
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# Enable OpenEXR reading in OpenCV before the library is imported elsewhere.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def load_depth(view_dir: str) -> np.ndarray:
    """
    Load mesh depth from view_dir.  Tries depth.npy first, then depth.exr.

    For EXR files the array may be single-channel (H, W) or multi-channel
    (H, W, C); only the first channel is used.

    Note: Objaverse depth.exr uses a large sentinel value (~1e10) for
    background pixels, not 0.

    Returns
    -------
    depth : (H, W) float32
    """
    npy_path = os.path.join(view_dir, "depth.npy")
    exr_path = os.path.join(view_dir, "depth.exr")

    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)

    if os.path.exists(exr_path):
        try:
            import cv2
            arr = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr is None:
                raise IOError(f"cv2 failed to read: {exr_path}")
        except ImportError:
            raise ImportError(
                "Cannot load depth.exr: cv2 is not available. "
                "Install opencv-python and ensure OpenEXR support is enabled "
                "(set OPENCV_IO_ENABLE_OPENEXR=1 before importing cv2)."
            )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]  # take first channel
        return arr

    raise FileNotFoundError(
        f"No depth file found in {view_dir} (tried depth.npy and depth.exr)"
    )


# ---------------------------------------------------------------------------
# Depth → canonical coordinate map
# ---------------------------------------------------------------------------

def depth_to_canonical_coord_map(
    depth: np.ndarray,           # (H, W) float32  – Z-depth, positive in front
    mask: np.ndarray,            # (H, W) uint8/bool – 1 = valid foreground
    fov_rad: float,              # horizontal FOV in radians
    T_cam_to_canonical: np.ndarray,  # (4, 4) float64 – camera → canonical
) -> np.ndarray:
    """
    Unproject a depth map to 3-D and transform to canonical space.

    Camera convention used for unprojection (OpenGL):
        X = (u - cx) * d / fx          (+X = right)
        Y = -(v - cy) * d / fx         (+Y = up,  image v is down)
        Z = -d                          (+Z = out of screen, -Z = into scene)

    Square pixels are assumed (fx == fy).

    Parameters
    ----------
    depth  : (H, W) float32  positive Z-depth values
    mask   : (H, W) bool/uint8  foreground mask (1 = valid)
    fov_rad: horizontal field of view in radians
    T_cam_to_canonical : (4, 4) transforms camera-space homogeneous points
                         to canonical space

    Returns
    -------
    canonical_map : (H, W, 3) float32
        3-D coordinates in canonical space; background pixels are zero.
    """
    H, W = depth.shape
    fx = 0.5 * W / np.tan(0.5 * fov_rad)
    cx = W / 2.0
    cy = H / 2.0

    # Valid pixels: foreground mask AND finite, positive, non-sentinel depth
    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth < 1e6)
    )

    canonical_map = np.zeros((H, W, 3), dtype=np.float32)
    if not np.any(valid):
        return canonical_map

    v_idx, u_idx = np.where(valid)          # pixel row/col
    d = depth[valid].astype(np.float64)

    # Unproject: camera OpenGL space
    X_cam = (u_idx.astype(np.float64) - cx) * d / fx
    Y_cam = -(v_idx.astype(np.float64) - cy) * d / fx   # flip v for OpenGL Y-up
    Z_cam = -d                                            # camera looks along -Z

    # Homogeneous [4, N]
    ones = np.ones(len(d), dtype=np.float64)
    cam_pts_h = np.stack([X_cam, Y_cam, Z_cam, ones], axis=0)   # (4, N)

    # Transform to canonical space
    canonical_pts_h = T_cam_to_canonical.astype(np.float64) @ cam_pts_h  # (4, N)
    canonical_pts = canonical_pts_h[:3].T.astype(np.float32)              # (N, 3)

    canonical_map[v_idx, u_idx] = canonical_pts
    return canonical_map


# ---------------------------------------------------------------------------
# Azimuth-based canonical rotation
# ---------------------------------------------------------------------------

def _compute_azimuth_and_rotation(
    c2w: np.ndarray,
    frame_scale: float,
    frame_translate: np.ndarray,
) -> tuple:
    """
    Compute camera azimuth relative to canonical space and the quantized
    rotation angle needed to bring the object's front toward the camera.

    Returns (azimuth_deg, rotation_deg).
    """
    cam_pos_world = c2w[:3, 3]
    T_st = np.eye(4, dtype=np.float64)
    T_st[:3, :3] *= frame_scale
    T_st[:3, 3] = frame_translate
    cam_canonical = (np.linalg.inv(T_st) @ np.append(cam_pos_world, 1.0))[:3]

    azimuth_deg = float(np.degrees(np.arctan2(cam_canonical[0], -cam_canonical[1])))

    if -45 <= azimuth_deg <= 45:
        rotation_deg = 0
    elif 45 < azimuth_deg <= 135:
        rotation_deg = -90
    elif -135 <= azimuth_deg < -45:
        rotation_deg = 90
    else:  # > 135 or < -135
        rotation_deg = 180

    return azimuth_deg, rotation_deg


_ROT_Z = {
    0: np.eye(3, dtype=np.float32),
    90: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    -90: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
}


def _rotate_coord_map_z(coord_map: torch.Tensor, rot_deg: int) -> torch.Tensor:
    """Rotate 3-D coordinate values in a [3, H, W] map around the Z axis."""
    if rot_deg == 0:
        return coord_map
    R = torch.from_numpy(_ROT_Z[rot_deg])          # (3, 3)
    C, H, W = coord_map.shape
    flat = coord_map.reshape(3, -1)                 # (3, H*W)
    rotated = R @ flat                              # (3, H*W)
    return rotated.reshape(C, H, W)


def _rotate_voxel_z(voxel: torch.Tensor, rot_deg: int) -> torch.Tensor:
    """Rotate a [R, R, R] occupancy grid around the Z axis by 0/±90/180°."""
    if rot_deg == 0:
        return voxel
    if rot_deg == -90:      # (x,y,z) → (y, -x, z)
        return voxel.permute(1, 0, 2).flip(1)
    if rot_deg == 90:       # (x,y,z) → (-y, x, z)
        return voxel.permute(1, 0, 2).flip(0)
    # 180°: (x,y,z) → (-x, -y, z)
    return voxel.flip(0).flip(1)


def _rotate_bbox_z(bbox: np.ndarray, rot_deg: int) -> np.ndarray:
    """Rotate a (2, 3) min/max bounding box around Z and recompute min/max."""
    if rot_deg == 0:
        return bbox
    R = _ROT_Z[rot_deg]
    corners = np.array([
        [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
    ], dtype=np.float32)
    rotated = (R @ corners.T).T
    return np.stack([rotated.min(axis=0), rotated.max(axis=0)])


# ---------------------------------------------------------------------------
# Mesh / voxel helpers
# ---------------------------------------------------------------------------

def load_glb_as_zup(glb_path: str) -> trimesh.Trimesh:
    """Load GLB and convert from glTF Y-up to Blender Z-up."""
    mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    new_verts = verts.copy()
    new_verts[:, 0] = verts[:, 0]
    new_verts[:, 1] = -verts[:, 2]
    new_verts[:, 2] = verts[:, 1]
    return trimesh.Trimesh(vertices=new_verts, faces=np.asarray(mesh.faces))


def voxelize_mesh(mesh: trimesh.Trimesh, voxel_res: int = 64) -> Dict[str, torch.Tensor]:
    """Voxelize a trimesh object within [-0.5, 0.5] bounds."""
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return {
            "voxel_indexes": torch.zeros((0, 3), dtype=torch.long),
            "voxel_centers": torch.zeros((0, 3), dtype=torch.float32),
            "voxel_res": voxel_res,
        }
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    voxel_size = 1.0 / voxel_res
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=voxel_size,
        min_bound=np.array([-0.5, -0.5, -0.5]),
        max_bound=np.array([0.5, 0.5, 0.5]),
    )
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        return {
            "voxel_indexes": torch.zeros((0, 3), dtype=torch.long),
            "voxel_centers": torch.zeros((0, 3), dtype=torch.float32),
            "voxel_res": voxel_res,
        }
    grid_index = np.stack([v.grid_index for v in voxels])
    origin = voxel_grid.origin
    voxel_size_f = voxel_size
    voxel_centers = origin + (grid_index + 0.5) * voxel_size_f
    voxel_indexes = np.floor((voxel_centers + 0.5) / voxel_size_f).astype(np.int32)
    valid = np.all((voxel_indexes >= 0) & (voxel_indexes < voxel_res), axis=1)
    voxel_indexes = np.unique(voxel_indexes[valid], axis=0)
    voxel_centers = origin + (voxel_indexes.astype(np.float32) + 0.5) * voxel_size_f
    return {
        "voxel_indexes": torch.from_numpy(voxel_indexes).long(),
        "voxel_centers": torch.from_numpy(voxel_centers).float(),
        "voxel_res": voxel_res,
    }


def gen_voxel_grid(voxel_dict: Dict) -> torch.Tensor:
    voxel_indexes = voxel_dict["voxel_indexes"].long()
    voxel_res = int(voxel_dict["voxel_res"])
    voxel = torch.zeros(voxel_res, voxel_res, voxel_res, dtype=torch.long)
    if voxel_indexes.numel() > 0:
        voxel[voxel_indexes[:, 0], voxel_indexes[:, 1], voxel_indexes[:, 2]] = 1
    return voxel


from UniDataset.utils.img_and_mask_transforms import crop_around_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ObjaverseSceneDepthDataset(Dataset):
    """
    Objaverse single-object scene dataset.

    Loads depth.exr (or depth.npy) from valid_scenes, filters it by mask.png,
    reprojects to 3-D camera space via the horizontal FOV, and transforms the
    result to canonical space using
        T_cam_to_canonical = inv(w2c @ T_st)
    where T_st encodes the canonical→world scale+translate and w2c is the
    world-to-camera matrix (both read from the align_summary JSON).

    Outputs canonical_coord_map : [1, 3, H, W]  (float32, 0 = invalid)
    """

    def __init__(
        self,
        summary_json: str,
        valid_scenes_dir: str,
        height: int = 518,
        width: int = 518,
        voxel_res: int = 64,
        voxel_cache_dir: str = "",
        split: str = "test",
        data_indices: Optional[List] = None,
        repeat: int = 1,
        with_mesh: bool = False,
        use_bbox_layout: bool = True,
        skip_exists_check: bool = False,
        canonicalize_azimuth: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.valid_scenes_dir = valid_scenes_dir
        self.height = height
        self.width = width
        self.voxel_res = voxel_res
        self.with_mesh = with_mesh
        self.use_bbox_layout = use_bbox_layout
        self.canonicalize_azimuth = canonicalize_azimuth

        # Load summary JSON
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Filter: valid=True, mesh_exists=True, scene_dir present
        entries = []
        skipped_no_mesh = 0
        for e in summary["entries"]:
            if "iou_ok" in e and not e["iou_ok"]:
                continue
            if not e.get("valid", False):
                continue
            if not e.get("mesh_exists", False):
                skipped_no_mesh += 1
                continue
            obj_id = e["obj_id"]
            view_str = str(e["view"]).zfill(3)
            scene_dir = os.path.join(valid_scenes_dir, f"{obj_id}_{view_str}")
            if not skip_exists_check and not os.path.isdir(scene_dir):
                continue
            entries.append({
                "obj_id": obj_id,
                "view": view_str,
                "scene_dir": scene_dir,
                "mesh_path": e["mesh_path"],
                "frame_transform": e["frame_transform"],   # {scale, translate}
                "camera": e["camera"],                     # {transform_matrix, fov}
            })

        if skipped_no_mesh > 0:
            print(
                f"ObjaverseSceneDepthDataset: skipped {skipped_no_mesh} "
                "entries with missing mesh"
            )

        # Optional index slicing
        if data_indices is not None:
            if len(data_indices) == 2:
                start, end = data_indices
                entries = entries[start:end]
            elif len(data_indices) == 3:
                start, end, gap = data_indices
                entries = entries[start:end:gap]

        self.entries = entries * repeat
        unique_objs = len(set(e["obj_id"] for e in entries))
        print(
            f"ObjaverseSceneDepthDataset: {len(self.entries)} entries "
            f"({unique_objs} unique objects)"
        )

        # Voxel cache directory
        if voxel_cache_dir:
            self.voxel_cache_dir = voxel_cache_dir
        elif entries:
            self.voxel_cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(entries[0]["mesh_path"])),
                "voxel_cache",
            )
        else:
            self.voxel_cache_dir = "/tmp/voxel_cache"

        self._voxel_cache: OrderedDict = OrderedDict()
        self._voxel_cache_maxsize = 100

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        if mesh_path.endswith((".glb", ".gltf")):
            return load_glb_as_zup(mesh_path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
            return mesh

    def _get_voxel(self, mesh_path: str, obj_id: str) -> Dict:
        """Return voxel dict (with canonical_bbox) for obj_id.

        Mesh is loaded from disk only on the very first call for each obj_id
        (i.e. when neither the in-memory LRU cache nor the on-disk .pt cache
        contains the result).  Subsequent calls are served entirely from cache
        without any mesh I/O.

        The returned dict contains:
            voxel_indexes   : (N, 3) long
            voxel_centers   : (N, 3) float32
            voxel_res       : int
            canonical_bbox  : (2, 3) float32  – mesh.bounds in canonical space
        """
        if obj_id in self._voxel_cache:
            self._voxel_cache.move_to_end(obj_id)
            return self._voxel_cache[obj_id]
        cache_path = os.path.join(
            self.voxel_cache_dir, obj_id, f"voxel_{self.voxel_res}.pt"
        )
        if os.path.exists(cache_path):
            try:
                vd = torch.load(cache_path, map_location="cpu")
                if "canonical_bbox" not in vd:
                    mesh = self._load_mesh(mesh_path)
                    vd["canonical_bbox"] = torch.from_numpy(
                        mesh.bounds.copy()
                    ).float()
                    try:
                        torch.save(vd, cache_path)
                    except Exception:
                        pass
                self._insert_voxel_cache(obj_id, vd)
                return vd
            except Exception:
                pass
        mesh = self._load_mesh(mesh_path)
        vd = voxelize_mesh(mesh, self.voxel_res)
        vd["canonical_bbox"] = torch.from_numpy(mesh.bounds.copy()).float()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            torch.save(vd, cache_path)
        except Exception:
            pass
        self._insert_voxel_cache(obj_id, vd)
        return vd

    def _insert_voxel_cache(self, obj_id: str, vd: Dict) -> None:
        self._voxel_cache[obj_id] = vd
        self._voxel_cache.move_to_end(obj_id)
        while len(self._voxel_cache) > self._voxel_cache_maxsize:
            self._voxel_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception as e:
            print(f"ObjaverseSceneDepthDataset: error at index {index}: {e}")
            traceback.print_exc()
            return self.__getitem__((index + 1) % len(self))

    def _get_item(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        obj_id  = entry["obj_id"]
        view_str = entry["view"]
        scene_id = f"{obj_id}_{view_str}"
        scene_dir = entry["scene_dir"]

        # ----------------------------------------------------------------
        # Camera + frame transform
        # ----------------------------------------------------------------
        ft  = entry["frame_transform"]
        cam = entry["camera"]

        frame_scale     = ft["scale"]
        frame_translate = np.array(ft["translate"], dtype=np.float64)
        c2w = np.array(cam["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        fov_deg = cam["fov"]
        fov_rad = np.radians(fov_deg)

        # T_st  : canonical Z-up → world Z-up  (scale + translate)
        T_st = np.eye(4, dtype=np.float64)
        T_st[:3, :3] *= frame_scale
        T_st[:3, 3]   = frame_translate

        # T_full            : canonical → camera (OpenGL)
        # T_cam_to_canonical: camera (OpenGL) → canonical
        T_full = w2c @ T_st
        T_cam_to_canonical = np.linalg.inv(T_full)

        # ----------------------------------------------------------------
        # Load depth and mask
        # ----------------------------------------------------------------
        depth_raw = load_depth(scene_dir)          # (H_raw, W_raw) float32

        mask_img = np.array(
            Image.open(os.path.join(scene_dir, "mask.png")).convert("L")
        )                                          # (H_raw, W_raw) uint8
        # Fall back: threshold depth when mask is missing / all-zero
        if mask_img.max() == 0:
            mask_img = ((depth_raw > 0) & (depth_raw < 1e6)).astype(np.uint8)
        else:
            mask_img = (mask_img > 0).astype(np.uint8)

        # Resize depth + mask to model resolution (bilinear for depth, NN for mask)
        H_raw, W_raw = depth_raw.shape
        if (H_raw, W_raw) != (self.height, self.width):
            depth_pil = Image.fromarray(depth_raw, mode="F")
            depth_pil = depth_pil.resize(
                (self.width, self.height), resample=Image.BILINEAR
            )
            depth = np.array(depth_pil, dtype=np.float32)

            mask_pil = Image.fromarray(mask_img)
            mask_pil = mask_pil.resize(
                (self.width, self.height), resample=Image.NEAREST
            )
            mask = np.array(mask_pil, dtype=np.uint8)
        else:
            depth = depth_raw
            mask  = mask_img

        # ----------------------------------------------------------------
        # Build canonical coordinate map  [H, W, 3] → tensor [3, H, W]
        # ----------------------------------------------------------------
        canonical_map_np = depth_to_canonical_coord_map(
            depth, mask, fov_rad, T_cam_to_canonical
        )                                             # (H, W, 3) float32
        canonical_coord_map = (
            torch.from_numpy(canonical_map_np).permute(2, 0, 1).float()
        )                                             # [3, H, W]

        # ----------------------------------------------------------------
        # Load RGB image
        # ----------------------------------------------------------------
        scene_img = np.array(
            Image.open(os.path.join(scene_dir, "scene.png")).convert("RGB")
        )
        resize_rgb  = transforms.Resize((self.height, self.width), antialias=True)
        resize_mask = transforms.Resize(
            (self.height, self.width),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        scene_tensor = torch.from_numpy(scene_img).permute(2, 0, 1).float() / 255.0
        scene_tensor = resize_rgb(scene_tensor)                       # [3, H, W]

        mask_tensor = torch.from_numpy(mask).float()
        mask_tensor = resize_mask(mask_tensor.unsqueeze(0))           # [1, H, W]

        part_image = scene_tensor * mask_tensor                       # [3, H, W]

        # ----------------------------------------------------------------
        # Resize canonical_coord_map to match part_image spatial size
        # ----------------------------------------------------------------
        canonical_coord_map = transforms.Resize(
            (self.height, self.width),
            interpolation=transforms.InterpolationMode.NEAREST,
        )(canonical_coord_map)                                    # [3, H, W]

        # ----------------------------------------------------------------
        # Crop around mask (RGB + mask + canonical_coord_map)
        # ----------------------------------------------------------------
        cropped = crop_around_mask(
            part_image, mask_tensor,
            target_h=self.height, target_w=self.width,
            extra_maps=[canonical_coord_map],
        )
        cropped_rgb, cropped_mask, cropped_canonical_coord_map, crop_params = cropped

        # ----------------------------------------------------------------
        # Mesh + voxel
        # ----------------------------------------------------------------
        voxel_dict   = self._get_voxel(entry["mesh_path"], obj_id)
        voxel_tensor = gen_voxel_grid(voxel_dict)    # [res, res, res]
        canonical_bbox = voxel_dict["canonical_bbox"].numpy()  # (2, 3)

        # ----------------------------------------------------------------
        # Optional: rotate canonical-space quantities based on azimuth
        # ----------------------------------------------------------------
        azimuth_rotation = 0
        if self.canonicalize_azimuth:
            _, azimuth_rotation = _compute_azimuth_and_rotation(
                c2w, frame_scale, frame_translate,
            )
            if azimuth_rotation != 0:
                canonical_coord_map = _rotate_coord_map_z(
                    canonical_coord_map, azimuth_rotation
                )
                cropped_canonical_coord_map = _rotate_coord_map_z(
                    cropped_canonical_coord_map, azimuth_rotation
                )
                voxel_tensor = _rotate_voxel_z(voxel_tensor, azimuth_rotation)
                canonical_bbox = _rotate_bbox_z(
                    canonical_bbox, azimuth_rotation
                )

        # ----------------------------------------------------------------
        # Pack output dict
        # ----------------------------------------------------------------
        rgb         = part_image.unsqueeze(0)          # [1, 3, H, W]
        masks       = mask_tensor.unsqueeze(0)         # [1, 1, H, W]
        rgb_scene   = scene_tensor.unsqueeze(0)        # [1, 3, H, W]
        rgb_cropped = cropped_rgb.unsqueeze(0)         # [1, 3, H, W]
        mask_cropped = cropped_mask.unsqueeze(0)       # [1, 1, H, W]

        pack = {
            "id":           scene_id,
            "num_instances": 1,
            "rgb":           rgb,
            "mask":          masks,
            "masks":         masks,
            "rgb_scene":     rgb_scene,
            "rgb_cropped":   rgb_cropped,
            "mask_cropped":  mask_cropped,
            "fov":           fov_rad,
            "height":        self.height,
            "width":         self.width,
            "voxel":         voxel_tensor.unsqueeze(0),    # [1, res, res, res]
            "voxel_res":     self.voxel_res,
            # Dense canonical coordinate map from GT depth
            "canonical_coord_map": canonical_coord_map.unsqueeze(0),   # [1, 3, H, W]
            "canonical_coord_map_cropped": cropped_canonical_coord_map.unsqueeze(0),  # [1, 3, H, W]
            "crop_params": crop_params,
            "select_indices": torch.tensor([0], dtype=torch.long),
            "azimuth_rotation": azimuth_rotation,
        }

        if self.with_mesh:
            mesh = self._load_mesh(entry["mesh_path"])
            pack["surface"] = [mesh]
            trimesh_scene = trimesh.Scene()
            trimesh_scene.add_geometry(mesh, transform=T_st)
            pack["scene_mesh"] = trimesh_scene

        if self.use_bbox_layout:
            pack["canonical_bboxes"] = (
                torch.from_numpy(canonical_bbox).float().unsqueeze(0)  # [1, 2, 3]
            )

        return pack

    # ------------------------------------------------------------------

    def collate(self, batch):
        """Custom collate function."""
        elem = batch[0]
        collated = {}
        for key in elem:
            if key in ("surface", "scene_mesh", "id", "crop_params"):
                collated[key] = [d[key] for d in batch]
            else:
                collated[key] = torch.utils.data.default_collate(
                    [d[key] for d in batch]
                )
        return collated
