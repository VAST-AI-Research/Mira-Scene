# UniDataset

Dataset library for Objaverse and 3D-FRONT (BlenderProc) 3D scene understanding training and evaluation.

## Structure

```
src/UniDataset/
├── __init__.py
├── Scene/
│   ├── objaverse_scene_depth_dataset.py       # Training dataset (single-object scenes)
│   ├── objaverse_scene_depth_dataset_alpha.py  # Training dataset (with alpha/multi-view)
│   ├── eval_dataset.py                        # Evaluation dataset (scene.png + mask + GT)
│   └── threedfrontv4/
│       └── blenderproc_scene_depth.py         # Training dataset (3D-FRONT multi-object scenes)
└── utils/
    ├── pcd_utils.py             # Voxel ↔ point cloud conversion
    └── img_and_mask_transforms.py  # Crop-around-mask, resize, transforms
```

## Datasets

### ObjaverseSceneDepthDataset (Training)

Loads Objaverse single-object scenes with:
- Scene RGB image + instance mask
- Depth map (`.exr` or `.npy`) → unprojected to canonical coordinate map
- Mesh → voxelized in `[-0.5, 0.5]³` at configurable resolution (default 64³)
- Optional azimuth canonicalization (rotate canonical space based on camera azimuth)

```yaml
# Config usage
dataset:
  objaverse65k:
    target: UniDataset.Scene.ObjaverseSceneDepthDataset
    params:
      summary_json: /path/to/align_summary.json
      valid_scenes_dir: /path/to/valid_scenes
      height: 518
      width: 518
      voxel_res: 64
      canonicalize_azimuth: true
```

Output dict per sample:
```python
{
    "image": [1, 3, H, W],              # Scene RGB (masked to instance)
    "mask": [1, 1, H, W],               # Instance binary mask
    "image_cropped": [1, 3, H, W],      # Cropped around mask
    "mask_cropped": [1, 1, H, W],       # Cropped mask
    "canonical_coord_map": [1, 3, H, W],# Canonical coordinate map
    "voxel": [Res, Res, Res],           # Occupancy grid
    ...
}
```

### BlenderProcSceneDepthDataset (Training, 3D-FRONT)

Loads BlenderProc-rendered 3D-FRONT multi-object scenes with:

- Scene RGB + per-instance masks from HDF5 renderings
- Depth map (`{view_samples_dir}/{unique_id}/depth.npy` or `depth.exr`) unprojected to a per-object canonical coordinate map
- Mesh → voxelized in `[-0.5, 0.5]³` (resolution 64³)
- Optional azimuth canonicalization (rotate each object's canonical space based on camera azimuth)

`unique_id` comes from the preprocess index (`preprocessed_view['id']`, e.g. `"scene__floor_0__0"`).

For each selected object instance `i`:

```
canonical_coord_map[i] = depth_to_canonical_coord_map(
    depth, mask[i], fov,
    T_cam_to_canonical[i] = inv(world_to_cam @ mesh_to_world[i])
)
```

```yaml
# Config usage
dataset:
  threedfront:
    target: UniDataset.Scene.BlenderProcSceneDepthDataset
    params:
      renderings_root: /path/to/blenderproc_renderings
      model_data_dir: /path/to/3dfront_models
      preprocess_json_path: /path/to/preprocess.json
      view_samples_dir: /path/to/view_samples
      height: 360
      width: 480
      canonicalize_azimuth: true
```

`preprocess_json_path` is required. `poses_dir` defaults to `{renderings_root}/poses`.

Output dict per sample (N = number of selected instances in the view):
```python
{
    "id": str,
    "num_instances": int,
    "rgb": [N, 3, H, W],                         # Per-instance RGB (masked)
    "mask": [N, 1, H, W],                        # Per-instance binary mask
    "rgb_scene": [N, 3, H, W],                   # Full scene RGB (repeated over N)
    "rgb_cropped": [N, 3, H, W],                 # Cropped around mask
    "mask_cropped": [N, 1, H, W],                # Cropped mask
    "canonical_coord_map": [N, 3, H, W],         # Per-object canonical coordinate map
    "canonical_coord_map_cropped": [N, 3, H, W], # Cropped coordinate map
    "voxel": [N, Res, Res, Res],                 # Occupancy grids
    "select_indices": [N],                       # Selected instance indices
    ...
}
```

Optional pack keys: `canonical_bboxes` (`use_bbox_layout=True`), `surface` / `scene_mesh` (`with_mesh=True`), `transformation` / `azimuth_estimate` (`include_transformation=True`).

### EvalDataset (Evaluation)

Loads evaluation scenes from the following directory format:

```
{eval_dir}/{scene_id}/
├── input/
│   ├── scene.png
│   └── mask.png (or mask_000.png, mask_001.png, ...)
└── gt/
    ├── canonical_coord_map.npy    # [3, H, W]
    ├── voxel.npy                  # [Res, Res, Res]
    └── camera.json                # (optional)
```

```yaml
val_dataset:
  objaverse_eval:
    target: UniDataset.Scene.EvalDataset
    params:
      eval_dir: /path/to/evaluation_dataset
      height: 518
      width: 518
      voxel_res: 64
```

## Installation

```bash
cd UniDataset
pip install -e .
```

## Key Utilities

| Function | Description |
|----------|-------------|
| `voxels_to_pcd(voxel, voxel_res, min_bound, max_bound)` | Occupancy grid → 3D point cloud |
| `crop_around_mask(image, mask, target_h, target_w)` | Crop image/mask to bounding box with padding |
| `compute_similarity_transform(src_pts, tgt_pts)` | Solve s, R, t between point sets |
