# BlendSwap evaluation

[English](eval.md) | [简体中文](eval_zh-CN.md)

## Download the benchmark

The benchmark is published in the `Yang-Tian/Mira-Scene-Dataset` Hugging Face
dataset repository. Install and authenticate the Hugging Face client, then
download the `blendswap_eval/` subdirectory:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download Yang-Tian/Mira-Scene-Dataset \
  --repo-type dataset \
  --include "blendswap_eval/*" \
  --local-dir data/mira-scene-dataset
```

Set `--data-dir` to the downloaded directory:

```text
data/mira-scene-dataset/blendswap_eval
```

`run_eval.py` provides the supported end-to-end BlendSwap evaluation:

```text
image + instance masks + depth
  -> CCM -> meshes -> graph-free scene -> metrics
```


## Data format

`--data-dir` must point to the directory containing all evaluation cases:

```text
blendswap_eval/
└── <case_id>/
    ├── input/
    │   ├── scene.png
    │   ├── mask_000.png
    │   ├── mask_001.png
    │   └── ...
    ├── depth/gt/
    │   ├── depth.npy                 # [H, W]
    │   ├── camera_pts_map.npy        # [H, W, 3]
    │   ├── valid_mask.npy            # [H, W]
    │   ├── intrinsics.npy            # [3, 3]
    │   └── fov_x_rad.txt
    └── gt/
        ├── scene_camera.glb
        └── camera.json
```

Masks must be numbered continuously from `mask_000.png`. Their order defines
the correspondence between predicted and GT objects. The image, masks, and
depth maps must have the same resolution.

`depth/gt` must have the same scale as `gt/scene_camera.glb`. An independently
metric-aligned depth can be kept under `depth/gt_metric`, but `run_eval.py`
does not read it. Replacing the scale-matched `depth/gt` changes scene scale
and makes placement metrics invalid.

## Run

Create and complete a local inference configuration first:

```bash
cp infer_scripts/config/example.yaml infer_scripts/config/local.yaml
```

Then run from the repository root:

```bash
python eval_scripts/run_eval.py \
  --data-dir /path/to/blendswap_eval \
  --config infer_scripts/config/local.yaml \
  --output-dir /path/to/eval_output \
  --ckpt-dir /path/to/ccm_checkpoint \
  --gpu-ids 0,1,2,3
```

`--ckpt-dir` overrides `checkpoints.ccm` in the YAML and may be omitted when
that value is already correct. `--output-dir` defaults to
`eval_output/blendswap_eval_sam3d_gt`.

Useful options:

| Option | Description |
| --- | --- |
| `--case NAME` | Evaluate one case; repeat to select multiple cases. |
| `--force-stage ccm\|mesh\|scene` | Rebuild a cached stage; repeat as needed. |
| `--gpu-ids 0,1,...` | Shard CCM/SAM3D cases across GPUs. |
| `--skip-inference` | Evaluate existing compatible scenes. |
| `--skip-eval` | Stop after scene construction. |
| `--no-2d-iou` | Skip the slow silhouette metric. |
| `--preflight-only` | Check configured environments without inference. |
| `--dry-run` | Print generated commands without model execution. |

Multi-GPU execution shards independent cases; it does not use multiple GPUs
for one case. Scene construction and metric computation are serial.

## Metrics

| Metric | Better | Description |
| --- | --- | --- |
| `object_cd` | Lower | Bidirectional squared Chamfer distance after per-object normalization and ICP. |
| `object_fscore` | Higher | Point precision/recall F-score at threshold `0.1`. |
| `object_emd` | Lower | Hungarian-assignment EMD after per-object normalization and ICP. |
| `iou_3d` | Higher | Axis-aligned 3D bounding-box IoU of each posed object pair. |
| `icp_rot_deg` | Lower | Rotation error obtained from per-object ICP, in degrees. |
| `iou_2d` | Higher | Object silhouette IoU rendered through the GT camera. |
| `add_s` | Lower | Symmetric closest-point distance normalized by GT diameter. |

Shape metrics normalize each object independently and mainly measure geometry.
The remaining metrics use posed objects and also measure layout or pose.

Predicted `geometry_N` and GT `object_NNN` nodes are paired by numeric index.
Object-count mismatches are skipped instead of silently truncating a scene.
Evaluation uses world space by default by applying `c2w` from `camera.json`.
Each scene contributes equally to the final `AVERAGE` row.

## Outputs

```text
<output_dir>/
├── <case_id>/scene/sam3d/gt_depth/
│   ├── scene.glb
│   └── scene_optimization.json
└── evaluation/sam3d_gt/
    ├── run.json
    ├── pipeline_driver.log
    ├── construct_scene.log
    ├── eval_scene.log
    ├── eval_scene_results.csv
    └── eval_scene_results_per_object.csv
```

The first CSV contains per-scene results and an `AVERAGE` row. The second
contains per-object results.
