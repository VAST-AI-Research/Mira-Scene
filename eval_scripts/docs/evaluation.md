# Mira-Scene evaluation scripts

This document describes the evaluation scripts and standalone metrics. The
short end-to-end guide is available in [`../README.md`](../README.md), and the
Chinese guide is available in [`eval_zh-CN.md`](eval_zh-CN.md).

## Directory layout

```text
eval_scripts/
├── run_eval.py                   # End-to-end BlendSwap evaluation driver
├── construct_scene_eval.py       # SAM3D + GT-depth graph-free construction
├── eval_scene.py                 # Scene reconstruction paper metrics
├── eval_ccm.py                   # Standalone CCM CD/F-score evaluation
├── eval_2d_3d_correspondence.py  # Standalone CCM/mesh consistency metrics
├── utils/                        # Metric and alignment implementations
├── docs/
│   ├── eval.md                   # English end-to-end evaluation guide
│   ├── eval_zh-CN.md             # 中文端到端评测文档
│   └── evaluation.md             # Detailed script and metric reference
└── tests/
```

## End-to-end BlendSwap evaluation

The recommended entry point is `run_eval.py`. This workflow:

- consumes masks, GT depth, camera metadata, and GT scenes from BlendSwap;
- segmentation and predicted-depth inference are skipped;
- CCM and SAM3D run through the normal resumable inference pipeline;
- only SAM3D is evaluated; TRELLIS.2 is not part of this benchmark;
- `construct_scene_eval.py` performs graph-free GT-depth scene construction
  with the historical joint similarity solver;
- `eval_scene.py` writes per-scene and per-object reports.

```bash
python eval_scripts/run_eval.py \
  --data-dir /path/to/blendswap_eval \
  --config infer_scripts/config/local.yaml \
  --output-dir /path/to/eval_output \
  --ckpt-dir /path/to/ccm_checkpoint \
  --gpu-ids 0,1,2,3
```

See [`eval.md`](eval.md) for the dataset format, benchmark download, complete
commands, metrics, and output layout.

## Standalone metric scripts

### Scene reconstruction

Use `eval_scene.py` when compatible predicted scenes already exist:

```bash
python eval_scripts/eval_scene.py \
  --pred_dir /path/to/predictions \
  --gt_dir /path/to/blendswap_eval \
  --pred_scene_file scene/sam3d/gt_depth/scene.glb
```

It reports `object_cd`, `object_fscore`, `object_emd`, `iou_3d`,
`icp_rot_deg`, `iou_2d`, and `add_s`. World space via `camera.json` `c2w` is
the default paper convention.

### CCM quality

`eval_ccm.py` compares predicted CCM arrays against GT CCM and performs
scale/translation alignment on shared valid pixels:

```bash
python eval_scripts/eval_ccm.py \
  --pred_dir /path/to/predictions \
  --gt_dir /path/to/ccm_ground_truth
```

It expects `canonical_coord_map_restored_000.npy` (preferred) or
`canonical_coord_map_000.npy` per scene and compatible GT coordinates. The
default output is `eval_ccm_results.csv` under the prediction root.

### 2D–3D correspondence consistency

This self-consistency metric needs no GT. It estimates a camera from predicted
CCM, raycasts a reconstructed mesh, then reports mask IoU, Chamfer distance,
and F-score:

```bash
python eval_scripts/eval_2d_3d_correspondence.py \
  --pred_dir /path/to/predictions \
  --recon_subdir moge_recon
```

It expects `<case>/<recon_subdir>/000.glb` and writes correspondence debug
artifacts below that reconstruction directory.
