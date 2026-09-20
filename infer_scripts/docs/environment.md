# Environment setup

[English](environment.md) | [简体中文](environment_zh-CN.md) · [Inference guide](../README.md)

Mira-Scene uses five Conda environments because its upstream projects have
different Python, PyTorch, and CUDA-extension requirements. `pipeline.py`
selects the configured interpreter for every stage.

## Environment map

| Environment | Stages | Main dependencies |
| --- | --- | --- |
| `mira-segmentation` | `segmentation` | SAM3, FastAPI/Uvicorn, VLM client |
| `mira-geometry` | `depth`, `floor`, `scene`, `environment` | MoGe/PPD, Open3D, Trimesh |
| `mira-ccm` | `ccm`, evaluation | Mira-CCM, UniDataset, diffusers, spconv |
| `mira-sam3d` | SAM3D `mesh` | SAM3D, PyTorch3D, Kaolin, gsplat, nvdiffrast, mip-splatting rasterizer |
| `mira-trellis2` | TRELLIS.2 `mesh` | TRELLIS.2, flash-attn, nvdiffrast, nvdiffrec, CuMesh, o-voxel, FlexGEMM |

Base manifests live in [`../environments/`](../environments/). External source
trees, model weights, and compiled extensions are intentionally not vendored.

## Prerequisites

- Linux, a compatible NVIDIA driver, and a CUDA toolkit;
- Conda or a compatible environment manager;
- external checkouts required by the selected stages (Mira-CCM,
  UniDataset, SAM3, SAM3D Objects, PPD/MoGe, and/or TRELLIS.2);
- downloaded model checkpoints and service credentials.

Build CUDA extensions on a machine compatible with the runtime driver. The
target environment's PyTorch build and the compilation toolkit must be
ABI-compatible.

## Create base environments

```bash
cd /path/to/Mira-Scene
MIRA_ENV_ROOT=/path/to/conda-envs \
CONDA_BIN=/path/to/conda \
  bash infer_scripts/environments/create_envs.sh
```

`CONDA_BIN` is optional when `conda` is on `PATH`. The command creates
`mira-segmentation`, `mira-geometry`, `mira-ccm`, `mira-sam3d`, and
`mira-trellis2` below `MIRA_ENV_ROOT`. A `.mira-base-environment-complete`
marker means only that the base manifest succeeded—not that external projects
or CUDA extensions are ready.

Optional controls include `MIRA_PIP_INDEX_URL` and
`MIRA_PIP_TRUSTED_HOST`. Set `MIRA_SAM3_REPO` to a SAM3 checkout containing
`pyproject.toml` if SAM3 should be installed during this step.

## Install external projects and CUDA extensions

Run on a GPU build machine after creating the base environments:

```bash
MIRA_ENV_ROOT=/path/to/conda-envs \
MIRA_SAM3D_REPO=/path/to/sam-3d-objects \
MIRA_TRELLIS2_REPO=/path/to/TRELLIS.2 \
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST=8.0 \
MAX_JOBS=4 \
  bash infer_scripts/environments/install_external_projects.sh
```

The script installs the upstream inference requirements, builds SAM3D's
nvdiffrast and pinned mip-splatting rasterizer, and builds TRELLIS.2's
flash-attn, nvdiffrast, nvdiffrec, CuMesh, o-voxel, and FlexGEMM dependencies.
It verifies representative imports before writing
`.mira-external-projects-complete`.

NVIDIA A800 has compute capability 8.0, hence the default
`TORCH_CUDA_ARCH_LIST=8.0`. Override it for other GPUs. If an extension was
built with another PyTorch/CUDA combination or architecture, reinstall it in
the runtime environment; copying compiled packages between incompatible stacks
can cause undefined symbols, invalid-device-function errors, or JIT failures.

Reusable extension checkouts can be selected with:

```text
MIRA_NVDIFFRAST_REPO
MIRA_NVDIFFREC_REPO
MIRA_CUMESH_REPO
MIRA_FLEXGEMM_REPO
MIRA_DIFF_GAUSSIAN_RASTERIZATION_URL
```

Keep their revisions compatible with the selected SAM3D/TRELLIS.2 versions.

## Configure the pipeline

```bash
cp infer_scripts/config/example.yaml infer_scripts/config/local.yaml
```

Fill `external`, `checkpoints`, and `environments`. A typical interpreter map
is:

```yaml
environments:
  segmentation: {python: /path/to/conda-envs/mira-segmentation/bin/python}
  depth: {python: /path/to/conda-envs/mira-geometry/bin/python}
  ccm: {python: /path/to/conda-envs/mira-ccm/bin/python}
  mesh: {python: /path/to/conda-envs/mira-sam3d/bin/python}
  trellis2: {python: /path/to/conda-envs/mira-trellis2/bin/python}
  floor: {python: /path/to/conda-envs/mira-geometry/bin/python}
  scene: {python: /path/to/conda-envs/mira-geometry/bin/python}
  environment: {python: /path/to/conda-envs/mira-geometry/bin/python}
```

## Checkpoint download and configuration

Install the Hugging Face client and authenticate once. Authentication is
required for gated repositories such as SAM 3, SAM 3D Objects, and DINOv3;
request access on each model page before downloading it.

```bash
python -m pip install -U huggingface_hub
hf auth login
mkdir -p checkpoints
```

Download only the checkpoints required by the selected backends. The paths
below are repository-local examples; use absolute paths in `local.yaml`.

### Core checkpoints

The Mira-Scene CCM model is required by every pipeline configuration. SAM 3 is
required when running the segmentation stage.

```bash
hf download Yang-Tian/Mira-Scene \
  --include "pipeline/*" --local-dir checkpoints/mira-scene

hf download facebook/sam3 sam3.pt \
  --local-dir checkpoints/sam3
```

```yaml
external:
  sam3:
    repo: /absolute/path/to/sam3-source
    checkpoint: /absolute/path/to/checkpoints/sam3/sam3.pt
checkpoints:
  ccm: /absolute/path/to/checkpoints/mira-scene/pipeline
```

`external.sam3.repo` is the official SAM 3 source checkout, whereas
`external.sam3.checkpoint` is the downloaded weight file. The CCM path must
point to the `pipeline/` subdirectory containing `model_index.json`, not its
parent.

### Depth checkpoints

Choose one depth method with `depth.method`.

For Pixel-Perfect Depth (`ppd`, the default), download all three files:

```bash
mkdir -p checkpoints/ppd
hf download gangweix/Pixel-Perfect-Depth ppd.pth \
  --local-dir checkpoints/ppd
hf download Ruicheng/moge-2-vitl-normal model.pt \
  --local-dir checkpoints/ppd/moge2
hf download depth-anything/Depth-Anything-V2-Large \
  depth_anything_v2_vitl.pth --local-dir checkpoints/ppd
```

```yaml
external:
  ppd: {repo: /absolute/path/to/pixel-perfect-depth}
checkpoints:
  ppd: /absolute/path/to/checkpoints/ppd/ppd.pth
  ppd_moge: /absolute/path/to/checkpoints/ppd/moge2/model.pt
  ppd_da2: /absolute/path/to/checkpoints/ppd/depth_anything_v2_vitl.pth
depth: {method: ppd, enable_metric: false}
```

For MoGe v1, cache the official checkpoint and select `moge`. The model loader
uses the Hugging Face model ID directly, so retaining it in the normal HF cache
is sufficient.

```bash
hf download Ruicheng/moge-vitl
```

```yaml
external:
  moge: {repo: /absolute/path/to/MoGe}
depth: {method: moge, enable_metric: false}
```

For metric MoGe-2, download the model to a stable directory and expose it to
the depth process with `MIRA_MOGE2_CHECKPOINT`. The same setting is used when
`enable_metric: true` scales a `moge` or `ppd` result using MoGe-2.

```bash
hf download Ruicheng/moge-2-vitl \
  --local-dir checkpoints/moge-2-vitl
export MIRA_MOGE2_CHECKPOINT=/absolute/path/to/checkpoints/moge-2-vitl
```

```yaml
external:
  moge: {repo: /absolute/path/to/MoGe}
depth: {method: moge2, enable_metric: false}
```

Set `MIRA_MOGE2_CHECKPOINT` in the shell that launches `pipeline.py`. Keep
`external.moge.repo` configured for both MoGe variants because it supplies the
model implementation.

### Mesh checkpoints

For the SAM3D backend, download the complete repository because
`checkpoints/pipeline.yaml` references sibling configs and weights:

```bash
hf download facebook/sam-3d-objects \
  --local-dir checkpoints/sam-3d-objects
```

```yaml
external:
  sam3d: {repo: /absolute/path/to/sam-3d-objects-source}
checkpoints:
  sam3d: /absolute/path/to/checkpoints/sam-3d-objects/checkpoints/pipeline.yaml
mesh: {backend: sam3d, bake_texture: true}
```

For the TRELLIS.2 backend, all three model repositories are required:

```bash
hf download microsoft/TRELLIS.2-4B \
  --local-dir checkpoints/TRELLIS.2-4B
hf download briaai/RMBG-2.0 \
  --local-dir checkpoints/RMBG-2.0
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir checkpoints/dinov3-vitl16
```

```yaml
external:
  trellis2: {repo: /absolute/path/to/TRELLIS.2-source}
checkpoints:
  trellis2: /absolute/path/to/checkpoints/TRELLIS.2-4B
  rmbg: /absolute/path/to/checkpoints/RMBG-2.0
  dinov3: /absolute/path/to/checkpoints/dinov3-vitl16
mesh: {backend: trellis2}
```

The source paths in `external` are separate from checkpoint directories. Do
not commit `local.yaml`, API keys, checkpoint credentials, or private paths.

Validate all resolved paths and imports without running model inference:

```bash
python infer_scripts/pipeline.py \
  --input /path/to/image_or_directory \
  --output /path/to/results \
  --config infer_scripts/config/local.yaml \
  --preflight-only
```
