# 环境配置

[English](environment.md) | [简体中文](environment_zh-CN.md) · [推理文档](../README_zh-CN.md)

Mira-Scene 使用五个 Conda 环境，因为各上游项目对 Python、PyTorch 和 CUDA 扩展的要求不同。`pipeline.py` 会为每个阶段选择配置中指定的解释器。

## 环境映射

| 环境 | 阶段 | 主要依赖 |
| --- | --- | --- |
| `mira-segmentation` | `segmentation` | SAM3、FastAPI/Uvicorn、VLM client |
| `mira-geometry` | `depth`、`floor`、`scene`、`environment` | MoGe/PPD、Open3D、Trimesh |
| `mira-ccm` | `ccm`、评测 | Mira-CCM、UniDataset、diffusers、spconv |
| `mira-sam3d` | SAM3D `mesh` | SAM3D、PyTorch3D、Kaolin、gsplat、nvdiffrast、mip-splatting rasterizer |
| `mira-trellis2` | TRELLIS.2 `mesh` | TRELLIS.2、flash-attn、nvdiffrast、nvdiffrec、CuMesh、o-voxel、FlexGEMM |

基础环境清单位于 [`../environments/`](../environments/)。外部源码、模型权重和已编译扩展不会包含在仓库中。

## 前置条件

- Linux、兼容的 NVIDIA 驱动与 CUDA toolkit；
- Conda 或兼容的环境管理器；
- 所选阶段需要的外部源码（Mira-CCM、UniDataset、SAM3、SAM3D Objects、PPD/MoGe 和/或 TRELLIS.2）；
- 已下载的 checkpoint 以及服务凭据。

应在与运行环境驱动兼容的 GPU 编译节点构建 CUDA 扩展。目标环境的 PyTorch 与用于编译的 toolkit 必须 ABI 兼容。

## 创建基础环境

```bash
cd /path/to/Mira-Scene
MIRA_ENV_ROOT=/path/to/conda-envs \
CONDA_BIN=/path/to/conda \
  bash infer_scripts/environments/create_envs.sh
```

若 `conda` 已在 `PATH` 中，可不设置 `CONDA_BIN`。脚本将在 `MIRA_ENV_ROOT` 下创建 `mira-segmentation`、`mira-geometry`、`mira-ccm`、`mira-sam3d` 和 `mira-trellis2`。`.mira-base-environment-complete` 只表示基础清单安装成功，并不表示外部项目或 CUDA 扩展已就绪。

可选变量包括 `MIRA_PIP_INDEX_URL` 和 `MIRA_PIP_TRUSTED_HOST`。若希望在此步骤安装 SAM3，可将 `MIRA_SAM3_REPO` 指向包含 `pyproject.toml` 的 SAM3 checkout。

## 安装外部项目和 CUDA 扩展

创建基础环境后，在 GPU 编译节点运行：

```bash
MIRA_ENV_ROOT=/path/to/conda-envs \
MIRA_SAM3D_REPO=/path/to/sam-3d-objects \
MIRA_TRELLIS2_REPO=/path/to/TRELLIS.2 \
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST=8.0 \
MAX_JOBS=4 \
  bash infer_scripts/environments/install_external_projects.sh
```

脚本会安装上游推理依赖，构建 SAM3D 的 nvdiffrast 与固定版本的 mip-splatting rasterizer，并构建 TRELLIS.2 所需的 flash-attn、nvdiffrast、nvdiffrec、CuMesh、o-voxel 和 FlexGEMM。代表性 import 全部成功后才会写入 `.mira-external-projects-complete`。

NVIDIA A800 的 compute capability 为 8.0，因此默认使用 `TORCH_CUDA_ARCH_LIST=8.0`。其他 GPU 需要覆盖此变量。如果扩展是用其他 PyTorch/CUDA 组合或 GPU 架构编译的，应在实际运行环境中重新安装；跨不兼容软件栈复制已编译包可能产生 undefined symbol、invalid device function 或 JIT 失败。

以下变量可指定可复用的扩展源码目录：

```text
MIRA_NVDIFFRAST_REPO
MIRA_NVDIFFREC_REPO
MIRA_CUMESH_REPO
MIRA_FLEXGEMM_REPO
MIRA_DIFF_GAUSSIAN_RASTERIZATION_URL
```

源码版本应与所用 SAM3D/TRELLIS.2 版本兼容。

## 配置 pipeline

```bash
cp infer_scripts/config/example.yaml infer_scripts/config/local.yaml
```

填写 `external`、`checkpoints` 和 `environments`。常见解释器映射为：

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

## Checkpoint 下载与配置

首先安装 Hugging Face 客户端并完成一次认证。SAM 3、SAM 3D Objects 和
DINOv3 等 gated 仓库需要先在各自的模型页面申请访问权限，然后才能下载。

```bash
python -m pip install -U huggingface_hub
hf auth login
mkdir -p checkpoints
```

只需下载所选后端需要的 checkpoint。下面使用仓库内的相对目录作为示例；
在 `local.yaml` 中建议填写绝对路径。

### 核心 checkpoint

所有 pipeline 配置都需要 Mira-Scene CCM 模型；运行 segmentation 阶段时还
需要 SAM 3。

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

`external.sam3.repo` 指向 SAM 3 官方源码 checkout，
`external.sam3.checkpoint` 指向下载的权重文件。CCM 路径必须指向包含
`model_index.json` 的 `pipeline/` 子目录，而不是它的上级目录。

### Depth checkpoint

通过 `depth.method` 选择一种深度模型。

Pixel-Perfect Depth（`ppd`，默认选项）需要下载全部三个文件：

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

使用 MoGe v1 时，将官方 checkpoint 下载到 Hugging Face 默认缓存并选择
`moge`。模型加载器直接使用 Hugging Face model ID，因此保留在默认缓存即可。

```bash
hf download Ruicheng/moge-vitl
```

```yaml
external:
  moge: {repo: /absolute/path/to/MoGe}
depth: {method: moge, enable_metric: false}
```

使用 metric MoGe-2 时，将模型下载到固定目录，并通过
`MIRA_MOGE2_CHECKPOINT` 暴露给 depth 进程。当 `enable_metric: true` 使用
MoGe-2 对 `moge` 或 `ppd` 结果定标时，也会读取同一设置。

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

请在启动 `pipeline.py` 的 shell 中设置 `MIRA_MOGE2_CHECKPOINT`。两种 MoGe
后端都应配置 `external.moge.repo`，因为该目录提供模型实现。

### Mesh checkpoint

SAM3D 后端需要下载完整仓库，因为 `checkpoints/pipeline.yaml` 会引用同目录
下的其他配置和权重：

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

TRELLIS.2 后端需要以下三个模型仓库：

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

`external` 中的源码路径与 checkpoint 目录是两个不同概念。不要提交
`local.yaml`、API key、checkpoint 凭据或私有路径。

使用以下命令检查所有路径与 import，不执行模型推理：

```bash
python infer_scripts/pipeline.py \
  --input /path/to/image_or_directory \
  --output /path/to/results \
  --config infer_scripts/config/local.yaml \
  --preflight-only
```
