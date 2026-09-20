#!/usr/bin/env bash
set -euo pipefail

# Run this on a GPU build node after create_envs.sh. External repositories are
# referenced, never copied, so the inference tree remains publishable.
PREFIX_ROOT="${MIRA_ENV_ROOT:-/mnt/pfs/data/sunyangtian/Experiments/conda_envs}"
SAM3D_REPO="${MIRA_SAM3D_REPO:-}"
SAM3_REPO="${MIRA_SAM3_REPO_SEGMENTATION:-${MIRA_SAM3_REPO_IMAGE:-}}"
TRELLIS2_REPO="${MIRA_TRELLIS2_REPO:-}"
# nvdiffrast is needed by SAM3D's utils3d rasterization path as well as by
# TRELLIS.2.  Keep the source checkout configurable so a shared checkout can
# be reused on a build pod without cloning it again.
NVDIFFRAST_REPO="${MIRA_NVDIFFRAST_REPO:-/tmp/extensions/nvdiffrast}"
NVDIFFREC_REPO="${MIRA_NVDIFFREC_REPO:-/tmp/extensions/nvdiffrec}"
CUMESH_REPO="${MIRA_CUMESH_REPO:-/tmp/extensions/CuMesh}"
FLEXGEMM_REPO="${MIRA_FLEXGEMM_REPO:-/tmp/extensions/FlexGEMM}"
# SAM3D's Gaussian renderer uses the mip-splatting fork rather than the
# original Inria rasterizer: it requires the additional ``kernel_size`` and
# ``subpixel_offset`` settings.  Pin the source revision so rebuilding an
# environment cannot silently change the CUDA extension ABI.
DIFF_GAUSSIAN_RASTERIZATION_URL="${MIRA_DIFF_GAUSSIAN_RASTERIZATION_URL:-git+https://github.com/autonomousvision/mip-splatting.git@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#subdirectory=submodules/diff-gaussian-rasterization}"
export PIP_INDEX_URL="${MIRA_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${MIRA_PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

if [[ -z "$SAM3D_REPO" || ! -f "$SAM3D_REPO/pyproject.toml" ]]; then
  echo "Set MIRA_SAM3D_REPO to the official sam-3d-objects checkout" >&2
  exit 2
fi
if [[ -z "$TRELLIS2_REPO" || ! -f "$TRELLIS2_REPO/setup.sh" ]]; then
  echo "Set MIRA_TRELLIS2_REPO to the official TRELLIS.2 checkout" >&2
  exit 2
fi

sam3d_python="$PREFIX_ROOT/mira-sam3d/bin/python"
trellis_python="$PREFIX_ROOT/mira-trellis2/bin/python"

add_source_path() {
  local python_bin="$1" source_dir="$2" name="$3"
  [[ -d "$source_dir" ]] || return 0
  local site_dir
  site_dir="$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "$source_dir" > "$site_dir/mira_${name}.pth"
}

clone_pinned_repo() {
  local target="$1" url="$2" revision="$3" recursive="${4:-false}"
  [[ -d "$target/.git" ]] && return 0
  if [[ -e "$target" ]]; then
    echo "extension source exists but is not a Git checkout: $target" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$target")"
  if [[ "$recursive" == true ]]; then
    git clone --recursive "$url" "$target"
  else
    git clone "$url" "$target"
  fi
  git -C "$target" checkout --detach "$revision"
  if [[ "$recursive" == true ]]; then
    git -C "$target" submodule update --init --recursive
  fi
}

# Keep all builds reproducible on the A800 pod.  The conda solver may choose a
# CPU-only torch build for TRELLIS.2, while its CUDA extensions require a
# matching CUDA wheel.  Install the wheel explicitly before compiling any
# extension.  ``--no-build-isolation`` also prevents build backends from
# silently pulling a second (often CPU) torch into the temporary build env.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
# NVIDIA A800 is compute capability 8.0. Override this variable explicitly if
# environments are built for another GPU architecture.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

"$trellis_python" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0"

# Follow upstream's pinned PyTorch3D source. SAM3D inference requirements also
# include Kaolin and other compiled packages, so use its published extras.
# SAM-3D uses a Hatch build backend that is not present in minimal conda
# prefixes.  A .pth file exposes the checked-out package without invoking a
# build or downloading anything; this is sufficient for inference and keeps
# the external checkout out of the repository.
add_source_path "$sam3d_python" "$SAM3D_REPO" sam3d
"$sam3d_python" -m pip install --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
  "$sam3d_python" -m pip install --no-build-isolation -r "$SAM3D_REPO/requirements.inference.txt"
# Some unbounded transitive packages currently upgrade NumPy past Kaolin's
# supported ABI. Reassert the upstream-compatible pair last.
"$sam3d_python" -m pip install "numpy<2" "opencv-python==4.9.0.80"

# Build the mip-splatting rasterizer against this environment's PyTorch.  It
# supplies GaussianRasterizationSettings for SAM3D's ``inria`` rendering
# backend and is exercised by texture baking/post-processing.
CUDA_HOME="$CUDA_HOME" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
MAX_JOBS="$MAX_JOBS" \
  "$sam3d_python" -m pip install --no-build-isolation \
    "$DIFF_GAUSSIAN_RASTERIZATION_URL"

# SAM3D's post-processing imports ``nvdiffrast.torch`` through utils3d.  It is
# a CUDA extension and is intentionally installed here (rather than as a
# plain conda/pip manifest dependency) after the target PyTorch and CUDA
# toolchain are available.  This also makes the setup reproducible for a
# freshly-created mira-sam3d environment.
if [[ ! -d "$NVDIFFRAST_REPO" ]]; then
  mkdir -p "$(dirname "$NVDIFFRAST_REPO")"
  git clone --depth 1 --branch v0.4.0 \
    https://github.com/NVlabs/nvdiffrast.git "$NVDIFFRAST_REPO"
fi
if [[ ! -f "$NVDIFFRAST_REPO/setup.py" && ! -f "$NVDIFFRAST_REPO/pyproject.toml" ]]; then
  echo "nvdiffrast source checkout is invalid: $NVDIFFRAST_REPO" >&2
  exit 2
fi
CUDA_HOME="$CUDA_HOME" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
MAX_JOBS="$MAX_JOBS" \
  "$sam3d_python" -m pip install --no-build-isolation "$NVDIFFRAST_REPO"
LIDRA_SKIP_INIT=true "$sam3d_python" -c \
  "import nvdiffrast.torch; from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings; from utils3d.torch import rasterization; from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap"
touch "$PREFIX_ROOT/mira-sam3d/.mira-external-projects-complete"

# Install TRELLIS.2 itself and its CUDA extensions without the upstream
# setup.sh wrapper.  That wrapper assumes sudo and a default pip index, neither
# of which is available in the shared pod.  Building each extension in place
# is equivalent to the upstream --flash-attn/--nvdiffrast/--nvdiffrec/
# --cumesh/--o-voxel/--flexgemm options, but remains non-root and uses the
# selected interpreter.
(
  cd "$TRELLIS2_REPO"
  export PATH="$PREFIX_ROOT/mira-trellis2/bin:$PATH"
  add_source_path "$trellis_python" "$TRELLIS2_REPO" trellis2
  # TRELLIS.2 sparse attention does not support PyTorch SDPA. Even when the
  # dense ATTN_BACKEND is set to sdpa it retains flash_attn as its default, so
  # compile the upstream-pinned package against the selected PyTorch first.
  CUDA_HOME="$CUDA_HOME" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  MAX_JOBS="$MAX_JOBS" \
    "$trellis_python" -m pip install --no-build-isolation "flash-attn==2.7.3"
  clone_pinned_repo "$NVDIFFREC_REPO" \
    https://github.com/JeffreyXiang/nvdiffrec.git \
    b296927cc7fd01c2ac1087c8065c4d7248f72da4
  clone_pinned_repo "$CUMESH_REPO" \
    https://github.com/JeffreyXiang/CuMesh.git \
    12289e1062f0603f2f0d0771b02e1395d247f26f true
  clone_pinned_repo "$FLEXGEMM_REPO" \
    https://github.com/JeffreyXiang/FlexGEMM.git \
    6dd94a859c26ee8246888502eada3dd8ad85532e
  declare -A ext_dirs=(
    [nvdiffrast]="$NVDIFFRAST_REPO"
    [nvdiffrec_render]="$NVDIFFREC_REPO"
    [cumesh]="$CUMESH_REPO"
    [o_voxel]="$TRELLIS2_REPO/o-voxel"
    [flex_gemm]="$FLEXGEMM_REPO"
  )
  for ext in nvdiffrast nvdiffrec_render cumesh o_voxel flex_gemm; do
    source_dir="${ext_dirs[$ext]}"
    if [[ -d "$source_dir" ]]; then
      CUDA_HOME="$CUDA_HOME" \
      TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
      MAX_JOBS="$MAX_JOBS" \
        "$trellis_python" -m pip install --force-reinstall --no-deps \
          --no-build-isolation "$source_dir"
    else
      echo "warning: TRELLIS.2 extension source not found: $ext" >&2
    fi
  done
)
PYTHONPATH="$TRELLIS2_REPO" ATTN_BACKEND=sdpa "$trellis_python" -c \
  "import flash_attn, o_voxel, cumesh, nvdiffrec_render.renderutils; from trellis2.pipelines import Trellis2ImageTo3DPipeline"
touch "$PREFIX_ROOT/mira-trellis2/.mira-external-projects-complete"

echo "SAM3D and TRELLIS.2 external project imports verified."
