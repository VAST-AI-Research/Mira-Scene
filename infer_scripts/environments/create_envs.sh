#!/usr/bin/env bash
set -euo pipefail
CONDA_BIN="${CONDA_BIN:-conda}"
PREFIX_ROOT="${MIRA_ENV_ROOT:-/mnt/pfs/data/sunyangtian/Experiments/conda_envs}"
# Aliyun is often reachable from restricted GPU pods. Override with
# MIRA_PIP_INDEX_URL/MIRA_PIP_TRUSTED_HOST when another mirror is preferred.
export PIP_INDEX_URL="${MIRA_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${MIRA_PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in segmentation geometry ccm sam3d trellis2; do
  prefix="$PREFIX_ROOT/mira-$name"
  if [[ -x "$prefix/bin/python" ]] && { [[ -f "$prefix/.mira-base-environment-complete" ]] || [[ -f "$prefix/.mira-environment-complete" ]]; }; then echo "exists: $prefix"; continue; fi
  if [[ -e "$prefix" ]]; then
    echo "incomplete environment exists: $prefix; remove it explicitly or finish its pip transaction" >&2
    exit 1
  fi
  created=0
  for attempt in 1 2 3; do
    if "$CONDA_BIN" env create --prefix "$prefix" --file "$HERE/$name.yml"; then
      created=1
      break
    fi
    echo "environment $name attempt $attempt failed; retrying"
  done
  if [[ "$created" != 1 ]]; then
    echo "failed to create $prefix after 3 attempts" >&2
    exit 1
  fi
  # This marker covers this manifest only. It intentionally does not imply
  # that an external project's CUDA extensions were built successfully.
  touch "$prefix/.mira-base-environment-complete"
done
SAM3_REPO="${MIRA_SAM3_REPO:-}"
if [[ -n "$SAM3_REPO" ]] && [[ -f "$SAM3_REPO/pyproject.toml" ]]; then
  "$PREFIX_ROOT/mira-segmentation/bin/python" -m pip install -e "$SAM3_REPO"
fi
echo "Base environments are complete. Install external repositories according to their pinned upstream instructions."
echo "SAM3D additionally requires: PyTorch3D Kaolin gsplat and the CUDA nvdiffrast and diff-gaussian-rasterization extensions."
echo "TRELLIS.2 additionally requires: flash-attn nvdiffrast nvdiffrec CuMesh o-voxel FlexGEMM."
