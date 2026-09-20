# Mira-Scene evaluation

This directory contains the supported BlendSwap benchmark evaluation driver.
The complete end-to-end guide is in [`docs/eval.md`](docs/eval.md), with a
[中文版本](docs/eval_zh-CN.md). A detailed reference for standalone metric
scripts is available in [`docs/evaluation.md`](docs/evaluation.md).

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

The evaluation root is then:

```text
data/mira-scene-dataset/blendswap_eval
```

## Run

After configuring the stage environments and CCM checkpoint:

```bash
python eval_scripts/run_eval.py \
  --data-dir data/mira-scene-dataset/blendswap_eval \
  --config infer_scripts/config/local.yaml \
  --output-dir eval_output/blendswap_eval_sam3d_gt \
  --ckpt-dir /path/to/ccm_checkpoint \
  --gpu-ids 0,1,2,3
```

See [`docs/eval.md`](docs/eval.md) for the required data format, options,
metrics, and output files.
