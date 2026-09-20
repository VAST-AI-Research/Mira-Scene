# Mira-Scene 评测

本目录包含推荐的 BlendSwap benchmark 评测入口。完整的端到端说明见
[`docs/eval_zh-CN.md`](docs/eval_zh-CN.md)，英文版本见
[`docs/eval.md`](docs/eval.md)。独立指标脚本的详细说明见
[`docs/evaluation.md`](docs/evaluation.md)。

## 下载 benchmark

benchmark 发布在 Hugging Face dataset 仓库
`Yang-Tian/Mira-Scene-Dataset` 中。安装并登录 Hugging Face 客户端后，下载
`blendswap_eval/` 子目录：

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download Yang-Tian/Mira-Scene-Dataset \
  --repo-type dataset \
  --include "blendswap_eval/*" \
  --local-dir data/mira-scene-dataset
```

下载后评测根目录为：

```text
data/mira-scene-dataset/blendswap_eval
```

## 运行

配置各阶段环境和 CCM checkpoint 后运行：

```bash
python eval_scripts/run_eval.py \
  --data-dir data/mira-scene-dataset/blendswap_eval \
  --config infer_scripts/config/local.yaml \
  --output-dir eval_output/blendswap_eval_sam3d_gt \
  --ckpt-dir /path/to/ccm_checkpoint \
  --gpu-ids 0,1,2,3
```

数据格式、参数、指标和输出文件说明见
[`docs/eval_zh-CN.md`](docs/eval_zh-CN.md)。
