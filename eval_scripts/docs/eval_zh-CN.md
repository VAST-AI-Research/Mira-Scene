# BlendSwap 推理与评测

[English](eval.md) | [简体中文](eval_zh-CN.md)

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

下载后，将 `--data-dir` 指向：

```text
data/mira-scene-dataset/blendswap_eval
```

`run_eval.py` 提供完整的 BlendSwap 评测流程：

```text
输入图 + 实例 mask + depth
  -> CCM -> meshes -> 无 graph 场景构建 -> 指标计算
```

## 数据格式

`--data-dir` 指向包含所有评测 case 的目录：

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

Mask 必须从 `mask_000.png` 开始连续编号，其顺序决定预测物体与 GT
物体的对应关系。输入图、mask 和深度数据必须具有相同分辨率。

`depth/gt` 必须与 `gt/scene_camera.glb` 保持相同尺度。单独进行 metric
对齐的深度可以放在 `depth/gt_metric`，但 `run_eval.py` 不会读取它。
如果用它替换原始 `depth/gt`，会导致生成场景的尺度错误，使布局指标失效。

## 运行方法

首先复制并填写本地推理配置：

```bash
cp infer_scripts/config/example.yaml infer_scripts/config/local.yaml
```

在仓库根目录运行：

```bash
python eval_scripts/run_eval.py \
  --data-dir /path/to/blendswap_eval \
  --config infer_scripts/config/local.yaml \
  --output-dir /path/to/eval_output \
  --ckpt-dir /path/to/ccm_checkpoint \
  --gpu-ids 0,1,2,3
```

`--ckpt-dir` 会覆盖 YAML 中的 `checkpoints.ccm`；若 YAML 已正确配置，
可以省略该参数。`--output-dir` 默认为
`eval_output/blendswap_eval_sam3d_gt`。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--case NAME` | 仅处理指定 case；可重复传入。 |
| `--force-stage ccm\|mesh\|scene` | 强制重新执行指定缓存阶段；可重复传入。 |
| `--gpu-ids 0,1,...` | 将不同 case 的 CCM/SAM3D 推理分配到多张 GPU。 |
| `--skip-inference` | 直接评测已有且兼容的场景。 |
| `--skip-eval` | 完成场景构建后停止，不计算指标。 |
| `--no-2d-iou` | 跳过耗时较长的轮廓 IoU。 |
| `--preflight-only` | 只检查环境配置，不执行推理。 |
| `--dry-run` | 只打印实际命令，不执行模型。 |

多卡模式按 case 分片，并不会用多张 GPU 加速单个 case。场景构建和指标
计算仍为串行。

## 评测指标

| 指标 | 趋势 | 含义 |
| --- | --- | --- |
| `object_cd` | 越低越好 | 每个物体归一化并 ICP 对齐后的双向平方 Chamfer Distance。 |
| `object_fscore` | 越高越好 | 距离阈值为 `0.1` 的点精确率/召回率 F-score。 |
| `object_emd` | 越低越好 | 每个物体归一化并 ICP 对齐后的匈牙利匹配 EMD。 |
| `iou_3d` | 越高越好 | 已放置物体对的轴对齐 3D 包围盒 IoU。 |
| `icp_rot_deg` | 越低越好 | 每个物体 ICP 得到的旋转误差，单位为度。 |
| `iou_2d` | 越高越好 | 通过 GT 相机渲染得到的物体轮廓 IoU。 |
| `add_s` | 越低越好 | 使用 GT 物体直径归一化的对称最近点距离。 |

前三项会独立归一化每个物体，主要衡量几何形状；其余指标使用已放置的
物体，同时衡量布局或姿态。预测节点 `geometry_N` 和 GT 节点
`object_NNN` 按数字编号配对，数量不一致的场景不会被截断后评测。
默认根据 `camera.json` 中的 `c2w` 在世界坐标系计算，最终 `AVERAGE`
对每个有效场景等权平均。

## 输出

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

第一个 CSV 包含逐场景结果和 `AVERAGE` 行，第二个 CSV 包含逐物体结果。
