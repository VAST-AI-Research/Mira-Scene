# Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction

Mira-Scene reconstructs an editable 3D scene from a single image. The pipeline
combines interactive instance segmentation, depth estimation, canonical
coordinate and voxel prediction, object mesh generation, scene assembly, and
environment-map generation.

![Mira-Scene teaser](assets/teaser.jpg)

## Inference

The stage-wise pipeline supports resumable execution, multiple depth and mesh
backends, and case-level multi-GPU sharding. See the [inference
guide](infer_scripts/README.md) ([中文](infer_scripts/README_zh-CN.md)).

## Visualization

See the [visualization guide](visualization/README.md)
([中文](visualization/README_zh-CN.md)).

## Evaluation

See the [evaluation guide](eval_scripts/docs/eval.md)
([中文](eval_scripts/docs/eval_zh-CN.md)).

| Method | Data Availability | 3D Data Scale | Data Preparation | CD↓ | FS@0.1↑ | EMD↓ | 3D-IoU↑ | ICP-Rot↓ | 2D-IoU↑ | ADD-S↓ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM3D | Closed-source | Million+ | Extensive manual layout annotation | 0.027 | 0.817 | **0.163** | 0.520 | 7.566 | 0.672 | 0.078 |
| Ours | Open-source | 80k | Fully automatic pipeline | **0.021** | **0.843** | 0.169 | **0.727** | **5.616** | **0.783** | **0.031** |

Please refer to the [evaluation guide](eval_scripts/docs/eval.md)
for details.

## ToDo

- Release the training code & example training data.
