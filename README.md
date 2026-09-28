# Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction

📄 [Paper](https://arxiv.org/abs/2609.23796) | 🌐 [Project Page](https://sunyangtian.github.io/Mira-Scene-web/)

Mira-Scene reconstructs an editable 3D scene from a single image. The pipeline
combines interactive instance segmentation, depth estimation, canonical
coordinate and voxel prediction, object mesh generation, scene assembly, and
environment-map generation.

![Mira-Scene teaser](assets/teaser.jpg)

## 📰 News

> **🔥 Mira-Scene is fully open-source**
>
> The Mira-Scene inference pipeline, Mira-CCM, UniDataset, evaluation and
> visualization tools, example cases, and both the **Stage 1 pretrained
> checkpoint** and **Stage 2 final checkpoint** are publicly available in this
> repository and through [Hugging Face](https://huggingface.co/Yang-Tian/Mira-Scene).
>
> **✅ Finetuning data is now available**
>
> The released Scene data and outpainted Objaverse data are available from the
> [Mira-Scene Dataset repository](https://huggingface.co/datasets/Yang-Tian/Mira-Scene-Dataset).
> The large Objaverse rendering data used for pretraining is not included;
> it can be generated conveniently from the source assets.

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

## Training

Training is organized into two stages:

1. Pretraining with [`example_train/configs/pretrain_l1.yaml`](example_train/configs/pretrain_l1.yaml);
2. Finetuning with [`example_train/configs/finetune.yaml`](example_train/configs/finetune.yaml).

A pretrained Mira-CCM checkpoint is available from
[Hugging Face](https://huggingface.co/Yang-Tian/Mira-Scene) and can be used
directly for the second-stage finetuning.

See the [training guide](example_train/README.md) for installation,
configuration, checkpoint download, and launch instructions.


## Citation

```bibtex
@article{sun2026mira,
  title={Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction},
  author={Sun Yang-Tian and Liu Tianjia and Huang Zehuan and Huang Yi-Hua and Lyu Xiaoyang and Yang Ziyi and Zou Zi-Xin and Guo Yuan-Chen and Cao Yan-Pei and Qi Xiaojuan},
  journal={arXiv preprint arXiv:2609.23796},
  year={2026}
}
```

## License

The Mira-Scene code in this repository is released under the [MIT License](LICENSE).
Third-party components, pretrained models, and datasets retain their respective
licenses and terms of use.
