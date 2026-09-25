---
pretty_name: Mira-Scene — Outpaint and 3D-FRONT Training Data
language:
- en
size_categories:
- 100K<n<1M
tags:
- 3d
- computer-vision
- scene-reconstruction
- depth
- point-cloud
- voxel
---

# Mira-Scene Training Data

Training data for [Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction](https://github.com/VAST-AI-Research/Mira-Scene), including RGB images, depth, masks, camera/scene metadata and mesh dependencies for the Outpaint and 3D-FRONT loaders.

## Dataset

| Subset | Views | Objects / scenes | Archives | Download size |
| --- | ---: | ---: | ---: | ---: |
| Outpaint | 109,944 | 42,972 objects | 41 | 129.4 GB |
| 3D-FRONT | 40,043 | 9,481 scenes | 75 | 268.0 GB |
| **Total** | **149,987** | — | **116** | **397.4 GB** |

Extracted data occupies approximately 605.9 GB. Allow about 1 TB to keep both archives and extracted files, plus space for runtime caches.

This release contains training data only, with no additional validation/test split. Orbit, Infinigen and TableVerse are not included.

```text
objaverse_outpaint/   # metadata.jsonl.gz and shards/*.tar.gz
3dfront/             # preprocess_train.json and shards/*.tar.gz
manifests/           # Archive/member checksums and code compatibility
```

## Download and use

Start with the first 8 views per source (about 6.7 GB to download). Replace `owner/dataset-name` with this repository's Hugging Face identifier.

```bash
python -m pip install -U huggingface_hub
export DATASET_REPO_ID="owner/dataset-name"

hf download "$DATASET_REPO_ID" --repo-type dataset \
  --local-dir ./mira-data-archives \
  --include "README.md" "manifests/*" \
    "objaverse_outpaint/metadata.jsonl.gz" "3dfront/preprocess_train.json" \
    "objaverse_outpaint/shards/part-00000.tar.gz" \
    "3dfront/shards/part-00000.tar.gz"

git clone https://github.com/VAST-AI-Research/Mira-Scene.git
python Mira-Scene/hf_release/data_tools/tools/extract_sample.py \
  ./mira-data-archives ./mira-data-smoke --views-per-subset 8
```

Extraction requires Python 3.10+ and an empty destination. The script verifies checksums and includes each selected view's geometry dependencies. To download the complete release, repeat `hf download` without `--include`.

See the [data tools guide](https://github.com/VAST-AI-Research/Mira-Scene/tree/main/hf_release/data_tools) for full extraction, sample checks and loading a batch, and the [training guide](https://github.com/VAST-AI-Research/Mira-Scene/blob/main/example_train/README.md) for finetuning. **Keep FRONT first when mixing FRONT and Outpaint.** These archives must be extracted before loading; they are not a `datasets.load_dataset()` or WebDataset dataset.

Release version: `mira-scene-outpaint-front-v1`. The matching code revision is recorded in `manifests/code_compatibility.json`.

## Sources and usage terms

The data derives from [Objaverse](https://objaverse.allenai.org/docs/intro/), [3D-FRONT](https://arxiv.org/abs/2011.09127) and [3D-FUTURE](https://arxiv.org/abs/2009.09633). Please acknowledge these datasets alongside Mira-Scene. Source assets retain their applicable licenses and usage terms; the code repository's license does not relicense the data.

## Citation

```bibtex
@article{sun2026mira,
  title={Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction},
  author={Sun Yang-Tian and Liu Tianjia and Huang Zehuan and Huang Yi-Hua and Lyu Xiaoyang and Yang Ziyi and Zou Zi-Xin and Guo Yuan-Chen and Cao Yan-Pei and Qi Xiaojuan},
  journal={arXiv preprint arXiv:2609.23796},
  year={2026}
}
```
