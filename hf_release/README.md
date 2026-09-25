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

This release provides the **Outpaint** and **3D-FRONT** training subsets for [Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction](https://github.com/VAST-AI-Research/Mira-Scene). It includes rendered observations, camera/geometry metadata and the mesh dependencies needed by the corresponding Mira-Scene data loaders.

The data supports object-conditioned canonical coordinate map (CCM) and voxel prediction from scene images. CCM and voxel training targets are prepared by the loaders from the supplied observations and geometry; this is not a collection of final reconstructed scenes or pretrained model weights.

- **Model and training code:** [VAST-AI-Research/Mira-Scene](https://github.com/VAST-AI-Research/Mira-Scene)
- **Training guide:** [example_train/README.md](https://github.com/VAST-AI-Research/Mira-Scene/blob/main/example_train/README.md)
- **Data tools in the code checkout:** `hf_release/data_tools/`
- **Dataset version:** `mira-scene-outpaint-front-v1`
- **Exact data/code compatibility:** [`manifests/code_compatibility.json`](manifests/code_compatibility.json)

## Contents and scale

| Subset | Views | Objects / scenes | Archives | Compressed archive size |
| --- | ---: | ---: | ---: | ---: |
| `objaverse_outpaint` | 109,944 | 42,972 objects | 41 | 129.4 GB |
| `3dfront` | 40,043 | 9,481 scenes | 75 | 268.0 GB |
| **Total** | **149,987** | — | **116** | **397.4 GB** |

Sizes use decimal GB. The archives contain **550,962 payload files**. The exact compressed size is **397,406,644,971 bytes**; the extracted payload occupies approximately **605.9 GB**, before filesystem overhead and runtime caches. Keeping both archives and extracted data requires about **1.0 TB**, plus space for caches and training outputs.

This version contains only the two subsets above. It does not include Objaverse Orbit, Infinigen or TableVerse. It supplies training data and **does not introduce a validation or test split**. The supplied FRONT training selection retains its source split policy; do not treat randomly selected training views as an independent evaluation set.

## Repository layout

```text
README.md
objaverse_outpaint/
  metadata.jsonl.gz
  shards/part-00000.tar.gz ... part-00040.tar.gz
3dfront/
  preprocess_train.json
  shards/part-00000.tar.gz ... part-00074.tar.gz
manifests/
  shards.json
  integrity_summary.json
  code_compatibility.json
  files/*.jsonl.gz
  shards/*.json
  ...
```

These are filesystem archives, **not WebDataset sample-format shards**. Extract them before constructing the Mira-Scene datasets. The repository is not a Parquet dataset intended to be loaded directly with `datasets.load_dataset()`.

The Outpaint index contains object/view identifiers, camera parameters, frame transforms and portable mesh paths. The FRONT index contains scene/object metadata, view references and the eligible object indices for each training view. Preserve these indices and their referenced files together; one view can depend on geometry stored separately from its RGB/depth observations.

After extraction, the main directories are:

```text
objaverse_outpaint/
  metadata.jsonl.gz
  valid_scenes/<object-id>_<view>/
    scene.png
    mask.png
    depth.exr or depth.npy
  meshes/<object-id>/mesh_normalized.ply
3dfront/
  preprocess_train.json
  renderings/<scene-id>/.../*.hdf5
  view_samples/<view-id>/depth.npy
  poses/<scene-id>_scene_state.json
  models/<model-id>/raw_model.obj
```

Use the supplied loader's camera, scale and canonicalization conventions. RGB/depth native resolutions and source geometry conventions should not be inferred from the 518×518 training input resolution.

## Quick start: download a small training sample

The examples target Linux with Python 3.10 or newer. The companion extraction utilities use the Python standard library. Dataset loading additionally requires the packages listed in `hf_release/data_tools/requirements.txt`; model training has further dependencies documented in the training guide.

Set `DATASET_REPO_ID` to this dataset's Hugging Face identifier, as shown in its URL (`owner/dataset-name`).

```bash
python -m pip install -U huggingface_hub
export DATASET_REPO_ID="owner/dataset-name"

# Obtain the companion code. For reproducibility, use the published commit
# recorded in manifests/code_compatibility.json.
git clone https://github.com/VAST-AI-Research/Mira-Scene.git
export DATA_TOOLS="$PWD/Mira-Scene/hf_release/data_tools"

# Download the indices/manifests and the two archives needed for the first
# eight views per subset in this release (about 6.7 GB of archives).
hf download "$DATASET_REPO_ID" --repo-type dataset \
  --local-dir ./mira-data-archives \
  --include "README.md" "manifests/*" \
    "objaverse_outpaint/metadata.jsonl.gz" "3dfront/preprocess_train.json" \
    "objaverse_outpaint/shards/part-00000.tar.gz" \
    "3dfront/shards/part-00000.tar.gz"

python "$DATA_TOOLS/tools/extract_sample.py" \
  ./mira-data-archives ./mira-data-smoke --views-per-subset 8
```

The sample destination must be empty. The extractor verifies the selected archive hashes, extracts the selected views and their dependencies, verifies member hashes, and writes reduced indices plus `SAMPLE_MANIFEST.json`. Its selection is the **first N index entries**, not a random or stratified sample. Increasing N may require additional archives; use the complete download below in that case.

### Check the extracted samples

Run dependency installation in a suitable dedicated environment. Outpaint EXR depth decoding requires an OpenCV build with OpenEXR support.

```bash
python -m pip install -r "$DATA_TOOLS/requirements.txt"
export OPENCV_IO_ENABLE_OPENEXR=1

python "$DATA_TOOLS/tools/check_samples.py" ./mira-data-smoke \
  --subset 3dfront --count 8
python "$DATA_TOOLS/tools/check_samples.py" ./mira-data-smoke \
  --subset objaverse_outpaint --count 8
```

These checks read the requested samples directly, check finite tensors, and fail rather than silently substituting another sample.

## Load a mixed PyTorch batch

The following example uses the companion `load_dataset` adapter, which resolves portable paths and builds loader caches under the extracted directory's `.runtime/<subset>/` by default. The extracted directory must therefore be writable; alternatively pass `cache_dir=` to `load_dataset`.

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["DATA_TOOLS"]) / "tools"))
from load_dataset import load_dataset
from torch.utils.data import ConcatDataset, DataLoader

front = load_dataset("./mira-data-smoke", "3dfront")
outpaint = load_dataset("./mira-data-smoke", "objaverse_outpaint")

# Example repetition policy: FRONT 5, Outpaint 2.
# Repetition changes sampling frequency; it does not duplicate payload files.
mixed = ConcatDataset([front] * 5 + [outpaint] * 2)
loader = DataLoader(
    mixed,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=front.collate,
)
batch = next(iter(loader))
```

**Keep FRONT first and use its collator when mixing these two sources.** In the training repository's `UniDataModule`, the first configured dataset supplies the collator. The FRONT collator preserves per-sample metadata such as `azimuth_rotation`, whose representation differs between the two sources.

For sampling by concatenated dataset length, the source probabilities are proportional to `repeat × dataset length`. Repeat factors 5 and 2 do not, in general, imply a 5:2 probability ratio.

The example above checks dataset integration; it is not a complete model-training script. For finetuning, use the model, optimizer and training loop from the linked training guide. Bind the two dataset entries to the extracted roots, and place FRONT before Outpaint. The training repository's original multi-source configuration may reference datasets outside this release; those entries are not provided by this download.

| Loader input | Path relative to the extracted root |
| --- | --- |
| FRONT `renderings_root` | `3dfront/renderings/` |
| FRONT `view_samples_dir` | `3dfront/view_samples/` |
| FRONT `poses_dir` | `3dfront/poses/` |
| FRONT `model_data_dir` | `3dfront/models/` |
| FRONT `preprocess_json_path` | `3dfront/preprocess_train.json` |
| Outpaint portable index | `objaverse_outpaint/metadata.jsonl.gz` |
| Outpaint `valid_scenes_dir` | `objaverse_outpaint/valid_scenes/` |

The adapter resolves the Outpaint portable index into the loader's `summary_json`, including absolute mesh paths under the chosen extracted root. Do not pass the portable gzip index to a loader revision that expects an already-resolved JSON summary.

## Download and extract the complete release

```bash
hf download "$DATASET_REPO_ID" --repo-type dataset \
  --local-dir ./mira-data-archives

# Use a different, initially empty directory from the sample extraction.
python "$DATA_TOOLS/tools/extract.py" \
  ./mira-data-archives ./mira-data-full
```

To extract only one complete subset, add `--subset 3dfront` or `--subset objaverse_outpaint`. The full extractor can resume its own destination: it checks the manifest identity and existing file hashes, and rejects conflicting data. Keep the indices, manifests and selected archives from the same release revision.

## Integrity and training compatibility

During packaging, all 116 archives were read back and their member hashes checked against the frozen worklists. Archive hashes and sizes are recorded in `manifests/shards.json`; per-file hashes are recorded in `manifests/files/`. HDF5 packaging removes descriptive source-machine metadata while preserving numeric arrays. See the included integrity and retry audit summaries for details and the limits of historical source-byte comparisons.

A subsequent public-code training smoke test used **32 freshly extracted views per subset**, covering **8 FRONT scenes and 14 Outpaint objects**:

- All 64 requested sample reads, multi-worker loaders and an explicit mixed batch passed.
- FRONT-first finetuning completed **8 optimizer steps at batch size 4**, with finite losses and gradients and parameter changes in both shape and layout groups.
- Six training batches contained both sources.
- Training visualization remained enabled. The exported 10,560-point voxel preview reproduced the input coordinates exactly when read back.

That smoke test used public code commit `66d4c965cf2e68bf8d35fcf258aaaca6600a2e47` plus the [PLY visualization fix in PR #2](https://github.com/VAST-AI-Research/Mira-Scene/pull/2). It did not require the extra Outpaint-first collator patch. Use a training revision containing the PLY fix when enabling that visualization path.

These are **sampled compatibility checks**, not a full-corpus semantic quality audit, model-quality benchmark or held-out evaluation. The sampled views came from the first archive of each subset.

## Source data and attribution

Outpaint observations derive from Objaverse objects. Indoor observations derive from 3D-FRONT scenes and 3D-FUTURE geometry. Please acknowledge the source datasets as well as Mira-Scene:

- [Objaverse](https://objaverse.allenai.org/docs/intro/) and its [paper](https://arxiv.org/abs/2212.08051).
- [3D-FRONT](https://arxiv.org/abs/2011.09127).
- [3D-FUTURE](https://arxiv.org/abs/2009.09633).

The underlying assets retain their original rights and usage terms; this packaging does not assign a new blanket license to them. Consult the source dataset providers and applicable asset terms for permitted use and redistribution. A code repository's license must not be interpreted as automatically relicensing its referenced data assets.

## Citation

```bibtex
@article{sun2026mira,
  title={Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction},
  author={Sun Yang-Tian and Liu Tianjia and Huang Zehuan and Huang Yi-Hua and Lyu Xiaoyang and Yang Ziyi and Zou Zi-Xin and Guo Yuan-Chen and Cao Yan-Pei and Qi Xiaojuan},
  journal={arXiv preprint arXiv:2609.23796},
  year={2026}
}
```

For dataset or loader issues, open an issue in the [Mira-Scene code repository](https://github.com/VAST-AI-Research/Mira-Scene/issues) and include the dataset revision, subset, sample identifier and a minimal error trace.
