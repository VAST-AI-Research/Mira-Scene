# Mira Scene data tools

These companion tools extract and load the Outpaint and 3D-FRONT release described in [the dataset guide](../README.md). They include extraction scripts, dataset component configurations and loader dependencies. They do not run model training.

The Hugging Face dataset repository contains only the data shards, indices, integrity manifests and dataset card. Set `DATASET_REPO_ID` to the published dataset identifier. `CODE_VERSION.json` records this companion bundle's file hashes and dataset compatibility; the dataset's `manifests/code_compatibility.json` identifies the published code commit.

## Extract a training smoke set

From the Mira-Scene repository root, set the tool directory below. The scripts resolve their companion imports/configuration relative to their own location, so they can also be invoked by absolute path from another working directory. Source and destination arguments are resolved relative to your current directory. Extraction uses only Python's standard library (Python 3.10+):

```bash
export DATA_TOOLS="$PWD/hf_release/data_tools"
python3 "$DATA_TOOLS/tools/extract_sample.py" /path/to/downloaded-dataset /path/to/test-data --views-per-subset 8
```

Choose a new empty destination. It extracts only selected views and all their dependencies and writes reduced indices. `--views-per-subset 64` creates a larger test set. Extraction writes data only; it does not copy code into the data directory.

## Read training samples

Install `"$DATA_TOOLS/requirements.txt"` in a suitable Python environment. OpenCV must support OpenEXR for Outpaint depth decoding.

```python
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['DATA_TOOLS']) / 'tools'))
from load_dataset import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset('/path/to/test-data', 'objaverse_outpaint')
loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=dataset.collate)
batch = next(iter(loader))
```

Use `3dfront` for the other subset. Keep model, optimizer and training-loop code in the training repository. These tools only provide dataset inputs and collation; they do not run model forward/backward passes.

Dataset defaults come from this checkout's `configs/datasets.json`. An extracted directory's own `configs/datasets.json`, if present, overrides these defaults for compatibility with earlier prepared test sets. `load_dataset` resolves portable paths and places voxel caches, resolved summaries and error logs under the extracted data root's `.runtime/<subset>/` by default. An explicit `cache_dir` may be supplied. Sampling repetition defaults to one; configure training mixtures in the consuming training pipeline.

Strict sample checks bypass retry/fallback behavior:

```bash
python3 "$DATA_TOOLS/tools/check_samples.py" /path/to/test-data --subset objaverse_outpaint --count 8
python3 "$DATA_TOOLS/tools/check_samples.py" /path/to/test-data --subset 3dfront --count 8
```

## Full extraction

```bash
python3 "$DATA_TOOLS/tools/extract.py" /path/to/downloaded-dataset /path/to/extracted-data
```

An interrupted extraction may resume into its own destination after verifying existing content. Use `--subset objaverse_outpaint` or `--subset 3dfront` for a complete individual subset.

The adapter imports the loaders from this checkout's `UniDataset/src`, using a path resolved from its own location. Keep this directory inside the Mira-Scene checkout. It changes path resolution and runtime cache placement, not the loader's image, geometry or target computations. No new blanket license is assigned to the source data. See the dataset guide for source attribution and usage terms.

## Extraction regression tests

From the Mira-Scene repository root, run:

```bash
python3 -m unittest discover -s hf_release/data_tools/tests -v
```

These standard-library tests use small synthetic archives to check relocation, both extraction entry points, resume validation, corruption detection, and rejection of escaping paths. They complement real-data smoke checks; they do not test model quality.
