# Mira-Scene Data Tools

Extract and load the Outpaint / 3D-FRONT archives from the [dataset guide](../README.md). Run the commands below from the Mira-Scene repository root. Keep this directory in the checkout: the loader adapter uses its `UniDataset/src` package.

## Extract

Requires Python 3.10+; no third-party packages are needed for extraction.

```bash
export DATA_TOOLS="$PWD/hf_release/data_tools"
export ARCHIVES="/path/to/downloaded-dataset"

# First 8 views per source, including geometry dependencies.
python "$DATA_TOOLS/tools/extract_sample.py" "$ARCHIVES" ./mira-data-smoke --views-per-subset 8

# Complete release; download all archives before running this command.
python "$DATA_TOOLS/tools/extract.py" "$ARCHIVES" ./mira-data-full
```

Use separate, initially empty destinations. Both scripts verify checksums. Full extraction can resume into its own destination and accepts `--subset 3dfront` or `--subset objaverse_outpaint`. Sample extraction selects the first N index entries; larger samples may require additional archives.

## Check and load

Use a Linux environment with OpenEXR-capable OpenCV for Outpaint depth:

```bash
python -m pip install -r "$DATA_TOOLS/requirements.txt"
export OPENCV_IO_ENABLE_OPENEXR=1
python "$DATA_TOOLS/tools/check_samples.py" ./mira-data-smoke --subset 3dfront --count 8
python "$DATA_TOOLS/tools/check_samples.py" ./mira-data-smoke --subset objaverse_outpaint --count 8
```

```python
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["DATA_TOOLS"]) / "tools"))
from load_dataset import load_dataset
from torch.utils.data import ConcatDataset, DataLoader

front = load_dataset("./mira-data-smoke", "3dfront")
outpaint = load_dataset("./mira-data-smoke", "objaverse_outpaint")
loader = DataLoader(ConcatDataset([front, outpaint]), batch_size=4,
                    shuffle=True, num_workers=2, collate_fn=front.collate)
batch = next(iter(loader))
```

**Use FRONT's collator for mixed batches.** The adapter resolves portable paths and writes caches to `<data-root>/.runtime/`; pass `cache_dir=` to change this location. For model training, follow the [training guide](../../example_train/README.md).

Extraction regression tests: `python -m unittest discover -s hf_release/data_tools/tests -v`.
