# Mira-Scene Training

Mira-Scene training is organized into two stages. The public configuration
files in this directory contain `/path/to/...` placeholders; replace them with
paths available in your environment before launching training.

## Install

From the repository root, install both packages in editable mode:

```bash
cd example_train
pip install -e ../Mira-CCM
pip install -e ../UniDataset
pip install pytorch-lightning==2.5.0
```

## Stage 1: pretraining

`configs/pretrain_l1.yaml` trains the CCM and voxel model on the Objaverse
training mixture. Set the dataset paths and the initial TRELLIS checkpoint in
the configuration, then run:

```bash
python -m miraccm.launch \
    --config configs/pretrain_l1.yaml \
    --train
```

The resulting Diffusers pipeline can be used as the initialization for stage 2.
A released stage-1 checkpoint is also available from Hugging Face, so stage 1
can be skipped when only finetuning is needed.

## Stage 2: finetuning

`configs/finetune.yaml` mixes 3D-FRONT, Infinigen, and Objaverse data for the
second training stage. Set its dataset paths and point
`system.params.pretrained_model_name_or_path` to either a local stage-1
pipeline or the downloaded `pretrain_pipeline` checkpoint, then run:

```bash
python -m miraccm.launch \
    --config configs/finetune.yaml \
    --train
```

To use the released stage-1 checkpoint, download it from the
[Hugging Face model repository](https://huggingface.co/Yang-Tian/Mira-Scene):

```bash
hf download Yang-Tian/Mira-Scene \
    --include "pretrain_pipeline/*" \
    --local-dir checkpoints/mira-scene
```

Then set the finetuning configuration to the downloaded directory, for example:

```yaml
system:
  params:
    pretrained_model_name_or_path: /absolute/path/to/checkpoints/mira-scene/pretrain_pipeline
```

## Multi-node training

For multi-node training, use the same configuration with `torchrun`. For
example, with four nodes and eight GPUs per node:

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<IP> --master_port=12357 \
    -m miraccm.launch \
    --config configs/finetune.yaml \
    --train trainer.num_nodes=4
```

## Training data

The Outpaint and 3D-FRONT training archives are being released in
[Yang-Tian/Mira-Scene-Dataset](https://huggingface.co/datasets/Yang-Tian/Mira-Scene-Dataset)
alongside the BlendSwap evaluation set. See the dataset card for availability.
After downloading the training archives, extract them in place (bash):

```bash
export DATA_ROOT="/absolute/path/to/mira-scene-data"
for f in "$DATA_ROOT"/{3dfront,objaverse_outpaint}/shards/*.tar.gz; do tar -xzf "$f" -C "$DATA_ROOT" || exit 1; done
```

The package includes Outpaint's ready-to-use `summary.json`. In
`configs/finetune.yaml`, update these anchors, replacing `DATA_ROOT` below
with the actual absolute path.

| Anchor | Parameter | Path |
| --- | --- | --- |
| `3DFrontPano2Pesp` | `renderings_root` | `DATA_ROOT/3dfront/renderings` |
| `3DFrontPano2Pesp` | `view_samples_dir` | `DATA_ROOT/3dfront/view_samples` |
| `3DFrontPano2Pesp` | `poses_dir` | `DATA_ROOT/3dfront/poses` |
| `3DFrontPano2Pesp` | `model_data_dir` | `DATA_ROOT/3dfront/models` |
| `3DFrontPano2Pesp` | `preprocess_json_path` | `DATA_ROOT/3dfront/preprocess_train.json` |
| `3DFrontPano2Pesp` | `voxel_cache_dir` | `DATA_ROOT/.runtime/3dfront/voxels` |
| `3DFrontPano2Pesp` | `error_log_path` | `DATA_ROOT/.runtime/3dfront/errors.log` |
| `ObjaverseParam65k` | `summary_json` | `DATA_ROOT/objaverse_outpaint/summary.json` |
| `ObjaverseParam65k` | `valid_scenes_dir` | `DATA_ROOT/objaverse_outpaint/valid_scenes` |
| `ObjaverseParam65k` | `voxel_cache_dir` (add) | `DATA_ROOT/.runtime/objaverse_outpaint/voxels` |

Create writable cache/log directories. To train on these two sources only,
keep `3DFrontPano2Pesp` and `objaverse65kOutpaint` under `data.params.dataset`,
in that order, and remove the other training entries. **FRONT must come first**
so its collator handles both sources. The release does not supply the other
FRONT, Infinigen or Orbit data referenced by the original configurations.
Set validation's `eval_dir` separately to your downloaded `blendswap_eval/`.

Outpaint's mesh paths are relative to the dataset root. After installing the
training packages, **start training from that directory** (no index conversion
is needed):

```bash
export CONFIG="/absolute/path/to/Mira-Scene/example_train/configs/finetune.yaml"
cd "$DATA_ROOT"
python -m miraccm.launch --config "$CONFIG" --train
```
