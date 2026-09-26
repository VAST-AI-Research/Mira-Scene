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

The [HF dataset](https://huggingface.co/datasets/Yang-Tian/Mira-Scene-Dataset)
provides Outpaint / 3D-FRONT training archives and BlendSwap evaluation data.
Use [`configs/finetune_hf.yaml`](configs/finetune_hf.yaml) for this two-source
training mixture. It keeps FRONT first and uses BlendSwap for validation.
See the dataset card for download availability. Extract the training archives
in place (bash):

```bash
export DATA_ROOT="/absolute/path/to/mira-scene-data"
for f in "$DATA_ROOT"/{3dfront,objaverse_outpaint}/shards/*.tar.gz; do tar -xzf "$f" -C "$DATA_ROOT" || exit 1; done
```

The package includes the ready-to-use Outpaint `summary.json`. In the HF YAML,
set `anchors.data_root` to **`.` for relative paths** or **your absolute dataset
path**; all data/cache paths follow this setting. Set
`system.params.pretrained_model_name_or_path` to your stage-1 pipeline.

**For either YAML path style, launch from the dataset root:** the mesh paths
inside the supplied Outpaint index are relative to that directory.

```bash
cd "$DATA_ROOT"
python -m miraccm.launch \
  --config /absolute/path/to/Mira-Scene/example_train/configs/finetune_hf.yaml \
  --train
```
