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

The configuration files refer to the prepared training datasets used by the
project. The data preparation and release instructions will be added later.

**TODO**

- [ ] Publish the concrete training data and corresponding download and
      preparation instructions.
