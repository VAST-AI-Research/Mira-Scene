#!/usr/bin/env bash
unset http_proxy https_proxy
set -euo pipefail

# Run this script from example_train/.
# Both packages are installed editable so training imports the shared root
# UniDataset package rather than a copy inside Mira-CCM.
pip install -e ../Mira-CCM
pip install -e ../UniDataset
pip install pytorch-lightning==2.5.0

mkdir -p logs
python -m miraccm.launch \
  --config configs/finetune_local.yaml \
  --train tag=debug_finetune 2>&1 | tee logs/train.log
