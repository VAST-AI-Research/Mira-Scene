# Mira-CCM

Diffusion model library for 3D shape synthesis, built on diffusers and PyTorch Lightning.

## Structure

```
src/miraccm/
├── models/
│   ├── transformers/
│   │   ├── dit_ccm_voxel.py              # Dual-stream MMDiT (shape + layout)
│   │   ├── dit_block.py                  # DiT block (self-attn + cross-attn + MLP + adaLN)
│   │   └── trellis_sparse_structure_transformer.py  # Base TRELLIS transformer
│   └── autoencoders/
│       └── autoencoder_kl_sparse_structure.py  # 3D voxel VAE
│
├── pipelines/
│   └── shape_synthesis/
│       └── pipeline_ccm_voxel.py              # Full dual-stream inference pipeline
│
├── systems/
│   └── shape_synthesis/
│       ├── system_ccm_voxel.py            # Training system (velocity, MSE/L1, masked loss)
│       └── data_processor/
│           └── ccm_voxel.py               # Data preprocessing (crop, mask, transform solving)
│
├── schedulers/
│   └── scheduling_flow_match_euler_discrete.py  # Flow-matching Euler scheduler
│
└── utils/
    ├── system_utils/    # Optimizer, scheduler parsing, logging
    ├── image_utils/     # Segmentation masks, id-map palette
    └── torch_utils/     # SparseTensor, misc
```

## Model Architecture

### CCMVoxelDiTModel (Dual-Stream MMDiT)

Two parallel token streams processed by shared DiT blocks with bidirectional self-attention:

- **Shape stream**: 3D voxel latent `[B, 8, 16, 16, 16]` (from VAE encoder), patch_size=1 → 4096 tokens
- **Layout stream**: 2D CCM `[B, 3, 296, 296]`, patch_size=8 → 37×37 coarse tokens

Cascaded layout architecture:
1. First half blocks: coarse 37×37 tokens
2. Midpoint: pixel-shuffle 2× upsample → 74×74 tokens
3. Second half blocks: fine resolution
4. FinalLayerLayout: unpatchify with patch_size/2=4 → 296×296 output

## Pipeline Inference

```python
from miraccm.pipelines.shape_synthesis.pipeline_ccm_voxel import CCMVoxelPipeline

pipe = CCMVoxelPipeline.from_pretrained("/path/to/pipeline")
output = pipe(
    image=image_tensor,
    mask=mask_tensor,
    image_cropped=image_cropped,
    mask_cropped=mask_cropped,
    num_inference_steps=30,
    guidance_scale=3.0,
    layout_pred_mode="velocity",
)

ccm_pred = output.latent_voxel_cam_pts   # [B, 3, H', W']
voxel_pred = output.samples              # [B, 1, 64, 64, 64]
```

## Training

Training is organized into two stages: pretraining and finetuning. The project
also provides a released pretrain checkpoint that can be used directly for the
second-stage finetuning. See the [training guide](../example_train/README.md)
for installation, configuration, checkpoint, and launch instructions.
