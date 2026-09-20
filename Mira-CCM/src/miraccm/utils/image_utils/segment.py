import torch
from PIL import Image
import numpy as np

def create_palette():
    # Define a palette with 24 colors for labels 0-23 (example colors)
    palette = [
        0,
        0,
        0,  # Label 0 (black)
        255,
        0,
        0,  # Label 1 (red)
        0,
        255,
        0,  # Label 2 (green)
        0,
        0,
        255,  # Label 3 (blue)
        255,
        255,
        0,  # Label 4 (yellow)
        255,
        0,
        255,  # Label 5 (magenta)
        0,
        255,
        255,  # Label 6 (cyan)
        128,
        0,
        0,  # Label 7 (dark red)
        0,
        128,
        0,  # Label 8 (dark green)
        0,
        0,
        128,  # Label 9 (dark blue)
        128,
        128,
        0,  # Label 10
        128,
        0,
        128,  # Label 11
        0,
        128,
        128,  # Label 12
        64,
        0,
        0,  # Label 13
        0,
        64,
        0,  # Label 14
        0,
        0,
        64,  # Label 15
        64,
        64,
        0,  # Label 16
        64,
        0,
        64,  # Label 17
        0,
        64,
        64,  # Label 18
        192,
        192,
        192,  # Label 19 (light gray)
        128,
        128,
        128,  # Label 20 (gray)
        255,
        165,
        0,  # Label 21 (orange)
        75,
        0,
        130,  # Label 22 (indigo)
        238,
        130,
        238,  # Label 23 (violet)
    ]
    # Extend the palette to have 768 values (256 * 3)
    palette.extend([0] * (768 - len(palette)))
    return palette


PALETTE = create_palette()

def masks2idmap(masks: torch.Tensor) -> Image.Image:
    """
    Convert a tensor of shape (N, H, W) containing binary masks to a single
    tensor of shape (H, W) where each pixel's value corresponds to the index
    of the mask it belongs to. Background pixels are assigned a value of 0.

    Args:
        masks (torch.Tensor): A tensor of shape (N, H, W) with binary masks.

    Returns:
        Image.Image: A PIL Image of shape (H, W) with palette applied.
    """
    N, H, W = masks.shape
    id_map = torch.zeros((H, W), dtype=torch.long, device=masks.device)

    for i in range(N):
        id_map[masks[i] > 0.5] = i + 1  # Assign index i+1 to mask pixels
    id_map_np = id_map.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(id_map_np, mode='P')
    img.putpalette(PALETTE)
    return img