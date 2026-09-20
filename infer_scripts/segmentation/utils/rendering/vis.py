import torch
import cv2
import numpy as np

def visualize_masks_on_image_cv2(img, mask_list, out_path="sofa.png"):
    # --- PIL -> numpy ---
    img = np.array(img)  # RGB, uint8

    # --- RGB -> BGR ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    overlay = img.copy()
    alpha = 0.6

    for idx, mask in enumerate(mask_list):
        # torch.Tensor -> numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # ensure HxW
        mask = mask.reshape(h, w).astype(bool)

        color = np.random.randint(0, 256, size=3, dtype=np.uint8)

        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * color
        ).astype(np.uint8)

    cv2.imwrite(out_path, overlay)



def save_seg_obj(img, mask, out_path="sofa.png"):
    # Save RGBA image with only the masked objects shown
    # --- PIL -> numpy ---
    img = np.array(img)  # RGB, uint8

    # --- RGB -> BGR ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # RGBA output for vis_obj (with transparency)
    vis_obj = np.zeros((h, w, 4), dtype=np.uint8)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # ensure HxW
    mask = mask.reshape(h, w).astype(bool)

    vis_obj[..., :3][mask] = img[mask]
    vis_obj[..., 3][mask] = 255

    cv2.imwrite(out_path, vis_obj)
