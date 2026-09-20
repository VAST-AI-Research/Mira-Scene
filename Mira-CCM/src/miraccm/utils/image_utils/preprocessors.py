import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from ..typing import *


def array_to_tensor(np_array, normalize=True):
    image_pt = torch.tensor(np_array).float()
    if normalize:
        image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


def list_to_pils(images: IMAGE_TYPE) -> List[Image.Image]:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    if isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            images = images[None]
        if images.shape[1] in [1, 3, 4]:
            images = images.transpose(0, 2, 3, 1)
        if images.min() >= 0 and images.max() <= 1:
            images = (images * 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
    elif isinstance(images, list):
        images = [Image.fromarray(image) for image in images]
    elif isinstance(images, Image.Image):
        images = [images]

    return images


class ImageProcessor:
    def __init__(self, size=512, border_ratio=None, bg_color=[255, 255, 255]):
        self.size = size
        self.border_ratio = border_ratio
        self.bg_color = bg_color

    @staticmethod
    def recenter(image, border_ratio: float = 0.2, bg_color=[255, 255, 255]):
        """Recenter an image to leave some empty space at the image border."""
        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]
        H, W, C = image.shape
        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError("input image is empty")
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2
        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_CUBIC
        )
        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * bg_color
        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 1)
        return result, mask

    def __call__(
        self,
        image_path,
        border_ratio=0.15,
        return_mask=False,
        **kwargs,
    ):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
            print(f"Using border_ratio from init: {border_ratio}")
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGBA")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGBA")
        else:
            raise ValueError("Unsupported image input type")

        image = np.asarray(image)
        image, mask = self.recenter(
            image, border_ratio=border_ratio, bg_color=self.bg_color
        )
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.size, self.size), Image.BICUBIC)
        pil_mask = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((self.size, self.size), Image.NEAREST)

        if return_mask:
            return pil_image, pil_mask
        return pil_image


class ImageEncoderPreprocessor:
    def __init__(
        self,
        size: int = 224,
        mean: List[float] = [0, 0, 0],
        std: List[float] = [1, 1, 1],
        do_resize: bool = True,
    ):
        self.size = size
        self.mean = torch.as_tensor(mean, dtype=torch.float32)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=torch.float32)[None, :, None, None]
        self.do_resize = do_resize

    def __call__(self, image: Float[Tensor, "B C H W"]):
        if self.do_resize:
            image = F.interpolate(
                image, size=self.size, mode="bilinear", antialias=True
            )
        image = (image - self.mean.to(image)) / self.std.to(image)
        return image


class DINOv2Preprocessor(ImageEncoderPreprocessor):
    def __init__(self, size: int = 224, do_resize: bool = True):
        super().__init__(
            size=size,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            do_resize=do_resize,
        )


class CLIPPreprocessor(ImageEncoderPreprocessor):
    def __init__(self, size: int = 224, do_resize: bool = True):
        super().__init__(
            size=size,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
            do_resize=do_resize,
        )
