import cv2
import numpy as np
import PIL.Image
import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BRIARMBG:
    def __init__(self, model_name_or_path="briaai/RMBG-2.0", device="cuda"):
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.birefnet.to(device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.device = device

    def __call__(self, image):
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image


class REMBG:
    def __init__(self):
        from rembg import new_session, remove

        self.session = new_session()

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output
