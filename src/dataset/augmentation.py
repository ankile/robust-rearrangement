import torch
import random
import torchvision.transforms.functional as F
from ipdb import set_trace as bp
import numpy as np
from PIL import Image


class ImageAugmentation:
    # Defaults
    max_translate = 10

    brightness = 0.1
    contrast = 0.1
    saturation = 0.1
    hue = 0.1

    degrees = 5

    def __init__(
        self,
        random_translate=False,
        color_jitter=False,
        random_rotate=False,
    ):
        self.augmentations = []

        if random_translate:
            self.augmentations.append(self.random_translate)

        if color_jitter:
            self.augmentations.append(self.random_color_jitter)

        if random_rotate:
            self.augmentations.append(self.random_rotation)

    def random_translate(self, img):
        H, W, _ = img.shape
        translated_images = np.zeros_like(img)

        translation_height = np.random.randint(-self.max_translate, self.max_translate)
        translation_width = np.random.randint(-self.max_translate, self.max_translate)

        # Calculate the indices for zero-padded array
        start_height = max(translation_height, 0)
        end_height = H + min(translation_height, 0)
        start_width = max(translation_width, 0)
        end_width = W + min(translation_width, 0)

        # Calculate the indices for the original image
        start_height_orig = -min(translation_height, 0)
        end_height_orig = H - max(translation_height, 0)
        start_width_orig = -min(translation_width, 0)
        end_width_orig = W - max(translation_width, 0)

        # Index into the zero-padded array and place the original image
        translated_images[start_height:end_height, start_width:end_width, :] = img[
            start_height_orig:end_height_orig, start_width_orig:end_width_orig, :
        ]

        return translated_images

    def random_crop(self, img, size=224):
        h, w = img.shape[-2:]

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        # Let all leading dimensions remain
        img_cropped = img[..., top : top + size, left : left + size]
        return img_cropped

    def random_color_jitter(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = F.adjust_brightness(
            img, random.uniform(1 - self.brightness, 1 + self.brightness)
        )
        img = F.adjust_contrast(
            img, random.uniform(1 - self.contrast, 1 + self.contrast)
        )
        img = F.adjust_saturation(
            img, random.uniform(1 - self.saturation, 1 + self.saturation)
        )
        img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        return img

    def random_rotation(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle)

    def random_grayscale(self, img, p=0.1):
        if random.random() < p:
            return F.rgb_to_grayscale(img)
        return img

    def random_flip(self, img):
        if random.random() > 0.5:
            return torch.flip(img, [2])  # Horizontal flip
        return img

    def random_color_cutout(self, img, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        if random.random() < p:
            h, w = img.shape[1], img.shape[2]
            area = h * w
            target_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])
            cut_w = int((target_area * aspect_ratio) ** 0.5)
            cut_h = int((target_area / aspect_ratio) ** 0.5)
            if cut_h <= h and cut_w <= w:
                top = random.randint(0, h - cut_h)
                left = random.randint(0, w - cut_w)
                img[:, top : top + cut_h, left : left + cut_w] = 0
        return img

    def __call__(self, img):
        for aug in self.augmentations:
            img = aug(img)
        return img
