import torch
import random
import torchvision.transforms.functional as F
from ipdb import set_trace as bp


class ImageAugmentation:
    def random_crop(self, img, size=224):
        h, w = img.shape[-2:]

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        # Let all leading dimensions remain
        img_cropped = img[..., top : top + size, left : left + size]
        return img_cropped

    def color_jitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        img = F.adjust_brightness(img, random.uniform(1 - brightness, 1 + brightness))
        img = F.adjust_contrast(img, random.uniform(1 - contrast, 1 + contrast))
        img = F.adjust_saturation(img, random.uniform(1 - saturation, 1 + saturation))
        img = F.adjust_hue(img, random.uniform(-hue, hue))
        return img

    def random_rotation(self, img, degrees=10):
        angle = random.uniform(-degrees, degrees)
        return F.rotate(img, angle)

    def random_grayscale(self, img, p=0.1):
        if random.random() < p:
            return F.rgb_to_grayscale(img)
        return img

    def random_flip(self, img):
        if random.random() > 0.5:
            return torch.flip(img, [2])  # Horizontal flip
        return img

    def color_cutout(self, img, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
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
        img = self.random_crop(img)
        # img = self.color_jitter(img)
        # img = self.random_rotation(img)
        # img = self.random_grayscale(img)
        # img = self.random_flip(img)
        # img = self.color_cutout(img)
        return img
