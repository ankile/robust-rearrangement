import pickle
import tarfile
from typing import Union
import numpy as np
import torch
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image


def zipped_img_generator(filename, max_samples=1000):
    n_samples = 0
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar:
            if (
                member.isfile() and ".pkl" in member.name
            ):  # Replace 'your_condition' with actual condition
                with tar.extractfile(member) as f:
                    if f is not None:
                        content = f.read()
                        data = pickle.loads(content)
                        n_samples += 1

                        yield data

                        if n_samples >= max_samples:
                            break


def resize(img: Union[np.ndarray, torch.Tensor]):
    """Resizes `img` into 320x240."""
    th, tw = 240, 320

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        img = img.permute(0, 3, 1, 2)

    img = F.resize(
        img, (th, tw), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)
    return img


def resize_crop(img: Union[np.ndarray, torch.Tensor]):
    """
    Resizes `img` and center crops into 320x240.

    Assumes that the channel is last.
    """
    # Must account for maybe having batch dimension
    th, tw = 240, 320

    if isinstance(img, np.ndarray):
        ch, cw = img.shape[:2]
        img = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        img = img.permute(0, 3, 1, 2)
        ch, cw = img.shape[-2:]

    # Calculate the aspect ratio of the original image.
    aspect_ratio = cw / ch

    # Resize based on the width, keeping the aspect ratio constant.
    new_width = int(th * aspect_ratio)
    img = F.resize(
        img, (th, new_width), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    # Calculate the crop size.
    crop_size = (new_width - tw) // 2

    # Crop the resized image.
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = img[:, crop_size : new_width - crop_size]
    elif isinstance(img, torch.Tensor):
        img = img[:, :, :, crop_size : new_width - crop_size]

        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)

    return img
