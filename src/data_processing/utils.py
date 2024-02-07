import pickle
import tarfile
from typing import Union
import numpy as np
import torch
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image

from src.common.geometry import np_rot_6d_to_rotvec, np_rotvec_to_rot_6d


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


def clip_axis_rotation(delta_action: np.ndarray, clip_mag=0.35, axis="z") -> np.ndarray:
    """
    Clips the rotation magnitude of the given axis.

    Args:
        delta_action: The action to clip.
        clip_mag: The magnitude to clip the rotation to.
        axis: The axis to clip the rotation magnitude of.

    Returns:
        The clipped action.
    """
    assert axis in "xyz", "Axis must be one of 'x', 'y', or 'z'."
    # Make a copy of the action
    delta_action = np.copy(delta_action)

    # Convert to rotation vectors
    rot_vec = np_rot_6d_to_rotvec(delta_action[:, 3:9])

    # Clip the axis specified of the magnitude of the rotation vectors
    rot_vec[:, "xyz".index(axis)] = np.clip(
        rot_vec[:, "xyz".index(axis)], -clip_mag, clip_mag
    )

    # Convert back to 6D
    delta_action[:, 3:9] = np_rotvec_to_rot_6d(rot_vec)

    return delta_action
