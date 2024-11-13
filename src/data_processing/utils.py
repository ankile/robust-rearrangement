import pickle
import tarfile
from typing import Dict, Union
from src.common.robot_state import ROBOT_STATES
import numpy as np
import torch
from torchvision.transforms import functional as F, InterpolationMode

from scipy.spatial.transform import Rotation as R

from ipdb import set_trace as bp


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
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        img = img.permute(0, 3, 1, 2)

    img = F.resize(
        img, (th, tw), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    if isinstance(img, torch.Tensor):
        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)

    if was_numpy:
        img = img.numpy()

    return img


def resize_crop(img: Union[np.ndarray, torch.Tensor]):
    """
    Resizes `img` and center crops into 320x240.

    Assumes that the channel is last.
    """
    # Must account for maybe having batch dimension
    th, tw = 240, 320
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
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

    if isinstance(img, torch.Tensor):
        img = img[..., crop_size : new_width - crop_size]

        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)

    if was_numpy:
        img = img.numpy()

    return img


def clip_quat_xyzw_magnitude(delta_quat_xyzw: np.ndarray, clip_mag=0.35) -> np.ndarray:
    """
    Clips the rotation magnitude
    """
    assert delta_quat_xyzw.shape[-1] == 4

    delta_rotvec = R.from_quat(delta_quat_xyzw).as_rotvec()

    magnitude = np.linalg.norm(delta_rotvec)
    if magnitude > clip_mag:
        scale_factor = clip_mag / magnitude
        delta_rotvec = scale_factor * delta_rotvec

    delta_quat_xyzw = R.from_rotvec(delta_rotvec).as_quat()

    return delta_quat_xyzw


def filter_and_concat_robot_state(robot_state: Dict[str, torch.Tensor]):
    current_robot_state = []
    for rs in ROBOT_STATES:
        if rs not in robot_state:
            continue

        # if rs == "gripper_width":
        #     robot_state[rs] = robot_state[rs].reshape(-1, 1)
        current_robot_state.append(robot_state[rs])
    return torch.cat(current_robot_state, dim=-1)
