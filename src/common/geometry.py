import numpy as np
import torch
import pytorch3d.transforms as pt


def isaac_quat_to_pytorch3d_quat(quat):
    """Converts IsaacGym quaternion to PyTorch3D quaternion.

    IsaacGym quaternion is (x, y, z, w) while PyTorch3D quaternion is (w, x, y, z).
    """
    return torch.cat([quat[3:], quat[:3]])


def pytorch3d_quat_to_isaac_quat(quat):
    """Converts PyTorch3D quaternion to IsaacGym quaternion.

    PyTorch3D quaternion is (w, x, y, z) while IsaacGym quaternion is (x, y, z, w).
    """
    return torch.cat([quat[1:], quat[:1]])


def isaac_quat_to_rot_6d(quat: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    quat = isaac_quat_to_pytorch3d_quat(quat)

    # Convert each quaternion to a rotation matrix
    rot_mats = pt.quaternion_to_matrix(quat)

    # Extract the first two columns of each rotation matrix
    rot_6d = pt.matrix_to_rotation_6d(rot_mats)

    return rot_6d


def np_isaac_quat_to_rot_6d(quat: np.ndarray) -> np.ndarray:
    """Converts IsaacGym quaternion to rotation 6D."""
    quat = torch.from_numpy(quat)
    rot_6d = isaac_quat_to_rot_6d(quat)
    return rot_6d.numpy()
