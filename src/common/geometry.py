import numpy as np
import torch
import pytorch3d.transforms as pt
from ipdb import set_trace as bp
from scipy.spatial.transform import Rotation as R


def isaac_quat_to_pytorch3d_quat(quat):
    """Converts IsaacGym quaternion to PyTorch3D quaternion.

    IsaacGym quaternion is (x, y, z, w) while PyTorch3D quaternion is (w, x, y, z).
    """
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def pytorch3d_quat_to_isaac_quat(quat):
    """Converts PyTorch3D quaternion to IsaacGym quaternion.

    PyTorch3D quaternion is (w, x, y, z) while IsaacGym quaternion is (x, y, z, w).
    """
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


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


def rot_6d_to_isaac_quat(rot_6d: torch.Tensor) -> torch.Tensor:
    # Convert 6D rotation back to a full rotation matrix
    rot_mats = pt.rotation_6d_to_matrix(rot_6d)

    # Convert rotation matrix to quaternion
    quat = pt.matrix_to_quaternion(rot_mats)

    # Convert quaternion from PyTorch3D format to IsaacGym format
    quat = pytorch3d_quat_to_isaac_quat(quat)

    return quat


def np_rot_6d_to_isaac_quat(rot_6d: np.ndarray) -> np.ndarray:
    rot_6d_tensor = torch.from_numpy(rot_6d)
    quat = rot_6d_to_isaac_quat(rot_6d_tensor)
    return quat.numpy()


def action_to_6d_rotation(action: torch.tensor) -> torch.tensor:
    """
    Convert the 8D action space to 10D action space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 1D gripper

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    assert action.shape[-1] == 8, "Action must be 8D"

    # Get each part of the action
    delta_pos = action[..., :3]  # (x, y, z)
    delta_quat = action[..., 3:7]  # (x, y, z, w)
    delta_gripper = action[..., 7:]  # (width)

    # Convert quaternion to 6D rotation
    delta_rot = isaac_quat_to_rot_6d(delta_quat)

    # Concatenate all parts
    action_6d = torch.cat([delta_pos, delta_rot, delta_gripper], dim=1)

    return action_6d


def np_action_to_6d_rotation(action: np.ndarray) -> np.ndarray:
    """
    Convert the 8D action space to 10D action space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 1D gripper

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    action = torch.from_numpy(action)
    action_6d = action_to_6d_rotation(action)
    action_6d = action_6d.numpy()
    return action_6d


def np_rot_6d_to_rotvec(rot_6d: np.ndarray) -> np.ndarray:
    # convert to isaac quat (which means that the real part is last)
    quat = np_rot_6d_to_isaac_quat(rot_6d)

    # convert to rot_vec using scipy (which also uses real part last)
    r = R.from_quat(quat)
    rot_vec = r.as_rotvec()

    return rot_vec


def np_rotvec_to_rot_6d(rot_vec: np.ndarray) -> np.ndarray:
    # Convert rotation vector to quaternion using scipy
    r = R.from_rotvec(rot_vec)
    quat = r.as_quat()  # This will produce a quaternion with the real part last

    # Convert this isaac quat back to 6D rotation
    rot_6d = np_isaac_quat_to_rot_6d(quat)

    return rot_6d


def proprioceptive_to_6d_rotation(robot_state: torch.tensor) -> torch.tensor:
    """
    Convert the 14D proprioceptive state space to 16D state space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 3D linear velocity
        - 3D angular velocity
        - 1D gripper width

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    assert robot_state.shape[-1] == 14, "Robot state must be 14D"

    # Get each part of the robot state
    pos = robot_state[..., :3]  # (x, y, z)
    ori_quat = robot_state[..., 3:7]  # (x, y, z, w)
    pos_vel = robot_state[..., 7:10]  # (x, y, z)
    ori_vel = robot_state[..., 10:13]  # (x, y, z)
    gripper = robot_state[..., 13:]  # (width)

    # Convert quaternion to 6D rotation
    ori_6d = isaac_quat_to_rot_6d(ori_quat)

    # Concatenate all parts
    robot_state_6d = torch.cat([pos, ori_6d, pos_vel, ori_vel, gripper], dim=-1)

    return robot_state_6d


def np_proprioceptive_to_6d_rotation(robot_state: np.ndarray) -> np.ndarray:
    """
    Convert the 14D proprioceptive state space to 16D state space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 3D linear velocity
        - 3D angular velocity
        - 1D gripper width

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    robot_state = torch.from_numpy(robot_state)
    robot_state_6d = proprioceptive_to_6d_rotation(robot_state)
    robot_state_6d = robot_state_6d.numpy()
    return robot_state_6d


def extract_ee_pose_6d(robot_state: torch.tensor) -> torch.tensor:
    """
    Extract the end effector pose from the 6D robot state.

    Accepts any number of leading dimensions.
    """
    assert robot_state.shape[-1] == 16, "Robot state must be 16D"

    # Get each part of the robot state
    pos = robot_state[..., :3]
    ori_6d = robot_state[..., 3:9]

    # Concatenate all parts
    ee_pose_6d = torch.cat([pos, ori_6d], dim=-1)

    return ee_pose_6d


def np_extract_ee_pose_6d(robot_state: np.ndarray) -> np.ndarray:
    """
    Extract the end effector pose from the 6D robot state.

    Accepts any number of leading dimensions.
    """
    robot_state = torch.from_numpy(robot_state)
    ee_pose_6d = extract_ee_pose_6d(robot_state)
    ee_pose_6d = ee_pose_6d.numpy()
    return ee_pose_6d
