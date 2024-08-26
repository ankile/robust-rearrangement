import numpy as np
import torch
import pytorch3d.transforms as pt
from ipdb import set_trace as bp
from scipy.spatial.transform import Rotation as R
import furniture_bench.controllers.control_utils as C


def quat_xyzw_error(from_quat_xyzw, to_quat_xyzw):
    """Computes the quaternion error between two quaternions."""
    from_quat_wxyz = C.quat_xyzw_to_wxyz(from_quat_xyzw)
    to_quat_wxyz = C.quat_xyzw_to_wxyz(to_quat_xyzw)
    rel_quat_wxyz = pt.quaternion_multiply(
        pt.quaternion_invert(from_quat_wxyz), to_quat_wxyz
    )
    rel_quat_xyzw = C.quat_wxyz_to_xyzw(rel_quat_wxyz)
    return rel_quat_xyzw


def pose_error(from_pose, to_pose):
    """
    Computes the pose error between two poses.

    The pose is represented as a 7D vector: (x, y, z, qx, qy, qz, qw)
    """
    from_pos, from_quat = from_pose[..., :3], from_pose[..., 3:]
    to_pos, to_quat = to_pose[..., :3], to_pose[..., 3:]

    pos_error = to_pos - from_pos
    quat_error = quat_xyzw_error(from_quat, to_quat)

    return torch.cat([pos_error, quat_error], dim=-1)


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


def isaac_quat_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = pt.quaternion_to_matrix(quat_wxyz)

    # Extract the first two columns of each rotation matrix
    rot_6d = pt.matrix_to_rotation_6d(rot_mats)

    return rot_6d


def quat_xyzw_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = pt.quaternion_to_matrix(quat_wxyz)

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


def action_quat_to_6d_rotation(action: torch.tensor) -> torch.tensor:
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


def np_action_quat_to_6d_rotation(action: np.ndarray) -> np.ndarray:
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
    action_6d = action_quat_to_6d_rotation(action)
    action_6d = action_6d.numpy()
    return action_6d


def action_6d_to_quat(action_6d: torch.tensor) -> torch.tensor:
    assert action_6d.shape[-1] == 10, "Action must be 10D"

    delta_pos = action_6d[..., :3]  # 3D position
    delta_rot_6d = action_6d[..., 3:9]  # 6D rotation
    delta_gripper = action_6d[..., 9:]  # 1D gripper

    delta_quat = rot_6d_to_isaac_quat(delta_rot_6d)

    action_quat = torch.cat([delta_pos, delta_quat, delta_gripper], dim=-1)
    return action_quat


def np_action_6d_to_quat(action_6d: np.ndarray) -> np.ndarray:
    action_6d_torch = torch.from_numpy(action_6d)
    action_quat = action_6d_to_quat(action_6d_torch)
    return action_quat.numpy()


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


def proprioceptive_quat_to_6d_rotation(robot_state: torch.tensor) -> torch.tensor:
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
    # assert robot_state.shape[-1] == 14, "Robot state must be 14D"

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


def isaac_quat_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    # quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = quaternion_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    return torch.stack((o1, o2, o3, o0), -1)


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :-1], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., -1:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :-1] / sin_half_angles_over_angles


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def np_proprioceptive_quat_to_6d_rotation(robot_state: np.ndarray) -> np.ndarray:
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
    # assert robot_state.shape[-1] == 14, "Robot state must be 14D"

    robot_state = torch.from_numpy(robot_state)
    robot_state_6d = proprioceptive_quat_to_6d_rotation(robot_state)
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


def np_apply_quat(state_quat: np.ndarray, action_quat: np.ndarray) -> np.ndarray:
    state_rot = R.from_quat(state_quat)
    action_rot = R.from_quat(action_quat)

    new_state_rot = state_rot * action_rot

    return new_state_rot.as_quat()
