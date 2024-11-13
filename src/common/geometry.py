import numpy as np
import torch

# import pytorch3d.transforms as pt
from ipdb import set_trace as bp
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


# import furniture_bench.controllers.control_utils as C


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., -1:] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ow = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((ox, oy, oz, ow), -1)


def quaternion_xyzw_to_rotation_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternions to 6D rotation representation by Zhou et al. [1]
    by converting the quaternion to a rotation matrix and then to 6D.
    Args:
        quat: batch of quaternions of size (*, 4)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    rot_mats = quaternion_xyzw_to_matrix(quat_xyzw)
    return matrix_to_rotation_6d(rot_mats)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            last, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([-1, -1, -1, 1], device=quaternion.device)
    return quaternion * scaling


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def np_rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = torch.from_numpy(d6)
    rot_mats = rotation_6d_to_matrix(d6)
    return rot_mats.numpy()


def quat_xyzw_error(from_quat_xyzw, to_quat_xyzw):
    """Computes the quaternion error between two quaternions."""
    rel_quat_xyzw = quaternion_multiply(quaternion_invert(from_quat_xyzw), to_quat_xyzw)
    return rel_quat_xyzw


@torch.jit.script
def quat_xyzw_to_wxyz(quat_xyzw):
    """
    Convert quaternions from (x, y, z, w) order to (w, x, y, z) order.

    Args:
        quat_xyzw (torch.Tensor): Quaternions in (x, y, z, w) order, shape (..., 4).

    Returns:
        torch.Tensor: Quaternions in (w, x, y, z) order, shape (..., 4).
    """
    inds = torch.tensor([3, 0, 1, 2], dtype=torch.long, device=quat_xyzw.device)
    return torch.index_select(quat_xyzw, dim=-1, index=inds)


@torch.jit.script
def quat_wxyz_to_xyzw(quat_wxyz):
    """
    Convert quaternions from (w, x, y, z) order to (x, y, z, w) order.

    Args:
        quat_wxyz (torch.Tensor): Quaternions in (w, x, y, z) order, shape (..., 4).

    Returns:
        torch.Tensor: Quaternions in (x, y, z, w) order, shape (..., 4).
    """
    inds = torch.tensor([1, 2, 3, 0], dtype=torch.long, device=quat_wxyz.device)
    return torch.index_select(quat_wxyz, dim=-1, index=inds)


def np_quat_wxyz_to_xyzw(quat):
    return np.roll(quat, -1)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def matrix_to_quaternion_xyzw(matrix):
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


def rotation_6d_to_quaternion_xyzw(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to quaternions.
    Args:
        rot_6d: 6D rotation representation, of size (*, 6)

    Returns:
        batch of quaternions of size (*, 4) with real part last
    """
    rot_mats = rotation_6d_to_matrix(rot_6d)
    return matrix_to_quaternion_xyzw(rot_mats)


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

    # Convert each quaternion to a rotation matrix
    rot_mats = quaternion_xyzw_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def quat_xyzw_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Convert each quaternion to a rotation matrix
    rot_mats = quaternion_xyzw_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def np_isaac_quat_to_rot_6d(quat: np.ndarray) -> np.ndarray:
    """Converts IsaacGym quaternion to rotation 6D."""
    quat = torch.from_numpy(quat)
    rot_6d = isaac_quat_to_rot_6d(quat)
    return rot_6d.numpy()


def rot_6d_to_isaac_quat(rot_6d: torch.Tensor) -> torch.Tensor:
    # Convert 6D rotation back to a full rotation matrix
    rot_mats = rotation_6d_to_matrix(rot_6d)

    # Convert rotation matrix to quaternion
    quat = matrix_to_quaternion(rot_mats)

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
    rot_mats = quaternion_xyzw_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def quaternion_xyzw_to_matrix(quaternions):
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


def np_matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Numpy version of matrix_to_rotation_6d function.
    Converts rotation matrices to 6D rotation representation.

    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    matrix_tensor = torch.from_numpy(matrix)
    rotation_6d_tensor = matrix_to_rotation_6d(matrix_tensor)
    return rotation_6d_tensor.numpy()


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
