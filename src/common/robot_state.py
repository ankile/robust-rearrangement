"""Define the types of robot state"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ipdb import set_trace as bp


# List of robot state we are going to use during training and testing.
ROBOT_STATES = [
    "ee_pos",
    "ee_quat",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
]

ROBOT_STATE_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "ee_pos_vel": 3,
    "ee_ori_vel": 3,
    "joint_positions": 7,
    "joint_velocities": 7,
    "joint_torques": 7,
    "gripper_width": 1,
}


def filter_and_concat_robot_state(robot_state):
    current_robot_state = []
    for rs in ROBOT_STATES:
        if rs not in robot_state:
            continue

        if rs == "gripper_width":
            robot_state[rs] = np.array([robot_state[rs]]).reshape(1)
        current_robot_state.append(robot_state[rs])
    return np.concatenate(current_robot_state, axis=-1)


@dataclass
class PandaState:
    """Define state of Panda arm and end-effector."""

    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_pos_vel: np.ndarray
    ee_ori_vel: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_width: np.ndarray


class PandaError(Enum):
    OLD_GRIPPER_ERROR = 1
    OK = "Successful"
    Gripper = "Panda gripper server stopped."
    Arm = 2
