from datetime import datetime
from typing import List
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.visualization.render_mp4 import pickle_data
from src.common.types import Trajectory, Observation
from src.common.geometry import np_action_6d_to_quat

from ipdb import set_trace as bp


def save_raw_rollout(
    robot_states: np.ndarray,
    imgs1: np.ndarray,
    imgs2: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    parts_poses: np.ndarray,
    success: bool,
    task: str,
    action_type: str,
    rollout_save_dir: Path,
    compress_pickles: bool = False,
):
    observations: List[Observation] = list()

    for robot_state, image1, image2, parts_pose in zip(
        robot_states, imgs1, imgs2, parts_poses
    ):
        observations.append(
            {
                "robot_state": robot_state,
                "color_image1": image1,
                "color_image2": image2,
                "parts_poses": parts_pose,
            }
        )

    if action_type == "pos":

        assert actions.shape[1] == 10
        # If we've used rot_6d convert to quat
        actions = np_action_6d_to_quat(actions)
        assert actions.shape[1] == 8

        # Get the action quat
        pos_action_quat = R.from_quat(actions[:, 3:7])

        # Get the position quat from the robot state
        pos_quat = R.from_quat([rs["ee_quat"] for rs in robot_states[:-1]])

        # The action quat was calculated as pos_quat * action_quat
        # Calculate the delta quat between the pos_quat and the action_quat
        delta_action_quat = pos_quat.inv() * pos_action_quat

        # Also calculate the delta position
        delta_action_pos = actions[:, :3] - np.array(
            [rs["ee_pos"] for rs in robot_states[:-1]]
        )

        # Insert the delta quat into the actions
        actions = np.concatenate(
            [delta_action_pos, delta_action_quat.as_quat(), actions[:, -1:]], axis=1
        )

    data: Trajectory = {
        "observations": observations,
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "success": success,
        "task": task,
        "action_type": action_type,
    }

    output_path = rollout_save_dir / ("success" if success else "failure")
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.pkl"

    if compress_pickles:
        output_path = output_path.with_suffix(".pkl.xz")

    pickle_data(data, output_path)
