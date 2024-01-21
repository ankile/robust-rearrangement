from datetime import datetime
from typing import List

from src.visualization.render_mp4 import pickle_data
from src.common.types import Trajectory, Observation


def save_raw_rollout(
    robot_states,
    imgs1,
    imgs2,
    actions,
    rewards,
    parts_poses,
    success,
    furniture,
    rollout_save_dir,
):
    compress_pickles = True
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

    data: Trajectory = {
        "observations": observations,
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "success": success,
        "furniture": furniture,
    }

    output_path = rollout_save_dir / ("success" if success else "failure")
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

    if compress_pickles:
        output_path = output_path.with_suffix(".pkl.xz")

    pickle_data(data, output_path)
