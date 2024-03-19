from pathlib import Path
from src.visualization.render_mp4 import (
    unpickle_data,
    pickle_data,
)
from src.common.files import get_raw_paths
from tqdm import tqdm

from furniture_bench.robot.robot_state import ROBOT_STATES, ROBOT_STATE_DIMS


rollout_paths = get_raw_paths(
    environment="sim",
    demo_source="rollout",
    demo_outcome="success",
    task="round_table",
    randomness="med",
)


for path in tqdm(rollout_paths):
    data = unpickle_data(path)

    # Check if we have already converted this one
    if isinstance(data["observations"][0]["robot_state"], dict):
        continue

    for obs in data["observations"]:
        robot_state_flat = obs["robot_state"]
        robot_state_dict = {}

        start = 0
        for state, dim in map(lambda s: (s, ROBOT_STATE_DIMS[s]), ROBOT_STATES):
            end = start + dim
            robot_state_dict[state] = robot_state_flat[start:end]
            start = end

        obs["robot_state"] = robot_state_dict

    pickle_data(data, path)
