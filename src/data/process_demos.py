import pickle
from glob import glob

import numpy as np
import zarr
from tqdm import tqdm


def get_concatenated_observation(obs, obs_type):
    robot_state = obs["robot_state"]

    if obs_type == "state":
        parts_poses = obs["parts_poses"]

        # Add the observation to the overall list.
        observation = np.concatenate((robot_state, parts_poses))
    elif obs_type == "feature":
        img1 = obs["image1"]
        img2 = obs["image2"]

        # Add the observation to the overall list.
        observation = np.concatenate((robot_state, img1, img2))

    else:
        raise ValueError(f"Invalid observation type: {obs_type}")

    return observation


def process_demos(input_path, output_path):
    file_paths = glob(f"{input_path}/**/*.pkl", recursive=True)
    n_state_action_pairs = len(file_paths)

    print(f"Number of state-action pairs: {n_state_action_pairs}")

    obs_type = (
        "state"
        if "state" in input_path
        else "feature"
        if "feature" in input_path
        else "image"
    )

    observations = []
    actions = []
    episode_ends = []

    end_index = 0
    for path in tqdm(file_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        for obs, action in zip(data["observations"], data["actions"]):
            # Each observation is just a concatenation of the robot state and the object state.
            # Collect the robot state.
            observation = get_concatenated_observation(obs, obs_type)
            observations.append(observation)

            # Add the action to the overall list.
            actions.append(action)

            # Increment the end index.
            end_index += 1

        # Add the end index to the overall list.
        episode_ends.append(end_index)

    # Convert the lists to numpy arrays.
    observations = np.array(observations)
    actions = np.array(actions)
    episode_ends = np.array(episode_ends)

    # Save the data to a zarr file.
    zarr.save(
        f"{output_path}/data.zarr",
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
    )


if __name__ == "__main__":
    raw_data_path = "data/raw/sim/feature/one_leg"
    output_path = "data/processed/sim/feature/one_leg"

    process_demos(raw_data_path, output_path)
