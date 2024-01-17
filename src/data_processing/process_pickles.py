from pathlib import Path
import pickle
from glob import glob
import argparse
import os
import numpy as np
import zarr
from ipdb import set_trace as bp  # noqa
from tqdm import tqdm
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from datetime import datetime


# === Image obs ===
def initialize_zarr_store(out_path, initial_data, chunksize=32):
    """
    Initialize the Zarr store with datasets based on the initial data sample.
    """
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = str(np.datetime64("now"))

    images_shape = initial_data["observations"][0]["color_image1"].shape
    actions_shape = initial_data["actions"][0].shape
    robot_state_shape = (
        len(
            filter_and_concat_robot_state(
                initial_data["observations"][0]["robot_state"]
            )
        ),
    )

    # Initialize datasets with shapes based on the initial data
    print("Chunksize", (chunksize,) + robot_state_shape)
    z.create_dataset(
        "robot_state",
        shape=(0,) + robot_state_shape,
        dtype=np.float32,
        chunks=(chunksize,) + robot_state_shape,
    )
    z.create_dataset(
        "color_image1",
        shape=(0,) + images_shape,
        dtype=np.uint8,
        chunks=(chunksize,) + images_shape,
    )
    z.create_dataset(
        "color_image2",
        shape=(0,) + images_shape,
        dtype=np.uint8,
        chunks=(chunksize,) + images_shape,
    )
    z.create_dataset(
        "action",
        shape=(0,) + actions_shape,
        dtype=np.float32,
        chunks=(chunksize,) + actions_shape,
    )
    # Setting chunking to True in the below is a mistake
    # Since we're appending to the dataset, the best Zarr
    # can do is to have chunksize 1, meaning we get no.
    # episodes times episode length chunks (too many).
    z.create_dataset(
        "reward",
        shape=(0,),
        dtype=np.float32,
        chunks=(chunksize,),
    )
    z.create_dataset(
        "skill",
        shape=(0,),
        dtype=np.float32,
        chunks=(chunksize,),
    )
    # It doesn't really matter what this does wrt. chunking, since
    # the number of elements is small and each element is small.
    z.create_dataset(
        "episode_ends",
        shape=(0,),
        dtype=np.uint32,
    )
    z.create_dataset(
        "furniture",
        shape=(0,),
        dtype=str,
    )

    # Add an array that stores a list of the pickle files used to create the dataset to make it easier to extend with more data later
    z.create_dataset(
        "pickle_files",
        shape=(0,),
        dtype=str,
    )

    return z


def process_pickle_file(z, pickle_path, noop_threshold):
    """
    Process a single pickle file and append data to the Zarr store.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    obs = data["observations"][:-1]

    assert len(obs) == len(
        data["actions"]
    ), f"Mismatch in {pickle_path}, lengths differ by {len(obs) - len(data['actions'])}"

    # Convert all lists to numpy arrays
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)
    robot_state = np.array(
        [filter_and_concat_robot_state(o["robot_state"]) for o in obs], dtype=np.float32
    )
    action = np.array(data["actions"], dtype=np.float32)
    reward = np.array(data["rewards"], dtype=np.float32)
    skill = np.array(data["skills"], dtype=np.float32)

    if len(action.shape) < 2:
        bp()

    # Find the indexes where no action is taken
    moving = np.linalg.norm(action[:, :6], axis=1) >= noop_threshold

    # Filter out the non-moving observations and add to the Zarr store
    z["robot_state"].append(robot_state[moving])
    z["color_image1"].append(color_image1[moving])
    z["color_image2"].append(color_image2[moving])
    z["action"].append(action[moving])
    z["reward"].append(reward[moving])
    z["skill"].append(skill[moving])

    # Add the episode ends, furniture, and pickle file name to the Zarr store
    curr_index = z["episode_ends"][-1] if len(z["episode_ends"]) > 0 else 0
    z["episode_ends"].append([curr_index + len(action[moving])])
    z["furniture"].append([data["furniture"]])

    # Keep only everything after `raw` in the path (the relevant part)
    pickle_file = pickle_path.parts[pickle_path.parts.index("raw") + 1 :]
    z["pickle_files"].append(["/".join(pickle_file)])


def create_zarr_dataset(pickle_paths, out_path, chunksize=32, noop_threshold=0):
    """
    Create a Zarr dataset from multiple pickle files in a directory.
    """
    print(f"Number of trajectories: {len(pickle_paths)}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Load one file to initialize the Zarr store
    with open(pickle_paths[0], "rb") as f:
        initial_data = pickle.load(f)

    z = initialize_zarr_store(out_path, initial_data, chunksize=chunksize)

    # Process the first file
    process_pickle_file(z, pickle_paths[0], noop_threshold=noop_threshold)

    # Process the remaining files
    for path in tqdm(pickle_paths[1:], desc="Processing pickle files"):
        process_pickle_file(z, path, noop_threshold=noop_threshold)

    # Update any final metadata if necessary
    # Set the time finished to now with timezone info
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop"],
        default="scripted",
        nargs="+",
    )
    parser.add_argument("--furniture", "-f", type=str, default=None)
    parser.add_argument("--randomness", "-r", type=str, default=None)

    args = parser.parse_args()

    data_base_path_in = Path(os.environ["DATA_DIR_RAW"])
    data_base_path_out = Path(os.environ["DATA_DIR_PROCESSED"])

    raw_data_paths = [data_base_path_in / "raw" / args.env / s for s in args.source]
    output_path = data_base_path_out / "processed" / args.env / "image"

    if args.furniture is not None:
        raw_data_paths = [p / args.furniture for p in raw_data_paths]
        output_path = output_path / args.furniture

    if args.randomness is not None:
        assert (
            args.furniture is not None
        ), "Must specify furniture when using randomness"
        assert args.randomness in ["low", "med", "high"], "Invalid randomness level"
        raw_data_paths = [p / args.randomness for p in raw_data_paths]
        output_path = output_path / args.randomness

    print(f"Raw data paths: {raw_data_paths}")

    pickle_paths = []
    for path in raw_data_paths:
        pickle_paths += list(path.glob("**/*_success.pkl"))

    sources = "_".join(sorted(args.source))

    chunksize = 32
    noop_threshold = 0.0
    output_path = output_path / f"{sources}.zarr"

    print(f"Output path: {output_path}")
    create_zarr_dataset(
        pickle_paths,
        output_path,
        chunksize=chunksize,
        noop_threshold=noop_threshold,
    )
