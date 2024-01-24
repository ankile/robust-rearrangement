import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import numpy as np
import zarr
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from numcodecs import Blosc, blosc
from tqdm import tqdm, trange
from src.common.types import Trajectory
from src.common.files import get_processed_path, get_raw_paths
from src.visualization.render_mp4 import unpickle_data
from src.common.geometry import (
    np_proprioceptive_to_6d_rotation,
    np_action_to_6d_rotation,
    np_extract_ee_pose_6d,
)

from ipdb import set_trace as bp  # noqa


# === Modified Function to Initialize Zarr Store with Full Dimensions ===
def initialize_zarr_store(out_path, full_data_shapes, chunksize=32):
    """
    Initialize the Zarr store with full dimensions for each dataset.
    """
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()

    # Define the compressor
    # compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)
    compressor = Blosc(cname="lz4")

    # Initialize datasets with full shapes
    for name, shape, dtype in full_data_shapes:
        if "color_image" in name:  # Apply compression to image data
            z.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                chunks=(chunksize,) + shape[1:],
                compressor=compressor,
            )
        else:
            z.create_dataset(
                name, shape=shape, dtype=dtype, chunks=(chunksize,) + shape[1:]
            )

    return z


def process_pickle_file(pickle_path: Path, noop_threshold: float):
    """
    Process a single pickle file and return processed data.
    """
    data: Trajectory = unpickle_data(pickle_path)
    obs = data["observations"]

    # Extract the observations from the pickle file and convert to 6D rotation
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)[:-1]
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)[:-1]
    all_robot_state = np.array(
        [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
        dtype=np.float32,
    )

    all_robot_state = np_proprioceptive_to_6d_rotation(all_robot_state)
    robot_state = all_robot_state[:-1]
    parts_poses = np.array([o["parts_poses"] for o in obs], dtype=np.float32)[:-1]

    # Extract the delta actions from the pickle file and convert to 6D rotation
    action_delta = np.array(data["actions"], dtype=np.float32)
    action_delta = np_action_to_6d_rotation(action_delta)

    # Extract the position control actions from the pickle file
    action_pos = np_extract_ee_pose_6d(all_robot_state[1:])

    # Extract the rewards, skills, and parts_poses from the pickle file
    reward = np.array(data["rewards"], dtype=np.float32)
    skill = np.array(data["skills"], dtype=np.float32)

    # Sanity check that all arrays are the same length
    assert len(robot_state) == len(
        action_delta
    ), f"Mismatch in {pickle_path}, lengths differ by {len(robot_state) - len(action_delta)}"

    # Extract the pickle file name as the path after `raw` in the path
    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    processed_data = {
        "robot_state": robot_state,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "action/delta": action_delta,
        "action/pos": action_pos,
        "reward": reward,
        "skill": skill,
        "parts_poses": parts_poses,
        "episode_length": len(action_delta),
        "furniture": data["furniture"],
        "success": data["success"],
        "pickle_file": pickle_file,
    }

    return processed_data


def parallel_process_pickle_files(pickle_paths, noop_threshold, num_threads):
    """
    Process all pickle files in parallel and aggregate results.
    """
    # Initialize empty data structures to hold aggregated data
    aggregated_data = {
        "robot_state": [],
        "color_image1": [],
        "color_image2": [],
        "action/delta": [],
        "action/pos": [],
        "reward": [],
        "skill": [],
        "parts_poses": [],
        "episode_ends": [],
        "furniture": [],
        "success": [],
        "pickle_file": [],
    }

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_pickle_file, path, noop_threshold)
            for path in pickle_paths
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            data = future.result()
            # Aggregate data from each file
            for key in data:
                if key == "episode_length":
                    # Calculate and append to episode_ends
                    last_end = (
                        aggregated_data["episode_ends"][-1]
                        if len(aggregated_data["episode_ends"]) > 0
                        else 0
                    )
                    aggregated_data["episode_ends"].append(last_end + data[key])
                else:
                    aggregated_data[key].append(data[key])

    # Convert lists to numpy arrays for numerical data
    for key in [
        "robot_state",
        "color_image1",
        "color_image2",
        "action/delta",
        "action/pos",
        "reward",
        "skill",
        "parts_poses",
    ]:
        aggregated_data[key] = np.concatenate(aggregated_data[key])

    return aggregated_data


def write_to_zarr_store(z, key, value):
    """
    Function to write data to a Zarr store.
    """
    z[key][:] = value


def parallel_write_to_zarr(z, aggregated_data, num_threads):
    """
    Write aggregated data to the Zarr store in parallel.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for key, value in aggregated_data.items():
            # Schedule the writing of each dataset
            futures.append(executor.submit(write_to_zarr_store, z, key, value))

        # Wait for all futures to complete and track progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Writing to Zarr store"
        ):
            future.result()


# === Entry Point of the Script ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, nargs="+", default=None)
    parser.add_argument("--furniture", "-f", type=str, default=None, nargs="+")
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop"],
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--randomness",
        "-r",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--demo-outcome",
        "-d",
        type=str,
        choices=["success", "failure"],
        default=None,
        nargs="+",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pickle_paths: List[Path] = get_raw_paths(
        environment=args.env,
        task=args.furniture,
        demo_source=args.source,
        randomness=args.randomness,
        demo_outcome=args.demo_outcome,
    )[:10]

    print(f"Found {len(pickle_paths)} pickle files")

    output_path = get_processed_path(
        environment=args.env,
        task=args.furniture,
        demo_source=args.source,
        randomness=args.randomness,
        demo_outcome=args.demo_outcome,
    )

    print(f"Output path: {output_path}")

    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"Output path already exists: {output_path}. Use --overwrite to overwrite."
        )

    # Process all pickle files
    chunksize = 1_000
    noop_threshold = 0.0
    # n_cpus = min(os.cpu_count(), 64)
    n_cpus = 1

    print(
        f"Processing pickle files with {n_cpus} CPUs, chunksize={chunksize}, noop_threshold={noop_threshold}"
    )

    all_data = parallel_process_pickle_files(pickle_paths, noop_threshold, n_cpus)

    # Define the full shapes for each dataset
    full_data_shapes = [
        # These are of length: number of timesteps
        ("robot_state", all_data["robot_state"].shape, np.float32),
        ("color_image1", all_data["color_image1"].shape, np.uint8),
        ("color_image2", all_data["color_image2"].shape, np.uint8),
        ("action/delta", all_data["action/delta"].shape, np.float32),
        ("action/pos", all_data["action/pos"].shape, np.float32),
        ("skill", all_data["skill"].shape, np.float32),
        ("reward", all_data["reward"].shape, np.float32),
        ("parts_poses", all_data["parts_poses"].shape, np.float32),
        # These are of length: number of episodes
        ("episode_ends", (len(all_data["episode_ends"]),), np.uint32),
        ("furniture", (len(all_data["furniture"]),), str),
        ("success", (len(all_data["success"]),), np.uint8),
        ("pickle_file", (len(all_data["pickle_file"]),), str),
    ]

    # Initialize Zarr store with full dimensions
    z = initialize_zarr_store(output_path, full_data_shapes, chunksize=chunksize)

    blosc.use_threads = True
    blosc.set_nthreads(n_cpus)

    # Write the data to the Zarr store
    it = tqdm(all_data)
    for name in it:
        it.set_description(f"Writing data to zarr: {name}")
        for i in trange(
            0, len(all_data[name]), chunksize, desc="Writing chunks", leave=False
        ):
            end_idx = min(i + chunksize, len(all_data[name]))
            z[name][i : i + end_idx] = all_data[name][i : i + end_idx]

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
