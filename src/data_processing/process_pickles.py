import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
from typing import List

import numpy as np
import torch
import zarr
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from numcodecs import Blosc, blosc
from tqdm import tqdm, trange
from src.common.types import Trajectory
from src.common.files import get_processed_path, get_raw_paths
from src.visualization.render_mp4 import unpickle_data
from src.common.geometry import (
    np_proprioceptive_quat_to_6d_rotation,
    np_action_quat_to_6d_rotation,
    np_extract_ee_pose_6d,
    np_apply_quat,
    np_action_6d_to_quat,
)
from src.data_processing.utils import clip_axis_rotation

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
    compressor = Blosc(cname="lz4", clevel=5)

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


def process_pickle_file(
    pickle_path: Path,
    noop_threshold: float,
    calculate_pos_action_from_delta: bool = False,
):
    """
    Process a single pickle file and return processed data.
    """
    data: Trajectory = unpickle_data(pickle_path)
    obs = data["observations"]

    # Extract the observations from the pickle file and convert to 6D rotation
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)[:-1]
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)[:-1]

    if isinstance(obs[0]["robot_state"], dict):
        # Convert the robot state to a numpy array
        all_robot_state_quat = np.array(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
            dtype=np.float32,
        )
    else:
        all_robot_state_quat = np.array(
            [o["robot_state"] for o in obs], dtype=np.float32
        )

    all_robot_state_6d = np_proprioceptive_quat_to_6d_rotation(all_robot_state_quat)

    robot_state_6d = all_robot_state_6d[:-1]
    parts_poses = np.array([o["parts_poses"] for o in obs], dtype=np.float32)[:-1]

    # Extract the delta actions from the pickle file and convert to 6D rotation
    action_delta = np.array(data["actions"], dtype=np.float32)
    if action_delta.shape[-1] == 8:
        action_delta_quat = action_delta
        action_delta_6d = np_action_quat_to_6d_rotation(action_delta_quat)
    elif action_delta.shape[-1] == 10:
        raise Exception("This was unexpected")
        action_delta_6d = action_delta
        action_delta_quat = np_action_6d_to_quat(action_delta_6d)
    else:
        raise ValueError(
            f"Unexpected action shape: {action_delta.shape}. Expected (N, 8) or (N, 10)"
        )

    # TODO: Make sure this is rectified in the controller-end and
    # figure out what to do with the corrupted raw data
    # For now, clip the z-axis rotation to 0.35
    action_delta_6d = clip_axis_rotation(action_delta_6d, clip_mag=0.35, axis="z")

    # Clip xyz delta position actions to ±0.025
    action_delta_6d[:, :3] = np.clip(action_delta_6d[:, :3], -0.025, 0.025)

    # Calculate the position actions
    if calculate_pos_action_from_delta:
        action_pos = np.concatenate(
            [
                all_robot_state_quat[:-1, :3] + action_delta_quat[:, :3],
                np_apply_quat(
                    all_robot_state_quat[:-1, 3:7], action_delta_quat[:, 3:7]
                ),
                # Append the gripper action
                action_delta_quat[:, -1:],
            ],
            axis=1,
        )
        action_pos_6d = np_action_quat_to_6d_rotation(action_pos)

    else:
        # Extract the position control actions from the pickle file
        # and concat onto the position actions the gripper actions
        action_pos_6d = np_extract_ee_pose_6d(all_robot_state_6d[1:])
        action_pos_6d = np.concatenate([action_pos, action_delta_6d[:, -1:]], axis=1)

    # Extract the rewards, skills, and parts_poses from the pickle file
    reward = np.array(data["rewards"], dtype=np.float32)
    skill = (
        np.array(data["skills"], dtype=np.float32)
        if "skills" in data
        else np.zeros_like(reward)
    )

    # Sanity check that all arrays are the same length
    assert len(robot_state_6d) == len(
        action_delta_6d
    ), f"Mismatch in {pickle_path}, lengths differ by {len(robot_state_6d) - len(action_delta_6d)}"

    # Extract the pickle file name as the path after `raw` in the path
    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    processed_data = {
        "robot_state": robot_state_6d,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "action/delta": action_delta_6d,
        "action/pos": action_pos_6d,
        "reward": reward,
        "skill": skill,
        "parts_poses": parts_poses,
        "episode_length": len(action_delta_6d),
        "furniture": data["furniture"],
        "success": 1 if data["success"] == "partial_success" else int(data["success"]),
        "failure_idx": data.get("failure_idx", -1),
        "pickle_file": pickle_file,
    }

    return processed_data


def parallel_process_pickle_files(
    pickle_paths,
    noop_threshold,
    num_threads,
    calculate_pos_action_from_delta=False,
):
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
        "failure_idx": [],  # This will be -1 if no failure
        "pickle_file": [],
    }

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_pickle_file,
                path,
                noop_threshold,
                calculate_pos_action_from_delta,
            )
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
    for key in tqdm(
        [
            "robot_state",
            "color_image1",
            "color_image2",
            "action/delta",
            "action/pos",
            "reward",
            "skill",
            "parts_poses",
        ],
        desc="Converting lists to numpy arrays",
    ):
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
        choices=["scripted", "rollout", "teleop", "augmentation"],
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
        choices=["success", "failure", "partial_success"],
        default=None,
        nargs="+",
    )
    parser.add_argument("--calculate-pos-action-from-delta", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    args = parser.parse_args()

    pickle_paths: List[Path] = sorted(
        get_raw_paths(
            environment=args.env,
            task=args.furniture,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
        )
    )

    if args.randomize_order:
        print(f"Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        random.shuffle(pickle_paths)

    if args.max_files is not None:
        pickle_paths = pickle_paths[: args.max_files]

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
    n_cpus = min(os.cpu_count(), args.n_cpus)

    print(
        f"Processing pickle files with {n_cpus} CPUs, chunksize={chunksize}, noop_threshold={noop_threshold}"
    )

    all_data = parallel_process_pickle_files(
        pickle_paths,
        noop_threshold,
        n_cpus,
        calculate_pos_action_from_delta=args.calculate_pos_action_from_delta,
    )

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
        ("failure_idx", (len(all_data["failure_idx"]),), np.int32),
        ("pickle_file", (len(all_data["pickle_file"]),), str),
    ]

    # Initialize Zarr store with full dimensions
    z = initialize_zarr_store(output_path, full_data_shapes, chunksize=chunksize)

    # Write the data to the Zarr store
    it = tqdm(all_data)
    for name in it:
        it.set_description(f"Writing data to zarr: {name}")
        dataset = z[name]
        data = all_data[name]
        for i in trange(
            0, len(all_data[name]), chunksize, desc="Writing chunks", leave=False
        ):
            dataset[i : i + chunksize] = all_data[name][i : i + chunksize]

        # Free memory
        # del all_data[name]

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = len(z["action/delta"])
    z.attrs["calculated_pos_action_from_delta"] = args.calculate_pos_action_from_delta
