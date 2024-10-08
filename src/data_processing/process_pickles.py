import argparse
import array
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
from numcodecs import Blosc, JSON
from tqdm import tqdm, trange
from src.common.types import Trajectory
from src.common.files import get_processed_path, get_raw_paths
from src.visualization.render_mp4 import unpickle_data
from src.common.geometry import (
    np_proprioceptive_quat_to_6d_rotation,
    np_action_quat_to_6d_rotation,
    np_apply_quat,
)
from src.data_processing.utils import resize, resize_crop
from src.data_processing.utils import clip_quat_xyzw_magnitude

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
        elif dtype == object:
            z.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=shape,
            object_codec=JSON(),
            )
        else:
            z.create_dataset(
            name, shape=shape, dtype=dtype, chunks=shape
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

    action_delta_quat = np.array(data["actions"], dtype=np.float32)
    assert (
        action_delta_quat.shape[-1] == 8
    ), "Expecting actions to be 8D (pos, quat, gripper)"

    if len(obs) == len(action_delta_quat) + 1:
        # The simulator data collection stores the observation received after
        # the last action. We need to remove this observation to match the lengths
        obs = obs[:-1]
    if len(obs) == len(action_delta_quat):
        # In the real world, we apparently don't do that
        pass
    else:
        raise ValueError(
            f"Observations and actions have different lengths: {len(obs)} vs {len(action_delta_quat)}"
        )

    # Extract the observations from the pickle file and convert to 6D rotation
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)

    assert (
        color_image1.shape == color_image2.shape
    ), "Color images have different shapes"

    if color_image1.shape[1:] != (240, 320, 3):
        # We only resize the wrist image to keep the gripper fingers in view
        color_image1 = resize(color_image1)

        # The front camera is also cropped as we don't need the edges
        color_image2 = resize_crop(color_image2)

    assert color_image1.shape[1:] == (
        240,
        320,
        3,
    ), f"Color image 1 has shape {color_image1.shape[1:]}"

    if isinstance(obs[0]["robot_state"], dict):
        # Convert the robot state to a numpy array
        robot_state_quat = np.array(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
            dtype=np.float32,
        )
    else:
        robot_state_quat = np.array([o["robot_state"] for o in obs], dtype=np.float32)

    robot_state_6d = np_proprioceptive_quat_to_6d_rotation(robot_state_quat)
    parts_poses = (
        np.array([o["parts_poses"] for o in obs], dtype=np.float32)
        if "parts_poses" in obs[0]
        else np.array([], dtype=np.float32)
    )

    # TODO: Make sure this is rectified in the controller-end and
    # Clip xyz delta position actions to ±0.025
    action_delta_quat[:, :3] = np.clip(action_delta_quat[:, :3], -0.025, 0.025)

    # figure out what to do with the corrupted raw data
    # For now, clip the z-axis rotation to 0.35
    action_delta_quat[:, 3:7] = clip_quat_xyzw_magnitude(
        action_delta_quat[:, 3:7], clip_mag=0.35
    )

    # Take the sign of the gripper action
    action_delta_quat[:, -1] = np.sign(action_delta_quat[:, -1])

    # Calculate the position actions
    if calculate_pos_action_from_delta:
        action_pos = np.concatenate(
            [
                robot_state_quat[:, :3] + action_delta_quat[:, :3],
                np_apply_quat(robot_state_quat[:, 3:7], action_delta_quat[:, 3:7]),
                # Append the gripper action
                action_delta_quat[:, -1:],
            ],
            axis=1,
        )
        action_pos_6d = np_action_quat_to_6d_rotation(action_pos)

    else:
        raise NotImplementedError(
            "This script only supports calculating position actions from delta actions."
        )

    # Convert delta action to use 6D rotation
    action_delta_6d = np_action_quat_to_6d_rotation(action_delta_quat)

    # Extract the rewards and skills from the pickle file
    reward = (
        np.array(data["rewards"], dtype=np.float32)
        if "rewards" in data
        else np.zeros(len(action_delta_6d))
    )
    skill = (
        np.array(data["skills"], dtype=np.float32)
        if "skills" in data
        else np.zeros_like(reward)
    )
    augment_states = (
        data["augment_states"] if "augment_states" in data else np.zeros_like(reward)
    )

    # Sanity check that all arrays are the same length
    assert len(robot_state_6d) == len(
        action_delta_6d
    ), f"Mismatch in {pickle_path}, lengths differ by {len(robot_state_6d) - len(action_delta_6d)}"

    # Extract the pickle file name as the path after `raw` in the path
    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    task = data.get("task", data.get("furniture"))

    processed_data = {
        "robot_state": robot_state_6d,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "action/delta": action_delta_6d,
        "action/pos": action_pos_6d,
        "reward": reward,
        "skill": skill,
        "augment_states": augment_states,
        "parts_poses": parts_poses,
        "episode_length": len(action_delta_6d),
        "task": task,
        "success": 1 if data["success"] == "partial_success" else int(data["success"]),
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
        "augment_states": [],
        "parts_poses": [],
        "episode_ends": [],
        "task": [],
        "success": [],
        "pickle_file": [],
    }

    def aggregate_data(data):
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

    if num_threads == 1:
        # Run synchronous version
        for path in tqdm(pickle_paths, desc="Processing files"):
            data = process_pickle_file(
                path, noop_threshold, calculate_pos_action_from_delta
            )
            aggregate_data(data)
    else:
        # Run threaded version
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
                aggregate_data(data)

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
            "augment_states",
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
    parser.add_argument(
        "--controller",
        "-c",
        type=str,
        required=True,
        choices=["osc", "diffik"],
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["sim", "real", "distillation"],
        required=True,
    )
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop", "augmentation"],
        required=True,
    )
    parser.add_argument(
        "--randomness",
        "-r",
        type=str,
        choices=["low", "low_perturb", "med", "med_perturb", "high", "high_perturb"],
        required=True,
    )
    parser.add_argument(
        "--demo-outcome",
        "-o",
        type=str,
        choices=["success", "failure", "partial_success"],
        required=True,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
    )
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    assert not args.randomize_order or args.offset == 0, "Cannot offset with randomize"

    pickle_paths: List[Path] = sorted(
        get_raw_paths(
            controller=args.controller,
            domain=args.domain,
            task=args.task,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
            suffix=args.suffix,
        )
    )

    total_files = len(pickle_paths)

    if args.randomize_order:
        print(f"Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        random.shuffle(pickle_paths)
    start = args.offset
    end = (
        args.offset + args.max_files
        if args.max_files is not None
        else len(pickle_paths)
    )
    pickle_paths = pickle_paths[start:end]

    print(f"Found {len(pickle_paths)} pickle files")

    output_path = get_processed_path(
        controller=args.controller,
        domain=args.domain,
        task=args.task,
        demo_source=args.source,
        randomness=args.randomness,
        demo_outcome=args.demo_outcome,
        suffix=args.output_suffix,
    )

    print(f"Output path: {output_path}")

    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"Output path already exists: {output_path}. Use --overwrite to overwrite."
        )

    # Process all pickle files
    chunksize = args.chunk_size
    noop_threshold = 0.0
    n_cpus = min(os.cpu_count(), args.n_cpus)

    print(
        f"Processing pickle files with {n_cpus} CPUs, chunksize={chunksize}, noop_threshold={noop_threshold}\n"
        f"randomize_order={args.randomize_order}, random_seed={args.random_seed}\n"
        f"from file nr. {start} to {end} out of {total_files}"
    )

    all_data = parallel_process_pickle_files(
        pickle_paths,
        noop_threshold,
        n_cpus,
        calculate_pos_action_from_delta=True,
    )

    # Define the full shapes for each dataset
    full_data_shapes = [
        # These are of length: number of timesteps
        ("robot_state", all_data["robot_state"].shape, np.float32),
        ("color_image1", all_data["color_image1"].shape, np.uint8),
        ("color_image2", all_data["color_image2"].shape, np.uint8),
        ("action/delta", all_data["action/delta"].shape, np.float32),
        ("action/pos", all_data["action/pos"].shape, np.float32),
        ("parts_poses", all_data["parts_poses"].shape, np.float32),
        ("reward", all_data["reward"].shape, np.float32),
        ("skill", all_data["skill"].shape, np.float32),
        ("augment_states", all_data["augment_states"].shape, np.float32),
        # These are of length: number of episodes
        ("episode_ends", (len(all_data["episode_ends"]),), np.uint32),
        ("task", (len(all_data["task"]),), str),
        ("success", (len(all_data["success"]),), np.uint8),
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

        for i in trange(0, len(data), chunksize, desc="Writing chunks", leave=False):
            dataset[i : i + chunksize] = data[i : i + chunksize]

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = len(z["action/delta"])
    z.attrs["mean_episode_length"] = round(
        len(z["action/delta"]) / len(z["episode_ends"])
    )
    z.attrs["calculated_pos_action_from_delta"] = True
    z.attrs["randomize_order"] = args.randomize_order
    z.attrs["random_seed"] = args.random_seed
    z.attrs["demo_source"] = args.source
    z.attrs["controller"] = args.controller
    z.attrs["domain"] = args.domain if args.domain == "real" else "sim"
    z.attrs["task"] = args.task
    z.attrs["randomness"] = args.randomness
    z.attrs["demo_outcome"] = args.demo_outcome
    z.attrs["suffix"] = args.suffix
