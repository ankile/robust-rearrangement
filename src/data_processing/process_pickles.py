import argparse
import os
from pathlib import Path
import pickle
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from datetime import datetime


# === Modified Function to Initialize Zarr Store with Full Dimensions ===
def initialize_zarr_store(out_path, full_data_shapes, chunksize=32):
    """
    Initialize the Zarr store with full dimensions for each dataset.
    """
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()

    # Define the compressor
    compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)

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


# === Modified Function to Process All Pickle Files into Memory First ===
def process_all_pickle_files(pickle_paths, noop_threshold):
    """
    Process all pickle files and store data in memory.
    """
    all_data = {
        "robot_state": [],
        "color_image1": [],
        "color_image2": [],
        "action": [],
        "reward": [],
        "skill": [],
        "episode_ends": [],
        "furniture": [],
        "pickle_files": [],
    }

    for path in tqdm(pickle_paths, desc="Processing pickle files"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        obs = data["observations"][:-1]
        assert len(obs) == len(
            data["actions"]
        ), f"Mismatch in {path}, lengths differ by {len(obs) - len(data['actions'])}"

        color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
        color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)
        robot_state = np.array(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
            dtype=np.float32,
        )
        action = np.array(data["actions"], dtype=np.float32)
        reward = np.array(data["rewards"], dtype=np.float32)
        skill = np.array(data["skills"], dtype=np.float32)

        moving = np.linalg.norm(action[:, :6], axis=1) >= noop_threshold

        all_data["robot_state"].append(robot_state[moving])
        all_data["color_image1"].append(color_image1[moving])
        all_data["color_image2"].append(color_image2[moving])
        all_data["action"].append(action[moving])
        all_data["reward"].append(reward[moving])
        all_data["skill"].append(skill[moving])

        curr_index = (
            all_data["episode_ends"][-1] if len(all_data["episode_ends"]) > 0 else 0
        )
        all_data["episode_ends"].append(curr_index + len(action[moving]))
        all_data["furniture"].append(data["furniture"])

        pickle_file = path.parts[path.parts.index("raw") + 1 :]
        all_data["pickle_files"].append("/".join(pickle_file))

    # Concatenate lists into numpy arrays
    for key in tqdm(all_data, desc="Concatenating data"):
        if key not in ["furniture", "pickle_files", "episode_ends"]:
            all_data[key] = np.concatenate(all_data[key])

    return all_data


def process_pickle_file(pickle_path, noop_threshold):
    """
    Process a single pickle file and return processed data.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    obs = data["observations"][:-1]
    assert len(obs) == len(
        data["actions"]
    ), f"Mismatch in {pickle_path}, lengths differ by {len(obs) - len(data['actions'])}"

    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)
    robot_state = np.array(
        [filter_and_concat_robot_state(o["robot_state"]) for o in obs], dtype=np.float32
    )
    action = np.array(data["actions"], dtype=np.float32)
    reward = np.array(data["rewards"], dtype=np.float32)
    skill = np.array(data["skills"], dtype=np.float32)

    moving = np.linalg.norm(action[:, :6], axis=1) >= noop_threshold

    processed_data = {
        "robot_state": robot_state[moving],
        "color_image1": color_image1[moving],
        "color_image2": color_image2[moving],
        "action": action[moving],
        "reward": reward[moving],
        "skill": skill[moving],
        "episode_length": len(action[moving]),
        "furniture": data["furniture"],
        "pickle_file": pickle_path.name,
    }

    return processed_data


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def parallel_process_pickle_files(pickle_paths, noop_threshold, num_threads):
    """
    Process all pickle files in parallel and aggregate results.
    """
    # Initialize empty data structures to hold aggregated data
    aggregated_data = {
        "robot_state": [],
        "color_image1": [],
        "color_image2": [],
        "action": [],
        "reward": [],
        "skill": [],
        "episode_ends": [],  # Start with 0
        "furniture": [],
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
        "action",
        "reward",
        "skill",
    ]:
        aggregated_data[key] = np.concatenate(aggregated_data[key])

    return aggregated_data


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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
# ... (Your argument parsing code remains the same) ...

if __name__ == "__main__":
    # ... (Your argument parsing code remains the same) ...
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

    print(f"Found {len(pickle_paths)} pickle files")

    sources = "_".join(sorted(args.source))

    chunksize = 10_000
    noop_threshold = 0.0
    output_path = output_path / f"{sources}.zarr"

    print(f"Output path: {output_path}")

    # Process all pickle files
    all_data = parallel_process_pickle_files(pickle_paths, noop_threshold, 32)

    # Define the full shapes for each dataset
    full_data_shapes = [
        ("robot_state", all_data["robot_state"].shape, np.float32),
        ("color_image1", all_data["color_image1"].shape, np.uint8),
        ("color_image2", all_data["color_image2"].shape, np.uint8),
        ("action", all_data["action"].shape, np.float32),
        ("reward", all_data["reward"].shape, np.float32),
        ("skill", all_data["skill"].shape, np.float32),
        ("episode_ends", (len(all_data["episode_ends"]),), np.uint32),
        ("furniture", (len(all_data["furniture"]),), str),
        ("pickle_file", (len(all_data["pickle_file"]),), str),
    ]

    # Initialize Zarr store with full dimensions
    z = initialize_zarr_store(output_path, full_data_shapes, chunksize=chunksize)

    from numcodecs import blosc

    blosc.use_threads = True
    blosc.set_nthreads(32)

    # Write the data to the Zarr store
    it = tqdm(all_data)
    for name in it:
        it.set_description(f"Writing data to zarr: {name}")
        z[name][:] = all_data[name]

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
