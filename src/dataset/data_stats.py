"""
Script for looping over all the zarr datasets and calculating the min, max, mean, and std across the entire dataset.

These values are then written to a json file for use in the normalizer.
"""

from typing import Dict, List
import zarr
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def get_stats_json_path() -> Path:
    return (
        Path(os.environ["DATA_DIR_PROCESSED"]) / "processed" / "sim" / "data_stats.json"
    )


def get_stats_for_field(data: np.ndarray) -> Dict[str, List]:
    data = data.reshape(-1, data.shape[-1])
    stats = {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
    }

    return stats


def get_zarr_files(root_dir: Path):
    zarr_files = []
    for path in root_dir.rglob("*.zarr"):
        zarr_files.append(str(path))
    return zarr_files


def accumulate_values(zarr_files: List[str], keys: List[str]):
    # Loop through all the zarr files and calculate the stats as:
    # - Min-max: the minimum and maximum values for each feature across all datasets
    # - Mean: the mean value for each feature across all datasets
    # - Std: the standard deviation for each feature across all datasets

    # Populate the data dictionary with the keys and initial data we can compare with
    # Accumulate the data from each zarr file before calculating the stats
    data = defaultdict(list)

    for zarr_file in zarr_files:
        z = zarr.open(zarr_file, "r")
        for key in keys:
            data[key].append(z[key][:])

    # Cnvert the lists to numpy arrays
    for key in keys:
        data[key] = np.concatenate(data[key], axis=0)

    return data


def stats_for_keys(data: Dict[str, np.ndarray], keys: List[str]):
    stats = {}
    for key in keys:
        stats[key] = get_stats_for_field(data[key])
    return stats


def calculate_data_stats():
    # Define the keys we want to calculate the stats for
    keys = ["action/delta", "action/pos", "robot_state"]

    # Get the input and output directories
    root_dir = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed" / "sim"
    stats_save_filename = get_stats_json_path()

    # Get the zarr files
    zarr_files = get_zarr_files(root_dir)
    print(f"Found {len(zarr_files)} zarr files")

    # Accumulate the values from the zarr files
    values: Dict[str, np.ndarray] = accumulate_values(zarr_files, keys)

    # Calculate the stats for the keys
    stats = stats_for_keys(values, keys)

    # Save the stats to a json file
    with open(stats_save_filename, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Stats saved to {stats_save_filename}")


def get_data_stats() -> Dict[str, List]:
    # Get the data stats from the json file
    stats_save_filename = get_stats_json_path()
    with open(stats_save_filename, "r") as f:
        data_stats = json.load(f)

    return data_stats


if __name__ == "__main__":
    calculate_data_stats()
