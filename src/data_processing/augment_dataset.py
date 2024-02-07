import pickle
from glob import glob
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import zarr
import numpy as np

from src.models import get_encoder
from src.data_processing.process_pickles import encode_demo
from src.visualization.render_mp4 import create_mp4

base_dir = Path(os.environ["ROLLOUT_SAVE_DIR"])
rollout_dir = base_dir / "raw" / "sim_rollouts"
file_path = rollout_dir / "index.csv"

print(f"Reading index file from {file_path}")
print(f"Loading rollouts from {rollout_dir}")

# Augment an existing Zarr array with new data from the index
base_dir = Path(os.environ["FURNITURE_DATA_DIR_PROCESSED"])
zarr_path = (
    base_dir
    / "processed"
    / "sim"
    / "feature_separate_small"
    / "vip"
    / "one_leg"
    / "data_aug.zarr"
)

print(f"Loading zarr file from {zarr_path}")
store = zarr.open(str(zarr_path), mode="a")

print(f"Currently {store['episode_ends'].shape[0]} trajectories in dataset")

if "rollout_paths" not in store:
    print("Creating rollout_paths dataset")
    store.create_dataset("rollout_paths", shape=(0,), dtype=str)
else:
    print("rollout_paths dataset already exists, nothing to do")

# Remove the skills dataset if it exists
if "skills" in store:
    print("Removing skills dataset")
    del store["skills"]
else:
    print("skills dataset does not exist, nothing to do")

# Read in the index file as a dataframe
index = pd.read_csv(file_path)
index = index[index["success"] == True]

print(f"Total of {len(index)} successsful rollouts in index file")

# Get the paths to all the successful rollouts
paths = index["path"].values

# Compare with the paths already in the zarr file
zarr_paths = store["rollout_paths"][:]
paths = [p for p in paths if p not in zarr_paths]

print(f"Adding {len(paths)} new rollouts to zarr file")
# Get an encoder
print("Loading encoder")
encoder = get_encoder("vip", freeze=True, device="cuda:0")
batch_size = 1024

# Iterate over the paths and add them to the zarr file
end_index = store["episode_ends"][-1]

for path in tqdm(paths):
    with open(path, "rb") as f:
        data = pickle.load(f)

    end_idx = np.argmax(data["rewards"]) + 1

    obs = data["observations"][:end_idx]

    demo_robot_states, demo_features1, demo_features2 = encode_demo(
        encoder, batch_size, obs
    )

    # Defer all the appending to last in case something goes wrong with the encoding
    store["action"].append(data["actions"][:end_idx])
    store["rewards"].append(data["rewards"][:end_idx])

    store["episode_ends"].append([end_index := end_index + end_idx])
    store["furniture"].append([data["furniture"]])

    store["robot_state"].append(demo_robot_states)
    store["feature1"].append(demo_features1)
    store["feature2"].append(demo_features2)
    store["rollout_paths"].append([path])
