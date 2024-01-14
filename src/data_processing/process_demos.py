from pathlib import Path
import pickle
from glob import glob
import argparse
import os
from turtle import color
from furniture_bench.robot import robot_state
import numpy as np
import zarr
from tqdm import tqdm, trange
from src.models.vision import get_encoder
import torch
import torchvision.transforms.functional as F
from furniture_bench.robot.robot_state import filter_and_concat_robot_state

from ipdb import set_trace as bp


# === Feature obs ===
@torch.no_grad()
def process_buffer(buffer, encoder):
    # Move buffer to same device as encoder
    tensor = torch.stack(buffer).to(encoder.device)
    return encoder(tensor).cpu().numpy()


def process_pickle_to_feature(input_path, output_path, encoder, batch_size=256):
    file_paths = sorted(list(glob(f"{input_path}/**/*.pkl", recursive=True)))[:16]

    actions, rewards, skills, episode_ends, furniture = [], [], [], [], []
    robot_states, features1, features2 = [], [], []

    end_index = 0

    for path in tqdm(file_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Cut off the last observation because it is not used
        data["observations"] = data["observations"][:-1]
        assert len(data["actions"]) == len(data["observations"]), f"Mismatch in {path}"

        actions.extend(data["actions"])
        rewards.extend(data["rewards"])
        skills.extend(data["skills"])
        episode_ends.append(end_index := end_index + len(data["actions"]))
        furniture.append(data["furniture"])

        obs = data["observations"]
        demo_robot_states, demo_features1, demo_features2 = encode_demo(
            encoder, batch_size, obs
        )
        robot_states.extend(demo_robot_states)
        features1.extend(demo_features1)
        features2.extend(demo_features2)

    obs_dict = {
        "robot_state": np.array(robot_states, dtype=np.float32),
        "feature1": np.array(features1, dtype=np.float32),
        "feature2": np.array(features2, dtype=np.float32),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    zarr.save(
        str(output_path / "data.zarr"),
        action=np.array(actions, dtype=np.float32),
        episode_ends=np.array(episode_ends, dtype=np.uint32),
        reward=np.array(rewards, dtype=np.float32),
        skills=np.array(skills, dtype=np.float32),
        furniture=furniture,
        time_created=np.datetime64("now"),
        **obs_dict,
    )


def encode_demo(encoder, batch_size, obs):
    robot_states, features1, features2 = [], [], []

    robot_state_buffer = []
    for o in obs:
        if isinstance(o["robot_state"], dict):
            robot_state_buffer.append(filter_and_concat_robot_state(o["robot_state"]))
        else:
            robot_state_buffer.append(o["robot_state"])

    img_buffer1 = [torch.from_numpy(o["color_image1"]) for o in obs]
    img_buffer2 = [torch.from_numpy(o["color_image2"]) for o in obs]

    for i in range(0, len(img_buffer1), batch_size):
        slice_end = min(i + batch_size, len(img_buffer1))
        robot_states.extend(robot_state_buffer[i:slice_end])
        features1.extend(process_buffer(img_buffer1[i:slice_end], encoder).tolist())
        features2.extend(process_buffer(img_buffer2[i:slice_end], encoder).tolist())

    return robot_states, features1, features2


# === Image Zarr to feature Zarr ===
@torch.no_grad()
def encode_numpy_batch(buffer, encoder):
    # Move buffer to same device as encoder
    tensor = torch.from_numpy(buffer).to(encoder.device)
    return encoder(tensor).cpu().numpy()


def process_zarr_to_feature(zarr_input_path, zarr_output_path, encoder, batch_size=256):
    zarr_group = zarr.open(zarr_input_path, mode="r")
    color_image1 = zarr_group["color_image1"]
    color_image2 = zarr_group["color_image2"]

    # Assuming other data like actions, rewards, etc. are also stored in the zarr file
    action = zarr_group["action"]
    episode_ends = zarr_group["episode_ends"]
    furniture = zarr_group["furniture"]
    reward = zarr_group["reward"]
    robot_state = zarr_group["robot_state"]
    skills = zarr_group["skill"]

    # Process images and create features
    features1, features2 = [], []
    for i in trange(0, len(color_image1), batch_size):
        slice_end = min(i + batch_size, len(color_image1))

        features1.extend(
            encode_numpy_batch(color_image1[i:slice_end], encoder).tolist()
        )
        features2.extend(
            encode_numpy_batch(color_image2[i:slice_end], encoder).tolist()
        )

    # Create a new Zarr file for output
    output_group = zarr.open(zarr_output_path, mode="w")
    output_group.array("feature1", np.array(features1, dtype=np.float32))
    output_group.array("feature2", np.array(features2, dtype=np.float32))
    output_group.array("action", action)
    output_group.array("episode_ends", episode_ends)
    output_group.array("furniture", furniture, dtype=str)
    output_group.array("reward", reward)
    output_group.array("robot_state", robot_state)
    output_group.array("skill", skills)
    output_group.attrs["time_created"] = str(np.datetime64("now"))


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


def process_pickle_file(z, pickle_path):
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
    
    # Find the indexes where no action is taken
    moving = np.linalg.norm(action[:, :6], axis=1) > 0.02

    # Filter out the non-moving observations and add to the Zarr store
    z["robot_state"].append(robot_state[moving])
    z["color_image1"].append(color_image1[moving])
    z["color_image2"].append(color_image2[moving])
    z["action"].append(action[moving])
    z["reward"].append(reward[moving])
    z["skill"].append(skill[moving])
    
    # Add the episode ends, furniture, and pickle file name to the Zarr store
    curr_index = z["episode_ends"][-1] if len(z["episode_ends"]) > 0 else 0
    z["episode_ends"].append([curr_index + len(data["actions"])])
    z["furniture"].append([data["furniture"]])

    # Keep only everything after `raw` in the path (the relevant part)
    pickle_file = pickle_path.split("raw")[-1]
    z["pickle_files"].append([pickle_file])


def create_zarr_dataset(in_dir, out_path, chunksize=32):
    """
    Create a Zarr dataset from multiple pickle files in a directory.
    """
    file_paths = sorted(list(glob(f"{in_dir}/**/*.pkl", recursive=True)))
    print(f"Number of trajectories: {len(file_paths)}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Load one file to initialize the Zarr store
    with open(file_paths[0], "rb") as f:
        initial_data = pickle.load(f)

    z = initialize_zarr_store(out_path, initial_data, chunksize=chunksize)

    # Process the first file
    process_pickle_file(z, file_paths[0])

    # Process the remaining files
    for path in tqdm(file_paths[1:], desc="Processing pickle files"):
        process_pickle_file(z, path)

    # Update any final metadata if necessary
    z.attrs["time_finished"] = str(np.datetime64("now"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument("--obs-in", "-i", type=str)
    parser.add_argument("--obs-out", "-o", type=str)
    parser.add_argument("--encoder", "-c", default=None, type=str)
    parser.add_argument("--furniture", "-f", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--randomness", "-r", type=str, default=None)

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = os.environ.get("FURNITURE_DATA_DIR", "data")
    data_base_path_in = Path(os.environ.get("FURNITURE_DATA_DIR_RAW", data_base_path))
    data_base_path_out = Path(
        os.environ.get("FURNITURE_DATA_DIR_PROCESSED", data_base_path)
    )

    obs_out_path = args.obs_out

    raw_data_path = data_base_path_in / "raw" / args.env / args.furniture
    output_path = data_base_path_out / "processed" / args.env / obs_out_path

    encoder = None
    if args.encoder is not None:
        output_path = output_path / args.encoder
        encoder = get_encoder(args.encoder, freeze=True, device=device)
        encoder.eval()

    output_path = output_path / args.furniture

    if args.randomness is not None:
        assert args.randomness in ["low", "med", "high"], "Invalid randomness level"
        raw_data_path = raw_data_path / args.randomness
        output_path = output_path / args.randomness

    print(f"Raw data path: {raw_data_path}")

    if args.obs_out == "feature":
        output_path = output_path / "data.zarr"
        print(f"Output path: {output_path}")
        process_zarr_to_feature(
            "/data/scratch/ankile/furniture-data/data/processed/sim/image/round_table/data_batch_32.zarr",
            output_path,
            encoder,
            batch_size=args.batch_size,
        )
    elif args.obs_out == "image":
        chunksize = 32
        output_path = output_path / f"data_batch_{chunksize}_noop.zarr"
        print(f"Output path: {output_path}")
        create_zarr_dataset(raw_data_path, output_path, chunksize=chunksize)
