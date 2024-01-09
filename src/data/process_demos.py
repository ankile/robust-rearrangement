from pathlib import Path
import pickle
from glob import glob
import argparse
import os
import numpy as np
import zarr
from tqdm import tqdm
from src.models.vision import get_encoder
import torch
import torchvision.transforms.functional as F
from furniture_bench.robot.robot_state import filter_and_concat_robot_state

from ipdb import set_trace as bp


@torch.no_grad()
def process_buffer(buffer, encoder):
    # Move buffer to same device as encoder
    tensor = torch.stack(buffer).to(encoder.device)
    return encoder(tensor).cpu().numpy()


def process_demos_to_feature(input_path, output_path, encoder, batch_size=256):
    file_paths = glob(f"{input_path}/**/*.pkl", recursive=True)

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
        str(output_path / "data_new.zarr"),
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


def initialize_zarr_store(out_dir, initial_data):
    """
    Initialize the Zarr store with datasets based on the initial data sample.
    """
    chunksize = 32
    z = zarr.open(str(out_dir / f"data_batch_{chunksize}.zarr"), mode="w")
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
    # z.create_dataset(
    #     "furniture",
    #     shape=(0,),
    #     dtype=np.object,
    #     chunks=True,
    #     maxshape=(None,),
    # )

    return z


def process_pickle_file(z, pickle_path):
    """
    Process a single pickle file and append data to the Zarr store.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    obs = data["observations"]
    z["robot_state"].append(
        [filter_and_concat_robot_state(o["robot_state"]) for o in obs]
    )
    z["color_image1"].append([o["color_image1"] for o in obs])
    z["color_image2"].append([o["color_image2"] for o in obs])
    z["action"].append(data["actions"])
    z["reward"].append(data["rewards"])
    z["skill"].append(data["skills"])
    curr_index = z["episode_ends"][-1] if len(z["episode_ends"]) > 0 else 0
    z["episode_ends"].append([curr_index + len(data["actions"])])
    # z["furniture"].append(data["furniture"])


def create_zarr_dataset(in_dir, out_dir):
    """
    Create a Zarr dataset from multiple pickle files in a directory.
    """
    file_paths = glob(f"{in_dir}/**/*.pkl", recursive=True)
    print(f"Number of trajectories: {len(file_paths)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load one file to initialize the Zarr store
    with open(file_paths[0], "rb") as f:
        initial_data = pickle.load(f)

    z = initialize_zarr_store(out_dir, initial_data)

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
    parser.add_argument("--highres", action="store_true")
    parser.add_argument("--lowres", action="store_true")

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    assert not (args.highres and args.lowres), "Cannot have both highres and lowres"

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = os.environ.get("FURNITURE_DATA_DIR", "data")
    data_base_path_in = Path(os.environ.get("FURNITURE_DATA_DIR_RAW", data_base_path))
    data_base_path_out = Path(
        os.environ.get("FURNITURE_DATA_DIR_PROCESSED", data_base_path)
    )

    obs_in_path = args.obs_in
    obs_out_path = args.obs_out

    if args.highres:
        obs_in_path = obs_in_path + "_highres"
        obs_out_path = obs_out_path + "_highres"

    if args.lowres:
        obs_in_path = obs_in_path + "_small"
        obs_out_path = obs_out_path + "_small"

    raw_data_path = data_base_path_in / "raw" / args.env / obs_in_path / args.furniture
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
    print(f"Output path: {output_path}")

    if args.obs_out == "feature":
        process_demos_to_feature(
            raw_data_path,
            output_path,
            encoder,
            batch_size=args.batch_size,
        )
    elif args.obs_out == "image":
        create_zarr_dataset(raw_data_path, output_path)
