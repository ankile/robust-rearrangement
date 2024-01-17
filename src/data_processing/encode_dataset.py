from pathlib import Path
import pickle
from glob import glob
import argparse
import os
import numpy as np
import zarr
from tqdm import tqdm, trange
from src.models.vision import get_encoder
import torch
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from datetime import datetime

from ipdb import set_trace as bp


# === Image Zarr to feature Zarr ===
@torch.no_grad()
def encode_numpy_batch(buffer, encoder):
    # Move buffer to same device as encoder
    tensor = torch.from_numpy(buffer).to(encoder.device)
    return encoder(tensor).cpu().numpy()


def process_zarr_to_feature(zarr_input_path, zarr_output_path, encoder, batch_size=256):
    zarr_group = zarr.open(zarr_input_path, mode="r")
    episode_ends = zarr_group["episode_ends"]
    print(f"Number of episodes: {len(episode_ends)}")

    color_image1 = zarr_group["color_image1"]
    color_image2 = zarr_group["color_image2"]

    # Assuming other data like actions, rewards, etc. are also stored in the zarr file
    action = zarr_group["action"]
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

    # Add time created with timezone info
    output_group.attrs["time_created"] = datetime.now().astimezone().isoformat()
    output_group.attrs["noop_threshold"] = zarr_group.attrs["noop_threshold"]
    output_group.attrs["encoder"] = encoder.__class__.__name__


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
    parser.add_argument("--obs-in", "-i", type=str)
    parser.add_argument("--obs-out", "-o", type=str)
    parser.add_argument("--encoder", "-c", default=None, type=str)
    parser.add_argument("--furniture", "-f", type=str, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--randomness", "-r", type=str, default=None)

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path_in = Path(os.environ["DATA_DIR_RAW"])
    data_base_path_out = Path(os.environ["DATA_DIR_PROCESSED"])

    obs_out_path = args.obs_out

    raw_data_path = data_base_path_in / "raw" / args.env / args.source
    output_path = data_base_path_out / "processed" / args.env / obs_out_path

    encoder = None
    if args.encoder is not None:
        output_path = output_path / args.encoder
        encoder = get_encoder(args.encoder, freeze=True, device=device)
        encoder.eval()

    if args.furniture is not None:
        raw_data_path = raw_data_path / args.furniture
        output_path = output_path / args.furniture

    if args.randomness is not None:
        assert (
            args.furniture is not None
        ), "Must specify furniture when using randomness"
        assert args.randomness in ["low", "med", "high"], "Invalid randomness level"
        raw_data_path = raw_data_path / args.randomness
        output_path = output_path / args.randomness

    print(f"Raw data path: {raw_data_path}")

    output_path = output_path / "data.zarr"
    print(f"Output path: {output_path}")
    process_zarr_to_feature(
        f"/data/scratch/ankile/furniture-data/data/processed/sim/image/{args.furniture}/data_batch_32.zarr",
        output_path,
        encoder,
        batch_size=args.batch_size,
    )
