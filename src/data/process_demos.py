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
from furniture_bench.robot.robot_state import filter_and_concat_robot_state

from ipdb import set_trace as bp

device = torch.device("cuda:1")


def process_demos_to_feature(input_path, output_path, encoder, batch_size=256):
    file_paths = glob(f"{input_path}/**/*.pkl", recursive=True)
    n_state_action_pairs = len(file_paths)

    print(f"Number of trajectories: {n_state_action_pairs}")

    observations = []
    actions = []
    episode_ends = []

    end_index = 0
    for path in tqdm(file_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        actions += data["actions"]
        end_index += len(data["actions"])
        obs = data["observations"]
        robot_state = np.stack(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs]
        )

        imgs1 = torch.from_numpy(np.stack([o["color_image1"] for o in obs])).to(device)
        imgs2 = torch.from_numpy(np.stack([o["color_image2"] for o in obs])).to(device)

        def process_in_batches(tensor, encoder):
            n = len(tensor)
            features_list = []

            for i in range(0, n, batch_size):
                batch = tensor[i : i + batch_size]
                features = encoder(batch)
                features_list.append(features)

            return torch.cat(features_list, dim=0)

        features1 = process_in_batches(imgs1, encoder)
        features2 = process_in_batches(imgs2, encoder)

        # Concatenate the robot state and the image features
        obs = np.concatenate(
            [robot_state, features1.cpu().numpy(), features2.cpu().numpy()], axis=-1
        )

        # Extend the observations list with the current trajectory.
        observations += obs.tolist()

        # Add the end index to the overall list.
        episode_ends.append(end_index)

    # Convert the lists to numpy arrays.
    observations = np.array(observations)
    actions = np.array(actions)
    episode_ends = np.array(episode_ends)

    # Save the data to a zarr file.
    zarr.save(
        f"{output_path}/data.zarr",
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument("--obs-in", "-i", type=str)
    parser.add_argument("--obs-out", "-o", type=str)
    parser.add_argument("--encoder", "-c", type=str)
    parser.add_argument("--furniture", "-f", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=256)

    args = parser.parse_args()

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = Path(os.environ.get("FURNITURE_DATA_PATH", "data"))

    raw_data_path = data_base_path / "raw" / args.env / args.obs_in / args.furniture
    output_path = (
        data_base_path / "processed" / args.env / args.obs_out / args.furniture
    )

    encoder = None
    if args.encoder is not None:
        output_path = output_path / args.encoder
        encoder = get_encoder(args.encoder, freeze=True, device=device)

    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {output_path}")

    process_demos_to_feature(
        raw_data_path, output_path, encoder, batch_size=args.batch_size
    )
