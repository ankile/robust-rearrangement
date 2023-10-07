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


def process_demos_to_feature(input_path, output_path, encoder, batch_size=256):
    file_paths = glob(f"{input_path}/**/*.pkl", recursive=True)
    print(f"Number of trajectories: {len(file_paths)}")

    observations, actions, episode_ends = [], [], []
    robot_state_buffer, img_buffer1, img_buffer2 = [], [], []
    robot_state_list, features_list1, features_list2 = [], [], []
    end_index = 0

    def process_buffer(buffer, encoder):
        tensor = torch.stack(buffer).to(device)
        features = encoder(tensor)
        return features.cpu().numpy()

    for path in tqdm(file_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        actions += data["actions"]
        end_index += len(data["actions"])

        obs = data["observations"]

        robot_state_buffer += [
            filter_and_concat_robot_state(o["robot_state"]) for o in obs
        ]
        img_buffer1 += [torch.from_numpy(o["color_image1"]) for o in obs]
        img_buffer2 += [torch.from_numpy(o["color_image2"]) for o in obs]

        while len(img_buffer1) >= batch_size:
            features1 = process_buffer(img_buffer1[:batch_size], encoder)
            features2 = process_buffer(img_buffer2[:batch_size], encoder)
            robot_state_list.append(np.stack(robot_state_buffer[:batch_size]))
            features_list1.append(features1)
            features_list2.append(features2)
            robot_state_buffer, img_buffer1, img_buffer2 = (
                robot_state_buffer[batch_size:],
                img_buffer1[batch_size:],
                img_buffer2[batch_size:],
            )

        # Concatenate the robot state and image features
        features1 = (
            np.concatenate(features_list1, axis=0) if features_list1 else np.array([])
        )
        features2 = (
            np.concatenate(features_list2, axis=0) if features_list2 else np.array([])
        )
        if features1.size == 0 or features2.size == 0:
            continue  # Skip the current file as it has no valid features

        robot_state_array = np.concatenate(robot_state_list, axis=0)
        obs = np.concatenate([robot_state_array, features1, features2], axis=-1)

        observations += obs.tolist()
        episode_ends.append(end_index)

        # Reset the features_list for the next iteration
        robot_state_list, features_list1, features_list2 = [], [], []

    # Don't forget to process any remaining items in the buffer
    if len(img_buffer1) > 0:
        features1 = process_buffer(img_buffer1, encoder)
        features2 = process_buffer(img_buffer2, encoder)
        features_list1.append(features1)
        features_list2.append(features2)

    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)
    episode_ends = np.array(episode_ends)

    # Save to file
    output_path.mkdir(parents=True, exist_ok=True)
    zarr.save(
        str(output_path / "data.zarr"),
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
        time_created=np.datetime64("now"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str)
    parser.add_argument("--obs-in", "-i", type=str)
    parser.add_argument("--obs-out", "-o", type=str)
    parser.add_argument("--encoder", "-c", type=str)
    parser.add_argument("--furniture", "-f", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))

    raw_data_path = data_base_path / "raw" / args.env / args.obs_in / args.furniture
    output_path = data_base_path / "processed" / args.env / args.obs_out

    encoder = None
    if args.encoder is not None:
        output_path = output_path / args.encoder
        encoder = get_encoder(args.encoder, freeze=True, device=device)

    output_path = output_path / args.furniture

    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {output_path}")

    process_demos_to_feature(
        raw_data_path, output_path, encoder, batch_size=args.batch_size
    )
