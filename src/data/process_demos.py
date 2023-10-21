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


def process_buffer(buffer, encoder):
    tensor = torch.stack(buffer).to(device)
    return encoder(tensor).cpu().numpy()


def process_demos_to_feature(input_path, output_path, encoder, batch_size=256, separate=False):
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
        robot_state_buffer = [filter_and_concat_robot_state(o["robot_state"]) for o in obs]
        img_buffer1 = [torch.from_numpy(o["color_image1"]) for o in obs]
        img_buffer2 = [torch.from_numpy(o["color_image2"]) for o in obs]

        for i in range(0, len(img_buffer1), batch_size):
            slice_end = min(i + batch_size, len(img_buffer1))
            robot_states.extend(robot_state_buffer[i:slice_end])
            features1.extend(process_buffer(img_buffer1[i:slice_end], encoder).tolist())
            features2.extend(process_buffer(img_buffer2[i:slice_end], encoder).tolist())

    if separate:
        obs_dict = {
            "robot_state": np.array(robot_states, dtype=np.float32),
            "feature1": np.array(features1, dtype=np.float32),
            "feature2": np.array(features2, dtype=np.float32),
        }
    else:
        observations = np.concatenate([robot_states, features1, features2], axis=-1)
        obs_dict = {"observations": observations.astype(np.float32)}

    output_path.mkdir(parents=True, exist_ok=True)
    zarr.save(
        str(output_path / "data.zarr"),
        action=np.array(actions, dtype=np.float32),
        episode_ends=np.array(episode_ends, dtype=np.uint32),
        rewards=np.array(rewards, dtype=np.float32),
        skills=np.array(skills, dtype=np.float32),
        furniture=furniture,
        time_created=np.datetime64("now"),
        **obs_dict,
    )


def process_demos_to_image(in_dir, out_dir, highres=False):
    file_paths = glob(f"{in_dir}/**/*.pkl", recursive=True)
    print(f"Number of trajectories: {len(file_paths)}")

    (
        robot_state,
        color_image1,
        color_image2,
        actions,
        rewards,
        skills,
        episode_ends,
        furniture,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    end_index = 0

    for path in tqdm(file_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        obs = data["observations"]
        robot_state += [filter_and_concat_robot_state(o["robot_state"]) for o in obs]
        color_image1 += [o["color_image1"] for o in obs]
        color_image2 += [o["color_image2"] for o in obs]

        actions += data["actions"]
        rewards += data["rewards"]
        skills += data["skills"]

        end_index += len(data["actions"])
        episode_ends.append(end_index)
        furniture.append(data["furniture"])

    color_image1 = (np.array(color_image1, dtype=np.uint8),)
    color_image2 = (np.array(color_image2, dtype=np.uint8),)

    if highres:
        color_image1 = F.resize(torch.from_numpy(color_image1), (228, 405)).numpy()
        color_image2 = F.resize(torch.from_numpy(color_image2), (228, 405)).numpy()

    # Save to file
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr.save(
        str(out_dir / "data.zarr"),
        robot_state=np.array(robot_state, dtype=np.float32),
        color_image1=color_image1,
        color_image2=np.array(color_image2, dtype=np.uint8),
        action=np.array(actions, dtype=np.float32),
        reward=np.array(rewards, dtype=np.float32),
        skills=np.array(skills, dtype=np.float32),
        episode_ends=np.array(episode_ends, dtype=np.uint32),
        furniture=furniture,
        time_created=np.datetime64("now"),
    )


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
    parser.add_argument("--features-separate", "-s", action="store_true")
    parser.add_argument("--highres", action="store_true")

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))

    obs_out_path = args.obs_out + ("_separate" if args.features_separate else "")
    obs_out_path = obs_out_path + ("_highres" if args.highres else "")
    obs_in_path = args.obs_in + ("_highres" if args.highres else "")

    raw_data_path = data_base_path / "raw" / args.env / obs_in_path / args.furniture
    output_path = data_base_path / "processed" / args.env / obs_out_path

    encoder = None
    if args.encoder is not None:
        output_path = output_path / args.encoder
        encoder = get_encoder(args.encoder, freeze=True, device=device)

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
            separate=args.features_separate,
        )
    elif args.obs_out == "image":
        process_demos_to_image(raw_data_path, output_path)
