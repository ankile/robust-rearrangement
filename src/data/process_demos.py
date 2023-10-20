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


# def process_demos_to_feature(
#     input_path,
#     output_path,
#     encoder,
#     batch_size=256,
#     separate=False,
# ):
#     file_paths = glob(f"{input_path}/**/*.pkl", recursive=True)
#     print(f"Number of trajectories: {len(file_paths)}")

#     # Storage for final output
#     robot_states, features1, features2 = [], [], []
#     actions, episode_ends, rewards, skills, furniture = [], [], [], [], []

#     # Temporary buffers for processing
#     robot_state_buffer, img_buffer1, img_buffer2 = [], [], []
#     robot_state_list, features_list1, features_list2 = [], [], []
#     end_index = 0

#     def process_buffer(buffer, encoder):
#         tensor = torch.stack(buffer).to(device)
#         features = encoder(tensor)
#         return features.cpu().numpy()

#     for path in tqdm(file_paths):
#         with open(path, "rb") as f:
#             data = pickle.load(f)

#         # These grow with the number of observations in each trajectory
#         actions += data["actions"]
#         rewards += data["rewards"]
#         skills += data["skills"]

#         # These grow by one per trajectory
#         end_index += len(data["actions"])
#         episode_ends.append(end_index)
#         furniture.append(data["furniture"])

#         # Extract the robot state and image features
#         obs = data["observations"]

#         robot_state_buffer += [filter_and_concat_robot_state(o["robot_state"]) for o in obs]
#         img_buffer1 += [torch.from_numpy(o["color_image1"]) for o in obs]
#         img_buffer2 += [torch.from_numpy(o["color_image2"]) for o in obs]

#         while len(img_buffer1) >= batch_size:
#             feature_batch1 = process_buffer(img_buffer1[:batch_size], encoder)
#             feature_batch2 = process_buffer(img_buffer2[:batch_size], encoder)
#             robot_state_list.append(np.stack(robot_state_buffer[:batch_size]))
#             features_list1.append(feature_batch1)
#             features_list2.append(feature_batch2)
#             robot_state_buffer, img_buffer1, img_buffer2 = (
#                 robot_state_buffer[batch_size:],
#                 img_buffer1[batch_size:],
#                 img_buffer2[batch_size:],
#             )

#         # Concatenate the robot state and image features
#         traj_features1 = np.concatenate(features_list1, axis=0) if features_list1 else np.array([])
#         traj_features2 = np.concatenate(features_list2, axis=0) if features_list2 else np.array([])
#         if traj_features1.size == 0 or traj_features2.size == 0:
#             # Skip this trajectory because it was not long enough to fill a batch
#             continue

#         robot_state_array = np.concatenate(robot_state_list, axis=0)

#         # observations += obs.tolist()
#         robot_states += robot_state_array.tolist()
#         features1 += traj_features1.tolist()
#         features2 += traj_features2.tolist()

#         # Reset the features_list for the next iteration
#         robot_state_list, features_list1, features_list2 = [], [], []

#     # Don't forget to process any remaining items in the buffer
#     if len(img_buffer1) > 0:
#         feature_batch1 = process_buffer(img_buffer1, encoder)
#         feature_batch2 = process_buffer(img_buffer2, encoder)
#         features_list1.append(feature_batch1)
#         features_list2.append(feature_batch2)

#     # Convert to numpy arrays
#     if separate:
#         obs_dict = {
#             "robot_state": np.array(robot_states, dtype=np.float32),
#             "feature1": np.array(features1, dtype=np.float32),
#             "feature2": np.array(features2, dtype=np.float32),
#         }
#     else:
#         observations = np.concatenate([robot_states, features1, features2], axis=-1).tolist()
#         obs_dict = {"observations": np.array(observations, dtype=np.float32)}

#     actions = np.array(actions, dtype=np.float32)
#     episode_ends = np.array(episode_ends, dtype=np.uint32)

#     # Save to file
#     output_path.mkdir(parents=True, exist_ok=True)
#     zarr.save(
#         str(output_path / "data.zarr"),
#         **obs_dict,
#         action=actions,
#         episode_ends=episode_ends,
#         rewards=np.array(rewards, dtype=np.float32),
#         skills=np.array(skills, dtype=np.float32),
#         furniture=furniture,
#         time_created=np.datetime64("now"),
#     )


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


def process_demos_to_image(in_dir, out_dir):
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

    # Save to file
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr.save(
        str(out_dir / "data.zarr"),
        robot_state=np.array(robot_state, dtype=np.float32),
        color_image1=np.array(color_image1, dtype=np.uint8),
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

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.obs_out == "feature":
        assert args.encoder is not None, "Must specify encoder when using feature obs"

    data_base_path = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))

    obs_out_path = args.obs_out + ("_separate" if args.features_separate else "")

    raw_data_path = data_base_path / "raw" / args.env / args.obs_in / args.furniture
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
