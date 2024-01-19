from pathlib import Path
import argparse
import os
import numpy as np
import zarr
from tqdm import trange
from src.models.vision import get_encoder
import torch
from datetime import datetime
from src.common.tasks import simple_task_descriptions


# === Image Zarr to feature Zarr ===
@torch.no_grad()
def encode_numpy_batch(encoder, buffer, lang=None):
    # Move buffer to same device as encoder
    tensor = torch.from_numpy(buffer).to(device)
    if lang is not None:
        return encoder(tensor, lang=lang).cpu().numpy()

    return encoder(tensor).cpu().numpy()


def process_zarr_to_feature(
    zarr_input_path,
    zarr_output_path,
    encoder,
    batch_size=256,
    use_language=False,
):
    zarr_group = zarr.open(zarr_input_path, mode="r")
    episode_ends = zarr_group["episode_ends"]
    print(f"Number of episodes: {len(episode_ends)}")

    from numcodecs import blosc

    blosc.use_threads = True
    blosc.set_nthreads(32)

    color_image1 = np.zeros(zarr_group["color_image1"].shape, dtype=np.uint8)
    color_image2 = np.zeros(zarr_group["color_image2"].shape, dtype=np.uint8)

    # Load images into memory
    for i in trange(0, len(color_image1), 10_000, desc="Loading images"):
        slice_end = min(i + 10_000, len(color_image1))

        color_image1[i:slice_end] = zarr_group["color_image1"][i:slice_end]
        color_image2[i:slice_end] = zarr_group["color_image2"][i:slice_end]

    # Assuming other data like actions, rewards, etc. are also stored in the zarr file
    action = zarr_group["action"]
    furniture = zarr_group["furniture"]
    reward = zarr_group["reward"]
    robot_state = zarr_group["robot_state"]
    skills = zarr_group["skill"]

    encoding_dim = encoder.encoding_dim

    # Process images and create features
    features1 = np.zeros((len(color_image1), encoding_dim), dtype=np.float32)
    features2 = np.zeros((len(color_image2), encoding_dim), dtype=np.float32)

    for i in trange(0, len(color_image1), batch_size):
        slice_end = min(i + batch_size, len(color_image1))

        language = None
        if use_language:
            # Load language
            language = zarr_group["language"][i:slice_end]
            # Use only the first simeple task description for now
            language = [simple_task_descriptions[l][0] for l in language]

        features1[i:slice_end] = encode_numpy_batch(
            encoder, color_image1[i:slice_end], lang=language
        )
        features2[i:slice_end] = encode_numpy_batch(
            encoder, color_image2[i:slice_end], lang=language
        )

    # Create a new Zarr file for output
    output_group = zarr.open(zarr_output_path, mode="w")
    output_group.array("feature1", features1)
    output_group.array("feature2", features2)

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
    output_group.attrs["encoding_dim"] = encoding_dim
    output_group.attrs["use_language"] = use_language


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-c", default=None, type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--zarr-path", "-z", type=str, required=True)
    parser.add_argument("--use-language", "-l", action="store_true")

    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    assert args.encoder is not None, "Must specify encoder when using feature obs"
    assert (
        not args.use_language or args.encoder == "voltron"
    ), "Only voltron supports language"

    input_path = Path(args.zarr_path)

    path_parts = list(input_path.parts)

    # Find index of "image" in the path
    image_index = path_parts.index("image")

    # Insert "feature" instead of "image" and the encoder name after "feature"
    path_parts[image_index] = "feature"
    path_parts.insert(
        image_index + 1, args.encoder + ("_lang" if args.use_language else "")
    )

    # Turn it back into a path
    output_path = Path(os.path.join(*path_parts))

    encoder = get_encoder(args.encoder, freeze=True, device=device)
    encoder.eval()

    print(f"Raw data path: {args.zarr_path}")
    print(f"Output path: {output_path}")

    process_zarr_to_feature(
        args.zarr_path,
        output_path,
        encoder,
        batch_size=args.batch_size,
        use_language=args.use_language,
    )
