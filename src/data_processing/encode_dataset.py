import argparse
from typing import List, Union
import numpy as np
from tqdm import trange, tqdm
from src.models import get_encoder
import torch
from datetime import datetime
from ipdb import set_trace as bp
import zarr

from src.common.tasks import simple_task_descriptions, furniture2idx, idx2furniture
from src.common.files import get_processed_paths
from src.common.vision import FrontCameraTransform, WristCameraTransform


# Create the transforms
camera1_transform = FrontCameraTransform(mode="eval")
camera2_transform = WristCameraTransform(mode="eval")


# === Image Zarr to feature Zarr ===
@torch.no_grad()
def encode_numpy_batch(
    encoder,
    image1: np.ndarray,
    image2: np.ndarray,
    lang: Union[List[str], None] = None,
):
    # Move images to same device as encoder
    image1 = torch.from_numpy(image1).to(device)
    image2 = torch.from_numpy(image2).to(device)

    # Move the channel to the front (B * obs_horizon, H, W, C) -> (B * obs_horizon, C, H, W)
    image1 = image1.permute(0, 3, 1, 2)
    image2 = image2.permute(0, 3, 1, 2)

    # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
    image1 = camera1_transform(image1)
    image2 = camera2_transform(image2)

    # Place the channel back to the end (B * obs_horizon, C, 224, 224) -> (B * obs_horizon, 224, 224, C)
    image1 = image1.permute(0, 2, 3, 1)
    image2 = image2.permute(0, 2, 3, 1)

    if lang is not None:
        image1 = encoder(image1, lang=lang).cpu().numpy()
        image2 = encoder(image2, lang=lang).cpu().numpy()
    else:
        image1 = encoder(image1).cpu().numpy()
        image2 = encoder(image2).cpu().numpy()

    return image1, image2


def process_zarr_to_feature(
    input_group,
    output_group,
    encoder,
    batch_size=256,
    use_language=False,
):
    # Open in append mode to read data and be able to write back the features
    episode_ends = input_group["episode_ends"]
    chunksize = input_group["color_image1"].chunks[0]
    print(
        f"Number of episodes: {len(episode_ends)}, chunksize: {chunksize}, batch_size: {batch_size}"
    )

    from numcodecs import blosc

    blosc.use_threads = True
    blosc.set_nthreads(32)

    color_image1 = np.zeros(input_group["color_image1"].shape, dtype=np.uint8)
    color_image2 = np.zeros(input_group["color_image2"].shape, dtype=np.uint8)

    # Load images into memory
    for i in trange(
        0, len(color_image1), chunksize, desc="Loading images", leave=False
    ):
        slice_end = min(i + chunksize, len(color_image1))

        color_image1[i:slice_end] = input_group["color_image1"][i:slice_end]
        color_image2[i:slice_end] = input_group["color_image2"][i:slice_end]

    # Load other data into memory
    furniture = input_group["furniture"]
    furniture_idxs = np.zeros(color_image1.shape[0], dtype=np.uint8)

    prev_idx = 0
    for i, f in zip(episode_ends, furniture):
        furniture_idxs[prev_idx:i] = furniture2idx[f]

    encoding_dim = encoder.encoding_dim

    # Process images and create features
    features1 = np.zeros((len(color_image1), encoding_dim), dtype=np.float32)
    features2 = np.zeros((len(color_image2), encoding_dim), dtype=np.float32)

    for i in trange(
        0, len(color_image1), batch_size, desc="Encoding images", leave=False
    ):
        slice_end = min(i + batch_size, len(color_image1))

        language = None
        if use_language:
            # Use only the first simeple task description for now
            language = [
                simple_task_descriptions[idx2furniture[f]][0]
                for f in furniture_idxs[i:slice_end]
            ]

        features1[i:slice_end], features2[i:slice_end] = encode_numpy_batch(
            encoder,
            color_image1[i:slice_end],
            color_image2[i:slice_end],
            lang=language,
        )

    # Add a new group to the Zarr store all features should be stored under the key "feature"
    # where the the two features are stored under the key of the encoder name
    # then "feature1" and "feature2"
    output_group.array("feature1", features1)
    output_group.array("feature2", features2)

    # # Add time created with timezone info
    output_group.attrs["time_created"] = datetime.now().astimezone().isoformat()
    output_group.attrs["noop_threshold"] = input_group.attrs["noop_threshold"]
    output_group.attrs["encoder"] = encoder.__class__.__name__
    output_group.attrs["encoding_dim"] = encoding_dim
    output_group.attrs["use_language"] = use_language


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-c", type=str, required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)

    parser.add_argument("--environment", "-e", default="sim", type=str, nargs="+")
    parser.add_argument("--task", "-t", default=None, type=str, nargs="+")
    parser.add_argument("--randomness", "-r", type=str, default=None, nargs="+")
    parser.add_argument("--demo-source", "-s", type=str, default=None, nargs="+")
    parser.add_argument("--demo-outcome", "-o", type=str, default="success", nargs="+")

    parser.add_argument("--use-language", "-l", action="store_true")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{args.gpu_id}")

    assert args.use_language is False, "Language not supported yet"
    assert args.encoder is not None, "Must specify encoder when using feature obs"
    assert (
        not args.use_language or args.encoder == "voltron"
    ), "Only voltron supports language"
    assert (not args.overwrite) and (
        not args.skip_existing
    ), "Ambiguous to both skip and overwrite"

    encoder = get_encoder(args.encoder, freeze=True, device=device)
    encoder.eval()

    new_group_name = f"feature/{args.encoder}"

    paths = get_processed_paths(
        environment=args.environment,
        task=args.task,
        demo_source=args.demo_source,
        randomness=args.randomness,
        demo_outcome=args.demo_outcome,
    )

    for path in tqdm(paths, desc="Processing datasets"):
        input_group = zarr.open(path, mode="a")

        # Check if the group already exists
        exists = new_group_name in input_group
        if exists:
            # If it exists, we skip it if that flag is set
            if args.skip_existing:
                print(f"Skipping {new_group_name} because it already exists.")
                continue
            # Otherwise we delete it if the overwrite flag is set
            elif not args.overwrite:
                raise ValueError(
                    f"Group {new_group_name} already exists. Use --overwrite to overwrite."
                )

        output_group = input_group.create_group(
            new_group_name, overwrite=args.overwrite
        )

        process_zarr_to_feature(
            input_group,
            output_group,
            encoder,
            batch_size=args.batch_size,
            use_language=args.use_language,
        )
