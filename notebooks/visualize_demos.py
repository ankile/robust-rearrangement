import os
from src.common.files import get_processed_paths
from src.visualization.render_mp4 import create_mp4_jupyter, create_mp4
import numpy as np

import zarr

os.environ["DATA_DIR_PROCESSED"] = "/home/gridsan/groups/furniture-diffusion/rr-data/"
paths = get_processed_paths(
    controller="diffik",
    domain="sim",
    task=["one_leg", "one_leg_simple"],
    demo_source=["teleop", "rollout"],
    randomness=["med", "med_perturb", "low"],
    demo_outcome="success",
)
for path in paths:
    print(path)

for path in paths:
    print("\n" * 10)
    print(f"Processing {path}")

    z = zarr.open(path)

    ep_ends = z["episode_ends"][:-1]
    imgs2 = z["color_image2"][:]
    img_episodes = np.split(imgs2, ep_ends, axis=0)

    for i, img_episode in enumerate(img_episodes):
        # create_mp4_jupyter(img_episode, f"{path}_episode_{i}.mp4", fps=50)
        create_mp4(img_episode, f"{path}_episode_{i}.mp4", fps=50)
