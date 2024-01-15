from datetime import datetime
import imageio
from io import BytesIO
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import numpy as np


def create_mp4(np_images, filename, fps=10):
    # duration = 1000 / fps
    with imageio.get_writer(filename, fps=fps) as writer:
        for img in tqdm(np_images):
            writer.append_data(img)
    print(f"File saved as {filename}")


def mp4_from_pickle(pickle_path, filename=None):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    ims1 = np.array([o["color_image1"] for o in data["observations"]])
    ims2 = np.array([o["color_image2"] for o in data["observations"]])
    ims = np.concatenate([ims1, ims2], axis=2)

    create_mp4(ims, filename)


def create_in_memory_mp4(np_images, fps=10):
    output = BytesIO()

    writer_options = {"fps": fps}
    writer_options["format"] = "mp4"
    writer_options["codec"] = "libx264"
    writer_options["pixelformat"] = "yuv420p"

    with imageio.get_writer(output, **writer_options) as writer:
        for img in np_images:
            writer.append_data(img)

    output.seek(0)
    return output


def render_mp4(ims1, ims2, filename=None):
    # Initialize plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Function to update plot
    def update(num):
        ax1.clear()
        ax2.clear()
        ax1.axis("off")
        ax2.axis("off")

        img_array1 = ims1[num]
        if isinstance(img_array1, torch.Tensor):
            img_array1 = img_array1.squeeze(0).cpu().numpy()

        img_array2 = ims2[num]
        if isinstance(img_array2, torch.Tensor):
            img_array2 = img_array2.squeeze(0).cpu().numpy()

        ax1.imshow(img_array1)
        ax2.imshow(img_array2)

    frame_indices = range(0, len(ims1), 1)

    framerate_hz = 10

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(frame_indices),
        interval=1000 // framerate_hz,
    )

    if not filename:
        filename = f"render-{datetime.now()}.mp4"

    ani.save(filename)
