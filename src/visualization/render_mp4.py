import gzip
import lzma
import pickle
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from tqdm import tqdm


def format_speedup(fps):
    speedup = fps / 10
    if speedup.is_integer():
        return f"{int(speedup)}x"
    else:
        return f"{speedup:.1f}x"


def annotate_frames_with_speed(frames: np.ndarray, fps: int) -> np.ndarray:
    assert (
        len(frames.shape) == 4
    ), "Frames must be a 4D array (batch, height, width, channels)"

    frames = np.array(
        [
            cv2.putText(
                frame,
                format_speedup(fps),
                (
                    frame.shape[1] - 55,
                    frame.shape[0] - 15,
                ),  # Adjusted position of the text
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                1,  # Font scale (doubled)
                (255, 255, 255),  # Text color (white)
                2,  # Text thickness (doubled)
                cv2.LINE_AA,  # Line type for better rendering
            )
            for frame in frames
        ]
    )

    return frames


def create_mp4_jupyter(
    np_images,
    filename,
    fps=10,
    speed_annotation=False,
):
    if speed_annotation:
        np_images = annotate_frames_with_speed(np_images, fps)

    with imageio.get_writer(filename, fps=fps) as writer:
        for img in np_images:
            writer.append_data(img)
    print(f"File saved as {filename}")
    # Display the video in the Jupyter Notebook
    video_tag = f'<video controls src="{filename}" width="640" height="480"></video>'

    return display(HTML(video_tag))


def mp4_from_pickle_jupyter(
    pickle_path: Union[str, Path],
    filename=None,
    fps=10,
    speed_annotation=False,
    cameras=[1, 2],
):
    ims = extract_numpy_frames(pickle_path, cameras)

    if speed_annotation:
        ims = annotate_frames_with_speed(ims, fps)

    return create_mp4_jupyter(ims, filename, fps)


def mp4_from_data_dict_jupyter(data_dict: dict, filename, fps=10):
    ims = data_to_video(data_dict)
    return create_mp4_jupyter(ims, filename, fps)


def create_mp4(
    np_images: np.ndarray, filename: Union[str, Path], fps=10, verbose=False
) -> None:
    # duration = 1000 / fps
    with imageio.get_writer(filename, fps=fps) as writer:
        for img in tqdm(np_images, disable=not verbose):
            writer.append_data(img)

    if verbose:
        print(f"File saved as {filename}")


def mp4_from_pickle(pickle_path, filename=None, fps=10, cameras=[1, 2]):
    ims = extract_numpy_frames(pickle_path, cameras)
    create_mp4(ims, filename, fps)


def extract_numpy_frames(pickle_path: Union[str, Path], cameras=[1, 2]) -> np.ndarray:
    data = unpickle_data(pickle_path)
    ims = data_to_video(data, cameras)

    return ims


def data_to_video(data: dict, cameras=[1, 2]) -> np.ndarray:
    ims = []

    for camera in cameras:
        ims.append(np.array([o[f"color_image{camera}"] for o in data["observations"]]))

    ims = np.concatenate(ims, axis=2)

    return ims


def unpickle_data(pickle_path: Union[Path, str]):
    pickle_path = Path(pickle_path)
    if pickle_path.suffix == ".gz":
        with gzip.open(pickle_path, "rb") as f:
            return pickle.load(f)
    elif pickle_path.suffix == ".pkl":
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    elif pickle_path.suffix == ".xz":
        with lzma.open(pickle_path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Invalid file extension: {pickle_path.suffix}")


def pickle_data(data, pickle_path: Union[Path, str]):
    pickle_path = Path(pickle_path)
    if pickle_path.suffix == ".gz":
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    elif pickle_path.suffix == ".pkl":
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    elif pickle_path.suffix == ".xz":
        with lzma.open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Invalid file extension: {pickle_path.suffix}")


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
