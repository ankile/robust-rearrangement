import cv2
import numpy as np
from src.common.files import get_raw_paths
from src.visualization.render_mp4 import (
    pickle_data,
    unpickle_data,
    data_to_video,
)
from src.common.tasks import task_phases

from ipdb import set_trace as bp

annotate = "success"


# 1. Load the video
pkl_paths = get_raw_paths(
    environment="sim",
    demo_source="teleop",
    demo_outcome=annotate,
    randomness="low",
    task="round_table",
)

for i, pkl_path in enumerate(pkl_paths, start=0):

    # Cut off everything after `raw` in the path
    path_name = "/".join(pkl_path.parts[pkl_path.parts.index("raw") :])

    print(f"Processing {i+1}/{len(pkl_paths)}: {path_name}")

    data = unpickle_data(pkl_path)

    # Check if this data has already been annotated
    # TODO make this be adaptive
    if ("augment_states" in data) or ("failure_idx" in data):
        print(f"Data {path_name} has already been annotated")
        continue

    video = data_to_video(data)
    total_frames = video.shape[0]

    # Initialize the annotation array
    annotations = []
    failure_idx = None

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 3 * 640, 3 * 240)

    current_frame = 0
    while True:
        frame = video[current_frame]
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):  # Press 'q' to exit
            raise SystemExit
        elif key == ord("s"):
            # Save the annotations to a file
            # Convert the indexes to a list of 0s and 1s
            data["augment_states"] = np.zeros(total_frames)
            for annotation in annotations:
                data["augment_states"][annotation] = 1

            if failure_idx is not None:
                data["failure_idx"] = failure_idx

            pickle_data(data, pkl_path)
            print(f"Saved annotations to {pkl_path}")

            break

        elif key == ord(" "):  # Press space to mark the frame
            annotations.append(current_frame)
            print(f"Marked frame {current_frame}, frames marked: {annotations}")
        elif key == ord("f"):
            # Mark the frame as a failure
            failure_idx = current_frame
            print(f"Marked frame {current_frame} as failure")

        elif key == ord("u"):
            # Undo the last marking
            if len(annotations) > 0:
                removed = annotations.pop()
                print(f"Removed frame {removed}, frames marked: {annotations}")
        elif key == ord("k"):
            # Jump 1 frame forward
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord("l"):
            # Jump 10 frames forward
            current_frame = min(current_frame + 10, total_frames - 1)
        elif key == ord("j"):
            current_frame = max(current_frame - 1, 0)
        elif key == ord("h"):
            current_frame = max(current_frame - 10, 0)

    cv2.destroyAllWindows()

    # annotations now contains your markings
    print(annotations)
