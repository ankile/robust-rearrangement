import argparse
import os
from pathlib import Path

from furniture_bench.data.data_collector import DataCollector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resize-sim-img", action="store_true")
    parser.add_argument("--furniture", type=str, default="one_leg")

    args = parser.parse_args()

    BASE = Path(os.environ.get("ROLLOUT_SAVE_DIR", "data"))

    # TODO: Consider what we do with images of full size and if that's needed
    # For now, we assume that images are stored in 224x224 and we know that as`image`
    # # Add the suffix _highres if we are not resizing images in or after simulation
    # if not args.resize_img_after_sim and not args.small_sim_img_size:
    #     obs_type = obs_type + "_highres"

    data_path = BASE / "raw" / "sim" / args.furniture / args.randomness
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {data_path}")

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=args.furniture,
        device_interface=None,
        headless=True,
        draw_marker=True,
        manual_label=False,
        scripted=True,
        randomness=args.randomness,
        pkl_only=True,
        save_failure=False,
        num_demos=args.num_demos,
        resize_sim_img=args.resize_sim_img,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
    )

    collector.collect()
