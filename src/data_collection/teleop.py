import argparse
import random

import furniture_bench
from furniture_bench.device import make_device
from furniture_bench.config import config

from src.data_collection.data_collector_sm import DataCollectorSpaceMouse
from src.data_collection.keyboard_interface import KeyboardInterface
from src.common.files import trajectory_save_dir


def main():
    parser = argparse.ArgumentParser(description="Collect IL data")
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=list(config["furniture"].keys()),
        required=True,
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help="Type of low level controller to use.",
        choices=["osc", "diffik"],
        default="osc",
    )
    parser.add_argument(
        "--draw-marker",
        action="store_true",
        help="If set, will draw an AprilTag marker on the furniture",
    )
    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        help="Directory to resume trajectories from",
        default=None,
    )

    args = parser.parse_args()

    keyboard_device_interface = KeyboardInterface()
    keyboard_device_interface.print_usage()

    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source="teleop",
        randomness=args.randomness,
    )

    from pathlib import Path

    if args.resume_dir is not None:
        pickle_paths = list(Path(args.resume_dir).rglob("*.pkl*"))
        random.shuffle(pickle_paths)
        pickle_paths = pickle_paths[: args.num_demos]
        print("loaded num trajectories", len(pickle_paths))
    else:
        pickle_paths = None

    data_collector = DataCollectorSpaceMouse(
        is_sim=True,
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        headless=False,
        draw_marker=args.draw_marker,
        manual_label=True,
        resize_sim_img=False,
        scripted=False,
        randomness=args.randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
        compress_pickles=False,
        resume_trajectory_paths=pickle_paths,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
