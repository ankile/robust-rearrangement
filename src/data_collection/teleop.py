import argparse
import random

import furniture_bench
from furniture_bench.device import make_device
from furniture_bench.config import config

from pathlib import Path

from src.data_collection.data_collector_sm import DataCollectorSpaceMouse
from src.data_collection.keyboard_interface import KeyboardInterface
from furniture_bench.envs.initialization_mode import Randomness

from src.common.files import trajectory_save_dir
from src.gym import turn_off_april_tags

from ipdb import set_trace as bp


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
        required=True,
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
    parser.add_argument(
        "--sample-perturbations",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.draw_marker:
        turn_off_april_tags()

    keyboard_device_interface = KeyboardInterface()
    keyboard_device_interface.print_usage()

    # Ensure valid randomness
    randomness = Randomness.str_to_enum(args.randomness)

    data_path: Path = trajectory_save_dir(
        controller=args.ctrl_mode,
        domain="sim",
        task=args.furniture,
        demo_source="teleop",
        randomness=args.randomness + ("_perturb" if args.sample_perturbations else ""),
    )

    if args.resume_dir is not None:
        pickle_paths = list(Path(args.resume_dir).rglob("*.pkl*"))
        random.shuffle(pickle_paths)
        pickle_paths = pickle_paths[: args.num_demos]
        print("loaded num trajectories", len(pickle_paths))
    else:
        pickle_paths = None

    data_collector = DataCollectorSpaceMouse(
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        draw_marker=args.draw_marker,
        resize_sim_img=False,
        randomness=randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
        compress_pickles=False,
        resume_trajectory_paths=pickle_paths,
        sample_perturbations=args.sample_perturbations,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
