import argparse
from pathlib import Path

from furniture_bench.data.data_collector import DataCollector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--num-demos", type=int, default=100)

    args = parser.parse_args()

    furniture = "one_leg"
    randomness = args.randomness

    BASE = Path(".")

    data_path = BASE / "scripted_sim_demo_state" / furniture / randomness

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=furniture,
        device_interface=None,
        headless=True,
        draw_marker=True,
        manual_label=True,
        scripted=True,
        randomness=randomness,
        gpu_id=1,
        pkl_only=True,
        save_failure=False,
        num_demos=args.num_demos,
        verbose=False,
        show_pbar=True,
        use_state=True,
    )

    collector.collect()
