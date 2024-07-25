# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import argparse
import time
import os
from pathlib import Path

from omni.isaac.kit import SimulationApp

# from furniture_bench.utils.pose import get_mat, rot_mat

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Force display off at all times.",
)
parser.add_argument("-f", "--furniture", type=str, required=True)

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        "width": 640,
        "height": 480,
    }
)

from omni.isaac.core.simulation_context import SimulationContext


def run_until_quit(simulation_app, world):
    import omni.appwindow
    import carb.input

    global RP_RUNNING, RP_PAUSE
    RP_RUNNING = True
    RP_PAUSE = False

    def on_kb_input(e):
        global RP_RUNNING, RP_PAUSE
        if e.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if e.input == carb.input.KeyboardInput.Q:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
            elif e.input == carb.input.KeyboardInput.P:
                print(f'[Run until quit] Caught "P" keypress, pausing')
                RP_PAUSE = True

    app_window = omni.appwindow.get_default_app_window()
    kb = app_window.get_keyboard()
    carb_input = carb.input.acquire_input_interface()
    _ = carb_input.subscribe_to_keyboard_events(kb, on_kb_input)

    while RP_RUNNING:
        # simulation_app.update()
        world.render()
        time.sleep(1 / 120.0)

        if RP_PAUSE:
            print(f"[Run until quit] Paused!")
            from IPython import embed

            embed()
            RP_PAUSE = False

    return


def main():

    sim = SimulationContext(physics_dt=0.0001, rendering_dt=0.0001, backend="torch")

    # Parts
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("omni.isaac.shapenet", True)
    manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    from omni.isaac.shapenet.shape import convert

    from src.sim2real.part_config_render import part_config_dict

    part_config = part_config_dict[args_cli.furniture]

    furniture_assets_usd_root_path = (
        Path(f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd")
        / args_cli.furniture
    )
    furniture_assets_usd_root_path.mkdir(parents=True, exist_ok=True)

    obj_root_path = (
        Path(f"{os.getenv('RARL_SOURCE_DIR')}/assets/furniture/mesh")
        / args_cli.furniture
    )
    assert os.path.exists(
        str(obj_root_path)
    ), f"Path to original .obj files {str(obj_root_path)} not found"

    obj_fname_list = [
        fn for fn in os.listdir(str(obj_root_path)) if fn.endswith(".obj")
    ]
    for i, fn in enumerate(obj_fname_list):
        obj_fn_full = str(obj_root_path / fn)
        usd_fn_full = str(furniture_assets_usd_root_path / part_config["usd_names"][i])

        print(f"Saving USD to {usd_fn_full}")
        asyncio.get_event_loop().run_until_complete(convert(obj_fn_full, usd_fn_full))

    run_until_quit(simulation_app=simulation_app, world=sim)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
