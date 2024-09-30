Steps for real world demo collection
- Set up a few things we use in the real world (our lab's "real deployment tools", `meshcat`, `polymetis`, `pyrealsense2`, etc.). Detailed setup instructions still needed here, but it's essentially using `conda` to install `polymetis`, using `pip` to install `meshcat` and `pyrealsense2`, and using `pip` to install some of our [RDT](https://github.com/anthonysimeonov/improbable_rdt) tools. 
- Set up the [Spacemouse](https://3dconnexion.com/us/product/spacemouse-wireless/). Packages to install + commands to run to make it work are below (borrowed from the [diffusion policy repo](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file)):
```
# Needed for spacemouse
pip install numpy termcolor atomics scipy
pip install git+https://github.com/cheng-chi/spnav
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```
- Run `teleop_sm.py`. Example usage:
```
python teleop_sm.py -p 6000 --save_dir teleop_data/one_leg_color --furniture one_leg
```
- `-p` indicates what port to use for `meshcat` visualization. Make sure you have run `meshcat-server` in a background terminal (after `pip install meshcat`), and that the port that gets printed out matches what you use with `-p`

Steps for real world eval
- Run `minimal.py`. Example usage:
```
python minimal.py -p 6000 --run-id real-one_leg-cotrain-2/paxnbwsu # -w _1199.pt
```
- `--run-id` corresponds to the `wandb` run you want to evaluate
- `-w` indicates which specific checkpoint you may want to test out (optional - by default it uses the `best_test_loss` or `best_success_rate` checkpoint)