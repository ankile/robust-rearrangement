# From Imitation to Refinement


_**NOTE** (updated Sept 1, 2024): The repo is still under active development and we are working on making reproducing the experiments [in the paper](https://arxiv.org/pdf/2407.16677) straightforward and hosting and making available the demonstration data we collected for learning the imitation policies from._

_**Update Sept 20, 2024:**_ The data used to train the models in this project is now available in an [S3 bucket](https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/index.html). We also have a script to download data for the different tasks in the right places.

_**Update Sept 27, 2024:**_ Commands for starting the BC pre-training on the data is added, as well as started adding the model weights from the paper.

**Update Oct 7, 2024:** The model weights of pre-trained and fine-tuned models are now available for download. The commands for evaluating the fine-tuned models are the same as the pre-trained models. COmmands for starting the RL fine-tuning are added.


## Installation Instructions


### Install Conda

First, install Conda by following the instructions on the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) (here using Miniconda).

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

To activate the changes, restart your shell or run:

```bash
source ~/.bashrc
source ~/.zshrc
```

### Create a Conda Environment

Create a new Conda environment by running:

```bash
conda create -n rr python=3.8 -y
```

Activate the environment by running:

```bash
conda activate rr
```


### Install IsaacGym

Download the IsaacGym installer from the [IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running (also refer to the [FurnitureBench installlation instructions](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html#download-isaac-gym)):

- Click "Join now" and log into your NVIDIA account.
- Click "Member area".
- Read and check the box for the license agreement.
- Download and unzip `Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release`.

You can also download a copy of the file from our AWS S3 bucket for your convenience:

```bash
wget https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/packages/IsaacGym_Preview_4_Package.tar.gz
```

Once the zipped file is downloaded, move it to the desired location and unzip it by running:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```


Now, you can install the IsaacGym package by navigating to the `isaacgym` directory and running:

```bash
pip install -e isaacgym/python --no-cache-dir --force-reinstall
```

_Note: The `--no-cache-dir` and `--force-reinstall` flags are used to avoid potential issues with the installation we encountered._

_Note: Please ignore Pip's notice that `[notice] To update, run: pip install --upgrade pip` as the current version of Pip is necessary for compatibility with the codebase._

_Tip: The documentation for IsaacGym  is located inside the `docs` directory in the unzipped folder and is not available online. You can open the `index.html` file in your browser to access the documentation._

You can now safely delete the downloaded zipped file and navigate back to the root directory for your project. 


### Install FurnitureBench

To allow for data collection with the SpaceMouse, etc. we used a [custom fork](https://github.com/ankile/furniture-bench/tree/iros-2024-release-v1) of the [FurnitureBench code](https://github.com/clvrai/furniture-bench). The fork is included in this codebase as a submodule. To install the FurnitureBench package, first run:

```bash
git clone --recursive git@github.com:ankile/robust-rearrangement.git
```

_Note: If you forgot to clone the submodule, you can run `git submodule update --init --recursive` to fetch the submodule._

Then, install the FurnitureBench package by running:

```bash
cd robust-rearrangement/furniture-bench
pip install -e .
```

To test the installation of FurnitureBench, run:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
```

This should open a window with the simulated environment and the robot in it.

If you encounter the error `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`, this might be remedied by adding the conda environment's library path to the `LD_LIBRARY_PATH` environment variable. This can be done by, e.g., running:

```bash
export LD_LIBRARY_PATH=YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib
```

If you encounter `[Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999` (and you're using a Nvidia GTX 3070), try running:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```


Some context on this error: https://forums.developer.nvidia.com/t/cudaimportexternalmemory-failed-on-rgbimage/212944/4

### Install the robust-rearrangement Package

Finally, install the `robust-rearrangement` package by running:

```bash
cd ..
pip install -e .
```

### Data Collection: Install the SpaceMouse Driver

```bash
pip install numpy termcolor atomics scipy
pip install git+https://github.com/cheng-chi/spnav
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```

### Install Additional Dependencies

Depending on what parts of the codebase you want to run, you may need to install additional dependencies. Especially different vision encoders might require additional dependencies. To install the R3M or VIP encoder, respectively, run:

```bash
pip install -e robust-rearrangement/furniture-bench/r3m
```


## Download the Data

We provide an S3 bucket that contains all the data. You can browse the contents of the bucket [here](https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/index.html).

Then, for the code to know where to look for the data, please set the environment variables `DATA_DIR_PROCESSED` to the path of the processed data directories. This can be done by running or adding the following lines to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export DATA_DIR_PROCESSED=/path/to/processed-data
```

The raw data, i.e., trajectories stored as `.pkl` files according to the file format used in [FurnitureBench](https://github.com/clvrai/furniture-bench), is also available. Before we train policies on this data, we process it into flat `.zarr` files with `src/data_processing/process_pickles.py` so it's easier to deal with in BC training. Please set the `DATA_DIR_RAW` environment variable before downloading the raw data.

All parts of the code (data collection, training, evaluation rollout storage, data processing, etc.) use these environment variables to locate the data.

_Note: The code uses the directory structure in the folders to locate the data. If you change the directory structure, you may need to update the code accordingly._

To download the data, you can call the downloading script and specify the appropriate `task` name. At this point, these are the options:

```bash
python scripts/download_data.py --task one_leg
python scripts/download_data.py --task lamp
python scripts/download_data.py --task roundd_table
python scripts/download_data.py --task mug_rack
python scripts/download_data.py --task factory_peg_hole
```

For each of these, the 50 demos we collected for each randomness level will be downloaded.



## Training models

We heavily rely on WandB as a tracking service and a way to organize runs and model weights. So, for the most streamlined experience with the below training, ensure that you've set `WANDB_ENTITY` environment variable, e.g.:

```bash
export WANDB_ENTITY=your-entity-name
```

This will log training runs to this entity, and we will later use the weights from those runs to evaluate the runs and load weights for RL fine-tuning.

_NOTE:_ We are also working on releasing our weights in a way that's independent of our specific WandB projects and a way to specify local weight paths instead of a WandB run ID.


### BC Pre-training

#### Training from scratch

To pre-train the models, please ensure that you've downloaded the relevant data and that the `DATA_DIR_PROCESSED` environment variable is set correctly.

The pre-training runs can then be launched with one of these commands (the `dryrun` flag is nice for debugging as it turns off WandB, loads less data, and makes epochs shorter):

**`one_leg`**

```bash
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=low dryrun=false
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=med dryrun=false
```

**`lamp`**

```bash
python -m src.train.bc +experiment=state/diff_unet task=lamp randomness=low dryrun=false
python -m src.train.bc +experiment=state/diff_unet task=lamp randomness=med dryrun=false
```

**`round_table`**

```bash
python -m src.train.bc +experiment=state/diff_unet task=round_table randomness=low dryrun=false
python -m src.train.bc +experiment=state/diff_unet task=round_table randomness=med dryrun=false
```

**`mug_rack`**

```bash
python -m src.train.bc +experiment=state/diff_unet task=mug_rack randomness=low dryrun=false
```


**`peg_hole`**

```bash
python -m src.train.bc +experiment=state/diff_unet task=factory_peg_hole randomness=low dryrun=false
```

You can run evaluations with a command like:

```bash
python -m src.eval.evaluate_model --n-envs 128 --n-rollouts 128 -f one_leg --if-exists append --max-rollout-steps 700 --action-type pos --observation-space image --randomness low --wt-type best_success_rate --run-id <wandb-project>/<wandb-run-id>
```

You can add the following flags to visualize in the viewer or store the rollouts:

```bash
--observation-space image --save-rollouts --visualize
```



#### Evaluate pre-trained checkpoints

`one_leg` BC pre-trained weights:

```
https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/checkpoints/bc/one_leg/low/actor_chkpt.pt
https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/checkpoints/bc/one_leg/med/actor_chkpt.pt
```

The rest of the weights are available in the same bucket, just substitute `one_leg` with the respective task name and `low` with `med` for the medium randomness level.

Once these are downloaded, you can run evaluation of the weights in a very similar manner to the above, except that you can substitute `--run-id` and `wt-type` with `wt-path`, like so:

```bash
python -m src.eval.evaluate_model --n-envs 128 --n-rollouts 128 -f one_leg --if-exists append --max-rollout-steps 700 --action-type pos --randomness low --observation-space state --wt-path <path to checkpoints>/bc/one_leg/low/actor_chkpt.pt
```

Also, we used the following `--max-rollout-steps` for the different tasks:

- `one_leg`: 700
- `lamp`: 1000
- `round_table`: 1000
- `mug_rack`: 400
- `peg_hole`: 200




### RL Fine-tuning

#### Run full fine-tuning

Running the residual RL finet-tuning looks like the following:

```bash
python -m src.train.residual_ppo \
    base_policy.wandb_id=<wandb-project>/<run-id> \
    base_policy.wt_type=best_success_rate \
    env.task=one_leg env.randomness=low \
    num_env_steps=700 \
    debug=false
```

if you want to run the fine-tuning from a pre-training run you've run in a WandB project, or like this:

```bash
python -m src.train.residual_ppo \
    base_policy.wt_path=/path/to/actor_chkpt.pt \
    env.task=one_leg env.randomness=low \
    num_env_steps=700 \
    debug=false
```

Of course, to fine-tune the rest of the tasks, you can substitute `one_leg` with the respective task name and `low` with `med` for the medium randomness level.

Also, we used the following `num_env_steps` for the different tasks:

- `one_leg`: 700
- `lamp`: 1000
- `round_table`: 1000
- `mug_rack`: 400
- `peg_hole`: 200



#### Evaluate trained checkpoints

_Our RL fine-tuned weights are to be available for download shortly_


`one_leg` residual RL fine-tuned weights:

```
https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/checkpoints/rppo/one_leg/low/actor_chkpt.pt
https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/checkpoints/rppo/one_leg/med/actor_chkpt.pt
```

The rest of the weights are available in the same bucket, just substitute `one_leg` with the respective task name and `low` with `med` for the medium randomness level.

To evaluate the weights, you can run the evaluation script just like for the BC weights.


## Notes on sim-to-real (in development)
Please see [our notes on using Isaac Sim to re-render trajectories in service of visual sim-to-real](src/sim2real/readme.md). With the ongoing developments of Isaac Sim and IsaacLab, this area of the pipeline is not as mature and is still under ongoing development. The `src/sim2real` folder contains the scripts we used for converting assets to USD for use with Isaac Sim and re-rendering trajectories collected either via teleoperation or rolling out trained agents. 


## Notes on real world evaluation (in development)
Please see [our notes on running on the real world Franka Panda robot](src/real/readme.md). Our steps for reproducing the identical real world setup are still being developed, but in the `src/real` folder, we provide the scripts that we used along with some notes on the general process of getting set up to use the same tools.



## Citation

If you find the paper or the code useful, please consider citing the paper:

```tex      
@article{ankile2024imitation,
  title={From Imitation to Refinement--Residual RL for Precise Visual Assembly},
  author={Ankile, Lars and Simeonov, Anthony and Shenfeld, Idan and Torne, Marcel and Agrawal, Pulkit},
  journal={arXiv preprint arXiv:2407.16677},
  year={2024}
}```

