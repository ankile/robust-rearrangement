# JUICER: Data-Efficient Imitation Learning for Robotic Assembly

## Installation Instructions


### Install Conda

First, install Conda by following the instructions on the ![Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or the ![Miniconda website](https://docs.conda.io/en/latest/miniconda.html) (here using Miniconda).

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
conda create -n imitation-juicer python=3.8 -y
```

Activate the environment by running:

```bash
conda activate imitation-juicer
```

Once installed and activated, make some compatibility changes to the environment by running:

```bash
pip install --upgrade pip wheel
pip install setuptools==58
pip install --upgrade pip==22.2.2
```


### Install IsaacGym

Download the IsaacGym installer from the ![IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running (also refer to the ![FurnitureBench installlation instructions](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html#download-isaac-gym)):

- Click "Join now" and log into your NVIDIA account.
- Click "Member area".
- Read and check the box for the license agreement.
- Download and unzip `Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release`.


Now, you can install the IsaacGym package by running:

```bash
cd <path/to/isaacgym>
pip install -e python --no-cache-dir --force-reinstall
```

### Install FurnitureBench

To allow for data collection with the SpaceMouse, etc. we used a ![custom fork](https://github.com/ankile/furniture-bench/tree/iros-2024-release-v1) of the ![FurnitureBench code](https://github.com/clvrai/furniture-bench). The fork is included in this codebase as a submodule. To install the FurnitureBench package, first run:

```bash
git clone --recursive git@github.com:ankile/imitation-juicer.git
```

_Note: If you forgot to clone the submodule, you can run `git submodule update --init --recursive` to fetch the submodule._

Then, install the FurnitureBench package by running:

```bash
cd imitation-juicer/furniture-bench
pip install -e .
```

### Install the ImitationJuicer Package

Finally, install the ImitationJuicer package by running:

```bash
cd ..
pip install -e .
```

### Install Additional Dependencies

Depending on what parts of the codebase you want to run, you may need to install additional dependencies. Especially different vision encoders might require additional dependencies. To install the R3M or VIP encoder, respectively, run:

```bash
pip install -e imitation-juicer/furniture-bench/r3m
pip install -e imitation-juicer/furniture-bench/vip
```

The Spatial Softmax encoder and BC_RNN policy requires the `robomimic` package to be installed:

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```



## Download the Data

TODO: Write this up.



## Guide to Project Workflow

This README outlines the workflow for collecting demonstrations,
annotating them, augmenting trajectories, training models, and
evaluating the trained models. Below are the steps involved in the
process.

### Collect Demonstrations

To collect data, start by invoking the simulated environment. Input
actions are recorded using the 3DConnextion SpaceMouse. The source code
and command line arguments are available in
`src/data_collection/teleop.py`. An example command for collecting
demonstrations for the `one_leg` task is:

    python -m src.data_collection.teleop --furniture one_leg --pkl-only --num-demos 10 --randomness low [--save-failure --no-ee-laser]

Demonstrations are saved as `.pkl` files at:

    $DATA_DIR_RAW/raw/sim/one_leg/teleop/low/success/

By default, only successful demonstrations are stored. Failures can be
stored by using the `--save-failure` flag. The `--no-ee-laser` flag
disables the assistive red light.

To collect data, control the robot with the SpaceMouse. To store an
episode and reset the environment, press `t`. To discard an episode,
press `n`. To \"undo\" actions, press `b`. To toggle recording on and
off, use `c` and `p`, respectively.

### Annotate Demonstrations

Before trajectory augmentation, demos must be annotated at bottleneck
states. Use `src/data_collection/annotate_demo.py` for this purpose.
Here\'s how to invoke the tool:

    python -m src.data_collection.annotate_demo --furniture one_leg --rest-of-arguments-tbd

Use `k` and `j` to navigate frames, and `l` and `h` for faster
navigation. Press `space` to mark a frame and `u` to undo a mark. Press
`s` to save and move to the next trajectory.

### Augment Trajectories

After annotation, use `src/data_collection/backward_augment.py` to
generate counterfactual snippets. Example command:

    python -m src.data_collection.backward_augment --furniture one_leg --randomness low --demo-source teleop [--no-filter-pickles]

New demonstrations are stored at:

    $DATA_DIR_RAW/raw/sim/one_leg/augmentation/low/success/

### Train Models

Train models using `src/train/bc.py`. We use Hydra and OmegaConf for
hyperparameter management. Ensure WandB authentication before starting.

To train for the `one_leg` task:

    python -m src.train.bc +experiment=image_baseline furniture=one_leg

For a debug run, add `dryrun=true`. For rollouts during training, add
`rollout=rollout`.

### Evaluate Models

Evaluate trained models with `src/eval/evaluate_model.py`. For example:

    python -m src.eval.evaluate_model --run-id entity/project/run-id --furniture one_leg --n-envs 10 --n-rollouts 10 --randomness low [--save-rollouts --wandb --if-exists append --run-state finished]

To save rollout results, use `--save-rollouts`. For WandB logging, add
`--wandb`.






## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


```tex      
@article{ankile2024juicer,
    author    = {Ankile, Lars and Simeonov, Anthony and Shenfeld, Idan and Agrawal, Pulkit},
    title     = {JUICER: Data-Efficient Imitation Learning for Robotic Assembly},
    journal   = {arXiv},
    year      = {2024},
}
```
