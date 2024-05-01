# Robust Rearrangement

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

Once installed and activated, make some compatibility changes to the environment by running:

```bash
pip install setuptools==65.5.0
pip install --upgrade pip wheel==0.38.4
pip install termcolor
```


### Install IsaacGym

Download the IsaacGym installer from the [IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running (also refer to the [FurnitureBench installlation instructions](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html#download-isaac-gym)):

- Click "Join now" and log into your NVIDIA account.
- Click "Member area".
- Read and check the box for the license agreement.
- Download and unzip `Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release`.

Once the zipped file is downloaded, move it to the desired location and unzip it by running:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```


Now, you can install the IsaacGym package by navigating to the `isaacgym` directory and running:

```bash
pip install -e python --no-cache-dir --force-reinstall
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




10:45
https://forums.developer.nvidia.com/t/cudaimportexternalmemory-failed-on-rgbimage/212944/4

### Install the robust-rearrangement Package

Finally, install the ImitationJuicer package by running:

```bash
cd ..
pip install -e .
```

### Data Collection: Install the SpaceMouse Driver

TODO: Write this up.


### Install Additional Dependencies

Depending on what parts of the codebase you want to run, you may need to install additional dependencies. Especially different vision encoders might require additional dependencies. To install the R3M or VIP encoder, respectively, run:

```bash
pip install -e robust-rearrangement/furniture-bench/r3m
pip install -e robust-rearrangement/furniture-bench/vip
```

The Spatial Softmax encoder and BC_RNN policy require the `robomimic` package to be installed:

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```


## Download the Data

We provide a Google Drive folder that contains a zip file with the raw data and a zip file with the processed data. [Download the data](https://drive.google.com/drive/folders/13UqtMLXY1_8JCQOZf3j-YbZyMRTsgZ2K?usp=sharing).

The data files can be unzipped by running:

```bash
tar -xzvf imitation-juicer-data-raw.tar.gz
tar -xzvf imitation-juicer-data-processed.tar.gz
```

Then, for the code to know where to look for the data, please set the environment variables `DATA_DIR_RAW` and `DATA_DIR_PROCESSED` to the paths of the raw and processed data directories, respectively. This can be done by running or adding the following lines to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export DATA_DIR_RAW=/path/to/raw-data
export DATA_DIR_PROCESSED=/path/to/processed-data
```

In the above example, the folders `raw` and `processed` in the two zipped files should be placed immediately inside the above folder, e.g., `/path/to/raw-data/raw` and `/path/to/processed-data/processed.`

All parts of the code (data collection, training, evaluation rollout storage, data processing, etc.) use these environment variables to locate the data.

_Note: The code uses the directory structure in the folders to locate the data. If you change the directory structure, you may need to update the code accordingly._



## Guide to Project Workflow


To be researched...



## Citation

If you find the paper or the code useful, please consider citing the paper:

```tex      
TBA
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
