# Go one level up
cd ..

# Download Isaac Gym
# curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1J4bb5SfY-8H05xXiyF4N1xUOas390tll" > /dev/null
# curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/confirm/ {print $NF}' ./cookie.txt)&id=1J4bb5SfY-8H05xXiyF4N1xUOas390tll" -o isaacgym.tar.gz

# Extract Isaac Gym
tar -xzf isaacgym.tar.gz

# Download Furniture-benchmark
git clone git@github.com:ankile/furniture-bench.git

# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create conda environment
conda create -n rlgpu python=3.8 -y
# some manual fixes
pip install setuptools==65.5.0
pip install --upgrade pip wheel==0.38.4


# Activate conda environment
conda activate rlgpu

# Install dependencies
# isaac gym (fresh)
cd isaacgym
pip install -e python --no-cache-dir --force-reinstall

cd ../furniture-bench
pip install -e .

# pip install -e r3m
# pip install -e vip

# Install AWS CLI
# cd ~
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# Do this if you have sudo access
# sudo ./aws/install

# Otherwise we can add the following to .bashrc to emulate the above
# export AWS_COMMAND='/home/larsankile/aws-cli/v2/current/bin/aws'
# alias aws='/home/larsankile/aws-cli/v2/current/bin/aws'

# Then run `aws configure` and enter the required information

# Install robomimic (for BC-RNN)
cd ..
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .

# Install the last required dependencies
cd furniture-diffusion
pip install -r requirements.txt