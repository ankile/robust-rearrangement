# isaac gym (fresh)
pip install -e . --no-cache-dir --force-reinstall

# some manual fixes
pip install setuptools==65.5.0
pip install --upgrade pip wheel==0.38.4

# furniture bench
cd /path/to/furniture-bench
pip install -e .

# furniture-diffusion reqs + src module
cd /path/to/furniture-diffusion
pip install -e .
