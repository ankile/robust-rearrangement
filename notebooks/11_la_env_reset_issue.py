import furniture_bench
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

from src.gym import get_env

from tqdm import trange, tqdm
import matplotlib.pyplot as plt

print("create 1. env")
env: FurnitureSimEnv = get_env(
    obs_type="image",
    num_envs=4,
    gpu_id=1,
)

print("reset 1. env")
obs = env.reset()

print("close 1. env")
env.close()


print("create 2. env")
env = get_env(
    obs_type="image",
    num_envs=4,
    gpu_id=1,
)

print("reset 2. env")
obs = env.reset()

print("close 2. env")
env.close()
