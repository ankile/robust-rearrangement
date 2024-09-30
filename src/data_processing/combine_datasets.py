from datetime import datetime
import zarr
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def combine_datasets(data_paths, output_path):
    last_episode_end = 0
    new_data = defaultdict(list)

    for path in tqdm(data_paths):
        dataset = zarr.open(path, "r")
        for key in tqdm(dataset.keys(), leave=False):
            if key == "episode_ends":
                # Increment episode_ends values and append
                incremented_ends = dataset[key][:] + last_episode_end
                new_data[key].extend(incremented_ends)
                last_episode_end = incremented_ends[-1]
            else:
                # Simply append for other keys
                new_data[key].extend(dataset[key][:])

    # Convert lists to arrays in the new dataset
    out_zarr = zarr.open(output_path, "w")
    for key in new_data.keys():
        out_zarr[key] = np.array(new_data[key])

    out_zarr.attrs["num_episodes"] = len(out_zarr["episode_ends"]) + 1
    out_zarr.attrs["created_at"] = str(datetime.now())
