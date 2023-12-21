import pickle
import tarfile


def zipped_img_generator(filename, max_samples=1000):
    n_samples = 0
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar:
            if (
                member.isfile() and ".pkl" in member.name
            ):  # Replace 'your_condition' with actual condition
                with tar.extractfile(member) as f:
                    if f is not None:
                        content = f.read()
                        data = pickle.loads(content)
                        n_samples += 1

                        yield data

                        if n_samples >= max_samples:
                            break


class ZarrSubsetView:
    def __init__(self, zarr_group, include_keys):
        """
        Create a view-like object for a Zarr group, excluding specified keys.
        :param zarr_group: The original Zarr group.
        :param exclude_keys: A set or list of keys to exclude.
        """
        self.zarr_group = zarr_group
        self.include_keys = set(include_keys)

    def __getitem__(self, key):
        return self.zarr_group[key]

    def observation_keys(self):
        """
        Return keys not excluded.
        """
        return [key for key in self.zarr_group.keys() if key in self.include_keys]

    def items(self):
        """
        Return items not excluded.
        """
        return [(key, self.zarr_group[key]) for key in self.observation_keys()]
