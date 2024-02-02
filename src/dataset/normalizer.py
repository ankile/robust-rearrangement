from typing import Union
import torch
import torch.nn as nn
import numpy as np

from src.dataset.data_stats import get_data_stats


# Define a normalizer base class
class Normalizer(nn.Module):
    def __init__(self, control_mode="delta"):
        super().__init__()
        assert control_mode in ["delta", "pos"]
        self.control_mode = control_mode

    def _normalize(
        self, x: Union[np.ndarray, torch.Tensor], key: str
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

    def _denormalize(
        self, x: Union[np.ndarray, torch.Tensor], key: str
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        key: str,
        forward: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize or denormalize the input data.

        It accepts either a numpy array or a torch tensor and will return the same type.
        """
        numpy = False
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            numpy = True

        if forward:
            x = self._normalize(x, key)
        else:
            x = self._denormalize(x, key)

        if numpy:
            return x.numpy()
        return x

    def keys(self):
        return self.stats.keys()

    @property
    def stats_dict(self):
        # Return the stats as a dict of numpy arrays
        stats = {}

        for key in self.stats.keys():
            stats[key] = {}
            for stat in self.stats[key].keys():
                stats[key][stat] = self.stats[key][stat].cpu().numpy()

        return stats

    @staticmethod
    def create_parameter_dict(stats, action_key, normalization_mode):
        param_dict = nn.ParameterDict()
        for key, value in stats.items():
            # Skip the action type that is not chosen
            if key in ["action/pos", "action/delta"] and key != action_key:
                continue

            # Rename the chosen action key
            if key == action_key:
                key = "action"

            if isinstance(value, dict) and (
                ("min" in value and "max" in value)
                or ("mean" in value and "std" in value)
            ):
                if normalization_mode == "min_max":
                    param_dict[key] = nn.ParameterDict(
                        {
                            "min": nn.Parameter(torch.tensor(value["min"])),
                            "max": nn.Parameter(torch.tensor(value["max"])),
                        }
                    )
                elif normalization_mode == "mean_std":
                    param_dict[key] = nn.ParameterDict(
                        {
                            "mean": nn.Parameter(torch.tensor(value["mean"])),
                            "std": nn.Parameter(torch.tensor(value["std"])),
                        }
                    )
                else:
                    raise ValueError(
                        f"Normalization mode {normalization_mode} not recognized."
                    )
            elif isinstance(value, dict):
                param_dict[key] = create_parameter_dict(
                    value, action_key, normalization_mode
                )
            else:
                raise ValueError(f"Value {value} of type {type(value)} not recognized.")
        return param_dict

    def _turn_off_gradients(self):
        # Turn off gradients for the stats
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    # Make a method that let's you get a copy of the class instance
    def get_copy(self):
        return self.__class__(control_mode=self.control_mode)


class LinearNormalizer(Normalizer):
    def __init__(self, control_mode="delta"):
        super().__init__(control_mode=control_mode)

        stats = get_data_stats()
        self.stats = self.create_parameter_dict(
            stats=stats,
            action_key=f"action/{control_mode}",
            normalization_mode="min_max",
        )

        # Turn off gradients for the stats
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x


class GaussianNormalizer(Normalizer):
    def __init__(self, control_mode="delta"):
        super().__init__(control_mode=control_mode)

        stats = get_data_stats()
        self.stats = self.create_parameter_dict(
            stats=stats,
            action_key=f"action/{control_mode}",
            normalization_mode="mean_std",
        )

        # Turn off gradients for the stats
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["mean"]) / stats["std"]
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = x * stats["std"] + stats["mean"]
        return x


if __name__ == "__main__":
    # Create the normalizers
    linear_normalizer = LinearNormalizer()
    gaussian_normalizer = GaussianNormalizer()

    # Print the stats
    print("Linear normalizer stats:")
    print(linear_normalizer.stats_dict)

    print("Gaussian normalizer stats:")
    print(gaussian_normalizer.stats_dict)
