from typing import Union
import torch
import torch.nn as nn
import numpy as np
from src.dataset.data_stats import data_stats


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


# For now, these stats come from calculating the min/max of data
# collected from the robot in the sim for the `one_leg` task,
# for ~200 episodes after the change in the controller
class StateActionNormalizer(nn.Module):
    def __init__(self, control_mode="delta"):
        super().__init__()
        assert control_mode in ["delta", "pos"]

        self.stats = nn.ParameterDict(
            {
                "robot_state": nn.ParameterDict(
                    {
                        "min": nn.Parameter(
                            torch.tensor(data_stats["robot_state"]["min"])
                        ),
                        "max": nn.Parameter(
                            torch.tensor(data_stats["robot_state"]["max"])
                        ),
                    }
                ),
                "action": nn.ParameterDict(
                    {
                        "min": nn.Parameter(
                            torch.tensor(data_stats[f"action/{control_mode}"]["min"])
                        ),
                        "max": nn.Parameter(
                            torch.tensor(data_stats[f"action/{control_mode}"]["max"])
                        ),
                    }
                ),
            }
        )

        # Turn off gradients for the stats
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

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
