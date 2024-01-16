import torch
import torch.nn as nn


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


# For now, these stats come from calculating the min/max of data
# collected from the robot in the sim for the `one_leg` task,
# for 490 trajectories, roughly evenly split between low, medium, and high randomness.
# The values were very similar to the values for the `lamp` task collected in the real world.
# TODO: Investigate more data from more tasks to see if these stats are good enough.
class StateActionNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.stats = nn.ParameterDict(
            {
                "action": nn.ParameterDict(
                    {
                        "min": nn.Parameter(
                            torch.tensor(
                                [
                                    -0.11999873,
                                    -0.11999782,
                                    -0.0910784,
                                    -0.41173494,
                                    -0.7986815,
                                    -0.73318267,
                                    -1.00000012,
                                    -1.0,
                                ]
                            )
                        ),
                        "max": nn.Parameter(
                            torch.tensor(
                                [
                                    0.11999907,
                                    0.11999977,
                                    0.1,
                                    0.27584794,
                                    0.80490655,
                                    0.75659704,
                                    1.00000024,
                                    1.0,
                                ]
                            )
                        ),
                    }
                ),
                "robot_state": nn.ParameterDict(
                    {
                        "min": nn.Parameter(
                            torch.tensor(
                                [
                                    2.80025989e-01,
                                    -1.65265590e-01,
                                    -1.33207440e-03,
                                    -9.99999881e-01,
                                    -8.32509935e-01,
                                    -5.51004350e-01,
                                    1.40711887e-08,
                                    -7.02747107e-01,
                                    -1.01964152e00,
                                    -7.42725849e-01,
                                    -2.45710993e00,
                                    -2.84063244e00,
                                    -3.71836829e00,
                                    4.75169145e-05,
                                ]
                            )
                        ),
                        "max": nn.Parameter(
                            torch.tensor(
                                [
                                    0.68205643,
                                    0.31372252,
                                    0.27053252,
                                    0.99999988,
                                    0.8431676,
                                    0.56648922,
                                    0.20231877,
                                    0.65723258,
                                    0.75370288,
                                    0.50734419,
                                    2.4507556,
                                    2.72471213,
                                    3.6940937,
                                    0.07003613,
                                ]
                            )
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

    def forward(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)

        return self._denormalize(x, key)

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
