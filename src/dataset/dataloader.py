import torch
import itertools


class FixedStepsDataloader(torch.utils.data.DataLoader):
    """
    Dataloader that always yields a fixed number of batches.
    If requested number of batches is smaller than available -> return a random subset
    If requedsted number is larger than available -> cycle through (like a new epoch, random order every time)
    """

    def __init__(self, *args, n_batches, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_batches = n_batches

    def __iter__(self):
        endless_dataloader = itertools.cycle(super().__iter__())
        for _ in range(self.n_batches):
            yield next(endless_dataloader)

    def __len__(self):
        return self.n_batches
