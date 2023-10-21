import torch
import itertools


class FixedStepsDataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, n_batches, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_batches = n_batches

    def __iter__(self):
        endless_dataloader = itertools.cycle(super().__iter__())
        for _ in range(self.n_batches):
            yield next(endless_dataloader)

    def __len__(self):
        return self.n_batches
