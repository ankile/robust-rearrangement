import torch
import itertools
import numpy as np


class FixedStepsDataloader(torch.utils.data.DataLoader):
    """
    Dataloader that always yields a fixed number of batches.
    If requested number of batches is smaller than available -> return a random subset
    If requested number is larger than available -> cycle through (like a new epoch, random order every time)
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


class EndlessDataloader(torch.utils.data.DataLoader):
    """
    Dataloader that cycles through the dataset indefinitely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        endless_dataloader = itertools.cycle(super().__iter__())
        for batch in endless_dataloader:
            yield batch

    def __len__(self):
        return float("inf")


class WeightedDataLoader:
    # Thanks to Lirui Wang for this code
    def __init__(self, dataloaders, weight_type="root"):
        """
        :param dataloaders: list of pytorch dataloaders
        :param weight_type: type of weighting, e.g., "square_root"
        """
        self.dataloaders = dataloaders
        if weight_type == "root":
            datasizes = [len(d) for d in dataloaders]
            datasizes = np.power(datasizes, 1.0 / 3)  # np.sqrt(datasizes)
            weights = datasizes / np.sum(datasizes)
            self.weights = weights
        else:
            print(f"weight type {weight_type} not defined")

        self.loader_iters = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        # Choose a dataloader based on weights
        chosen_dataloader_idx = np.random.choice(len(self.dataloaders), p=self.weights)
        chosen_loader_iter = self.loader_iters[chosen_dataloader_idx]
        try:
            data = next(chosen_loader_iter)
            return data
        except StopIteration:
            # Handle case where a dataloader is exhausted. Reinitialize the iterator.
            self.loader_iters[chosen_dataloader_idx] = iter(
                self.dataloaders[chosen_dataloader_idx]
            )
            return self.__next__()

    def __len__(self):
        return sum([len(dataloader) for dataloader in self.dataloaders])
