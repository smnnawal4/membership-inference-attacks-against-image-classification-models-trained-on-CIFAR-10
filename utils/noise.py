import torch

def add_noise_to_data(x, noise_std=0.1):
    noisy = x.clone()
    mask = torch.rand_like(noisy).lt(noise_std)
    noisy[mask] = 1.0 - noisy[mask]
    return noisy

from torch.utils.data import Dataset
import torch

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR10_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

class NoisyRealDataset(Dataset):
    def __init__(self, base_dataset, noise_std=0.1):
        self.base = base_dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        # de-normalizing
        x = x * CIFAR10_STD + CIFAR10_MEAN
        x = torch.clamp(x, 0.0, 1.0)

        # applying noise
        x = add_noise_to_data(x, self.noise_std)

        # re-normalizing
        x = (x - CIFAR10_MEAN) / CIFAR10_STD
        return x, y
