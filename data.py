import torch
from torch.utils.data import Dataset


class ArtificialDataset(Dataset):
    def __init__(self, path, N):
        d = torch.load(path)
        self.samples = d['samples'][:N]
        self.true_W = d['W']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item
