import os

import torch
from torch.utils.data import Dataset
from torch.distributions import Categorical

from constant import DEFAULT_TYPE, TREE_DATA
from src.configurations import load_configurations
from src.energy_based import distribution, energy_rbm


def load_artificial_data(n_v=2, n_h=3, N=1000000):
    if TREE_DATA:
        fn = 'data/{}_{}_tree.pt'.format(n_v, n_h)
    else:
        fn = 'data/{}_{}.pt'.format(n_v, n_h)

    if os.path.exists(fn):
        file = torch.load(fn)
    else:
        n = n_v + n_h

        W = torch.randn(n_h, n_v)  # * 1 / np.sqrt(n)
        if TREE_DATA:
            # W[1:, 1:] = 0.
            if n_v % n_h == 0:
                k = n_v // n_h
                for i in range(n_h):
                    W[i, : i * k] = 0.
                    W[i, (i + 1) * k:] = 0.
            elif n_h % n_v == 0:
                k = n_h // n_v
                for i in range(n_v):
                    W[: i * k, i] = 0.
                    W[(i + 1) * k:, i] = 0.

        x = load_configurations(n).type(DEFAULT_TYPE)
        p = distribution(W, torch.zeros(1, n), x, energy_rbm)[0]

        indices = Categorical(probs=p).sample((N,)).long()
        samples = x.T[indices]

        file = {
            'samples': samples[:, :n_v],
            'hiddens': samples[:, n_v:],
            'W': W,
        }
        os.makedirs('data', exist_ok=True)
        torch.save(file, fn)

        # phat = np.zeros_like(p)
        # for i in range(len(phat)):
        #     phat[i] = (indices == i).sum()
        # phat /= N

        # plt.bar(range(len(p)), p, alpha=0.5, label='True')
        # plt.bar(range(len(p)), phat, alpha=0.7, label='Sample')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.tight_layout()
        # plt.legend(loc='upper right', fontsize=20)
        # plt.show()

    dataset = ArtificialDataset(fn, N)
    return dataset


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
