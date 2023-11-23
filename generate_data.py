import numpy as np
import torch
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import os

from configurations import load_configurations
from energy_based import distribution, energy_rbm
from data import ArtificialDataset
from constant import DEFAULT_TYPE, TREE

torch.set_default_dtype(DEFAULT_TYPE)


def load_data(n_v=2, n_h=3, N=1000000):
    if os.path.exists('data/{}_{}.pt'.format(n_v, n_h)):
        file = torch.load('data/{}_{}.pt'.format(n_v, n_h))
    else:
        n = n_v + n_h

        W = torch.randn(n_h, n_v)  # * 1 / np.sqrt(n)
        if TREE:
            W[1:, 1:] = 0.

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
        torch.save(file, 'data/{}_{}.pt'.format(n_v, n_h))

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

    dataset = ArtificialDataset('data/{}_{}.pt'.format(n_v, n_h), N)
    return dataset
