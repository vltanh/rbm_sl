import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from sl import stochasic_localization_rbm
from energy_based import distribution, energy_rbm
from configurations import load_configurations

from constant import DEFAULT_TYPE

# Hyperparameters
n_v = 2
n_h = 4
n = n_v + n_h

delta = .1  # Step size
L = 100  # Number of steps each sample
N = 1000000  # Number of samples

# Generate random tree-structured RBM
W = torch.randn(n_v, n_h).type(DEFAULT_TYPE)
# W = torch.ones(n_v, n_h)
W[:-1, 1:] = 0.

# Generate samples
samples = stochasic_localization_rbm(W, L, delta, N, store_history=False)

# Check marginals
samples_marginals = (torch.mean(samples, dim=0) + 1) / 2
print('Marginals (samples)\n', samples_marginals)

if n <= 10:
    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Dummy tilting variable
    y = torch.zeros(1, n).type(DEFAULT_TYPE)

    # Compute the true distribution
    p = distribution(W, y, x, energy_fn=energy_rbm)[0]

    # Compute the sampled distribution
    t = np.argmin(pairwise_distances(x.T, samples), 0)
    phat = np.zeros_like(p)
    for i in range(len(phat)):
        phat[i] = np.sum(t == i)
    phat /= N

    # Compare
    print('Error:', torch.max(torch.abs(p - phat)).item())

    # Plot the distributions
    plt.bar(range(len(p)), p, alpha=0.5, label='True')
    plt.bar(range(len(p)), phat, alpha=0.7, label='Sample')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=20)
    plt.show()
