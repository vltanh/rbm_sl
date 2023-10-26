import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

from energy_based import distribution
from sl import stochasic_localization_rbm

from constant import DEFAULT_TYPE

# Hyperparameters
n_v, n_h = 2, 16  # Dimension
delta = .1  # Step size
L = 20  # Number of steps each sample
N = 1000  # Number of samples

# Generate random tree-structured RBM
W = torch.randn(n_v, n_h).type(DEFAULT_TYPE)
# W = torch.ones(n_v, n_h)
W[1:, 1:] = 0.

# Generate samples
samples, history = stochasic_localization_rbm(W, L, delta, N)

for sample, h in zip(samples, history.transpose(0, 1)):
    # Plot trajectory 
    fig, axes = plt.subplots(1, 1)
    axes.set_aspect('equal')
    axes.add_patch(
        patches.Rectangle(
            (-1, -1),   # (x,y)
            2,          # width
            2,          # height
            facecolor='white',
            edgecolor='black',
            linestyle='--',
            alpha=0.5
        )
    )

    axes.scatter(h[0, 0], h[0, 1], color='green', marker='o')
    axes.quiver(h[:-1, 0], h[:-1, 1],
                h[1:, 0]-h[:-1, 0], h[1:, 1]-h[:-1, 1],
                scale_units='xy', angles='xy', scale=1., alpha=0.5)
    axes.scatter(h[-1, 0], h[-1, 1], color='orange', marker='o')
    axes.set_xlim(-1.1, 1.1)
    axes.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()
