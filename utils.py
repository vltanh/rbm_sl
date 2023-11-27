import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.metrics import pairwise_distances
import numpy as np

from constant import DEFAULT_TYPE
from configurations import load_configurations


def plot_histogram(samples, p):
    B, n = samples.shape

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the sampled distribution
    t = np.argmin(pairwise_distances(x.T, samples), 0)
    phat = np.zeros_like(p)
    for i in range(len(phat)):
        phat[i] = np.sum(t == i)
    phat /= B

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


def plot_trajectory(history, max_samples):
    for i, h in enumerate(history.transpose(0, 1)):
        if i == max_samples:
            break

        # Plot trajectory
        _, axes = plt.subplots(1, 1)
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
