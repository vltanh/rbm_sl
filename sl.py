import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from constant import DEFAULT_TYPE, ALGO

from energy_based import energy_rbm
from brute_force import barycenter_bf
from factor_graph import TreeRBMFactorGraph


def barycenter(W, y):
    if ALGO == 'slfg':
        return TreeRBMFactorGraph(W, y).barycenter()
    elif ALGO == 'sl':
        return barycenter_bf(W, y, energy_fn=energy_rbm)
    else:
        raise NotImplementedError


def stochasic_localization_rbm(W, L, delta, B, store_history=True):
    '''
    Stochastic localization for RBM

    Args:
        W: [n_v, n_h]
        L: number of iterations
        delta: step size
        B: batch size

    Returns:
        samples: [B, n]
        history: [L+1, B, n]
    '''
    n_v, n_h = W.shape
    n = n_v + n_h

    if store_history:
        # Store the history
        history = torch.empty((L + 1, B, n)).type(DEFAULT_TYPE)

    # Initialize the tilting variable
    yhat = torch.zeros(B, n).type(DEFAULT_TYPE)
    # yhat = y.clone()

    # Iterations
    for l in tqdm(range(L)):
        # Generate noise
        w = torch.randn(B, n)

        # Compute the barycenter
        mhat = barycenter(W, yhat)

        if store_history:
            # Store the history
            history[l] = mhat

        # Update the tilting variable
        yhat = yhat + mhat * delta + np.sqrt(delta) * w

    # Compute the barycenter
    mhat = barycenter(W, yhat)

    if store_history:
        # Store the history
        history[L] = mhat

    # Compute the marginal distribution
    p = (1 + mhat) / 2

    # Generate samples
    samples = 2 * torch.bernoulli(p.clamp(0, 1)) - 1

    # Return
    if store_history:
        return samples, history
    return samples


class StochasticLocalization(nn.Module):
    """Stochastic Localization.
    Args:
        rbm (RBM): The RBM model.
        L (int): The number of iterations.
        delta (float): The step size.
    """

    def __init__(self, rbm, L, delta):
        """Create a Stochastic Localization."""
        super().__init__()
        self.rbm = rbm
        self.L = L
        self.delta = delta
        self.n = self.rbm.n_v + self.rbm.n_h

    def free_energy(self, v):
        return self.rbm.free_energy(v)

    def forward(self, v, B=None):
        r"""Compute the generated examples.
        Args:
            v (Tensor): The visible variable.
        Returns:
            (Tensor, Tensor): The generated variables.
        """
        B = B if B else v.shape[0]
        yhat = torch.zeros(B, self.n)
        for _ in tqdm(range(self.L), leave=False):
            w = torch.randn(B, self.n)
            mhat = barycenter(self.rbm.W, yhat)
            yhat = yhat + mhat * self.delta + np.sqrt(self.delta) * w
        mhat = barycenter(self.rbm.W, yhat)
        p = (1 + mhat) / 2
        samples = 2 * torch.bernoulli(p.clamp(0, 1)) - 1
        # samples = torch.bernoulli(mhat.clamp(0, 1))
        return samples[:, :self.rbm.n_v]
