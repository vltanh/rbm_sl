import torch

from constant import DEFAULT_TYPE
from energy_based import energy_rbm
from energy_based import distribution
from configurations import load_configurations


def marginal_rbm_bf(W, y):
    '''
    Compute the marginal distribution of the visible states

    Args:
        W: [n_v, n_h]
        y: [B, n]

    Returns:
        p: [B, 2^n_v]
    '''
    n_v, n_h = W.shape
    n = n_v + n_h

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(W, y, x, energy_rbm)  # [B, ]

    # Compute the marginal distribution
    p = p.reshape(-1, 2 ** n_v, 2 ** n_h)  # [B, 2^n_v, 2^n_h]
    p = torch.sum(p, dim=2)  # [B, 2^n_v]
    return p


def barycenter_bf(A, y, energy_fn):
    '''
    Compute the barycenter of a tilted SK

    Args:
        A: [n, n]
        y: [B, n]
        energy_fn: energy function

    Returns:
        barycenter: [B, n]
    '''
    n = y.shape[1]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(A, y, x, energy_fn)  # [B, 2^n]

    # Compute the barycenter
    barycenter = torch.matmul(p, x.transpose(0, 1))
    return barycenter
