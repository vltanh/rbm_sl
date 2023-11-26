import torch

from constant import DEFAULT_TYPE
from energy_based import energy_rbm
from energy_based import distribution
from configurations import load_configurations


def marginal_v_rbm_bf(W, y):
    '''
    Compute the marginal distribution of the visible states

    Args:
        W: [n_h, n_v]
        y: [B, n]

    Returns:
        p: [B, 2^n_v]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(W, y, x, energy_rbm)  # [B, 2^n]

    # Compute the marginal distribution
    p = p.reshape(-1, 2 ** n_v, 2 ** n_h)  # [B, 2^n_v, 2^n_h]
    p = torch.sum(p, dim=2)  # [B, 2^n_v]
    return p


def marginal_h_rbm_bf(W, y):
    '''
    Compute the marginal distribution of the hidden states

    Args:
        W: [n_h, n_v]
        y: [B, n]

    Returns:
        p: [B, 2^n_h]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(W, y, x, energy_rbm)  # [B, 2^n]

    # Compute the marginal distribution
    p = p.reshape(-1, 2 ** n_v, 2 ** n_h)  # [B, 2^n_v, 2^n_h]

    p = torch.sum(p, dim=1)  # [B, 2^n_h]
    return p


def barycenter_v_rbm_bf(W, y):
    '''
    Compute the barycenter of the visible states

    Args:
        W: [n_h, n_v]
        y: [B, n]

    Returns:
        barycenter: [B, n_v]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Compute the marginal distribution
    p = marginal_v_rbm_bf(W, y)  # [B, 2^n_v]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the barycenter
    x = x[:n_v, ::2**n_h]  # [n_v, 2^n_v]

    # Compute the barycenter
    # [B, 2^n_v] @ [2^n_v, n_v] = [B, n_v]
    barycenter = torch.matmul(p, x.transpose(0, 1))  # [B, n_v]

    return barycenter


def barycenter_h_rbm_bf(W, y):
    '''
    Compute the barycenter of the hidden states

    Args:
        W: [n_h, n_v]
        y: [B, n]

    Returns:
        barycenter: [B, n_h]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Compute the marginal distribution
    p = marginal_h_rbm_bf(W, y)  # [B, 2^n_h]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the barycenter
    x = x[n_v:, :2**n_h]  # [n_v, 2^n_v]

    # Compute the barycenter
    # [B, 2^n_v] @ [2^n_v, n_v] = [B, n_v]
    barycenter = torch.matmul(p, x.transpose(0, 1))  # [B, n_v]

    return barycenter


def conditional_h_rbm_bf(W, y, h):
    '''
    Compute the conditional distribution p(v | h)

    Args:
        W: [n_h, n_v]
        y: [B, n]
        h: [B, n_h]

    Returns:
        p: [B, 2^n_v]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h
    B = h.shape[0]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(W, y, x, energy_rbm)  # [B, 2^n]

    # Extract the configurations that satisfy the condition
    x_h = x[n_v:, :]  # [n_h, 2^n]
    # [B, n_h, 1] - [1, n_h, 2^n]
    diff = h[:, :, None] - x_h[None, :, :]  # [B, n_h, 2^n]
    mask = (diff == 0).all(dim=1)  # [B, 2^n]

    # Compute the conditional distribution
    p = p[mask].reshape(B, -1)  # [B, 2^n_v]
    p = p / torch.sum(p, dim=1, keepdim=True)  # [B, 2^n_v]

    return p


def conditional_v_rbm_bf(W, y, v):
    '''
    Compute the conditional distribution p(h | v)

    Args:
        W: [n_h, n_v]
        y: [B, n]
        v: [B, n_v]

    Returns:
        p: [B, 2^n_h]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h
    B = v.shape[0]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the distribution
    p = distribution(W, y, x, energy_rbm)  # [B, 2^n]

    # Extract the configurations that satisfy the condition
    x_v = x[:n_v, :]  # [n_v, 2^n]
    # [B, n_v, 1] - [1, n_v, 2^n]
    diff = v[:, :, None] - x_v[None, :, :]  # [B, n_v, 2^n]
    mask = (diff == 0).all(dim=1)  # [B, 2^n]

    # Compute the conditional distribution
    p = p[mask].reshape(B, -1)  # [B, 2^n_h]
    p = p / torch.sum(p, dim=1, keepdim=True)  # [B, 2^n_h]

    return p


def barycenter_conditional_h_rbm_bf(W, y, h):
    '''
    Compute the barycenter of the conditional distribution p(v | h)

    Args:
        W: [n_h, n_v]
        y: [B, n]
        h: [B, n_h]

    Returns:
        m: [B, n_v]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Compute the conditional distribution
    p = conditional_h_rbm_bf(W, y, h)  # [B, 2^n_v]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Extract the configurations that satisfy the condition
    x_h = x[n_v:, :]  # [n_h, 2^n]
    # [B, n_h, 1] - [1, n_h, 2^n] = [B, n_h, 2^n]
    diff = h[:, :, None] - x_h[None, :, :]
    mask = (diff == 0).all(dim=1)  # [B, 2^n]
    x_v = x[:n_v, mask[0]]  # [n_v, 2^n_v]

    # Compute the barycenter
    m = torch.matmul(p, x_v.transpose(0, 1))  # [B, n_v]
    return m


def barycenter_conditional_v_rbm_bf(W, y, v):
    '''
    Compute the barycenter of the conditional distribution p(h | v)

    Args:
        W: [n_h, n_v]
        y: [B, n]
        v: [B, n_v]

    Returns:
        m: [B, n_h]
    '''
    n_h, n_v = W.shape
    n = n_v + n_h

    # Compute the conditional distribution
    p = conditional_v_rbm_bf(W, y, v)  # [B, 2^n_h]

    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Extract the configurations that satisfy the condition
    x_v = x[:n_v, :]  # [n_v, 2^n]
    # [B, n_v, 1] - [1, n_v, 2^n]
    diff = v[:, :, None] - x_v[None, :, :]  # [B, n_v, 2^n]
    mask = (diff == 0).all(dim=1)  # [B, 2^n]
    x_h = x[n_v:, mask[0]]  # [n_h, 2^n_h]

    # Compute the barycenter
    m = torch.matmul(p, x_h.transpose(0, 1))  # [B, n_v]
    return m


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
