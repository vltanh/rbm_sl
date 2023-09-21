import torch

from configurations import load_configurations


def energy(A, y, x):
    '''
    Compute the energy of a SK

    Args:
        A: [n, n]
        y: [B, n]
        x: [n, 2^n]

    Returns:
        energy: [B, 2^n]
    '''
    xAx = torch.sum(torch.matmul(A.T, x) * x, dim=0)  # [2^n, ]
    xy = torch.matmul(y, x)  # [B, 2^n]
    return - xAx - xy  # [B, 2^n]


def energy_rbm(W, y, x):
    '''
    Compute the energy of a RBM

    Args:
        W: [n_v, n_h]
        y: [B, n]
        x: [n, 2^n]

    Returns:
        energy: [B, 2^n]
    '''
    # Split x into v and h
    n_v, n_h = W.shape
    v = x[:n_v, :]
    h = x[n_v:, :]

    # Compute A * x (Ax)
    print(W.shape, v.shape, h.shape)
    Av = torch.matmul(W.T, v)  # [n_h, 2^n]
    Ah = torch.matmul(W, h)  # [n_v, 2^n]
    Ax = torch.cat((Ah, Av), dim=0) / 2  # [n, 2^n]

    # Compute dot(x, Ax)
    xAx = torch.sum(x * Ax, dim=0)  # [2^n, ]

    # Compute dot(x, y)
    xy = torch.matmul(y, x)  # [B, 2^n]

    # Return the sum
    return - xAx - xy  # [B, 2^n]


def distribution(A, y, x, energy_fn=energy):
    '''
    Compute the distribution

    Args:
        A: [n, n]
        y: [B, n]
        x: [n, 2^n]

    Returns:
        distribution: [B, 2^n]
    '''
    E = -energy_fn(A, y, x)  # [B, 2^n]
    log_partition = torch.logsumexp(E, 1)  # [B, ]
    distribution = torch.exp(E - log_partition.unsqueeze(1))  # [B, 2^n]
    return distribution  # [B, 2^n]


def barycenter(A, y, x, energy_fn=energy):
    '''
    Compute the barycenter of a tilted SK

    Args:
        A: [n, n]
        y: [B, n]
        x: [n, 2^n]
        energy_fn: energy function

    Returns:
        barycenter: [B, n]
    '''
    p = distribution(A, y, x, energy_fn)  # [B, 2^n]
    barycenter = torch.matmul(p, x.transpose(0, 1))
    return barycenter
