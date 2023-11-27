import torch


def energy(A, y, x):
    '''
    Compute the energy of a SK
        E(x; A, y) = - <x, Ax> - <x, y>

    Args:
        A: [n, n]
        y: [B_y, n]
        x: [n, B_x]

    Returns:
        energy: [B_y, B_x]
    '''
    xAx = torch.sum(torch.matmul(A.T, x) * x, dim=0)  # [B_x, ]
    xy = torch.matmul(y, x)  # [B_y, B_x]
    return - xAx - xy  # [B_y, B_x]


def energy_rbm(W, y, x):
    '''
    Compute the energy of a RBM
        x = (v, h)
        E(x; W, y) = - 1/2 <v, Wh> - <x, y>

    Args:
        W: [n_h, n_v]
        y: [B_y, n]
        x: [n_v + n_h, B_x]

    Returns:
        energy: [B_y, B_x]
    '''
    # Split x into v and h
    n_h, n_v = W.shape
    v = x[:n_v, :]  # [n_v, B_x]
    h = x[n_v:, :]  # [n_h, B_x]

    # Compute A * x (Ax)
    Av = torch.matmul(W, v)  # [n_h, B_x]
    Ah = torch.matmul(W.T, h)  # [n_v, B_x]
    Ax = torch.cat((Ah, Av), dim=0) / 2  # [n, B_x]

    # Compute dot(x, Ax)
    xAx = torch.sum(x * Ax, dim=0)  # [B_x, ]

    # Compute dot(x, y)
    xy = torch.matmul(y, x)  # [B_y, B_x]

    # Return the sum
    return - xAx - xy  # [B_y, B_x]


def partition(A, y, x, energy_fn):
    '''
    Compute the partition

    Args:
        A: [n, n]
        y: [B_y, n]
        x: [n, B_x]

    Returns:
        distribution: [B_y, B_x]
    '''
    E = -energy_fn(A, y, x)  # [B_y, B_x]
    log_partition = torch.logsumexp(E, 1)  # [B_y, ]
    return log_partition  # [B_y, B_x]


def distribution(A, y, x, energy_fn):
    '''
    Compute the distribution 
        p(x) = \exp(-E(x)) / Z
    where 
        Z = \sum_x \exp(-E(x))

    Args:
        A: [n, n]
        y: [B_y, n]
        x: [n, B_x]

    Returns:
        distribution: [B_y, B_x]
    '''
    E = -energy_fn(A, y, x)  # [B_y, B_x]
    log_Z = torch.logsumexp(E, 1)  # [B_y, ]
    p = torch.exp(E - log_Z.unsqueeze(1))  # [B_y, B_x]
    return p  # [B_y, B_x]
