import torch
import matplotlib.pyplot as plt
import networkx as nx
import time

from configurations import load_configurations
from energy_based import energy_rbm, barycenter

# Set seed to be consistent
torch.manual_seed(3698)

DEBUG = False


def visualize_factorgraph(W, y):
    G = nx.Graph()

    for i in range(n_v):
        G.add_node(f'$v_{i}$')
        G.add_node(f'$f^v_{i}$')
        G.add_edge(f'$v_{i}$', f'$f^v_{i}$', weight=y[i])

    for i in range(n_h):
        G.add_node(f'$h_{i}$')
        G.add_node(f'$f^h_{i}$')
        G.add_edge(f'$h_{i}$', f'$f^h_{i}$', weight=y[i + n_v])

    for i in range(n_v):
        for j in range(n_h):
            if W[i, j] != 0:
                G.add_node(f'$f_{{{i},{j}}}$')
                G.add_edge(f'$v_{i}$', f'$f_{{{i},{j}}}$', weight=W[i, j])
                G.add_edge(f'$h_{j}$', f'$f_{{{i},{j}}}$', weight=W[i, j])

    nx.draw_planar(G, with_labels=True)
    plt.show()


def marginals_factorgraph(W, y):
    '''
    Compute the marginals of a tree-RBM using factor graph and sum-product

    Args:
        W: [n_v, n_h]
        y: [B, n]

    Returns:
        m: [B, n]
    '''
    n_v, n_h = W.shape
    n = n_v + n_h

    # (2 * #edges * #directions + #nodes) * #values
    messages = {
        'fv2v': dict(),
        'fh2h': dict(),
        'f2v': dict(),
        'f2h': dict(),
        'h2f': dict(),
        'v2f': dict(),
    }

    def compute_message_fv2v(i_v, v):
        if DEBUG:
            _id = f'f^v_{i_v} -> v_{i_v} ({v})'

        if (i_v, v) not in messages['fv2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['fv2v'][(i_v, v)] = \
                (y[:, [i_v]] * v).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fv2v'][(i_v, v)]

    def compute_message_fh2h(i_h, h):
        if DEBUG:
            _id = f'f^h_{i_h} -> h_{i_h} ({h})'

        if (i_h, h) not in messages['fh2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['fh2h'][(i_h, h)] = \
                (y[:, [n_v + i_h]] * h).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fh2h'][(i_h, h)]

    def compute_message_f2v(i_v, i_h, v):
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> v_{i_v} ({v})'

        if (i_v, i_h, v) not in messages['f2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['f2v'][(i_v, i_h, v)] = torch.logsumexp(torch.hstack([
                W[i_v, i_h] * v * 1. + compute_message_h2f(i_v, i_h, 1.),
                W[i_v, i_h] * v * (-1.) + compute_message_h2f(i_v, i_h, -1.)
            ]), dim=1, keepdim=True).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2v'][(i_v, i_h, v)]

    def compute_message_f2h(i_v, i_h, h):
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> h_{i_h} ({h})'

        if (i_v, i_h, h) not in messages['f2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['f2h'][(i_v, i_h, h)] = torch.logsumexp(torch.hstack([
                W[i_v, i_h] * h * 1. + compute_message_v2f(i_v, i_h, 1.),
                W[i_v, i_h] * h * (-1.) + compute_message_v2f(i_v, i_h, -1.)
            ]), dim=1, keepdim=True).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2h'][(i_v, i_h, h)]

    def compute_message_h2f(i_v, i_h, h):
        if DEBUG:
            _id = f'h_{i_h} -> f_{{{i_v},{i_h}}} ({h})'

        if (i_v, i_h, h) not in messages['h2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['h2f'][(i_v, i_h, h)] = \
                compute_message_fh2h(i_h, h).clone()

            for i in range(n_v):
                if i != i_v and W[i, i_h] != 0:
                    messages['h2f'][(i_v, i_h, h)] += \
                        compute_message_f2h(i, i_h, h).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['h2f'][(i_v, i_h, h)]

    def compute_message_v2f(i_v, i_h, v):
        if DEBUG:
            _id = f'v_{i_v} -> f_{{{i_v},{i_h}}} ({v})'

        if (i_v, i_h, v) not in messages['v2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages['v2f'][(i_v, i_h, v)] = \
                compute_message_fv2v(i_v, v).clone()

            for j in range(n_h):
                if j != i_h and W[i_v, j] != 0:
                    messages['v2f'][(i_v, i_h, v)] += \
                        compute_message_f2v(i_v, j, v).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['v2f'][(i_v, i_h, v)]

    def compute_logmarginal_v(i_v, v):
        if DEBUG:
            _id = f'v_{i_v} ({v})'
            print(f'[COMPUTE] {_id}')

        # v_ = torch.FloatTensor([1.0, -1.0])
        m = compute_message_fv2v(i_v, v).clone()
        # m = m[:, [0]] if v == 1.0 else m[:, [1]]

        for j in range(n_h):
            if W[i_v, j] != 0:
                m += compute_message_f2v(i_v, j, v).clone()

        return m

    def compute_logmarginal_h(i_h, h):
        if DEBUG:
            _id = f'h_{i_h} ({h})'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fh2h(i_h, h).clone()

        for i in range(n_v):
            if W[i, i_h] != 0:
                m += compute_message_f2h(i, i_h, h).clone()

        return m

    m = torch.zeros(y.size(0), 2, n)
    for i in range(n_v):
        m[:, [0], [i]] = compute_logmarginal_v(i, 1.0)
        m[:, [1], [i]] = compute_logmarginal_v(i, -1.0)

    for j in range(n_v, n):
        m[:, [0], [j]] = compute_logmarginal_h(j - n_v, 1.0)
        m[:, [1], [j]] = compute_logmarginal_h(j - n_v, -1.0)

    m_ = torch.exp(m - torch.logsumexp(m, dim=1, keepdim=True))
    return m_[:, 0]


def barycenter_factorgraph(W, y):
    marginals = marginals_factorgraph(W, y)
    barycenter = 2 * marginals - 1
    return barycenter


# Set dimensions
n_v, n_h = 2, 1
n = n_v + n_h
B = 3

# Generate random tree-structured RBM
W = torch.randn(n_v, n_h)
# W = torch.ones(n_v, n_h)
W[1:, 1:] = 0.

y = torch.randn(B, n)  # assume to be (n_v, n_h)
# y = torch.ones(2, n)

# print('W\n', W)
# print('y\n', y)

# Generate all configurations
x = load_configurations(n)  # [n, 2^n]

# Compute the barycenter through brute force
m = barycenter(W, y, x, energy_fn=energy_rbm)
marginals = (m + 1) / 2

print('Marginals\n', marginals)
print('Barycenter (brute-force)\n', m)

# # print(x)
# # print('Distribution', distribution(W, y, x, energy_fn=energy_rbm))

# # print('Partition\n', partition(W, y, x, energy_fn=energy_rbm))

start = time.time()
m_ = barycenter_factorgraph(W, y)
elapsed = time.time() - start

print('Barycenter (factor graph)\n', m_)
print(elapsed)
print(torch.max(torch.abs(m - m_)))
print(m - m_)
