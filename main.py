import torch
import matplotlib.pyplot as plt
import networkx as nx
import time

from configurations import load_configurations
from energy_based import energy_rbm, barycenter

# Set seed to be consistent
torch.manual_seed(0)

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

    messages = dict()

    def compute_message_fv2v(idx, v):
        if DEBUG:
            _id = f'f^v_{idx} -> v_{idx} ({v})'

        if (('fv2v', idx), idx, v) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages[(('fv2v', idx), idx, v)] = y[:, [idx]] * v
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages[(('fv2v', idx), idx, v)]

    def compute_message_fh2h(idx, h):
        if DEBUG:
            _id = f'f^h_{idx} -> h_{idx} ({h})'

        if (('fh2h', idx), idx, h) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages[(('fh2h', idx), idx, h)] = y[:, [n_v + idx]] * h
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages[(('fh2h', idx), idx, h)]

    def compute_message_f2v(src, dst, v):
        if DEBUG:
            _id = f'f_{{{src[0]},{src[1]}}} -> v_{dst} ({v})'

        if ('f2v', src, dst, v) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            i, j = src
            messages[('f2v', src, dst, v)] = torch.logsumexp(torch.hstack([
                W[i, j] * v * 1. + compute_message_h2f(j, src, 1.),
                W[i, j] * v * (-1.) + compute_message_h2f(j, src, -1.)
            ]), dim=1, keepdim=True)
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages[('f2v', src, dst, v)]

    def compute_message_f2h(src, dst, h):
        if DEBUG:
            _id = f'f_{{{src[0]},{src[1]}}} -> h_{dst} ({h})'

        if ('f2h', src, dst, h) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            i, j = src
            messages[('f2h', src, dst, h)] = torch.logsumexp(torch.hstack([
                W[i, j] * h * 1. + compute_message_v2f(i, src, 1.),
                W[i, j] * h * (-1.) + compute_message_v2f(i, src, -1.)
            ]), dim=1, keepdim=True)
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages[('f2h', src, dst, h)]

    def compute_message_h2f(src, dst, h):
        if DEBUG:
            _id = f'h_{src} -> f_{{{dst[0]},{dst[1]}}} ({h})'

        if ('h2f', src, dst) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages[('h2f', src, dst, h)] = \
                compute_message_fh2h(src, h).clone()

            for i in range(n_v):
                if (i, src) != dst and W[i, src] != 0:
                    messages[('h2f', src, dst, h)] += \
                        compute_message_f2h((i, src), src, h).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages[('h2f', src, dst, h)]

    def compute_message_v2f(src, dst, v):
        if DEBUG:
            _id = f'v_{src} -> f_{{{dst[0]},{dst[1]}}} ({v})'

        if ('v2f', src, dst, v) not in messages:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            messages[('v2f', src, dst, v)] = \
                compute_message_fv2v(src, v).clone()
            for j in range(n_h):
                if (src, j) != dst and W[src, j] != 0:
                    messages[('v2f', src, dst, v)] += \
                        compute_message_f2v((src, j), src, v).clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')
                
        return messages[('v2f', src, dst, v)]

    def compute_logmarginal_v(idx, v):
        if DEBUG:
            _id = f'v_{idx} ({v})'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fv2v(idx, v).clone()
        for j in range(n_h):
            if W[idx, j] != 0:
                m += compute_message_f2v((idx, j), idx, v).clone()
        return m

    def compute_logmarginal_h(idx, h):
        if DEBUG:
            _id = f'h_{idx} ({h})'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fh2h(idx, h).clone()
        for i in range(n_v):
            if W[i, idx] != 0:
                m += compute_message_f2h((i, idx), idx, h).clone()
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
n_v, n_h = 2, 3
n = n_v + n_h
B = 10

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
