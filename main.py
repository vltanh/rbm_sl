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

    values = torch.FloatTensor([1.0, -1.0])

    def compute_message_fv2v(i_v):
        if DEBUG:
            _id = f'f^v_{i_v} -> v_{i_v}'

        if i_v not in messages['fv2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = y[:, [i_v]] * values[None]

            messages['fv2v'][i_v] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fv2v'][i_v].clone()

    def compute_message_fh2h(i_h):
        if DEBUG:
            _id = f'f^h_{i_h} -> h_{i_h}'

        if i_h not in messages['fh2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = y[:, [n_v + i_h]] * values[None]

            messages['fh2h'][i_h] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fh2h'][i_h].clone()

    def compute_message_f2v(i_v, i_h):
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> v_{i_v}'

        if (i_v, i_h) not in messages['f2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            vWh = W[i_v, i_h] * values[:, None] * values[None, :]
            h2f = compute_message_h2f(i_v, i_h)
            vWh_h2f = vWh[:, None, :] + h2f[None, :, :]

            notsum = torch.logsumexp(vWh_h2f, dim=-1, keepdim=True)
            m = notsum.squeeze(-1).transpose(0, 1)

            messages['f2v'][(i_v, i_h)] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2v'][(i_v, i_h)].clone()

    def compute_message_f2h(i_v, i_h):
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> h_{i_h}'

        if (i_v, i_h) not in messages['f2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            vWh = W[i_v, i_h] * values[:, None] * values[None, :]
            v2f = compute_message_v2f(i_v, i_h)
            vWh_v2f = vWh[:, None, :] + v2f[None, :, :]

            notsum = torch.logsumexp(vWh_v2f, dim=-1, keepdim=True)
            m = notsum.squeeze(-1).transpose(0, 1)

            messages['f2h'][(i_v, i_h)] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2h'][(i_v, i_h)].clone()

    def compute_message_h2f(i_v, i_h):
        if DEBUG:
            _id = f'h_{i_h} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in messages['h2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = compute_message_fh2h(i_h)
            for i in range(n_v):
                if i != i_v and W[i, i_h] != 0:
                    m += compute_message_f2h(i, i_h).clone()

            messages['h2f'][(i_v, i_h)] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['h2f'][(i_v, i_h)].clone()

    def compute_message_v2f(i_v, i_h):
        if DEBUG:
            _id = f'v_{i_v} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in messages['v2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = compute_message_fv2v(i_v)
            for j in range(n_h):
                if j != i_h and W[i_v, j] != 0:
                    m += compute_message_f2v(i_v, j).clone()

            messages['v2f'][(i_v, i_h)] = m.clone()
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['v2f'][(i_v, i_h)].clone()

    def compute_logmarginal_v(i_v):
        if DEBUG:
            _id = f'v_{i_v}'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fv2v(i_v)

        for j in range(n_h):
            if W[i_v, j] != 0:
                m += compute_message_f2v(i_v, j)

        return m

    def compute_logmarginal_h(i_h):
        if DEBUG:
            _id = f'h_{i_h}'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fh2h(i_h)

        for i in range(n_v):
            if W[i, i_h] != 0:
                m += compute_message_f2h(i, i_h)

        return m

    m = torch.zeros(y.size(0), 2, n)
    for i in range(n_v):
        m[:, :, i] = compute_logmarginal_v(i)

    for j in range(n_v, n):
        m[:, :, j] = compute_logmarginal_h(j - n_v)

    m_ = torch.exp(m - torch.logsumexp(m, dim=1, keepdim=True))
    return m_[:, 0]


def barycenter_factorgraph(W, y):
    marginals = marginals_factorgraph(W, y)
    barycenter = 2 * marginals - 1
    return barycenter


# Set dimensions
n_v, n_h = 11, 11
n = n_v + n_h
B = 512

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

# # Compute the barycenter through brute force
m = barycenter(W, y, x, energy_fn=energy_rbm)
marginals = (m + 1) / 2

# print('Marginals\n', marginals)
# print('Barycenter (brute-force)\n', m)

start = time.time()
m_ = barycenter_factorgraph(W, y)
elapsed = time.time() - start

# print('Barycenter (factor graph)\n', m_)
print('Time:', elapsed)
print('Error (uniform):', torch.max(torch.abs(m - m_)))
