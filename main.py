import torch
import matplotlib.pyplot as plt
import networkx as nx
import time

from configurations import load_configurations
from energy_based import energy_rbm, barycenter

# Set seed to be consistent
torch.manual_seed(3698)

DEBUG = False
CUDA = False


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

    Let
        n_v: number of visible nodes
        n_h: number of hidden nodes
        n: n_v + n_h
        B: batch size

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

    if CUDA:
        W = W.cuda()
        y = y.cuda()
        values = values.cuda()

    def compute_message_fv2v(i_v: int) -> torch.Tensor:
        '''
        Compute the message sent from f^v_i to v_i

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f^v_{i_v} -> v_{i_v}'

        if i_v not in messages['fv2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = y[:, [i_v]] * values[None]  # [B, 1] * [1, 2] = [B, 2]

            messages['fv2v'][i_v] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fv2v'][i_v].clone()  # [B, 2]

    def compute_message_fh2h(i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f^h_i to h_i

        Args:
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f^h_{i_h} -> h_{i_h}'

        if i_h not in messages['fh2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = y[:, [n_v + i_h]] * values[None]  # [B, 1] * [1, 2] = [B, 2]

            messages['fh2h'][i_h] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['fh2h'][i_h].clone()  # [B, 2]

    def compute_message_f2v(i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f_{i_v, i_h} to v_i

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> v_{i_v}'

        if (i_v, i_h) not in messages['f2v']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [1, 1] * [2, 1] * [1, 2] = [2, 2]
            vWh = W[i_v, i_h] * values[:, None] * values[None, :]
            h2f = compute_message_h2f(i_v, i_h)  # [B, 2]
            # [2, 1, 2] + [1, B, 2] = [2, B, 2]
            vWh_h2f = vWh[:, None, :] + h2f[None, :, :]

            # [2, B, 2] -> [2, B, 1]
            notsum = torch.logsumexp(vWh_h2f, dim=-1, keepdim=True)
            # [2, B, 1] -> [B, 2]
            m = notsum.squeeze(-1).transpose(0, 1)

            messages['f2v'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2v'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_f2h(i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from f_{i_v, i_h} to h_i

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'f_{{{i_v},{i_h}}} -> h_{i_h}'

        if (i_v, i_h) not in messages['f2h']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            # [1, 1] * [2, 1] * [1, 2] = [2, 2]
            vWh = W[i_v, i_h] * values[:, None] * values[None, :]
            # [2, 1, 2] + [1, B, 2] = [2, B, 2]
            v2f = compute_message_v2f(i_v, i_h)
            # [2, B, 2] -> [2, B, 1]
            vWh_v2f = vWh[:, None, :] + v2f[None, :, :]

            # [2, B, 2] -> [2, B, 1]
            notsum = torch.logsumexp(vWh_v2f, dim=-1, keepdim=True)
            # [2, B, 1] -> [B, 2]
            m = notsum.squeeze(-1).transpose(0, 1)

            messages['f2h'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['f2h'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_h2f(i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from h_i to f_{i_v, i_h}

        Args:
            i_v (int): index of the visible node
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'h_{i_h} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in messages['h2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = compute_message_fh2h(i_h)  # [B, 2]
            for i in range(n_v):
                if i != i_v and W[i, i_h] != 0:
                    # [B, 2] + [B, 2] = [B, 2]
                    m += compute_message_f2h(i, i_h).clone()

            messages['h2f'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['h2f'][(i_v, i_h)].clone()  # [B, 2]

    def compute_message_v2f(i_v: int, i_h: int) -> torch.Tensor:
        '''
        Compute the message sent from v_i to f_{i_v, i_h}

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] message
        '''
        if DEBUG:
            _id = f'v_{i_v} -> f_{{{i_v},{i_h}}}'

        if (i_v, i_h) not in messages['v2f']:
            if DEBUG:
                print(f'[COMPUTE] {_id}')

            m = compute_message_fv2v(i_v)  # [B, 2]
            for j in range(n_h):
                if j != i_h and W[i_v, j] != 0:
                    m += compute_message_f2v(i_v, j).clone()  # [B, 2]

            messages['v2f'][(i_v, i_h)] = m.clone()  # [B, 2]
        else:
            if DEBUG:
                print(f'[CACHE] {_id}')

        return messages['v2f'][(i_v, i_h)].clone()  # [B, 2]

    def compute_logmarginal_v(i_v: int) -> torch.Tensor:
        '''
        Compute the log-marginal of v_i

        Args:
            i_v (int): index of the visible node

        Returns:
            m (torch.Tensor): [B, 2] log-marginal
        '''
        if DEBUG:
            _id = f'v_{i_v}'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fv2v(i_v)  # [B, 2]

        for j in range(n_h):
            if W[i_v, j] != 0:
                # [B, 2] + [B, 2] = [B, 2]
                m += compute_message_f2v(i_v, j)

        return m  # [B, 2]

    def compute_logmarginal_h(i_h: int) -> torch.Tensor:
        '''
        Compute the log-marginal of h_i

        Args:
            i_h (int): index of the hidden node

        Returns:
            m (torch.Tensor): [B, n] log-marginal
        '''
        if DEBUG:
            _id = f'h_{i_h}'
            print(f'[COMPUTE] {_id}')

        m = compute_message_fh2h(i_h)  # [B, 2]

        for i in range(n_v):
            if W[i, i_h] != 0:
                # [B, 2] + [B, 2] = [B, 2]
                m += compute_message_f2h(i, i_h)

        return m  # [B, 2]

    # Store the marginals
    m = torch.zeros(y.size(0), values.size(0), n)  # [B, 2, n]
    for i in range(n_v):
        m[:, :, i] = compute_logmarginal_v(i)  # [B, 2] -> [B, 2, n]

    for j in range(n_v, n):
        m[:, :, j] = compute_logmarginal_h(j - n_v)  # [B, 2] -> [B, 2, n]

    log_partition = torch.logsumexp(m, dim=1, keepdim=True)  # [B, 1, n]
    log_marginals = m - log_partition  # [B, 2, n] - [B, 1, n] = [B, 2, n]
    m_ = torch.exp(log_marginals)  # [B, 2, n] -> [B, 2, n]
    return m_[:, 0]  # [B, 2, n] -> [B, n]


def barycenter_factorgraph(W, y):
    marginals = marginals_factorgraph(W, y)
    barycenter = 2 * marginals - 1
    return barycenter


# Set dimensions
n_v, n_h = 4, 3
n = n_v + n_h
B = 1024

# Generate random tree-structured RBM
W = torch.randn(n_v, n_h)
# W = torch.ones(n_v, n_h)
W[1:, :-1] = 0.

y = torch.randn(B, n)  # assume to be (n_v, n_h)
# y = torch.ones(2, n)

# print('W\n', W)
# print('y\n', y)

# Generate all configurations
x = load_configurations(n)  # [n, 2^n]

# Compute the barycenter through brute force
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
