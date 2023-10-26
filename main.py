import torch
import time

from brute_force import barycenter_bf
from energy_based import energy_rbm

from factor_graph import TreeRBMFactorGraph

from constant import DEFAULT_TYPE, SEED, DEBUG, BRUTE_FORCE

# Set default float precision
torch.set_default_dtype(DEFAULT_TYPE)

# Set seed to be consistent
torch.manual_seed(SEED)

# Set dimensions
n_v, n_h = 4, 8
n = n_v + n_h
B = 1024

if n > 20:
    BRUTE_FORCE = False

# Generate random tree-structured RBM
W = torch.randn(n_v, n_h).type(DEFAULT_TYPE)
# W = torch.ones(n_v, n_h)
W[:-1, 1:] = 0.

# y = torch.randn(B, n)  # assume to be (n_v, n_h)
y = torch.zeros(B, n)

if DEBUG:
    print('W\n', W)
    print('y\n', y)

# Algorithms
fg = TreeRBMFactorGraph(W, y)

start = time.time()
m_factorgraph = fg.barycenter()
elapsed = time.time() - start
print('Time:', elapsed)

if DEBUG:
    print('Barycenter (factor graph)\n', m_factorgraph)

if BRUTE_FORCE:
    # Compute the barycenter through brute force
    m_bruteforce = barycenter_bf(W, y, energy_fn=energy_rbm)
    marginals = (m_bruteforce + 1) / 2

    if DEBUG:
        print('Marginals\n', marginals)
        print('Barycenter (brute-force)\n', m_bruteforce)

    print(
        'Error (uniform):',
        torch.max(torch.abs(m_bruteforce - m_factorgraph))
    )
