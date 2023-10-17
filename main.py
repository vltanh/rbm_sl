import torch
import time

from configurations import load_configurations
from energy_based import energy_rbm, barycenter

from factor_graph import TreeRBMFactorGraph

DEFAULT_TYPE = torch.float64
SEED = 0

DEBUG = False
CUDA = False
BRUTE_FORCE = True

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
W = torch.randn(n_v, n_h)
# W = torch.ones(n_v, n_h)
W[:-1, 1:] = 0.

y = torch.randn(B, n)  # assume to be (n_v, n_h)
# y = torch.ones(2, n)

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
    # Generate all configurations
    x = load_configurations(n).type(DEFAULT_TYPE)  # [n, 2^n]

    # Compute the barycenter through brute force
    m_bruteforce = barycenter(W, y, x, energy_fn=energy_rbm)
    marginals = (m_bruteforce + 1) / 2

if DEBUG:
    print('Marginals\n', marginals)
    print('Barycenter (brute-force)\n', m_bruteforce)

if BRUTE_FORCE:
    print(
        'Error (uniform):',
        torch.max(torch.abs(m_bruteforce - m_factorgraph))
    )
