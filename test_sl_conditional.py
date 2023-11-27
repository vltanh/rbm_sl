import torch

from constant import DEFAULT_TYPE, SEED
from src.brute_force import conditional_v_rbm_bf, conditional_h_rbm_bf
from src.sl import sample_h_given_v_sl_rbm, sample_v_given_h_sl_rbm
from src.utils import plot_histogram, plot_trajectory

# Set default float precision
torch.set_default_dtype(DEFAULT_TYPE)

# Set seed to be consistent
torch.manual_seed(SEED)


def main():
    # Set dimensions
    n_v, n_h = 2, 2
    n = n_v + n_h
    B = 1000000

    # Set parameters
    L = 100
    delta = .1

    # Generate a random RBM
    W = torch.randn(n_h, n_v).type(DEFAULT_TYPE)
    y = torch.randn(1, n).repeat(B, 1).type(DEFAULT_TYPE)
    # y = torch.zeros(B, n)

    print('W\n', W)
    print('y\n', y)

    # === Fix v ===
    v = torch.randint(0, 2, (1, n_v)).type(DEFAULT_TYPE)
    v = 2 * v - 1
    v = v.repeat(B, 1)
    print('v\n', v)

    # Compute the conditional distribution
    p_h_given_v = conditional_v_rbm_bf(W, y, v)[0]
    print('p(h|v)\n', p_h_given_v)

    # Sample from the conditional distribution
    samples, history = sample_h_given_v_sl_rbm(W, y, v, L, delta)

    plot_histogram(samples, p_h_given_v)
    plot_trajectory(history, max_samples=10)

    # === Fix h ===
    h = torch.randint(0, 2, (1, n_h)).type(DEFAULT_TYPE)
    h = 2 * h - 1
    h = h.repeat(B, 1)
    print('h\n', h[0])

    # Compute the conditional distribution
    p_v_given_h = conditional_h_rbm_bf(W, y, h)[0]
    print('p(v|h)\n', p_v_given_h)

    # Sample from the conditional distribution
    samples, history = sample_v_given_h_sl_rbm(W, y, h, L, delta)

    plot_histogram(samples, p_v_given_h)
    plot_trajectory(history, max_samples=10)


if __name__ == '__main__':
    main()
