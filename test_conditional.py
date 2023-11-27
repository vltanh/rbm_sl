import torch

from constant import DEFAULT_TYPE, SEED
from src.brute_force import marginal_h_rbm_bf, marginal_v_rbm_bf, \
    conditional_h_rbm_bf, conditional_v_rbm_bf, \
    barycenter_conditional_h_rbm_bf, barycenter_conditional_v_rbm_bf

# Set default float precision
torch.set_default_dtype(DEFAULT_TYPE)

# Set seed to be consistent
torch.manual_seed(SEED)


def main():
    # Set dimensions
    n_v, n_h = 5, 5
    n = n_v + n_h
    B = 1

    # Generate a random RBM
    W = torch.randn(n_h, n_v).type(DEFAULT_TYPE)
    y = torch.randn(B, n)
    # y = torch.zeros(B, n)

    print('W\n', W)
    print('y\n', y)

    # === Fix h ===
    print('=========')
    h = torch.randint(0, 2, (B, n_h)).type(DEFAULT_TYPE)
    h = 2 * h - 1
    print('h\n', h)

    # Compute the marginal distribution p(h)
    p_h = marginal_h_rbm_bf(W, y)
    print('p(h)\n', p_h)

    # Compute the conditional distribution
    p_v_given_h = conditional_h_rbm_bf(W, y, h)
    print('p(v|h)\n', p_v_given_h)

    # Compute the barycenter of the conditional distribution
    E_v_given_h = barycenter_conditional_h_rbm_bf(W, y, h)
    print('E(v|h)\n', E_v_given_h)

    # === Fix v ===
    print('=========')
    v = torch.randint(0, 2, (B, n_v)).type(DEFAULT_TYPE)
    v = 2 * v - 1
    print('v\n', v)

    # Compute the marginal distribution p(v)
    p_v = marginal_v_rbm_bf(W, y)
    print('p(v)\n', p_v)

    # Compute the conditional distribution
    p_h_given_v = conditional_v_rbm_bf(W, y, v)
    print('p(h|v)\n', p_h_given_v)

    # Compute the barycenter of the conditional distribution
    E_h_given_v = barycenter_conditional_v_rbm_bf(W, y, v)
    print('E(h|v)\n', E_h_given_v)


if __name__ == '__main__':
    main()
