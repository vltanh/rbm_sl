import torch
from torch import nn
from torch.nn import functional as F


class TreeRBM(nn.Module):
    def __init__(self, n_v: int, n_h: int):
        '''
        Tree-structured Restricted Boltzmann Machine

        Args:
            n_v (int): The number of visible nodes.
            n_h (int): The number of hidden nodes.
        '''
        super().__init__()

        # Set dimensions
        assert n_v % n_h == 0 or n_h % n_v == 0, \
            'The number of visible nodes and hidden nodes must be multiples of each other.'
        self.n_v = n_v
        self.n_h = n_h

        # # Initialize biases
        self.v = nn.Parameter(torch.zeros(1, n_v))
        self.h = nn.Parameter(torch.zeros(1, n_h))
        self.v.requires_grad = False
        self.h.requires_grad = False

        # Initialize weights
        self.W = torch.randn(n_h, n_v)

        if n_v % n_h == 0:
            k = n_v // n_h
            for i in range(n_h):
                self.W[i, : i * k] = 0.
                self.W[i, (i + 1) * k:] = 0.
        elif n_h % n_v == 0:
            k = n_h // n_v
            for i in range(n_v):
                self.W[: i * k, i] = 0.
                self.W[(i + 1) * k:, i] = 0.

        self.W = nn.Parameter(self.W)

    def mask_grad(self):
        # self.W.grad.data[1:, 1:].fill_(0)
        if self.n_v % self.n_h == 0:
            k = self.n_v // self.n_h
            for i in range(self.n_h):
                self.W.grad.data[i, : i * k].fill_(0)
                self.W.grad.data[i, (i + 1) * k:].fill_(0)
        elif self.n_h % self.n_v == 0:
            k = self.n_h // self.n_v
            for i in range(self.n_v):
                self.W.grad.data[: i * k, i].fill_(0)
                self.W.grad.data[(i + 1) * k:, i].fill_(0)

    def free_energy(self, v):
        r"""Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        """
        # v_term = torch.matmul(v, self.v.t())
        # w_x_h = F.linear(v, self.W, self.h)
        # h_term = torch.sum(F.softplus(w_x_h), dim=1)
        # return torch.mean(-h_term - v_term)
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def to_interaction_matrix(self):
        '''
        Convert the RBM to the interaction matrix.

        Returns:
            Tensor: The interaction matrix.
        '''
        return torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(self.n_v, self.n_v),
                        self.W.t(),
                    ],
                    dim=1
                ),
                torch.cat(
                    [
                        self.W,
                        torch.zeros(self.n_h, self.n_h),
                    ],
                    dim=1
                )
            ],
            dim=0
        ) / 2

    def forward(self):
        pass


class RBM(nn.Module):
    """Restricted Boltzmann Machine.
    Args:
        n_v (int, optional): The size of visible layer. Defaults to 784.
        n_h (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_v, n_h):
        """Create a RBM."""
        super().__init__()
        self.n_v = n_v
        self.n_h = n_h
        self.v = torch.zeros(1, n_v)
        self.h = torch.zeros(1, n_h)
        self.W = nn.Parameter(torch.randn(n_h, n_v))

    def free_energy(self, v):
        r"""Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        """
        # import pdb; pdb.set_trace()
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def to_interaction_matrix(self):
        """Convert the RBM to the interaction matrix.
        Returns:
            Tensor: The interaction matrix.
        """
        return torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(self.n_v, self.n_v),
                        self.W.t(),
                    ],
                    dim=1
                ),
                torch.cat(
                    [
                        self.W,
                        torch.zeros(self.n_h, self.n_h),
                    ],
                    dim=1
                )
            ],
            dim=0
        ) / 2

    def forward(self):
        pass
