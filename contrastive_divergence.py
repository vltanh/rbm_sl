import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveDivergence(nn.Module):
    """Contrastive Divergence.
    Args:
        rbm (RBM): The RBM model.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, rbm, k=1):
        """Create a Contrastive Divergence."""
        super().__init__()
        self.rbm = rbm
        self.k = k

    def free_energy(self, v):
        return self.rbm.free_energy(v)

    def visible_to_hidden(self, v):
        r"""Conditional sampling a hidden variable given a visible variable.
        Args:
            v (Tensor): The visible variable.
        Returns:
            Tensor: The hidden variable.
        """
        p = torch.sigmoid(F.linear(v, 2 * self.rbm.W, self.rbm.h))
        return 2 * p.bernoulli() - 1

    def hidden_to_visible(self, h):
        r"""Conditional sampling a visible variable given a hidden variable.
        Args:
            h (Tendor): The hidden variable.
        Returns:
            Tensor: The visible variable.
        """
        p = torch.sigmoid(F.linear(h, 2 * self.rbm.W.t(), self.rbm.v))
        return 2 * p.bernoulli() - 1

    def forward(self, v):
        r"""Compute the generated examples.
        Args:
            v (Tensor): The visible variable.
        Returns:
            (Tensor, Tensor): The generated variables.
        """
        h = self.visible_to_hidden(v)
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v_gibb)
        return v_gibb
