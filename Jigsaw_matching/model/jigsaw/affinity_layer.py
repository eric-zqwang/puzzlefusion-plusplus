import math
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class Affinity(nn.Module):
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d

        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hd)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.hd)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, self.A)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M


class AffinityDual(nn.Module):
    def __init__(self, d):
        super(AffinityDual, self).__init__()
        self.d = d
        assert d % 2 == 0
        self.hd = d // 2

        self.A = Parameter(Tensor(self.hd, self.hd))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hd)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.hd)

    def forward(self, X, Y):
        """
        Compute affinity metric based on the primal descriptor of X
        and the dual descriptor of Y.
        :param X: X[:, :, :self.hd] is the primal descriptor of X.
        :param Y: Y[:, :, self.hd:] is the dual descriptor of Y.
        :return: Affinity metric.
        """
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X[:, :, : self.hd], self.A)
        M = torch.matmul(M, Y[:, :, self.hd:].transpose(1, 2))
        return M


def build_affinity(affinity, dim):
    if affinity.lower() == "aff_dual":
        affinity_layer = AffinityDual(dim)
    elif affinity.lower() == "aff":
        affinity_layer = Affinity(dim)
    else:
        raise NotImplementedError(f"affinity {affinity} not implemented")
    return affinity_layer
