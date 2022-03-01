from common.math import euclidean_norm_squared, pairwise_distances
import torch
import math as m

# Based on https://github.com/gmum/cwae-pytorch/blob/master/src/metrics/cw.py


def cw_normality(X: torch.Tensor) -> torch.Tensor:
    assert len(X.size()) == 2

    N, D = X.size()

    y = __silverman_rule_of_thumb_normal(N)

    K = 1.0/(2.0*D-3.0)

    A1 = pairwise_distances(X)
    A = (1/torch.sqrt(y + K*A1)).mean()

    B1 = euclidean_norm_squared(X, axis=1)
    B = 2*((1/torch.sqrt(y + 0.5 + K*B1))).mean()

    return (1/m.sqrt(1+y)) + A - B


def __silverman_rule_of_thumb_normal(N: int) -> float:
    return (4/(3*N))**0.4
