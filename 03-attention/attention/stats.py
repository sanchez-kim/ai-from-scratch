"""Numbers that describe one row of attention weights."""

import math


def entropy_bits(p):
    """How spread out the weights are, in bits.

    log2(n) when all n weights are equal, 0 when one weight is 1.
    """
    return -sum(x * math.log2(x) for x in p if x > 0)


def softmax_jacobian(p):
    """d p_i / d s_j for p = softmax(s): p_i * (1 if i == j else 0) - p_i * p_j.

    This matrix is what a gradient passes through on its way back from the
    weights to the scores.
    """
    n = len(p)
    return [[p[i] * ((1.0 if i == j else 0.0) - p[j]) for j in range(n)] for i in range(n)]


def jacobian_norm(p):
    """Frobenius norm of softmax_jacobian(p), without building the n x n matrix.

    sum_ij (p_i d_ij - p_i p_j)^2 = sum p_i^2 - 2 sum p_i^3 + (sum p_i^2)^2
    """
    s2 = sum(x * x for x in p)
    s3 = sum(x * x * x for x in p)
    return math.sqrt(max(s2 - 2 * s3 + s2 * s2, 0.0))


def variance(xs):
    """Population variance."""
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)
