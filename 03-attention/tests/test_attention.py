import math
import random

import numpy as np
import pytest

from attention import (
    attention,
    attention_weights,
    entropy_bits,
    jacobian_norm,
    multi_head_attention,
    softmax,
    softmax_jacobian,
    variance,
)


def rand_matrix(rng, rows, cols):
    return [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


def np_attention(q, k, v):
    scores = q @ k.T / math.sqrt(q.shape[1])
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return w @ v


def np_multi_head(x, wq, wk, wv, wo, num_heads):
    seq, d_model = x.shape
    d = d_model // num_heads

    def split(m):  # (seq, d_model) -> (heads, seq, d)
        return m.reshape(seq, num_heads, d).transpose(1, 0, 2)

    q, k, v = split(x @ wq), split(x @ wk), split(x @ wv)
    heads = [np_attention(q[h], k[h], v[h]) for h in range(num_heads)]
    return np.concatenate(heads, axis=1) @ wo


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.array(a) - np.asarray(b))))


def test_matches_numpy_reference():
    rng = random.Random(1)
    q, k, v = rand_matrix(rng, 5, 16), rand_matrix(rng, 7, 16), rand_matrix(rng, 7, 4)
    assert max_abs_diff(attention(q, k, v), np_attention(*map(np.array, (q, k, v)))) < 1e-9

    x = rand_matrix(rng, 6, 8)
    wq, wk, wv, wo = (rand_matrix(rng, 8, 8) for _ in range(4))
    ours = multi_head_attention(x, wq, wk, wv, wo, num_heads=2)
    ref = np_multi_head(*map(np.array, (x, wq, wk, wv, wo)), num_heads=2)
    assert max_abs_diff(ours, ref) < 1e-9


def test_weight_rows_sum_to_one():
    rng = random.Random(2)
    for d_k in (8, 64, 512):
        q, k = rand_matrix(rng, 4, d_k), rand_matrix(rng, 12, d_k)
        for scale in (None, 1.0):  # scaled and unscaled
            for row in attention_weights(q, k, scale):
                assert len(row) == 12
                assert all(w >= 0 for w in row)
                assert abs(sum(row) - 1) < 1e-12


def test_scaling_off_vs_on():
    rng = random.Random(3)
    d_k = 512
    q, k = rand_matrix(rng, 1, d_k), rand_matrix(rng, 12, d_k)

    # Scaling is the same as shrinking the query by sqrt(d_k) before the dot product.
    shrunk = [[x / math.sqrt(d_k) for x in q[0]]]
    assert max_abs_diff(attention_weights(q, k), attention_weights(shrunk, k, scale=1.0)) < 1e-12

    off = attention_weights(q, k, scale=1.0)[0]
    on = attention_weights(q, k)[0]
    # Unscaled, one key takes almost all the weight. Scaled, the row stays spread out.
    assert max(off) > 0.99 and entropy_bits(off) < 0.1
    assert max(on) < 0.5 and entropy_bits(on) > 2.0


def split_heads_wrong_axis(m, num_heads):
    """The bug: numpy's m.reshape(num_heads, seq, d) without the transpose.

    It reads the matrix as one flat list and cuts it into equal pieces, so a
    "head" holds rows made of different tokens' features. Shapes stay right.
    """
    seq, d = len(m), len(m[0]) // num_heads
    flat = [x for row in m for x in row]
    return [
        [flat[(h * seq + i) * d:(h * seq + i + 1) * d] for i in range(seq)]
        for h in range(num_heads)
    ]


def test_wrong_head_axis_runs_but_gives_wrong_output():
    rng = random.Random(4)
    x = rand_matrix(rng, 6, 8)
    wq, wk, wv, wo = (rand_matrix(rng, 8, 8) for _ in range(4))
    ref = np_multi_head(*map(np.array, (x, wq, wk, wv, wo)), num_heads=2)

    wrong = multi_head_attention(x, wq, wk, wv, wo, num_heads=2, split=split_heads_wrong_axis)
    assert len(wrong) == 6 and len(wrong[0]) == 8  # no error, right shape
    assert max_abs_diff(wrong, ref) > 0.1          # wrong numbers

    # With one head the two splits are identical, so a one-head test cannot catch it.
    one_ref = np_multi_head(*map(np.array, (x, wq, wk, wv, wo)), num_heads=1)
    one_wrong = multi_head_attention(x, wq, wk, wv, wo, num_heads=1, split=split_heads_wrong_axis)
    assert max_abs_diff(one_wrong, one_ref) < 1e-9


def test_softmax_jacobian_matches_finite_differences():
    s = [0.3, -1.2, 2.0, 0.0, 0.7]
    p = softmax(s)
    jac = softmax_jacobian(p)
    eps = 1e-6
    for j in range(len(s)):
        up = softmax([x + (eps if i == j else 0) for i, x in enumerate(s)])
        down = softmax([x - (eps if i == j else 0) for i, x in enumerate(s)])
        for i in range(len(s)):
            assert abs((up[i] - down[i]) / (2 * eps) - jac[i][j]) < 1e-8
    frob = math.sqrt(sum(x * x for row in jac for x in row))
    assert abs(jacobian_norm(p) - frob) < 1e-12


def test_entropy_endpoints():
    assert abs(entropy_bits([1 / 12] * 12) - math.log2(12)) < 1e-12
    assert entropy_bits([1.0] + [0.0] * 11) == 0
    # A one-hot row passes no gradient back to the scores.
    assert jacobian_norm([1.0] + [0.0] * 11) == 0


def test_softmax_survives_scores_past_exp_overflow():
    # exp(710) overflows a float. The max-subtraction in softmax must keep it finite.
    scores = [1000.0, 999.0, 0.0, -1000.0]
    ours = softmax(scores)
    shifted = np.array(scores) - max(scores)  # numpy stable reference
    ref = np.exp(shifted) / np.exp(shifted).sum()
    assert max_abs_diff(ours, ref) < 1e-12
    # Hand value: e / (e + 1) = 0.7310586 for the top weight, 0.2689414 for the next.
    assert abs(ours[0] - 0.7310586) < 1e-6 and abs(ours[1] - 0.2689414) < 1e-6
    # Through attention_weights with no scaling: scores are 1000 and 998, a gap of 2.
    row = attention_weights([[1000.0]], [[1.0], [0.998]], scale=1.0)[0]
    assert abs(row[0] - 1 / (1 + math.exp(-2))) < 1e-12


def test_head_count_must_divide_d_model():
    rng = random.Random(5)
    x = rand_matrix(rng, 4, 8)
    w = [rand_matrix(rng, 8, 8) for _ in range(4)]
    with pytest.raises(ValueError, match="divisible"):
        multi_head_attention(x, *w, num_heads=3)
    # 8 divides evenly by 2 and 4, so those still run.
    assert len(multi_head_attention(x, *w, num_heads=4)) == 4


def test_variance_is_population_variance():
    # mean 2.5; squared gaps 2.25 + 0.25 + 0.25 + 2.25 = 5; population: 5 / 4 = 1.25.
    # (Sample variance, 5 / 3, would be 1.6667.)
    assert variance([1, 2, 3, 4]) == 1.25
    assert variance([7.0]) == 0.0
    rng = random.Random(6)
    xs = [rng.gauss(0, 1) for _ in range(50)]
    assert abs(variance(xs) - float(np.var(xs))) < 1e-12  # np.var defaults to ddof=0
