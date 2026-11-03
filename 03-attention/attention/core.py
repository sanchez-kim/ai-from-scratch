"""Scaled dot-product attention and multi-head attention in plain Python.

A matrix is a list of rows, and a row is a list of floats. No numpy, so every
multiply is visible. The tests compare this file with a numpy reference.
"""

import math


def matmul(a, b):
    cols = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in cols] for row in a]


def transpose(m):
    return [list(col) for col in zip(*m)]


def softmax(row):
    # Subtracting the largest score changes nothing mathematically
    # and keeps exp() from overflowing on large scores.
    top = max(row)
    exps = [math.exp(s - top) for s in row]
    total = sum(exps)
    return [e / total for e in exps]


def attention_weights(q, k, scale=None):
    """One row of weights per query. scale=None means 1/sqrt(d_k)."""
    d_k = len(q[0])
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = matmul(q, transpose(k))
    return [softmax([s * scale for s in row]) for row in scores]


def attention(q, k, v, scale=None):
    """Each output row is a weighted average of the rows of v."""
    return matmul(attention_weights(q, k, scale), v)


def split_heads(m, num_heads):
    """(seq, d_model) -> num_heads matrices of (seq, d_model / num_heads).

    Heads own columns, not rows: head h gets columns h*d to (h+1)*d
    of every token.
    """
    d = len(m[0]) // num_heads
    return [[row[h * d:(h + 1) * d] for row in m] for h in range(num_heads)]


def merge_heads(heads):
    """The inverse of split_heads: glue each token's head outputs side by side."""
    return [sum((head[i] for head in heads), []) for i in range(len(heads[0]))]


def multi_head_attention(x, wq, wk, wv, wo, num_heads, split=split_heads):
    if len(x[0]) % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    q, k, v = matmul(x, wq), matmul(x, wk), matmul(x, wv)
    heads = [
        attention(qh, kh, vh)
        for qh, kh, vh in zip(split(q, num_heads), split(k, num_heads), split(v, num_heads))
    ]
    return matmul(merge_heads(heads), wo)
