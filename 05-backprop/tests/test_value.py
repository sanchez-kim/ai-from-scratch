import math
import random

import pytest

from backprop import (AssignValue, Value, backward_forgot_reversed, backward_recursive,
                      central_diff, grad_check, numeric_grads)
from backprop.graphs import random_graph


def test_slope_of_x_squared_is_6_at_3():
    x = Value(3.0)
    (x * x).backward()
    assert x.grad == 6.0
    # Nudge x and watch f(x) = x^2 move: the same slope.
    assert central_diff(lambda t: t * t, 3.0, 1e-4) == pytest.approx(6.0, abs=1e-9)


def test_diamond_adds_both_paths_and_assign_bug_keeps_one():
    a = Value(2.0)
    (a * 3 + a * 5).backward()
    assert a.grad == 8.0
    # Bug 1: the path whose _backward runs last wins. No error either way.
    a = AssignValue(2.0)
    (a * 3 + a * 5).backward()
    assert a.grad == 3.0
    a = AssignValue(2.0)
    (a * 5 + a * 3).backward()
    assert a.grad == 5.0
    # x * x uses x twice inside one op; x ** 2 uses it once.
    x = AssignValue(3.0)
    (x * x).backward()
    assert x.grad == 3.0
    x = AssignValue(3.0)
    (x ** 2).backward()
    assert x.grad == 6.0


def test_backward_matches_finite_differences():
    def f(v):
        x, y, z = v
        return ((x * y + z.tanh()) / (y ** 2 + 1) - (x - z).exp() * 0.3) * x
    xs = [0.7, -1.3, 0.4]

    vals = [Value(t) for t in xs]
    f(vals).backward()
    expected = numeric_grads(lambda ts: f([Value(t) for t in ts]).data, xs)
    for v, g in zip(vals, expected):
        assert v.grad == pytest.approx(g, rel=1e-7, abs=1e-9)

    # Random graphs with shared nodes: 50 graphs of 8 ops.
    for seed in range(50):
        leaves, root, _ = random_graph(seed, 8)
        root.backward()

        def g(ts, seed=seed):
            ls, r, _ = random_graph(seed, 8)
            for leaf, t in zip(ls, ts):
                leaf.data = t
            # Recompute the forward pass with the nudged leaves.
            return _forward(r)
        num = numeric_grads(g, [v.data for v in leaves])
        for v, n in zip(leaves, num):
            assert v.grad == pytest.approx(n, rel=1e-6, abs=1e-8)


def _forward(node, memo=None):
    memo = {} if memo is None else memo
    if id(node) in memo:
        return memo[id(node)]
    kids = [_forward(c, memo) for c in node._prev]
    out = {"": lambda: node.data, "+": lambda: kids[0] + kids[1],
           "*": lambda: kids[0] * kids[1], "tanh": lambda: math.tanh(kids[0])}[node._op]()
    memo[id(node)] = out
    return out


def test_forgetting_reversed_leaves_gradients_at_zero():
    a = Value(2.0)
    L = a * 3 + a * 5
    backward_forgot_reversed(L)
    assert a.grad == 0.0  # no error, just zero


def test_recursive_backward_passes_simple_tests_and_fails_on_shared_inner_node():
    # Only a leaf is shared: the recursive version gets it right.
    a = Value(2.0)
    backward_recursive(a * 3 + a * 5)
    assert a.grad == 8.0
    # Share an inner node: a = x * x feeds two products.
    x = Value(2.0)
    a = x * x
    L = a * 3 + a * 5
    L.backward()
    assert (x.grad, a.grad) == (32.0, 8.0)
    x = Value(2.0)
    a = x * x
    backward_recursive(a * 3 + a * 5)
    assert a.grad == 8.0      # a itself is right
    assert x.grad == 44.0     # 12 + 32: a passed on 3 first, then all 8


def test_calling_backward_twice_without_zeroing_overcounts():
    a = Value(2.0)
    L = a * 3 + a * 5
    L.backward()
    L.backward()
    # Not 16: the inner nodes kept their grads from the first call too.
    assert a.grad == 24.0


def test_gradient_check_catches_both_bugs_on_a_shared_inner_node():
    # x feeds a = x * x, and a feeds two products: a shared inner node.
    def nested(v):
        a = v[0] * v[0]
        return a * 3 + a * 5
    assert grad_check(nested, [2.0]) < 1e-6
    assert grad_check(nested, [2.0], cls=AssignValue) > 1          # bug 1
    assert grad_check(nested, [2.0], backward=backward_forgot_reversed) > 1  # bug 2a
    assert grad_check(nested, [2.0], backward=backward_recursive) > 1        # bug 2b


def test_numpy_reference_matches_value_layer():
    np = pytest.importorskip("numpy")
    from bench_numpy import numpy_grad, value_grad
    rng = random.Random(0)
    n = 8
    w = [[rng.gauss(0, 0.3) for _ in range(n)] for _ in range(n)]
    x = [rng.gauss(0, 1) for _ in range(n)]
    g, nodes = value_grad(w, x)
    assert nodes == 225
    assert float(np.max(np.abs(np.array(g) - numpy_grad(np.array(w), np.array(x))))) < 1e-12
