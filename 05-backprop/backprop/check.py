"""Finite differences: nudge the input, watch the output."""


def forward_diff(f, x, eps):
    return (f(x + eps) - f(x)) / eps


def central_diff(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def numeric_grads(f, xs, eps=1e-6):
    """Central difference for each input of f(list_of_floats) -> float."""
    grads = []
    for i in range(len(xs)):
        def g(t, i=i):
            ys = list(xs)
            ys[i] = t
            return f(ys)
        grads.append(central_diff(g, xs[i], eps))
    return grads


def grad_check(expr, xs, cls=None, backward=None, eps=1e-6):
    """Largest |backward() gradient - central difference| over the inputs.

    expr takes a list of Values and returns one Value. cls and backward
    let a test swap in a buggy Value class or a buggy backward function.
    """
    from .value import Value
    cls = cls or Value
    vals = [cls(t) for t in xs]
    root = expr(vals)
    (backward or type(root).backward)(root)
    num = numeric_grads(lambda ts: expr([Value(t) for t in ts]).data, xs, eps)
    return max(abs(v.grad - n) for v, n in zip(vals, num))
