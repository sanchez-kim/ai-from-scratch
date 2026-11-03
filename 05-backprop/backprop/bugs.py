"""The two silent bugs, kept next to the correct code so tests can compare.

Bug 1, AssignValue: every _backward writes `grad = ...` instead of
`grad += ...`. A node used twice keeps only the last contribution.

Bug 2, wrong order: _backward runs before a node has collected its full
gradient. Two common ways to get there are below. Neither raises.
"""

import math

from .value import Value


class AssignValue(Value):
    """Value with bug 1. Only the primitives change; -, / and the r-ops
    are built from them, so they inherit the bug."""

    def _wrap(self, other):
        return other if isinstance(other, Value) else AssignValue(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = AssignValue(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = out.grad            # bug: should be +=
            other.grad = out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = AssignValue(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = other.data * out.grad   # bug: should be +=
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = AssignValue(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad = k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = AssignValue(t, (self,), "tanh")

        def _backward():
            self.grad = (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = AssignValue(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out


def topo_order(root):
    """Inputs first, root last: the same list Value.backward builds."""
    topo, seen = [], set()

    def build(v):
        if v not in seen:
            seen.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(root)
    return topo


def backward_forgot_reversed(root):
    """Bug 2a: walks the topological list from the inputs, not the root.
    Each _backward runs while its node's grad is still 0, except the root's,
    which runs last."""
    root.grad = 1.0
    for v in topo_order(root):
        v._backward()


def backward_recursive(root):
    """Bug 2b: depth-first, no topological sort. A node passes its grad on
    as soon as one parent reaches it, then again from the next parent, so a
    shared inner node sends its partial grad more than once. Correct on
    trees, and on graphs where only leaves are shared."""
    root.grad = 1.0

    def visit(v):
        v._backward()
        for child in v._prev:
            visit(child)
    visit(root)
