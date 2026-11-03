"""Random small expression graphs, built the same way for any Value class."""

import random

from .value import Value

OPS = ("+", "*", "tanh")


def random_graph(seed, n_ops, n_leaves=3, reuse=True, cls=Value):
    """Return (leaves, root, uses). The same seed gives the same graph for
    any cls, so the correct and buggy classes see identical inputs.

    reuse=True: each operand is drawn from all earlier nodes, so a node can
    feed several ops, or the same op twice (x * x).
    reuse=False: each operand is a node nobody uses yet (a fresh leaf if
    none is left), so the graph is a tree.
    The root is the sum of every node that no op uses.
    """
    rng = random.Random(seed)
    leaves = [cls(rng.uniform(-2, 2)) for _ in range(n_leaves)]
    nodes = list(leaves)
    uses = {id(v): 0 for v in nodes}
    for _ in range(n_ops):
        op = rng.choice(OPS)
        args = []
        for _ in range(1 if op == "tanh" else 2):
            if reuse:
                v = rng.choice(nodes)
            else:
                free = [u for u in nodes if uses[id(u)] == 0 and u not in args]
                if not free:
                    v = cls(rng.uniform(-2, 2))
                    leaves.append(v)
                    nodes.append(v)
                    uses[id(v)] = 0
                else:
                    v = rng.choice(free)
            uses[id(v)] += 1
            args.append(v)
        out = args[0] + args[1] if op == "+" else args[0] * args[1] if op == "*" else args[0].tanh()
        nodes.append(out)
        uses[id(out)] = 0
    unused = [v for v in nodes if uses[id(v)] == 0]
    root = unused[0]
    for v in unused[1:]:
        root = root + v
    for v in unused:
        uses[id(v)] += 1
    return leaves, root, {"nodes": nodes, "uses": uses}


def shared_nodes(leaves, info):
    """Count nodes used more than once, split into leaves and inner nodes."""
    leaf_ids = {id(v) for v in leaves}
    shared = [v for v in info["nodes"] if info["uses"][id(v)] >= 2]
    return (sum(id(v) in leaf_ids for v in shared),
            sum(id(v) not in leaf_ids for v in shared))


def count_paths(root):
    """Root-to-node paths, with multiplicity: the number of visit() calls
    backward_recursive makes. It can grow exponentially with depth."""
    from .bugs import topo_order
    paths = {id(root): 1}
    total = 0
    for v in reversed(topo_order(root)):
        p = paths.get(id(v), 0)
        total += p
        for c in v._prev:
            paths[id(c)] = paths.get(id(c), 0) + p
    return total
