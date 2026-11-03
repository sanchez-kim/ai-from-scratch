"""How often does each silent bug change a gradient, on random small graphs?

    uv run python measure_bugs.py      # writes results/bugs-<date>.json

For every graph the leaves' gradients from the correct code are compared
with bug 1 (AssignValue), bug 2a (forgot reversed) and bug 2b (recursive,
no topological sort). "Changed" means some leaf gradient moved by more than
1e-9 * max(1, |correct|). Also records the diamond examples quoted in the
post. Deterministic: random.Random(seed) for seeds 0..999.
"""

import datetime
import json
from pathlib import Path

from backprop import AssignValue, Value, backward_forgot_reversed, backward_recursive, grad_check
from backprop.graphs import count_paths, random_graph, shared_nodes
from backprop.runinfo import header, uptime

SIZES = [1, 2, 4, 8, 16, 32]
SEEDS = range(1000)
TOL = 1e-9


def grads(seed, n, reuse, cls=Value, backward=None):
    leaves, root, info = random_graph(seed, n, reuse=reuse, cls=cls)
    (backward or type(root).backward)(root)
    return [v.grad for v in leaves], leaves, root, info


def differs(a, b):
    return any(abs(x - y) > TOL * max(1.0, abs(y)) for x, y in zip(a, b))


def examples():
    out = {}
    a = Value(2.0); L = a * 3 + a * 5; L.backward()
    out["diamond_a*3+a*5_correct"] = a.grad
    a = AssignValue(2.0); L = a * 3 + a * 5; L.backward()
    out["diamond_a*3+a*5_bug1_assign"] = a.grad
    a = AssignValue(2.0); L = a * 5 + a * 3; L.backward()
    out["diamond_a*5+a*3_bug1_assign"] = a.grad
    x = Value(3.0); (x * x).backward(); out["x*x_at_3_correct"] = x.grad
    x = AssignValue(3.0); (x * x).backward(); out["x*x_at_3_bug1_assign"] = x.grad
    x = AssignValue(3.0); (x ** 2).backward(); out["x**2_at_3_bug1_assign"] = x.grad
    a = Value(2.0); L = a * 3 + a * 5; backward_forgot_reversed(L)
    out["diamond_bug2a_forgot_reversed"] = a.grad
    a = Value(2.0); L = a * 3 + a * 5; backward_recursive(L)
    out["diamond_bug2b_recursive"] = a.grad
    for name, bw in (("correct", None), ("bug2a_forgot_reversed", backward_forgot_reversed),
                     ("bug2b_recursive", backward_recursive)):
        x = Value(2.0); a = x * x; L = a * 3 + a * 5
        (bw or Value.backward)(L)
        out[f"nested_x=2,a=x*x,a*3+a*5_{name}"] = {"x.grad": x.grad, "a.grad": a.grad}
    a = Value(2.0); L = a * 3 + a * 5; L.backward(); L.backward()
    out["diamond_backward_called_twice"] = a.grad

    def nested(v):
        a = v[0] * v[0]
        return a * 3 + a * 5
    out["grad_check_nested_at_2"] = {
        "correct": grad_check(nested, [2.0]),
        "bug1_assign": grad_check(nested, [2.0], cls=AssignValue),
        "bug2a_forgot_reversed": grad_check(nested, [2.0], backward=backward_forgot_reversed),
        "bug2b_recursive": grad_check(nested, [2.0], backward=backward_recursive),
    }
    return out


def main():
    result = header("measure_bugs.py")
    result["setup"] = {"sizes_n_ops": SIZES, "seeds": [SEEDS.start, SEEDS.stop - 1],
                       "leaves": "3 to start; trees add a fresh leaf when no unused node is left", "ops": ["+", "*", "tanh"], "leaf_values": "uniform(-2, 2)",
                       "tolerance": "|bug - correct| > 1e-9 * max(1, |correct|) for some leaf"}
    result["examples"] = examples()
    rows = []
    for reuse in (True, False):
        for n in SIZES:
            c = {"bug1": 0, "bug2a": 0, "bug2b": 0, "shared_any": 0, "shared_inner": 0,
                 "bug1_given_shared": 0, "bug1_given_no_shared": 0,
                 "bug2b_given_shared_inner": 0, "bug2b_given_no_shared_inner": 0}
            paths = []
            for s in SEEDS:
                good, leaves, root, info = grads(s, n, reuse)
                b1 = grads(s, n, reuse, cls=AssignValue)[0]
                b2a = grads(s, n, reuse, backward=backward_forgot_reversed)[0]
                b2b, _, root_b, _ = grads(s, n, reuse, backward=backward_recursive)
                paths.append(count_paths(root_b))
                sh_leaf, sh_inner = shared_nodes(leaves, info)
                d1, d2a, d2b = differs(b1, good), differs(b2a, good), differs(b2b, good)
                c["bug1"] += d1
                c["bug2a"] += d2a
                c["bug2b"] += d2b
                c["shared_any"] += (sh_leaf + sh_inner) > 0
                c["shared_inner"] += sh_inner > 0
                c["bug1_given_shared" if sh_leaf + sh_inner else "bug1_given_no_shared"] += d1
                c["bug2b_given_shared_inner" if sh_inner else "bug2b_given_no_shared_inner"] += d2b
            paths.sort()
            rows.append({"graph": "reuse" if reuse else "tree", "n_ops": n, "graphs": len(SEEDS),
                         **c, "visits_recursive_median": paths[len(paths) // 2],
                         "visits_recursive_max": paths[-1]})
            print(rows[-1])
    result["sweep"] = rows
    result["uptime_after"] = uptime()
    out = Path("results") / f"bugs-{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
