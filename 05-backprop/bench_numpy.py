"""Pure-Python Value vs numpy: gradient of sum(tanh(W @ x)) with respect to W.

    uv run python bench_numpy.py --sizes 16 32 64 128 --repeats 7
    uv run python bench_numpy.py --sizes 4 8 --repeats 3 --smoke   # smoke test
    uv run python bench_numpy.py --nodes-only                      # no timing

Wall-clock timings depend on machine load. Run on an idle machine before
quoting them. --smoke marks the results file so it is not mistaken for a
real measurement. The node count of the Value graph is deterministic.
"""

import argparse
import datetime
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np

from backprop import Value, topo_order
from backprop.runinfo import header, uptime


def value_grad(w_rows, x):
    W = [[Value(w) for w in row] for row in w_rows]
    xs = [Value(v) for v in x]
    loss = sum((sum((wij * xj for wij, xj in zip(row, xs)), Value(0.0))).tanh() for row in W)
    loss.backward()
    return [[w.grad for w in row] for row in W], len(topo_order(loss))


def numpy_grad(W, x):
    # d/dW sum(tanh(W x)) = (1 - tanh(W x)^2) outer x
    t = np.tanh(W @ x)
    return np.outer(1.0 - t * t, x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--nodes-only", action="store_true",
                   help="record graph size and the numpy check, skip timing")
    args = p.parse_args()

    result = header("bench_numpy.py")
    result["smoke_test_only"] = args.smoke
    result["timed"] = not args.nodes_only
    result["numpy"] = np.__version__
    rows = []
    for n in args.sizes:
        rng = random.Random(n)
        w_rows = [[rng.gauss(0, 1 / math.sqrt(n)) for _ in range(n)] for _ in range(n)]
        x = [rng.gauss(0, 1) for _ in range(n)]
        W, xv = np.array(w_rows), np.array(x)
        g_val, nodes = value_grad(w_rows, x)
        diff = float(np.max(np.abs(np.array(g_val) - numpy_grad(W, xv))))
        if args.nodes_only:
            rows.append({"n": n, "value_graph_nodes": nodes, "max_abs_diff_vs_numpy": diff})
            print(rows[-1])
            continue
        tv, tn = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); value_grad(w_rows, x); tv.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); numpy_grad(W, xv); tn.append(time.perf_counter() - t0)
        rows.append({"n": n, "value_graph_nodes": nodes, "max_abs_diff_vs_numpy": diff,
                     "value_seconds_median": statistics.median(tv),
                     "numpy_seconds_median": statistics.median(tn)})
        print(rows[-1])
    result["rows"] = rows
    result["uptime_after"] = uptime()
    tag = "nodes" if args.nodes_only else "smoke" if args.smoke else "bench"
    out = Path("results") / f"{tag}-{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
