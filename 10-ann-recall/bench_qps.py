"""Wall-clock timing: index build time and queries per second (QPS).

Timings depend on the machine and on whatever else runs on it. Run this on an
idle machine. --n and --queries shrink the run for a smoke test.

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run python bench_qps.py --set real
"""
import argparse
import json
import pathlib
import platform
import subprocess
import time

import numpy as np

from ann.search import top_k_exact
from build_index import build
from measure_recall import EF_SEARCH, K

HERE = pathlib.Path(__file__).parent


def qps(fn, n_queries, repeats=3):
    best = min(_timed(fn) for _ in range(repeats))  # best of 3 hides one-off stalls
    return round(n_queries / best, 1)


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["real", "random"], default="real")
    ap.add_argument("--n", type=int, default=None, help="corpus size (default: all 100,000)")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(HERE / "data" / f"{args.set}.npz")
    corpus = d["corpus"][: args.n]
    queries = d["queries"][: args.queries]
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "machine": platform.platform(), "set": args.set, "n": len(corpus), "queries": len(queries),
           "smoke": args.n is not None or args.queries < 1000}

    start = time.perf_counter()
    index = build(corpus)
    out["build_seconds_1_thread"] = round(time.perf_counter() - start, 2)
    out["exact_numpy_qps"] = qps(lambda: top_k_exact(corpus, queries, K), len(queries))
    out["hnsw_qps_1_thread"] = []
    for ef in EF_SEARCH:
        index.set_ef(ef)
        out["hnsw_qps_1_thread"].append(
            {"ef_search": ef, "qps": qps(lambda: index.knn_query(queries, k=K, num_threads=1), len(queries))})
    print(json.dumps(out, indent=2))
    name = args.out or f"qps-{args.set}-{time.strftime('%Y-%m-%d-%H%M')}.json"
    (HERE / "results" / name).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
