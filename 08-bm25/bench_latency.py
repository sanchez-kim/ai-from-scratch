"""Time one query on the 10,000 paragraphs: ours, bm25s and rank_bm25.

    uv run python bench_latency.py --queries 1000 --repeats 5    # idle machine only
    uv run python bench_latency.py --queries 5 --repeats 1 --smoke

Each library returns the top 10 for every query; we keep the median of the
per-query times over all repeats. Index build time is recorded once each.
Writes results/latency-<date>.json (or latency-smoke.json with --smoke).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import bm25s
import numpy as np
from rank_bm25 import BM25Okapi

from bm25 import BM25, tokenize
from common import all_queries, corpus_tokens, uptime


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    up_before = uptime()
    _, toks = corpus_tokens()
    queries = [q for q in (tokenize(r["query"]) for r in all_queries()) if q][: args.queries]

    ours, t_ours = timed(lambda: BM25(toks))
    ref = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    _, t_bm25s = timed(lambda: ref.index(toks, show_progress=False))
    okapi, t_okapi = timed(lambda: BM25Okapi(toks))

    runners = {
        "ours": lambda q: ours.search(q, 10),
        "bm25s": lambda q: ref.retrieve([q], k=10, show_progress=False),
        "rank_bm25": lambda q: np.argsort(okapi.get_scores(q))[::-1][:10],
    }
    per_query = {name: [] for name in runners}
    for _ in range(args.repeats):
        for q in queries:
            for name, run in runners.items():
                per_query[name].append(timed(lambda: run(q))[1])
    result = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "smoke": args.smoke,
        "uptime_before": up_before, "uptime_after": uptime(),
        "queries": len(queries), "repeats": args.repeats, "documents": len(toks),
        "index_build_s": {"ours": t_ours, "bm25s": t_bm25s, "rank_bm25": t_okapi},
        "median_query_ms": {k: statistics.median(v) * 1000 for k, v in per_query.items()},
        "p90_query_ms": {k: float(np.percentile(v, 90)) * 1000 for k, v in per_query.items()},
    }
    name = "latency-smoke.json" if args.smoke else f"latency-{time.strftime('%Y-%m-%d')}.json"
    (Path(__file__).parent / "results" / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
