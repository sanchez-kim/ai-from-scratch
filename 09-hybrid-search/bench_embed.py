"""Embedding throughput on this machine (wall clock). Run on an idle machine.

    nice -n 10 uv run python bench_embed.py --docs 2000 --queries 500 --repeats 3
    uv run python bench_embed.py --smoke        # 64 docs, 32 queries, 1 repeat

CPU, 2 torch threads, batch size 32, the same function as embed_corpus.py.
Reports paragraphs per second and queries per second (median of repeats),
after one warm-up batch. Writes results/embed-throughput[-smoke].json.
"""

import argparse
import json
import statistics
import time

from common import KINDS, RESULTS, uptime
from hybrid.data import load_corpus, load_queries, searchable_text
from hybrid.embed import MODEL, REVISION, embed


def timed(texts, is_query):
    t = time.perf_counter()
    embed(texts, is_query=is_query)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.docs, args.queries, args.repeats = 64, 32, 1
    docs = [searchable_text(d) for d in load_corpus()[: args.docs]]
    queries = [q["query"] for k in KINDS for q in load_queries(k)][: args.queries]
    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "uptime_before": uptime(),
           "model": MODEL, "revision": REVISION, "device": "cpu", "torch_threads": 2,
           "batch_size": 32, "smoke": args.smoke, "docs": len(docs), "queries": len(queries)}
    embed(docs[:32])  # warm-up: loads the model and runs one batch
    doc_s = [timed(docs, False) for _ in range(args.repeats)]
    q_s = [timed(queries, True) for _ in range(args.repeats)]
    out["doc_seconds"], out["query_seconds"] = doc_s, q_s
    out["docs_per_second_median"] = len(docs) / statistics.median(doc_s)
    out["queries_per_second_median"] = len(queries) / statistics.median(q_s)
    out["uptime_after"] = uptime()
    name = "embed-throughput-smoke.json" if args.smoke else "embed-throughput.json"
    (RESULTS / name).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
