"""Filtered search on the real set: post-filtering vs hnswlib's filter callback.

The filter keeps a vector if (its Wikipedia article id mod 100) < pct, so about
pct % of the corpus passes, and all paragraphs of one article pass or fail together.
Ground truth: the exact top 10 among the vectors that pass.

- post-filter k=10: ask hnswlib for 10, drop what the filter rejects.
- post-filter k=100: ask for 100, drop, keep the best 10 that pass.
- callback: pass the filter to knn_query; hnswlib skips rejected vectors while it walks.
"""
import json
import pathlib
import subprocess

import numpy as np

from ann.search import recall_at_k, top_k_exact
from measure_recall import K, load

HERE = pathlib.Path(__file__).parent
SELECTIVITY = [1, 10, 50]
EF_SEARCH = [10, 40, 160]


def post_filter(index, queries, allowed, fetch, ef):
    index.set_ef(max(ef, fetch))
    labels, _ = index.knn_query(queries, k=fetch, num_threads=2)
    return [[i for i in row if allowed[i]][:K] for row in labels.tolist()]


def callback(index, queries, allowed, ef):
    index.set_ef(ef)
    calls = 0

    def keep(label):
        # hnswlib asks this for every vector it considers adding to its list,
        # so the call count shows how much of the graph the walk touched.
        nonlocal calls
        calls += 1
        return allowed[label]

    # num_threads=1, as the hnswlib README recommends for a Python filter.
    labels, _ = index.knn_query(queries, k=K, num_threads=1, filter=keep)
    return labels.tolist(), calls / len(queries)


def main():
    d = np.load(HERE / "data" / "real.npz")
    corpus, queries = d["corpus"], d["queries"]
    index = load("real-a.bin", corpus.shape[1], len(corpus))
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "filter": "wiki_id % 100 < pct", "k": K, "rows": []}
    for pct in SELECTIVITY:
        mask = d["corpus_wiki_id"] % 100 < pct
        allowed = mask.tolist()
        ids = np.flatnonzero(mask)
        truth = ids[top_k_exact(corpus[ids], queries, K)]
        for ef in EF_SEARCH:
            found_cb, calls = callback(index, queries, allowed, ef)
            runs = {"post_filter_k10": post_filter(index, queries, allowed, 10, ef),
                    "post_filter_k100": post_filter(index, queries, allowed, 100, ef),
                    "callback": found_cb}
            for method, found in runs.items():
                row = {"pct": pct, "corpus_passing": int(mask.sum()), "ef_search": ef, "method": method,
                       "recall_at_10": round(recall_at_k(found, truth, K), 4),
                       "mean_results_returned": round(float(np.mean([len(f) for f in found])), 2)}
                if method == "callback":
                    row["filter_calls_per_query"] = round(calls, 1)
                out["rows"].append(row)
                print(json.dumps(row))
    (HERE / "results" / "filter-sweep.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
