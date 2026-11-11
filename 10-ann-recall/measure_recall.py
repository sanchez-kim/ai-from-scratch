"""recall@10 of hnswlib against exact search, for each ef_search, on both sets.

Loads the two builds made by build_index.py (tags a and b), runs the 1,000
queries at every ef_search, and checks that both builds return the same ids.
Also counts how often "close to my nearest neighbour" means "close to me".
"""
import hashlib
import json
import pathlib
import subprocess

import hnswlib
import numpy as np

from ann.search import recall_at_k, top_k_exact
from build_index import EF_CONSTRUCTION, M, SEED

HERE = pathlib.Path(__file__).parent
EF_SEARCH = [10, 20, 40, 80, 160, 320]
K = 10


def load(name, dim, n):
    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(HERE / "data" / name), max_elements=n)
    return index


def sweep(index, queries, truth):
    rows, labels_by_ef = [], {}
    for ef in EF_SEARCH:
        index.set_ef(ef)
        labels, _ = index.knn_query(queries, k=K, num_threads=2)
        per_query = [len(set(f) & set(t)) / K for f, t in zip(labels, truth)]
        rows.append({"ef_search": ef, "recall_at_10": round(recall_at_k(labels, truth, K), 4),
                     "queries_with_all_10": sum(p == 1.0 for p in per_query),
                     "worst_query_recall": min(per_query)})
        labels_by_ef[ef] = labels
    return rows, labels_by_ef


def neighbour_of_neighbour(corpus, truth):
    """For each query: of its other 9 nearest, how many are in its nearest's top 10?"""
    nearest = truth[:, 0]
    their_top = top_k_exact(corpus, corpus[nearest], K + 1)[:, 1:]  # drop the vector itself
    shared = [len(set(t[1:]) & set(n)) for t, n in zip(truth, their_top)]
    return {"mean_shared_of_9": round(float(np.mean(shared)), 3), "median": float(np.median(shared))}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "hnswlib": "0.8.0", "space": "ip", "M": M, "ef_construction": EF_CONSTRUCTION,
           "random_seed": SEED, "build_threads": 1, "query_threads": 2, "k": K, "sets": {}}
    for name in ("real", "random"):
        d = np.load(HERE / "data" / f"{name}.npz")
        corpus, queries, truth = d["corpus"], d["queries"], d["truth"]
        a, b = (load(f"{name}-{t}.bin", corpus.shape[1], len(corpus)) for t in "ab")
        rows, labels_a = sweep(a, queries, truth)
        _, labels_b = sweep(b, queries, truth)
        out["sets"][name] = {
            "sweep": rows,
            "builds_return_identical_ids": all(np.array_equal(labels_a[e], labels_b[e]) for e in EF_SEARCH),
            "index_files_identical": sha256(HERE / "data" / f"{name}-a.bin") == sha256(HERE / "data" / f"{name}-b.bin"),
            "neighbour_of_neighbour": neighbour_of_neighbour(corpus, truth),
        }
        print(name, json.dumps(out["sets"][name]))
    (HERE / "results" / "recall-sweep.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
