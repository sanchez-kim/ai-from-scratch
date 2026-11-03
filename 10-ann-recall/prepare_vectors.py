"""Turn the two parquet shards into the arrays every other script uses.

Real set: 120,000 Simple English Wikipedia paragraph embeddings (768 numbers
each). A seeded shuffle picks 1,000 as queries and 100,000 others as the
corpus, so no query is in the index. Every vector is scaled to length 1, so
the dot product is the cosine similarity.

Random set: 101,000 Gaussian vectors of the same dimension, also length 1,
split the same way. It has no cluster structure at all.

Ground truth for both: the exact top 10 by brute force. Also saved: how far
the 1st, 10th, 100th and 1,000th neighbours are from the query compared with
a typical vector, which explains why recall differs between the two sets.
"""
import hashlib
import json
import pathlib
import subprocess

import numpy as np
import pyarrow.parquet as pq

from ann.search import top_k_exact

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
N_CORPUS, N_QUERIES, K = 100_000, 1_000, 10


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def load_real():
    vecs, wiki_ids = [], []
    for shard in ("0000.parquet", "0001.parquet"):
        table = pq.read_table(DATA / shard, columns=["meta", "embedding"])
        for meta, emb in zip(table["meta"].to_pylist(), table["embedding"].to_pylist()):
            wiki_ids.append(json.loads(meta)["wiki_id"])
            vecs.append(np.asarray(json.loads(emb), dtype=np.float32))
    return np.stack(vecs), np.asarray(wiki_ids, dtype=np.int64)


def neighbour_profile(corpus, queries):
    """Cosine of the 1st/10th/100th/1000th neighbour, and of an average vector."""
    ranks = [1, 10, 100, 1000]
    rows = {f"rank_{r}": [] for r in ranks} | {"mean": [], "std": []}
    for start in range(0, len(queries), 200):
        s = queries[start:start + 200] @ corpus.T
        top = -np.sort(-np.partition(s, -1000, axis=1)[:, -1000:], axis=1)
        for r in ranks:
            rows[f"rank_{r}"].extend(top[:, r - 1].tolist())
        rows["mean"].extend(s.mean(axis=1).tolist())
        rows["std"].extend(s.std(axis=1).tolist())
    return {name: round(float(np.median(v)), 4) for name, v in rows.items()}


def duplicate_count(x):
    _, counts = np.unique(x, axis=0, return_counts=True)
    return int(counts[counts > 1].sum())


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    raw, wiki_ids = load_real()
    norms = np.linalg.norm(raw, axis=1)
    perm = np.random.default_rng(0).permutation(len(raw))
    q_idx, c_idx = perm[:N_QUERIES], perm[N_QUERIES:N_QUERIES + N_CORPUS]
    real = {
        "corpus": unit(raw[c_idx]), "queries": unit(raw[q_idx]),
        "corpus_wiki_id": wiki_ids[c_idx], "corpus_row": c_idx, "query_row": q_idx,
    }
    g = unit(np.random.default_rng(1).standard_normal((N_CORPUS + N_QUERIES, raw.shape[1])))
    rand = {"corpus": g[N_QUERIES:], "queries": g[:N_QUERIES]}

    summary = {
        "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        "rows_read": int(len(raw)), "dim": int(raw.shape[1]),
        "raw_norm": {"min": round(float(norms.min()), 3), "median": round(float(np.median(norms)), 3),
                     "max": round(float(norms.max()), 3)},
        "split": {"seed": 0, "corpus": N_CORPUS, "queries": N_QUERIES},
        "datasets": {},
    }
    for name, d in (("real", real), ("random", rand)):
        d["truth"] = top_k_exact(d["corpus"], d["queries"], K)
        np.savez(DATA / f"{name}.npz", **d)
        summary["datasets"][name] = {
            "corpus_shape": list(d["corpus"].shape),
            "corpus_vectors_with_an_exact_duplicate": duplicate_count(d["corpus"]),
            "neighbour_cosine_median_over_queries": neighbour_profile(d["corpus"], d["queries"]),
            "truth_sha256": digest(d["truth"]),
        }
        print(name, json.dumps(summary["datasets"][name]))
    (HERE / "results" / "vectors.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
