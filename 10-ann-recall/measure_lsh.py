"""recall@10 of random-hyperplane LSH, and how many vectors it scores per query."""
import json
import math
import pathlib
import subprocess

import numpy as np

from ann.search import HyperplaneLSH, recall_at_k

HERE = pathlib.Path(__file__).parent
N_BITS, TABLES, K = 16, [1, 2, 4, 8, 16, 32], 10


def collision_chance(cosine, n_bits, n_tables):
    """Chance that a vector at this cosine shares the query's bucket in at least one table.

    One random plane separates two vectors with probability angle / pi
    (Goemans and Williamson 1995; Charikar 2002), so all n_bits agree with
    probability (1 - angle / pi) ** n_bits, per table.
    """
    one_table = (1 - math.acos(cosine) / math.pi) ** n_bits
    return 1 - (1 - one_table) ** n_tables


def main():
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "n_bits": N_BITS, "seed": 0, "k": K, "sets": {}}
    profile = json.loads((HERE / "results" / "vectors.json").read_text())["datasets"]
    for name in ("real", "random"):
        cos10 = profile[name]["neighbour_cosine_median_over_queries"]["rank_10"]
        d = np.load(HERE / "data" / f"{name}.npz")
        corpus, queries, truth = d["corpus"], d["queries"], d["truth"]
        rows = []
        for n_tables in TABLES:
            lsh = HyperplaneLSH(corpus.shape[1], n_tables, N_BITS, seed=0)
            lsh.add(corpus)
            results = [lsh.search(q, K) for q in queries]
            scanned = np.array([r[1] for r in results])
            rows.append({"tables": n_tables, "recall_at_10": round(recall_at_k([r[0] for r in results], truth, K), 4),
                         "median_vectors_scored": int(np.median(scanned)),
                         "mean_share_of_corpus_scored": round(float(scanned.mean() / len(corpus)), 4),
                         "largest_bucket_table_0": max(len(v) for v in lsh.tables[0].values()),
                         "formula_chance_for_median_10th_neighbour": round(collision_chance(cos10, N_BITS, n_tables), 4)})
            print(name, json.dumps(rows[-1]))
        out["sets"][name] = rows
    (HERE / "results" / "lsh-sweep.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
