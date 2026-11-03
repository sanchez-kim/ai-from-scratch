"""Sanity check for the random set: does recall reach 1 when ef_search grows far past 320?

If it does, the low recall at small ef_search is a property of the data, not a
broken index or a mismatch between the index and the ground truth.
"""
import json
import pathlib
import subprocess

import numpy as np

from ann.search import recall_at_k
from measure_recall import K, load

HERE = pathlib.Path(__file__).parent


def main():
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), "rows": []}
    for name in ("real", "random"):
        d = np.load(HERE / "data" / f"{name}.npz")
        index = load(f"{name}-a.bin", d["corpus"].shape[1], len(d["corpus"]))
        for ef in (1000, 3000, 10000):
            index.set_ef(ef)
            labels, _ = index.knn_query(d["queries"], k=K, num_threads=2)
            row = {"set": name, "ef_search": ef, "recall_at_10": round(recall_at_k(labels, d["truth"], K), 4)}
            out["rows"].append(row)
            print(json.dumps(row))
    (HERE / "results" / "high-ef-check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
