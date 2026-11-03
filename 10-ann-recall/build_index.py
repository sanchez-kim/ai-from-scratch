"""Build one hnswlib index and save it to data/<set>-<tag>.bin.

One thread on purpose: with more threads, the order in which vectors join the
graph depends on scheduling, so two builds can differ. With one thread and a
fixed random_seed, the build is repeatable; measure_recall.py checks that by
building every index twice (tags a and b) and comparing.
"""
import argparse
import pathlib

import hnswlib
import numpy as np

HERE = pathlib.Path(__file__).parent
M, EF_CONSTRUCTION, SEED = 16, 200, 100


def build(corpus):
    index = hnswlib.Index(space="ip", dim=corpus.shape[1])  # ip distance = 1 - dot
    index.init_index(max_elements=len(corpus), ef_construction=EF_CONSTRUCTION, M=M, random_seed=SEED)
    index.add_items(corpus, np.arange(len(corpus)), num_threads=1)
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["real", "random"], required=True)
    ap.add_argument("--tag", default="a")
    args = ap.parse_args()
    corpus = np.load(HERE / "data" / f"{args.set}.npz")["corpus"]
    build(corpus).save_index(str(HERE / "data" / f"{args.set}-{args.tag}.bin"))
    print("saved", args.set, args.tag)
