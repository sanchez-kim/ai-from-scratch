"""Shared helpers for the scripts: uptime, file names, ranking with numpy."""

import subprocess
from pathlib import Path

import numpy as np

from hybrid.embed import MODEL, REVISION

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
TAG = f"{MODEL.split('/')[-1]}-{REVISION[:12]}"
DOC_VECS = DATA / f"{TAG}-docs.npy"
QUERY_VECS = DATA / f"{TAG}-queries.npy"
QUERY_VECS_NOPREFIX = DATA / f"{TAG}-queries-noprefix.npy"
KINDS = ("lexical", "paraphrase")


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def top_k(scores, k):
    """Indices of the k highest scores; ties broken by lower index (as in 08-bm25)."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:k]]
