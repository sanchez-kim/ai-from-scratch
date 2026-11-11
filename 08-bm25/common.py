"""Shared helpers for the measurement scripts: load, tokenize, rank, uptime."""

import subprocess

import numpy as np

from bm25 import tokenize
from bm25.data import load_corpus, load_queries, searchable_text


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def corpus_tokens(name="corpus"):
    docs = load_corpus(name)
    return docs, [tokenize(searchable_text(d)) for d in docs]


def all_queries():
    return [q for kind in ("lexical", "paraphrase") for q in load_queries(kind)]


def dense(score_dict, n):
    out = np.zeros(n)
    for i, s in score_dict.items():
        out[i] = s
    return out


def top_k(scores, k=10):
    """Doc numbers of the k highest scores; ties broken by lower doc number."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=np.float64)))
    return [int(i) for i in order[:k]]
