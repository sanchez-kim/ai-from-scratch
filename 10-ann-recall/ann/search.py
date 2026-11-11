"""Exact nearest-neighbour search, recall@k, and random-hyperplane LSH."""
from collections import defaultdict

import numpy as np


def top_k_naive(corpus, query, k):
    """Plain Python: score every vector against the query, keep the k best."""
    scores = []
    for i, vec in enumerate(corpus):
        scores.append((sum(a * b for a, b in zip(vec, query)), i))
    # Highest score first; on a tie, the lower index wins, so results are stable.
    scores.sort(key=lambda s: (-s[0], s[1]))
    return [i for _, i in scores[:k]]


def top_k_exact(corpus, queries, k, block=200):
    """The same search with numpy: one matrix product per block of queries."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), block):
        scores = queries[start:start + block] @ corpus.T
        # argpartition finds the k best without sorting all N scores...
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        best = np.take_along_axis(scores, part, axis=1)
        # ...then sort only those k, highest first, lower index on a tie.
        order = np.lexsort((part, -best), axis=1)
        out[start:start + block] = np.take_along_axis(part, order, axis=1)
    return out


def recall_at_k(found, truth, k):
    """Share of the true top-k that the search returned, averaged over queries."""
    per_query = [len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]
    return sum(per_query) / len(per_query)


class HyperplaneLSH:
    """Random-hyperplane LSH: vectors on the same side of every plane share a bucket.

    One table with b planes gives each vector a b-bit key. Close vectors tend to
    get the same key. Several tables with different planes give a near vector
    several chances to land in the query's bucket. The final ranking is exact,
    over the union of the buckets the query falls into.
    """

    def __init__(self, dim, n_tables, n_bits, seed=0):
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((n_tables * n_bits, dim)).astype(np.float32)
        self.n_tables, self.n_bits = n_tables, n_bits
        self.powers = 1 << np.arange(n_bits)
        self.tables = [defaultdict(list) for _ in range(n_tables)]

    def keys(self, x):
        """(n, dim) vectors -> (n_tables, n) integer bucket keys."""
        bits = (x @ self.planes.T > 0).reshape(len(x), self.n_tables, self.n_bits)
        return (bits @ self.powers).T

    def add(self, corpus):
        self.corpus = corpus
        for table, row in zip(self.tables, self.keys(corpus)):
            for i, key in enumerate(row.tolist()):
                table[key].append(i)

    def search(self, query, k):
        """Returns (top-k ids, number of vectors scored)."""
        cand = set()
        for table, key in zip(self.tables, self.keys(query[None, :])[:, 0].tolist()):
            cand.update(table.get(key, ()))
        cand = np.fromiter(sorted(cand), dtype=np.int64)
        if len(cand) == 0:
            return [], 0
        best = top_k_exact(self.corpus[cand], query[None, :], min(k, len(cand)))[0]
        return cand[best].tolist(), len(cand)
