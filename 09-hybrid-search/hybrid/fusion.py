"""Similarity and two ways to merge ranked lists, in plain Python.

A ranking is a list of document ids, best first. A scored list is a list of
(doc id, score) pairs, best first.
"""

import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    """Cosine of the angle between a and b: 1 same direction, 0 unrelated.

    For vectors of length 1 (normalised), the two norms are 1 and cosine
    equals dot(a, b). That is why search code normalises once and then uses
    the cheaper dot product.
    """
    norm = math.sqrt(dot(a, a)) * math.sqrt(dot(b, b))
    return dot(a, b) / norm if norm else 0.0


def normalise(a):
    n = math.sqrt(dot(a, a))
    return [x / n for x in a] if n else list(a)


def rrf(rankings, k=60):
    """Reciprocal rank fusion: each list gives a document 1 / (k + rank).

    rank starts at 1. A document missing from a list gets nothing from it.
    Only ranks are used, so the lists' score scales never meet.
    Returns (doc id, fused score), best first; ties go to the smaller id.
    """
    fused = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def min_max(scored):
    """Rescale one list's scores to 0..1: best -> 1, worst kept -> 0."""
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc: 1.0 for doc, _ in scored}
    return {doc: (s - lo) / (hi - lo) for doc, s in scored}


def sum_scores(scored_lists):
    """The naive fusion: min-max each list, then add the scores.

    A document missing from a list gets 0 from it.
    Returns (doc id, fused score), best first; ties go to the smaller id.
    """
    fused = {}
    for scored in scored_lists:
        for doc, s in min_max(scored).items():
            fused[doc] = fused.get(doc, 0.0) + s
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
