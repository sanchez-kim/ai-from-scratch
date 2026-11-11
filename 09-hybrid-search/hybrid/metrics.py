"""Rank metrics with the same definitions as 08-bm25/measure_retrieval.py."""


def gold_rank(ranking, gold, k=10):
    """1-based position of gold in the first k ids, or None."""
    top = list(ranking[:k])
    return top.index(gold) + 1 if gold in top else None


def recall_at_k(ranking, gold, k=10):
    """1.0 if the one known answer is in the top k, else 0.0."""
    return 1.0 if gold_rank(ranking, gold, k) else 0.0


def reciprocal_rank(ranking, gold, k=10):
    """1 / rank of the known answer within the top k, else 0. Averaged = MRR@k."""
    r = gold_rank(ranking, gold, k)
    return 1.0 / r if r else 0.0


def kendall_tau_b(x, y):
    """Kendall's tau-b between two score lists for the same items.

    For every pair of items: concordant if both lists order them the same
    way, discordant if opposite. tau = (C - D) / sqrt(...) with the tie
    correction of tau-b, so tied scores (BM25 gives 0 to many documents)
    do not count as agreement. 1 = same order, 0 = unrelated, -1 = reversed.
    """
    n = len(x)
    c = d = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                c += 1
            else:
                d += 1
    denom = ((c + d + tie_x) * (c + d + tie_y)) ** 0.5
    return (c - d) / denom if denom else 0.0
