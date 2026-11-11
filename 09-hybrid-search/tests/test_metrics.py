import random

import pytest
import scipy.stats

from hybrid import kendall_tau_b, recall_at_k, reciprocal_rank


def test_recall_and_reciprocal_rank():
    ranking = [7, 3, 9, 1]
    assert recall_at_k(ranking, 9) == 1.0
    assert reciprocal_rank(ranking, 9) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, 5) == 0.0 and reciprocal_rank(ranking, 5) == 0.0
    assert recall_at_k(list(range(20)), 10) == 0.0    # rank 11 is outside the top 10


def test_kendall_tau_hand_examples():
    assert kendall_tau_b([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau_b([1, 2, 3], [30, 20, 10]) == -1.0
    # One swapped neighbour among 4 items: 5 concordant, 1 discordant pair.
    assert kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    # A tie in y: 2 concordant pairs, 1 pair tied in y only.
    assert kendall_tau_b([1, 2, 3], [0, 0, 1]) == pytest.approx(2 / (2 * 3) ** 0.5)


def test_kendall_tau_matches_scipy_with_ties():
    rng = random.Random(9)
    for _ in range(50):
        n = rng.randint(3, 20)
        x = [rng.choice([0.0, 0.0, rng.random()]) for _ in range(n)]
        y = [rng.random() for _ in range(n)]
        expected = scipy.stats.kendalltau(x, y).statistic
        if expected != expected:   # NaN when one list is constant
            continue
        assert kendall_tau_b(x, y) == pytest.approx(expected)
