import math

import pytest

from hybrid import cosine, dot, min_max, normalise, rrf, sum_scores


def test_cosine_hand_examples():
    assert cosine([1, 0], [0, 1]) == 0.0          # unrelated
    assert cosine([1, 2], [2, 4]) == pytest.approx(1.0)   # same direction, any length
    assert cosine([1, 0], [-1, 0]) == -1.0        # opposite
    assert cosine([0, 0], [1, 1]) == 0.0          # zero vector: defined as 0


def test_dot_equals_cosine_only_after_normalising():
    a, b = [3, 4], [4, 3]
    assert dot(a, b) == 24                        # depends on length
    assert cosine(a, b) == pytest.approx(24 / 25)
    na, nb = normalise(a), normalise(b)
    assert na == pytest.approx([0.6, 0.8])
    assert dot(na, nb) == pytest.approx(cosine(a, b))


def test_rrf_adds_one_over_k_plus_rank():
    fused = dict(rrf([["a", "b", "c"], ["c", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)    # missing from list 2: adds nothing
    assert [d for d, _ in rrf([["a", "b", "c"], ["c", "a", "d"]])] == ["a", "c", "b", "d"]


def test_rrf_k_decides_between_one_top_place_and_two_middle_places():
    # x is first in one list only; y is fourth in both lists.
    lists = [["x", "p", "q", "y"], ["r", "s", "t", "y"]]
    k1, k60 = dict(rrf(lists, k=1)), dict(rrf(lists, k=60))
    assert k1["x"] > k1["y"]                      # 1/2 = 0.5 beats 2 * 1/5 = 0.4
    assert k60["y"] > k60["x"]                    # 2/64 = 0.031 beats 1/61 = 0.016
    assert rrf(lists, k=60)[0][0] == "y"


def test_rrf_ties_go_to_smaller_id():
    assert [d for d, _ in rrf([[2, 1], [1, 2]])] == [1, 2]


def test_min_max():
    assert min_max([("a", 10), ("b", 5), ("c", 0)]) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert min_max([("a", 3), ("b", 3)]) == {"a": 1.0, "b": 1.0}


def test_sum_scores_hand_example():
    bm25 = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
    emb = [("c", 0.9), ("a", 0.5), ("d", 0.1)]
    fused = sum_scores([bm25, emb])
    assert [d for d, _ in fused] == ["a", "c", "b", "d"]
    assert dict(fused)["a"] == pytest.approx(1.0 + 0.5)
    assert dict(fused)["d"] == 0.0


def test_one_outlier_squashes_min_max_but_not_rrf():
    # BM25 gives one paragraph a huge score; the rest sit near the bottom.
    bm25 = [("a", 40.0), ("b", 4.0), ("c", 3.0), ("d", 2.0)]
    emb = [("b", 0.80), ("c", 0.79), ("d", 0.78), ("a", 0.50)]
    summed = dict(sum_scores([bm25, emb]))
    assert math.isclose(summed["a"], 1.0)         # 1.0 + 0.0
    assert summed["b"] < 1.06                     # 0.053 + 1.0: BM25's second place is worth almost nothing
    assert [d for d, _ in rrf([[d for d, _ in bm25], [d for d, _ in emb]])][0] == "b"


def test_rrf_default_k_is_60():
    # One list, rank 1: 1 / (60 + 1).
    assert dict(rrf([["a"]]))["a"] == pytest.approx(1 / 61, rel=1e-15)


def test_normalise_zero_vector_stays_zero():
    # Length 0 has no direction; the code returns the vector unchanged.
    assert normalise([0, 0]) == [0, 0]
    assert normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
