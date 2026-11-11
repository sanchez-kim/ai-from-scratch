import math

import hnswlib
import numpy as np
import pytest

from ann.search import HyperplaneLSH, recall_at_k, top_k_exact, top_k_naive
from measure_lsh import collision_chance


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_numpy_search_matches_the_plain_python_loop():
    rng = np.random.default_rng(1)
    corpus = unit(rng.standard_normal((300, 12)))
    queries = unit(rng.standard_normal((20, 12)))
    fast = top_k_exact(corpus, queries, k=10, block=7)  # odd block size on purpose
    for q, row in zip(queries, fast):
        assert row.tolist() == top_k_naive(corpus.tolist(), q.tolist(), 10)


def test_recall_at_10_on_a_hand_example():
    truth = [list(range(10))]
    # 8 of the true top 10, plus two vectors that are not in it.
    found = [[0, 1, 2, 3, 4, 5, 6, 7, 42, 99]]
    assert recall_at_k(found, truth, 10) == 0.8
    # Order inside the top 10 does not matter, and the result is a mean over queries.
    assert recall_at_k([list(reversed(range(10))), found[0]], truth * 2, 10) == 0.9


def test_more_lsh_tables_approach_exact_recall():
    # Clustered data, like real embeddings: 40 centres, 50 points around each.
    rng = np.random.default_rng(2)
    centres = rng.standard_normal((40, 32))
    corpus = unit(np.repeat(centres, 50, axis=0) + 0.35 * rng.standard_normal((2000, 32)))
    queries = unit(centres[:20] + 0.35 * rng.standard_normal((20, 32)))
    truth = top_k_exact(corpus, queries, 10)

    recalls, scanned = {}, {}
    for n_tables in (1, 4, 32):
        lsh = HyperplaneLSH(dim=32, n_tables=n_tables, n_bits=10, seed=0)
        lsh.add(corpus)
        results = [lsh.search(q, 10) for q in queries]
        recalls[n_tables] = recall_at_k([r[0] for r in results], truth, 10)
        scanned[n_tables] = np.mean([r[1] for r in results])

    assert recalls[1] < recalls[4] <= recalls[32]
    assert recalls[32] >= 0.95
    # The price: more tables means more vectors scored per query.
    assert scanned[1] < scanned[4] < scanned[32] < len(corpus)


def test_lsh_returns_exact_ranking_of_its_candidates():
    rng = np.random.default_rng(3)
    corpus = unit(rng.standard_normal((500, 16)))
    lsh = HyperplaneLSH(dim=16, n_tables=1, n_bits=0, seed=0)  # 0 bits: one bucket holds everything
    lsh.add(corpus)
    ids, scanned = lsh.search(corpus[7], 10)
    assert scanned == 500
    assert ids == top_k_exact(corpus, corpus[7:8], 10)[0].tolist()


def test_recall_counts_only_the_first_k_found():
    truth = [list(range(10))]
    # 12 ids returned: the first 10 hold 8 true ones; true ids 8 and 9 come
    # at places 11 and 12, which recall@10 must not count.
    found = [[0, 1, 2, 3, 4, 5, 6, 7, 42, 99, 8, 9]]
    assert recall_at_k(found, truth, 10) == 0.8


def test_lsh_key_is_the_binary_number_of_the_sides():
    # Two tables of two planes each. Bit i is "dot with plane i > 0" and is
    # worth 2**i. Rows 0-1 belong to table 0, which asks (x > 0, y > 0);
    # rows 2-3 to table 1, which asks (x < 0, y > 0).
    lsh = HyperplaneLSH(dim=2, n_tables=2, n_bits=2, seed=0)
    lsh.planes = np.array([[1, 0], [0, 1], [-1, 0], [0, 1]], dtype=np.float32)
    # Small coordinates on purpose: a threshold other than 0 changes the bits.
    x = np.array([[0.3, -0.2], [-0.4, -0.1], [0.2, 0.25], [-0.3, 0.4]], dtype=np.float32)
    keys = lsh.keys(x)
    assert keys.shape == (2, 4)
    # table 0: (1,0) -> 1, (0,0) -> 0, (1,1) -> 1 + 2 = 3, (0,1) -> 2
    assert keys[0].tolist() == [1, 0, 3, 2]
    # table 1: (0,0) -> 0, (1,0) -> 1, (0,1) -> 2, (1,1) -> 3
    assert keys[1].tolist() == [0, 1, 2, 3]



def test_lsh_key_with_three_bits_gives_eight_distinct_buckets():
    # One table of three planes = the three axes, so bit i is "x_i > 0".
    # The eight sign patterns of a cube's corners, written as (b0, b1, b2):
    # key = b0 * 1 + b1 * 2 + b2 * 4, so the keys are 0..7 with no repeats.
    lsh = HyperplaneLSH(dim=3, n_tables=1, n_bits=3, seed=0)
    lsh.planes = np.eye(3, dtype=np.float32)
    s = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.float32)
    # (0,0,0) -> 0, (1,0,0) -> 1, (0,1,0) -> 2, (1,1,0) -> 3,
    # (0,0,1) -> 4, (1,0,1) -> 5, (0,1,1) -> 6, (1,1,1) -> 7
    assert lsh.keys(s)[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

def test_collision_chance_formula_by_hand():
    # cosine 0.5 is 60 degrees: one plane splits the pair with chance 1/3.
    assert collision_chance(0.5, 1, 1) == pytest.approx(2 / 3)
    assert collision_chance(0.5, 2, 1) == pytest.approx(4 / 9)
    # Three tables: 1 - (1 - 4/9)^3 = 1 - 125/729.
    assert collision_chance(0.5, 2, 3) == pytest.approx(1 - 125 / 729)


@pytest.mark.parametrize("n_bits, n_tables", [(1, 2000), (2, 1000)])
def test_lsh_collision_rate_matches_one_minus_angle_over_pi(n_bits, n_tables):
    # Two unit vectors at 60 degrees. Each table is an independent trial that
    # collides with chance (1 - 1/3) ** n_bits: 2/3 for 1 bit, 4/9 for 2 bits.
    # Binomial sd of the rate: sqrt(p (1 - p) / n) = 0.0105 and 0.0157,
    # so the 4-sigma tolerance is 0.042 and 0.063.
    p = (1 - math.acos(0.5) / math.pi) ** n_bits
    assert p == pytest.approx(collision_chance(0.5, n_bits, 1))
    a = np.zeros(16, dtype=np.float32)
    b = np.zeros(16, dtype=np.float32)
    a[0] = 1.0
    b[0], b[1] = 0.5, math.sqrt(3) / 2
    lsh = HyperplaneLSH(dim=16, n_tables=n_tables, n_bits=n_bits, seed=0)
    keys = lsh.keys(np.stack([a, b]))
    rate = float(np.mean(keys[:, 0] == keys[:, 1]))
    sigma = math.sqrt(p * (1 - p) / n_tables)
    assert abs(rate - p) < 4 * sigma


def test_lsh_scores_only_the_query_bucket():
    # Planes x and y: the bucket is the quadrant. A query in the (+, +)
    # quadrant must score exactly the corpus points in that quadrant.
    rng = np.random.default_rng(4)
    corpus = rng.standard_normal((100, 2)).astype(np.float32)
    lsh = HyperplaneLSH(dim=2, n_tables=1, n_bits=2, seed=0)
    lsh.planes = np.eye(2, dtype=np.float32)
    lsh.add(corpus)
    query = np.array([0.6, 0.8], dtype=np.float32)
    ids, scanned = lsh.search(query, 5)
    same = np.flatnonzero((corpus[:, 0] > 0) & (corpus[:, 1] > 0))
    assert 5 < scanned == len(same) < 100
    order = same[np.argsort(-(corpus[same] @ query), kind="stable")]
    assert ids == order[:5].tolist()


def test_exact_search_agrees_with_hnswlib_at_full_ef():
    # With ef_search = n, hnswlib keeps every vector it reaches, so on a
    # connected graph it returns the exact top 10.
    rng = np.random.default_rng(5)
    corpus = unit(rng.standard_normal((200, 16)))
    queries = unit(rng.standard_normal((20, 16)))
    index = hnswlib.Index(space="ip", dim=16)
    index.init_index(max_elements=200, M=16, ef_construction=200, random_seed=100)
    index.set_num_threads(1)
    index.add_items(corpus, np.arange(200))
    index.set_ef(200)
    labels, _ = index.knn_query(queries, k=10)
    truth = top_k_exact(corpus, queries, 10)
    assert recall_at_k(labels.tolist(), truth.tolist(), 10) == 1.0
