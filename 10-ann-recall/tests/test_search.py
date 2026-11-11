import numpy as np

from ann.search import HyperplaneLSH, recall_at_k, top_k_exact, top_k_naive


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
