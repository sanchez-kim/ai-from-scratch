import math

import bm25s
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from bm25 import BM25, tokenize

TOY = [
    "the cat sat on the mat",
    "the dog sat",
    "cats and dogs and a cat",
]


def test_toy_score_matches_hand_computation():
    docs = [tokenize(t) for t in TOY]
    index = BM25(docs, k1=1.5, b=0.75)
    # Lengths 6, 3, 6 words, so avgdl = 15 / 3 = 5.
    assert index.avgdl == 5
    # "cat" is in documents 0 and 2 (doc 1 has "dog", doc 2 also has "cats").
    # df = 2, N = 3: idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    idf = math.log(1.6)
    assert index.idf("cat") == pytest.approx(0.47000362924573563)
    # Document 0: tf = 1, dl = 6. norm = 1 - 0.75 + 0.75 * 6 / 5 = 1.15
    # term = idf * 1 * 2.5 / (1 + 1.5 * 1.15) = idf * 2.5 / 2.725
    assert index.score(["cat"], 0) == pytest.approx(idf * 2.5 / 2.725)
    # Document 2: tf = 1 ("cats" is another word), same length, same score.
    assert index.score(["cat"], 2) == pytest.approx(index.score(["cat"], 0))
    # A word in every document still gets a small positive idf: ln(1 + 0.5 / 3.5).
    assert index.idf("sat") == pytest.approx(math.log(1 + 1.5 / 2.5))
    assert index.score(["cat"], 1) == 0.0


def test_idf_variant_is_the_one_bm25s_calls_lucene_and_never_negative():
    docs = [["a", "b"], ["a"], ["a", "c"], ["a"]]
    index = BM25(docs)
    # df("a") = 4 = N. Robertson idf ln((N - df + 0.5) / (df + 0.5)) = ln(0.5 / 4.5) < 0.
    assert math.log(0.5 / 4.5) < 0
    assert index.idf("a") == pytest.approx(math.log(1 + 0.5 / 4.5)) and index.idf("a") > 0


SMALL = [
    "bm25 ranks documents by matching words",
    "a search index maps every word to the documents that contain it",
    "long documents repeat words more often than short documents",
    "the length normalization in bm25 is controlled by b",
    "k1 controls how fast repeated words stop adding to the score",
    "rare words get a higher inverse document frequency",
    "an embedding model maps text to vectors",
    "retrieval augmented generation needs a search step before the model answers",
]
QUERIES = ["bm25 length normalization", "rare words score", "search index for documents", "vectors"]


def test_matches_bm25s_lucene_up_to_the_k1_plus_1_factor():
    docs = [tokenize(t) for t in SMALL]
    ours = BM25(docs, k1=1.5, b=0.75)
    ref = bm25s.BM25(k1=1.5, b=0.75, method="lucene", dtype="float64")
    ref.index(docs, show_progress=False)
    for q in QUERIES:
        theirs = ref.get_scores(tokenize(q))
        mine = np.array([ours.score(tokenize(q), i) for i in range(len(docs))])
        # bm25s "lucene" leaves out the (k1 + 1) factor in the numerator.
        np.testing.assert_allclose(mine / (1.5 + 1), theirs, rtol=0, atol=1e-12)


def test_matches_rank_bm25_when_using_its_idf():
    docs = [tokenize(t) for t in SMALL]
    ref = BM25Okapi(docs, k1=1.5, b=0.75)
    # On this set no word is in more than half of the 8 documents, so rank_bm25's
    # epsilon floor (for idf < 0) never fires and its idf is
    # ln((N - df + 0.5) / (df + 0.5)). "words" is in exactly 4: idf 0.
    assert all(v >= 0 for v in ref.idf.values())
    assert ref.idf["words"] == 0

    class RobertsonIdf(BM25):
        def idf(self, word):
            df = len(self.postings.get(word, ()))
            return math.log((self.N - df + 0.5) / (df + 0.5))

    ours = RobertsonIdf(docs, k1=1.5, b=0.75)
    for q in QUERIES:
        theirs = ref.get_scores(tokenize(q))
        mine = np.array([ours.score(tokenize(q), i) for i in range(len(docs))])
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)


def test_search_orders_by_score_and_breaks_ties_by_doc_number():
    docs = [tokenize(t) for t in TOY]
    hits = BM25(docs).search(["cat"], k=10)
    assert [i for i, _ in hits] == [0, 2]  # equal scores: lower number first
    assert hits[0][1] == hits[1][1] > 0


def test_tokenize_lowercases_and_keeps_numbers():
    # Runs of a-z and 0-9 after lowercasing; everything else splits.
    assert tokenize("BM25 Ranks, K1=1.5") == ["bm25", "ranks", "k1", "1", "5"]


def test_search_returns_exactly_k_hits_in_score_order():
    # Four documents of 4 words, so avgdl = 4 and norm = 1 - b + b * 4 / 4 = 1.
    # "cat" is in all four: idf = ln(1 + 0.5 / 4.5) = ln(10 / 9).
    # term = idf * tf * 2.5 / (tf + 1.5):
    #   tf 1: 2.5 / 2.5 = 1   tf 2: 5 / 3.5 = 10/7   tf 3: 7.5 / 4.5 = 5/3   tf 4: 10 / 5.5 = 20/11
    docs = [["cat", "cat", "cat", "x"], ["cat", "y", "y", "y"],
            ["cat", "cat", "z", "z"], ["cat", "cat", "cat", "cat"]]
    idf = math.log(10 / 9)
    hits = BM25(docs, k1=1.5, b=0.75).search(["cat"], k=2)
    assert [i for i, _ in hits] == [3, 0]
    assert [s for _, s in hits] == pytest.approx([idf * 20 / 11, idf * 5 / 3])
    hits = BM25(docs, k1=1.5, b=0.75).search(["cat"], k=3)
    assert [i for i, _ in hits] == [3, 0, 2]
    assert hits[0][1] > hits[1][1] > hits[2][1] == pytest.approx(idf * 10 / 7)


def test_a_number_in_the_query_finds_the_document_with_it():
    docs = [tokenize(t) for t in ["Released in 1999.", "Released in 2004.", "Never released."]]
    assert [i for i, _ in BM25(docs).search(tokenize("1999"))] == [0]


def test_a_repeated_query_word_counts_once_per_repeat():
    # The scores() docstring: a word repeated in the query counts once per
    # repeat, as in rank_bm25 and bm25s. Same TOY numbers as the first test:
    # one "cat" in document 0 is ln(1.6) * 2.5 / 2.725, so "cat cat" is twice that.
    index = BM25([tokenize(t) for t in TOY], k1=1.5, b=0.75)
    one = math.log(1.6) * 2.5 / 2.725
    assert index.score(["cat"], 0) == pytest.approx(one)
    assert index.score(["cat", "cat"], 0) == pytest.approx(2 * one)
    assert index.score(["cat", "sat", "cat"], 0) == pytest.approx(
        2 * one + index.score(["sat"], 0))


def test_defaults_are_k1_1_5_b_0_75_and_top_10():
    # README: "Defaults: k1 = 1.5, b = 0.75". The TOY hand value from the
    # first test uses exactly those, so the default index must reproduce it.
    index = BM25([tokenize(t) for t in TOY])
    assert (index.k1, index.b) == (1.5, 0.75)
    assert index.score(["cat"], 0) == pytest.approx(math.log(1.6) * 2.5 / 2.725)
    # search() without k returns the top 10. Twelve one-word documents tie,
    # so the lowest ten numbers win.
    many = BM25([["w"]] * 12)
    assert [i for i, _ in many.search(["w"])] == list(range(10))


def test_tokenize_splits_on_anything_outside_a_to_z_and_0_to_9():
    # README: "Tokenizer: lowercase, `[a-z0-9]+`". An underscore is not in
    # the set, so it splits; so do accented letters, which are dropped.
    assert tokenize("snake_case") == ["snake", "case"]
    assert tokenize("Naïve café") == ["na", "ve", "caf"]
