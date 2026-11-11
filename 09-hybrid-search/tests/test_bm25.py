import math
from pathlib import Path

import pytest

from hybrid.bm25 import BM25, tokenize

PART8 = Path(__file__).resolve().parents[2] / "08-bm25" / "bm25" / "bm25.py"
COPY = Path(__file__).resolve().parents[1] / "hybrid" / "bm25.py"


def test_copy_matches_part_8_except_the_header_line():
    if not PART8.exists():
        pytest.skip("08-bm25 is not next to this folder; the hand-value tests below still run")
    lines = COPY.read_text().splitlines()
    assert lines[0].startswith("# Copied unchanged from 08-bm25/bm25/bm25.py")
    assert lines[1:] == PART8.read_text().splitlines()


def test_tokenize_lowercases_and_keeps_numbers():
    assert tokenize("BM25 Ranks, K1=1.5") == ["bm25", "ranks", "k1", "1", "5"]


def test_toy_score_by_hand():
    # Same toy as Part 8. Lengths 6, 3, 6, so avgdl = 5.
    docs = [tokenize(t) for t in ["the cat sat on the mat", "the dog sat", "cats and dogs and a cat"]]
    index = BM25(docs, k1=1.5, b=0.75)
    # "cat": df = 2, N = 3, idf = ln(1 + 1.5 / 2.5) = ln(1.6).
    # Document 0: tf 1, dl 6, norm = 0.25 + 0.75 * 6 / 5 = 1.15,
    # term = ln(1.6) * 2.5 / (1 + 1.5 * 1.15) = ln(1.6) * 2.5 / 2.725.
    assert index.score(["cat"], 0) == pytest.approx(math.log(1.6) * 2.5 / 2.725)
    assert index.score(["cat"], 1) == 0.0


def test_search_returns_exactly_k_hits_in_score_order():
    # Four documents of 4 words: norm = 1. "cat" is in all four, idf = ln(10 / 9).
    # term = idf * tf * 2.5 / (tf + 1.5): tf 4 -> 20/11, tf 3 -> 5/3, tf 2 -> 10/7.
    docs = [["cat", "cat", "cat", "x"], ["cat", "y", "y", "y"],
            ["cat", "cat", "z", "z"], ["cat", "cat", "cat", "cat"]]
    idf = math.log(10 / 9)
    hits = BM25(docs, k1=1.5, b=0.75).search(["cat"], k=2)
    assert [i for i, _ in hits] == [3, 0]
    assert [s for _, s in hits] == pytest.approx([idf * 20 / 11, idf * 5 / 3])
