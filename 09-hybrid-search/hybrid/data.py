"""The corpus and queries come from Part 8 (08-bm25), read in place.

Run 08-bm25/build_corpus.py once to create 08-bm25/data/corpus.jsonl.
08-bm25/results/queries.jsonl is committed.
"""

import json
from pathlib import Path

PART8 = Path(__file__).resolve().parent.parent.parent / "08-bm25"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    """10,000 rows: id, title, url, text."""
    return read_jsonl(PART8 / "data" / "corpus.jsonl")


def load_queries(kind):
    """kind = "lexical" or "paraphrase". Rows: query_id, kind, query, gold_id."""
    return [q for q in read_jsonl(PART8 / "results" / "queries.jsonl") if q["kind"] == kind]


def searchable_text(doc):
    """Same text for both methods: the article title, then the paragraph."""
    return doc["title"] + "\n" + doc["text"]
