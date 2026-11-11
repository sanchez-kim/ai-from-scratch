"""Load the shared corpus and query sets. Parts 9 and 10 import this by path.

    import sys; sys.path.insert(0, "../08-bm25")
    from bm25.data import load_corpus, load_queries, searchable_text
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(name="corpus"):
    """name="corpus": the 10,000 paragraphs. name="extra": 1,000 more."""
    return read_jsonl(ROOT / "data" / f"{name}.jsonl")


def load_queries(kind="lexical"):
    """kind="lexical" or "paraphrase". Rows: query_id, query, gold_id, kind."""
    return [q for q in read_jsonl(ROOT / "results" / "queries.jsonl") if q["kind"] == kind]


def searchable_text(doc):
    """What gets indexed: the article title, then the paragraph."""
    return doc["title"] + "\n" + doc["text"]
