"""Embed the 10,000 paragraphs and the 1,000 queries once, cache them in data/.

    nice -n 10 uv run python embed_corpus.py

Queries are embedded twice: with the model card's query prefix (used
everywhere) and without it (one side check). The model revision is in every
file name, so a new revision never overwrites old vectors.
Writes results/embeddings.json: versions, shapes, sha256 of each file, a
re-run check. No timing (see bench_embed.py).
"""

import hashlib
import json
import platform
import time

import numpy as np

from common import DATA, DOC_VECS, KINDS, QUERY_VECS, QUERY_VECS_NOPREFIX, RESULTS, uptime
from hybrid.data import load_corpus, load_queries, searchable_text
from hybrid.embed import MODEL, QUERY_PREFIX, REVISION, embed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import sentence_transformers
    import torch
    import transformers

    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "uptime_before": uptime(),
           "model": MODEL, "revision": REVISION, "query_prefix": QUERY_PREFIX,
           "device": "cpu", "torch_threads": 2, "python": platform.python_version(),
           "versions": {"sentence_transformers": sentence_transformers.__version__,
                        "torch": torch.__version__, "transformers": transformers.__version__}}
    DATA.mkdir(exist_ok=True)
    docs = load_corpus()
    queries = [q for kind in KINDS for q in load_queries(kind)]
    texts = [q["query"] for q in queries]

    doc_vecs = embed([searchable_text(d) for d in docs])
    np.save(DOC_VECS, doc_vecs)
    np.save(QUERY_VECS, embed(texts, is_query=True))
    np.save(QUERY_VECS_NOPREFIX, embed(texts, is_query=True, prefix=""))
    (DATA / "ids.json").write_text(json.dumps({
        "docs": [d["id"] for d in docs], "queries": [q["query_id"] for q in queries]}))

    # Same input again, in a different batch: how far do the vectors move?
    again = embed([searchable_text(d) for d in docs[:64]], batch_size=8)
    out["rerun_max_abs_diff_first_64_docs_batch8_vs_batch32"] = float(np.abs(again - doc_vecs[:64]).max())
    out["norms"] = {"min": float(np.linalg.norm(doc_vecs, axis=1).min()),
                    "max": float(np.linalg.norm(doc_vecs, axis=1).max())}
    out["files"] = {str(p.relative_to(p.parent.parent)): {"shape": list(np.load(p).shape), "sha256": sha256(p)}
                    for p in (DOC_VECS, QUERY_VECS, QUERY_VECS_NOPREFIX)}
    out["max_seq_length"] = embed.__globals__["_model"].max_seq_length
    tok = embed.__globals__["_model"].tokenizer
    lengths = [len(tok(searchable_text(d))["input_ids"]) for d in docs]
    out["doc_token_lengths"] = {"max": max(lengths), "median": float(np.median(lengths)),
                                "over_max_seq_length": sum(n > out["max_seq_length"] for n in lengths)}
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["uptime_after"] = uptime()
    (RESULTS / "embeddings.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
