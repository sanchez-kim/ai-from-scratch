"""Compare our BM25 scores with bm25s and rank_bm25 on the 10,000 paragraphs.

    uv run python measure_agreement.py

Every library gets the same token lists (our tokenizer), k1 = 1.5, b = 0.75.
Queries: both synthetic sets (results/queries.jsonl). Deterministic: no timing.
Writes results/agreement.json.
"""

import json
from importlib.metadata import version
import time
from pathlib import Path

import bm25s
import numpy as np
import rank_bm25
from rank_bm25 import BM25Okapi

from bm25 import BM25, tokenize
from common import all_queries, corpus_tokens, dense, top_k, uptime

K1, B = 1.5, 0.75
OUT = Path(__file__).parent / "results" / "agreement.json"


def rank10(scores, gold):
    top = top_k(scores)
    return top.index(gold) + 1 if gold in top else None


def compare(name, ours_list, theirs_list, scale=1.0):
    """ours_list, theirs_list: one score array per query (all documents)."""
    diffs = [np.abs(o / scale - t) for o, t in zip(ours_list, theirs_list)]
    same_order = same_set = same_top1 = 0
    overlap = []
    for o, t in zip(ours_list, theirs_list):
        a, b = top_k(o), top_k(t)
        same_order += a == b
        same_set += set(a) == set(b)
        same_top1 += a[0] == b[0]
        overlap.append(len(set(a) & set(b)) / 10)
    n = len(ours_list)
    return {
        "name": name, "scale_applied_to_ours": scale, "queries": n,
        "max_abs_diff": float(max(d.max() for d in diffs)),
        "mean_abs_diff_over_matching_docs": float(np.mean(np.concatenate(
            [d[o > 0] for d, o in zip(diffs, ours_list)]))),
        "top10_same_order": same_order / n, "top10_same_set": same_set / n,
        "top1_same": same_top1 / n, "top10_mean_overlap": float(np.mean(overlap)),
    }


def main():
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    up_before = uptime()
    docs, toks = corpus_tokens()
    n = len(docs)
    pos = {d["id"]: i for i, d in enumerate(docs)}
    rows = [(r["kind"], pos[r["gold_id"]], tokenize(r["query"])) for r in all_queries()]
    rows = [r for r in rows if r[2]]
    queries = [q for _, _, q in rows]

    ours = BM25(toks, k1=K1, b=B)
    ours_scores = [dense(ours.scores(q), n) for q in queries]

    results = []
    for dtype in ("float32", "float64"):
        ref = bm25s.BM25(k1=K1, b=B, method="lucene", dtype=dtype)
        ref.index(toks, show_progress=False)
        theirs = [np.asarray(ref.get_scores(q), dtype=np.float64) for q in queries]
        results.append(compare(f"bm25s lucene {dtype}, raw", ours_scores, theirs))
        results.append(compare(f"bm25s lucene {dtype}, ours / (k1 + 1)", ours_scores, theirs, K1 + 1))

    okapi = BM25Okapi(toks, k1=K1, b=B)
    theirs = [okapi.get_scores(q) for q in queries]
    results.append(compare("rank_bm25 BM25Okapi", ours_scores, theirs))
    # Does the idf difference change how often the known answer is found?
    quality = {}
    for name, score_list in (("ours", ours_scores), ("rank_bm25", theirs)):
        for kind in ("lexical", "paraphrase"):
            ranks = [rank10(s, g) for (k, g, _), s in zip(rows, score_list) if k == kind]
            quality[f"{name}/{kind}"] = {
                "queries": len(ranks), "recall@10": sum(r is not None for r in ranks) / len(ranks),
                "mrr@10": sum(1 / r for r in ranks if r) / len(ranks)}

    # Where rank_bm25's idf differs most: words in more than half the documents.
    floored = sorted(w for w, df in ((w, len(p)) for w, p in ours.postings.items()) if df > n / 2)
    examples = {w: {"df": len(ours.postings[w]), "ours_idf": ours.idf(w),
                    "rank_bm25_idf": okapi.idf[w],
                    "robertson_idf_before_floor": float(np.log((n - len(ours.postings[w]) + 0.5)
                                                               / (len(ours.postings[w]) + 0.5)))}
                for w in floored}
    top_df = sorted(ours.postings, key=lambda w: (-len(ours.postings[w]), w))[:15]
    most_common = [{"word": w, "df": len(ours.postings[w]), "ours_idf": ours.idf(w),
                    "rank_bm25_idf": okapi.idf[w]} for w in top_df]
    rare = min(ours.postings, key=lambda w: (len(ours.postings[w]), w))
    OUT.write_text(json.dumps({
        "started": t0, "uptime_before": up_before, "uptime_after": uptime(),
        "versions": {"bm25s": bm25s.__version__, "rank_bm25": version("rank-bm25"), "numpy": np.__version__},
        "k1": K1, "b": B, "documents": n, "avgdl": ours.avgdl, "vocabulary": len(ours.postings),
        "queries_scored": len(queries),
        "comparisons": results,
        "retrieval_quality": quality,
        "rank_bm25": {"epsilon": okapi.epsilon, "average_idf": okapi.average_idf,
                      "floor_value": okapi.epsilon * okapi.average_idf,
                      "words_in_more_than_half_of_docs": examples,
                      "15_most_common_words": most_common,
                      "rare_word_example": {"word": rare, "df": len(ours.postings[rare]),
                                            "ours_idf": ours.idf(rare), "rank_bm25_idf": okapi.idf[rare]}},
        "rank_bm25_module_file": rank_bm25.__file__.split("site-packages/")[-1],
    }, indent=2) + "\n")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
