"""Recall@10 and MRR@10 of our BM25 on both query sets, plus two experiments.

    uv run python measure_retrieval.py

1. k1 and b: recall and MRR for a small grid, and what b = 0 does to the
   length of the paragraphs that reach the top 10.
2. Add 10% more documents (data/extra.jsonl) and see how avgdl, idf, scores
   and ranks of the original documents move.

Deterministic: no timing. Writes results/retrieval.json.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bm25 import BM25, tokenize
from bm25.data import load_queries, searchable_text
from common import corpus_tokens, top_k, uptime

OUT = Path(__file__).parent / "results" / "retrieval.json"
KINDS = ("lexical", "paraphrase")


def evaluate(index, queries, gold, k=10):
    hits, rr, top_lens = [], [], []
    for q, g in zip(queries, gold):
        top = [i for i, _ in index.search(q, k)]
        rank = top.index(g) + 1 if g in top else None
        hits.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
        top_lens.extend(index.doc_len[i] for i in top)
    return {"recall@10": float(np.mean(hits)), "mrr@10": float(np.mean(rr)),
            "mean_len_of_top10_docs": float(np.mean(top_lens)), "hits": hits}


def overlap(query_tokens, doc_tokens):
    doc = set(doc_tokens)
    return sum(w in doc for w in query_tokens) / len(query_tokens)


def main():
    t0, up_before = time.strftime("%Y-%m-%dT%H:%M:%S%z"), uptime()
    docs, toks = corpus_tokens()
    pos = {d["id"]: i for i, d in enumerate(docs)}
    sets = {k: load_queries(k) for k in KINDS}
    qtok = {k: [tokenize(q["query"]) for q in sets[k]] for k in KINDS}
    gold = {k: [pos[q["gold_id"]] for q in sets[k]] for k in KINDS}
    out = {"started": t0, "uptime_before": up_before, "documents": len(docs),
           "queries": {k: len(sets[k]) for k in KINDS}}

    # Worked examples for the article: one word, idf 1, k1 = 1.5, using term_score.
    unit = BM25([["w"], ["x"]], k1=1.5, b=0.75)  # avgdl = 1
    flat = BM25([["w"], ["x"]], k1=1.5, b=0.0)
    out["worked_examples"] = {
        "tf_saturation_k1_1.5": {tf: unit.term_score(tf, 1, 1.0) for tf in (1, 2, 4, 8)},
        "length_tf1_b0.75": {f"{f}x_avgdl": unit.term_score(1, f, 1.0) for f in (0.5, 1, 2)},
        "length_tf1_b0": {f"{f}x_avgdl": flat.term_score(1, f, 1.0) for f in (0.5, 1, 2)},
    }

    # How many query words appear in the gold paragraph (title + text)?
    out["query_word_overlap"] = {k: {
        "mean_share_of_query_words_in_gold": float(np.mean(
            [overlap(q, toks[g]) for q, g in zip(qtok[k], gold[k])])),
        "mean_query_words": float(np.mean([len(q) for q in qtok[k]])),
        "queries_with_zero_overlap": int(sum(overlap(q, toks[g]) == 0 for q, g in zip(qtok[k], gold[k]))),
        "queries_whose_text_repeats": int(sum(n for n in Counter(r["query"] for r in sets[k]).values() if n > 1)),
    } for k in KINDS}

    # Grid over k1 and b.
    grid = {}
    for k1 in (0.5, 1.2, 1.5, 2.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            index = BM25(toks, k1=k1, b=b)
            grid[f"k1={k1},b={b}"] = {k: {m: v for m, v in evaluate(index, qtok[k], gold[k]).items()
                                          if m != "hits"} for k in KINDS}
    out["grid"] = grid

    # b = 0 vs b = 0.75 at k1 = 1.5, split by the length of the gold paragraph.
    lens = np.array([len(t) for t in toks])
    q1, q3 = np.percentile(lens, [25, 75])
    out["length_quartiles_tokens"] = {"q1": float(q1), "median": float(np.median(lens)), "q3": float(q3)}
    by_len = {}
    for b in (0.0, 0.75):
        index = BM25(toks, k1=1.5, b=b)
        for k in KINDS:
            r = evaluate(index, qtok[k], gold[k])
            short = [h for h, g in zip(r["hits"], gold[k]) if lens[g] <= q1]
            long_ = [h for h, g in zip(r["hits"], gold[k]) if lens[g] >= q3]
            by_len[f"b={b}/{k}"] = {
                "recall@10": r["recall@10"], "mrr@10": r["mrr@10"],
                "mean_len_of_top10_docs": r["mean_len_of_top10_docs"],
                "recall@10_short_gold": float(np.mean(short)), "n_short": len(short),
                "recall@10_long_gold": float(np.mean(long_)), "n_long": len(long_)}
    out["b0_vs_b075"] = by_len

    # Add 10% more documents.
    extra_docs, extra_toks = corpus_tokens("extra")
    a = BM25(toks)
    b_index = BM25(toks + extra_toks)
    n = len(toks)
    changes, same_order, gold_rank_moves, new_in_top10 = [], 0, [], 0
    hits_before, hits_after = {k: [] for k in KINDS}, {k: [] for k in KINDS}
    all_q = [(q, g, k) for k in KINDS for q, g in zip(qtok[k], gold[k])]
    for q, g, kind in all_q:
        sa, sb = a.scores(q), b_index.scores(q)
        if g in sa:
            changes.append(sb[g] / sa[g] - 1)
        top_a = top_k(np.array([sa.get(i, 0.0) for i in range(n)]))
        full_b = np.array([sb.get(i, 0.0) for i in range(len(toks) + len(extra_toks))])
        top_b_all = top_k(full_b)
        new_in_top10 += sum(i >= n for i in top_b_all)
        top_b_orig = top_k(full_b[:n])
        same_order += top_a == top_b_orig
        hits_before[kind].append(g in top_a)
        hits_after[kind].append(g in top_b_all)
        ra = top_k(np.array([sa.get(i, 0.0) for i in range(n)]), n).index(g) + 1
        rb = top_k(full_b, len(full_b)).index(g) + 1
        gold_rank_moves.append(rb - ra)
    moves = np.array(gold_rank_moves)
    changes = np.array(changes)
    words = sorted({w for q, _, _ in all_q for w in q if w in a.postings})
    idf_shift = np.array([b_index.idf(w) - a.idf(w) for w in words])
    out["add_10_percent"] = {
        "documents_before": n, "documents_after": len(toks) + len(extra_toks),
        "avgdl_before": a.avgdl, "avgdl_after": b_index.avgdl,
        "avgdl_change": b_index.avgdl / a.avgdl - 1,
        "query_words": len(words),
        "idf_shift": {"min": float(idf_shift.min()), "median": float(np.median(idf_shift)),
                      "max": float(idf_shift.max()), "ln_1.1": float(np.log(1.1))},
        "gold_score_relative_change": {
            "min": float(changes.min()), "median": float(np.median(changes)),
            "max": float(changes.max()), "mean_abs": float(np.mean(np.abs(changes)))},
        "queries": len(all_q),
        "top10_of_original_docs_same_order": same_order / len(all_q),
        "new_docs_in_top10_total": int(new_in_top10),
        "gold_rank_unchanged": float(np.mean(moves == 0)),
        "gold_rank_worse": float(np.mean(moves > 0)), "gold_rank_better": float(np.mean(moves < 0)),
        "gold_rank_move_max": int(moves.max()),
        "recall@10_before_after": {k: [float(np.mean(hits_before[k])), float(np.mean(hits_after[k]))]
                                   for k in KINDS},
    }
    out["uptime_after"] = uptime()
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=2))


if __name__ == "__main__":
    main()
