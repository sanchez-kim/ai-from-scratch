"""BM25, embeddings, score sum and RRF on both query sets. Deterministic, no timing.

    nice -n 10 uv run python measure_fusion.py      # after embed_corpus.py

For every query:
1. BM25 (k1 1.5, b 0.75) top 100, and embedding (cosine) top 100.
2. Fuse those two lists: score sum (min-max), RRF k = 60, RRF k = 1,
   and a small sweep of k.
3. recall@10 and MRR@10 of each list against the one gold paragraph.
4. Kendall tau-b between the BM25 and embedding scores over the union of
   the two top-10 lists (the definition fixed before measuring), and, added
   after the first run, over each method's own top 10, plus a null check
   with random scores.
5. Added after the first run (exploratory): fusion depth 10/20/50/100.
Writes results/fusion.json, results/per-query.jsonl, results/worked-example.json.
"""

import hashlib
import json
import random
import time

import numpy as np

from common import DOC_VECS, KINDS, QUERY_VECS, QUERY_VECS_NOPREFIX, RESULTS, TAG, top_k, uptime
from hybrid import cosine, kendall_tau_b, reciprocal_rank, rrf, sum_scores
from hybrid.bm25 import BM25, tokenize
from hybrid.data import PART8, load_corpus, load_queries, searchable_text
from hybrid.metrics import gold_rank

DEPTH = 100          # each method hands this many results to fusion
K_SWEEP = (1, 5, 10, 20, 60, 100, 200)
DEPTH_SWEEP = (10, 20, 50, 100)
METHODS = ("bm25", "embedding", "sum", "rrf60", "rrf1")


def rank_in(ranking, gold):
    """1-based rank of gold anywhere in the list, or None."""
    return ranking.index(gold) + 1 if gold in ranking else None


def null_union_tau(n_docs=10_000, trials=1000, seed=9):
    """Tau over the union of two top-10 lists when the two scores are unrelated.

    If picking the union alone pushes tau below 0, a negative tau on real
    data says nothing about the methods.
    """
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(trials):
        a, b = rng.random(n_docs), rng.random(n_docs)
        union = sorted(set(top_k(a, 10)) | set(top_k(b, 10)))
        taus.append(kendall_tau_b([a[i] for i in union], [b[i] for i in union]))
    return {"trials": trials, "mean": float(np.mean(taus)), "p10": float(np.percentile(taus, 10)),
            "p90": float(np.percentile(taus, 90))}


def main():
    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "uptime_before": uptime(),
           "embeddings": TAG, "bm25": {"k1": 1.5, "b": 0.75}, "fusion_depth": DEPTH,
           "kendall_tau": "tau-b between BM25 score and cosine, over the union of both top-10 lists"}
    out["inputs_sha256"] = {f"08-bm25/{name}": hashlib.sha256((PART8 / name).read_bytes()).hexdigest()
                            for name in ("data/corpus.jsonl", "results/queries.jsonl")}
    docs = load_corpus()
    pos = {d["id"]: i for i, d in enumerate(docs)}
    index_tokens = [tokenize(searchable_text(d)) for d in docs]
    index = BM25(index_tokens)
    doc_vecs = np.load(DOC_VECS).astype(np.float64)
    all_q = np.load(QUERY_VECS).astype(np.float64)
    all_q_noprefix = np.load(QUERY_VECS_NOPREFIX).astype(np.float64)

    # The pure-Python cosine and the numpy product must agree.
    rng = random.Random(9)
    pairs = [(rng.randrange(len(all_q)), rng.randrange(len(docs))) for _ in range(200)]
    out["pure_cosine_vs_numpy_dot_max_abs_diff"] = max(
        abs(cosine(list(all_q[q]), list(doc_vecs[d])) - float(all_q[q] @ doc_vecs[d])) for q, d in pairs)

    per_query, row = [], 0
    for kind in KINDS:
        for q in load_queries(kind):
            gold = pos[q["gold_id"]]
            words = tokenize(q["query"])
            bm25_scored = index.search(words, DEPTH)
            bm25_ids = [i for i, _ in bm25_scored]
            cos = doc_vecs @ all_q[row]
            emb_ids = top_k(cos, DEPTH)
            emb_scored = [(i, float(cos[i])) for i in emb_ids]
            noprefix_ids = top_k(doc_vecs @ all_q_noprefix[row], 10)
            row += 1

            lists = {"bm25": bm25_ids, "embedding": emb_ids,
                     "sum": [d for d, _ in sum_scores([bm25_scored, emb_scored])],
                     "rrf60": [d for d, _ in rrf([bm25_ids, emb_ids], k=60)],
                     "rrf1": [d for d, _ in rrf([bm25_ids, emb_ids], k=1)]}
            sweep = {k: reciprocal_rank([d for d, _ in rrf([bm25_ids, emb_ids], k=k)], gold)
                     for k in K_SWEEP}
            depth = {d: {"rrf60": reciprocal_rank([x for x, _ in rrf([bm25_ids[:d], emb_ids[:d]])], gold),
                         "sum": reciprocal_rank([x for x, _ in sum_scores([bm25_scored[:d], emb_scored[:d]])], gold)}
                     for d in DEPTH_SWEEP}

            union = sorted(set(bm25_ids[:10]) | set(emb_ids[:10]))
            bm25_all = index.scores(words)
            tau = kendall_tau_b([bm25_all.get(i, 0.0) for i in union], [float(cos[i]) for i in union])
            tau_b10 = kendall_tau_b([bm25_all.get(i, 0.0) for i in bm25_ids[:10]],
                                    [float(cos[i]) for i in bm25_ids[:10]])
            tau_e10 = kendall_tau_b([bm25_all.get(i, 0.0) for i in emb_ids[:10]],
                                    [float(cos[i]) for i in emb_ids[:10]])
            per_query.append({
                "query_id": q["query_id"], "kind": kind, "gold": gold,
                "rank_top10": {m: gold_rank(lists[m], gold) for m in METHODS},
                "rank_top100": {m: rank_in(lists[m][:DEPTH], gold) for m in ("bm25", "embedding")},
                "rr_sweep": sweep, "rr_depth": depth,
                "noprefix_hit": gold in noprefix_ids,
                "tau": tau, "tau_in_bm25_top10": tau_b10, "tau_in_embedding_top10": tau_e10,
                "union_size": len(union),
                "top10_overlap": len(set(bm25_ids[:10]) & set(emb_ids[:10])),
                "bm25_matched_docs": len(bm25_all),
            })

    def agg(rows):
        res = {}
        for m in METHODS:
            ranks = [r["rank_top10"][m] for r in rows]
            res[m] = {"recall@10": float(np.mean([r is not None for r in ranks])),
                      "mrr@10": float(np.mean([1 / r if r else 0.0 for r in ranks]))}
        return res

    summary = {}
    for kind in KINDS:
        rows = [r for r in per_query if r["kind"] == kind]
        hit = {m: [r["rank_top10"][m] is not None for r in rows] for m in METHODS}
        taus = np.array([r["tau"] for r in rows])
        s = {"queries": len(rows), "methods": agg(rows)}
        s["rrf_k_sweep"] = {k: {"mrr@10": float(np.mean([r["rr_sweep"][k] for r in rows])),
                                "recall@10": float(np.mean([r["rr_sweep"][k] > 0 for r in rows]))}
                            for k in K_SWEEP}
        s["embedding_without_query_prefix"] = {"recall@10": float(np.mean([r["noprefix_hit"] for r in rows]))}
        s["gold_in_top10"] = {
            "both": int(sum(a and b for a, b in zip(hit["bm25"], hit["embedding"]))),
            "bm25_only": int(sum(a and not b for a, b in zip(hit["bm25"], hit["embedding"]))),
            "embedding_only": int(sum(b and not a for a, b in zip(hit["bm25"], hit["embedding"]))),
            "neither": int(sum(not a and not b for a, b in zip(hit["bm25"], hit["embedding"]))),
        }
        for m in ("sum", "rrf60", "rrf1"):
            s["gold_in_top10"][f"{m}_found_but_neither_single"] = int(sum(
                h and not a and not b for h, a, b in zip(hit[m], hit["bm25"], hit["embedding"])))
            s["gold_in_top10"][f"{m}_lost_from_a_single"] = int(sum(
                (not h) and (a or b) for h, a, b in zip(hit[m], hit["bm25"], hit["embedding"])))
        s["gold_in_top100"] = {m: float(np.mean([r["rank_top100"][m] is not None for r in rows]))
                               for m in ("bm25", "embedding")}
        s["kendall_tau"] = {"mean": float(taus.mean()), "median": float(np.median(taus)),
                            "p10": float(np.percentile(taus, 10)), "p90": float(np.percentile(taus, 90)),
                            "share_below_0": float(np.mean(taus < 0))}
        for key in ("tau_in_bm25_top10", "tau_in_embedding_top10"):
            v = np.array([r[key] for r in rows])
            s[key] = {"mean": float(v.mean()), "median": float(np.median(v))}
        s["depth_sweep"] = {d: {m: {"recall@10": float(np.mean([r["rr_depth"][d][m] > 0 for r in rows])),
                                    "mrr@10": float(np.mean([r["rr_depth"][d][m] for r in rows]))}
                                for m in ("rrf60", "sum")} for d in DEPTH_SWEEP}
        s["top10_overlap_mean"] = float(np.mean([r["top10_overlap"] for r in rows]))
        s["by_top10_overlap"] = {}
        for name, test in (("0", lambda n: n == 0), ("1-2", lambda n: 1 <= n <= 2), ("3+", lambda n: n >= 3)):
            group = [r for r in rows if test(r["top10_overlap"])]
            if group:
                a = agg(group)
                s["by_top10_overlap"][name] = {"queries": len(group),
                                               "recall@10": {m: a[m]["recall@10"] for m in METHODS}}
        s["union_size_mean"] = float(np.mean([r["union_size"] for r in rows]))
        s["bm25_matched_docs_median"] = float(np.median([r["bm25_matched_docs"] for r in rows]))
        # Split the queries into three equal groups by tau.
        order = sorted(rows, key=lambda r: (r["tau"], r["query_id"]))
        thirds = [order[: len(order) // 3], order[len(order) // 3: 2 * len(order) // 3],
                  order[2 * len(order) // 3:]]
        s["by_tau_third"] = []
        for name, group in zip(("low", "middle", "high"), thirds):
            a = agg(group)
            best = max(a["bm25"]["recall@10"], a["embedding"]["recall@10"])
            s["by_tau_third"].append({
                "third": name, "queries": len(group),
                "tau_range": [group[0]["tau"], group[-1]["tau"]],
                "recall@10": {m: a[m]["recall@10"] for m in METHODS},
                "mrr@10": {m: a[m]["mrr@10"] for m in METHODS},
                "rrf60_minus_best_single_recall": a["rrf60"]["recall@10"] - best})
        summary[kind] = s
    out["results"] = summary
    # Both sets together (500 + 500): what a mixed stream of queries would see.
    out["all_queries"] = {"queries": len(per_query), "methods": agg(per_query)}
    out["null_check_union_tau_random_scores"] = null_union_tau()

    # Worked example, chosen by a fixed rule: the first paraphrase query (by id)
    # whose gold is in both top-5 lists at different ranks, first in neither,
    # and first after RRF k = 60.
    ex = next(r for r in per_query if r["kind"] == "paraphrase"
              and r["rank_top10"]["bm25"] and r["rank_top10"]["embedding"]
              and r["rank_top10"]["bm25"] <= 5 and r["rank_top10"]["embedding"] <= 5
              and r["rank_top10"]["bm25"] != r["rank_top10"]["embedding"]
              and r["rank_top10"]["bm25"] > 1 and r["rank_top10"]["embedding"] > 1
              and r["rank_top10"]["rrf60"] == 1)
    q = next(x for x in load_queries("paraphrase") if x["query_id"] == ex["query_id"])
    qrow = [x["query_id"] for k in KINDS for x in load_queries(k)].index(ex["query_id"])
    words = tokenize(q["query"])
    bm25_scored = index.search(words, DEPTH)
    cos = doc_vecs @ all_q[qrow]
    emb_ids = top_k(cos, DEPTH)
    bm25_ids = [i for i, _ in bm25_scored]
    fused = rrf([bm25_ids, emb_ids], k=60)
    fused_k1 = rrf([bm25_ids, emb_ids], k=1)

    def describe(i):
        return {"doc": docs[i]["id"], "title": docs[i]["title"], "gold": i == ex["gold"],
                "bm25_rank": rank_in(bm25_ids, i), "embedding_rank": rank_in(emb_ids, i)}

    worked = {"rule": "first paraphrase query by id: gold in both top-5 at different ranks, "
                      "first in neither, first after RRF k=60",
              "query_id": q["query_id"], "query": q["query"], "gold_id": q["gold_id"],
              "gold_title": docs[ex["gold"]]["title"],
              "gold_text": docs[ex["gold"]]["text"],
              "bm25_top5": [dict(describe(i), score=s) for i, s in bm25_scored[:5]],
              "embedding_top5": [dict(describe(i), score=float(cos[i])) for i in emb_ids[:5]],
              "rrf60_top5": [dict(describe(d), rrf=s) for d, s in fused[:5]],
              "rrf1_top5": [dict(describe(d), rrf=s) for d, s in fused_k1[:5]],
              "kendall_tau": ex["tau"],
              "rrf60_rank_of_each_list_top2": {
                  "bm25": [[docs[i]["title"], [d for d, _ in fused].index(i) + 1] for i in bm25_ids[:2]],
                  "embedding": [[docs[i]["title"], [d for d, _ in fused].index(i) + 1] for i in emb_ids[:2]]},
              "query_words": words,
              "query_words_in_gold": [w for w in words if w in set(index_tokens[ex["gold"]])],
              "query_words_in_bm25_top1": [w for w in words if w in set(index_tokens[bm25_ids[0]])]}
    (RESULTS / "worked-example.json").write_text(json.dumps(worked, indent=2, ensure_ascii=False) + "\n")

    with open(RESULTS / "per-query.jsonl", "w") as f:
        for r in per_query:
            f.write(json.dumps(r) + "\n")
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["uptime_after"] = uptime()
    (RESULTS / "fusion.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: {m: v for m, v in s.items() if m in ("methods", "kendall_tau", "gold_in_top10")}
                      for k, s in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
