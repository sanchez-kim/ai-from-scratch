"""Count survivors per sampler order on the probabilities collect_probs.py saved.

    uv run python analyze.py results/probs-<model>-<stamp>.json [--example PASSAGE:PREFIX]

Every chain here contains top-k (40), and top-k runs before top-p in both
orders. So top-p only ever renormalizes over the top 40, and the top 100
candidates the server returned are enough for exact counts: the missing tail
never enters the sum.

Writes results/survivors-<same stamp>.json.
"""

import json
import math
import statistics
import sys
from pathlib import Path

from sampling import run_chain, softmax

TS = (0.7, 1.0, 1.3)
PS = (0.9, 0.95)
K = 40


def first(t, p):   # HF transformers: TemperatureLogitsWarper before TopK/TopP
    return [("temperature", t), ("top_k", K), ("top_p", p)]


def last(t, p):    # llama.cpp default chain: top_k, top_p, ..., temperature
    return [("top_k", K), ("top_p", p), ("temperature", t)]


LLAMACPP_DEFAULT = [("top_k", 40), ("top_p", 0.95), ("min_p", 0.05), ("temperature", 0.8)]
LLAMACPP_DEFAULT_TEMP_FIRST = [("temperature", 0.8), ("top_k", 40), ("top_p", 0.95), ("min_p", 0.05)]


def quantiles(xs):
    # Linear interpolation between order statistics (statistics.quantiles 'inclusive').
    q = statistics.quantiles(xs, n=10, method="inclusive")
    return {"p10": round(q[0], 3), "median": round(statistics.median(xs), 3), "p90": round(q[8], 3),
            "mean": round(statistics.fmean(xs), 3), "min": min(xs), "max": max(xs)}


def tvd(a, b):
    # Total variation distance between two final sampling distributions:
    # the share of probability that sits on different tokens.
    pa, pb = dict(softmax(a)), dict(softmax(b))
    return 0.5 * sum(abs(pa.get(t, 0) - pb.get(t, 0)) for t in set(pa) | set(pb))


def compare(positions, chain_a, chain_b, server_a=None, server_b=None):
    na, nb, diff, smaller, larger, dists, keeps_a, keeps_b = [], [], 0, 0, 0, [], 0, 0
    match_a = match_b = 0
    mismatches = []
    for pos in positions:
        cands = [(tok, lp) for tok, lp in pos["top"]]
        a, b = run_chain(cands, chain_a), run_chain(cands, chain_b)
        sa, sb = {t for t, _ in a}, {t for t, _ in b}
        na.append(len(sa))
        nb.append(len(sb))
        if sa != sb:
            diff += 1
            smaller += len(sa) < len(sb)
            larger += len(sa) > len(sb)
            dists.append(tvd(a, b))
        keeps_a += pos["actual_next"] in sa
        keeps_b += pos["actual_next"] in sb
        for name, mine, store in ((server_a, sa, "a"), (server_b, sb, "b")):
            if name is None:
                continue
            if set(pos["survivors"][name]) == mine:
                if store == "a":
                    match_a += 1
                else:
                    match_b += 1
            else:
                mismatches.append({"chain": name, "passage": pos["passage"], "prefix_len": pos["prefix_len"],
                                   "mine": len(mine), "server": len(pos["survivors"][name])})
    n = len(positions)
    out = {
        "positions": n,
        "survivors_a": quantiles(na), "survivors_b": quantiles(nb),
        "differ": diff, "differ_share": round(diff / n, 4),
        "a_smaller": smaller, "a_larger": larger,
        "tvd_when_differ": quantiles(dists) if len(dists) >= 2 else None,
        "actual_next_kept_a": keeps_a, "actual_next_kept_b": keeps_b,
    }
    if server_a or server_b:
        out["server_match_a"] = match_a if server_a else None
        out["server_match_b"] = match_b if server_b else None
        out["server_mismatches"] = mismatches
    return out


def worked_example(pos):
    # The beginner example in the post: keep only the top 5 candidates, rescale
    # them to add up to 1, then turn each knob on those five.
    five = [(tok, lp) for tok, lp in pos["top"][:5]]
    text = dict(zip([t for t, _ in pos["top"][:20]], pos["top_text"]))
    out = {"passage": pos["passage"], "prefix_len": pos["prefix_len"], "context_tail": pos["context_tail"],
           "actual_next_text": pos["actual_next_text"],
           "full_vocab_prob": [round(math.exp(lp), 4) for _, lp in five]}
    for t in (1.0, 0.7, 1.3):
        probs = softmax([(tok, lp / t) for tok, lp in five])
        out[f"T{t}"] = [[text[tok], round(p, 4)] for tok, p in probs]
        cum, run = [], 0.0
        for _, p in probs:
            run += p
            cum.append(round(run, 4))
        out[f"T{t}-cumulative"] = cum
    for t in (0.7, 1.3):
        out[f"top_p0.9-T{t}-first"] = [text[tok] for tok, _ in run_chain(five, [("temperature", t), ("top_p", 0.9)])]
        out[f"top_p0.9-T{t}-last"] = [text[tok] for tok, _ in run_chain(five, [("top_p", 0.9), ("temperature", t)])]
    out["top_k3"] = [text[tok] for tok, _ in run_chain(five, [("top_k", 3)])]
    # The same position with all 100 candidates, as drawn in the figure (k 15)
    # and as measured (k 40), at T 0.7 and top-p 0.9.
    cands = [(tok, lp) for tok, lp in pos["top"]]
    for k in (15, 40):
        out[f"k{k}-T0.7-p0.9-survivors"] = {
            "first": len(run_chain(cands, [("temperature", 0.7), ("top_k", k), ("top_p", 0.9)])),
            "last": len(run_chain(cands, [("top_k", k), ("top_p", 0.9), ("temperature", 0.7)])),
        }
    ranks = [tok for tok, _ in pos["top"]]
    out["actual_next_rank"] = ranks.index(pos["actual_next"]) + 1 if pos["actual_next"] in ranks else None
    out["actual_next_prob"] = round(math.exp(dict(pos["top"])[pos["actual_next"]]), 4) if out["actual_next_rank"] else None
    return out


def main():
    src = Path(sys.argv[1])
    data = json.loads(src.read_text())
    positions = data["positions"]

    top1 = [math.exp(pos["top"][0][1]) for pos in positions]
    tail = [1 - sum(math.exp(lp) for _, lp in pos["top"]) for pos in positions]
    in_top = [pos["actual_next"] in {t for t, _ in pos["top"]} for pos in positions]

    report = {
        "source": src.name,
        "server_build": data["server_build"],
        "server_default_samplers": data["server_default_samplers"],
        "uptime_start": data["uptime_start"], "uptime_end": data["uptime_end"],
        "positions": len(positions),
        "top1_prob": quantiles(top1),
        "top1_at_least_0.9": sum(p >= 0.9 for p in top1),
        "top1_at_least_0.5": sum(p >= 0.5 for p in top1),
        "mass_outside_top100": quantiles(tail),
        "actual_next_in_top100": sum(in_top),
        "settings": {},
    }
    for t in TS:
        for p in PS:
            name = f"T{t}-p{p}"
            report["settings"][name] = {
                "a": "temperature first", "b": "temperature last",
                **compare(positions, first(t, p), last(t, p), f"{name}-first", f"{name}-last"),
            }
    report["settings"]["llamacpp-default"] = {
        "a": "temperature first (same numbers)", "b": "llama.cpp default order",
        **compare(positions, LLAMACPP_DEFAULT_TEMP_FIRST, LLAMACPP_DEFAULT,
                  "llamacpp-default-temp-first", "llamacpp-default"),
    }
    # HF with no generation_config.json: do_sample defaults to False (greedy). With
    # do_sample=True and nothing else set: temperature 1.0, top_k 50, top_p 1.0.
    report["settings"]["hf-bare-sampling"] = {
        "a": "T 1.0, top_k 50, top_p 1.0", "b": "same",
        **compare(positions, [("top_k", 50)], [("top_k", 50)]),
    }

    if "--example" in sys.argv:
        pi, n = map(int, sys.argv[sys.argv.index("--example") + 1].split(":"))
        report["example"] = worked_example(next(x for x in positions if x["passage"] == pi and x["prefix_len"] == n))

    out = src.parent / src.name.replace("probs-", "survivors-")
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)
    for name, s in report["settings"].items():
        print(f'{name:18s} first {s["survivors_a"]["median"]:>5} [{s["survivors_a"]["p10"]}-{s["survivors_a"]["p90"]}]'
              f'  last {s["survivors_b"]["median"]:>5} [{s["survivors_b"]["p10"]}-{s["survivors_b"]["p90"]}]'
              f'  differ {s["differ"]}/{s["positions"]}  smaller {s["a_smaller"]} larger {s["a_larger"]}'
              f'  match {s.get("server_match_a")}/{s.get("server_match_b")}')


if __name__ == "__main__":
    main()
