"""Sweep d_k and measure what 1/sqrt(d_k) does to attention weights.

    uv run python measure_scaling.py

Queries and keys are random vectors with entries drawn from N(0, 1), the
assumption behind the 1/sqrt(d_k) footnote in Vaswani et al. 2017. For each
seed the script draws 64 queries and 64 keys of 512 dimensions and uses the
first d_k dimensions for each d_k, so every d_k sees the same vectors cut to
length. Everything is deterministic; nothing here is timed.

Writes results/scaling-<YYYY-MM-DD>.json.
"""

import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from attention import attention_weights, entropy_bits, jacobian_norm, matmul, softmax, transpose, variance

D_KS = [8, 16, 32, 64, 128, 256, 512]
SEEDS = list(range(10))
SEQ = 64          # queries per seed, and keys per query
FIG_KEYS = 12     # bars in the animated figure
VARIANTS = {      # name -> scale as a function of d_k
    "none": lambda d: 1.0,
    "sqrt": lambda d: 1.0 / math.sqrt(d),
    "linear": lambda d: 1.0 / d,
}


def rand_matrix(rng, rows, cols):
    return [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


def prefix(m, d):
    return [row[:d] for row in m]


def summary(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


SATURATED = 0.99  # a row counts as saturated when one weight is above this


def row_stats(weights):
    """Per seed: mean entropy (bits), mean largest weight, mean and median
    softmax Jacobian norm, and the share of rows with one weight above 0.99."""
    n = len(weights)
    jacs = [jacobian_norm(w) for w in weights]
    return (
        sum(entropy_bits(w) for w in weights) / n,
        sum(max(w) for w in weights) / n,
        sum(jacs) / n,
        statistics.median(jacs),
        sum(max(w) > SATURATED for w in weights) / n,
    )


def sweep():
    # per[variant][d_k] -> lists over seeds
    per = {v: {d: {"score_var": [], "entropy": [], "max_w": [], "jac": [], "jac_med": [], "sat": []} for d in D_KS} for v in VARIANTS}
    for seed in SEEDS:
        rng = random.Random(seed)
        q_all, k_all = rand_matrix(rng, SEQ, max(D_KS)), rand_matrix(rng, SEQ, max(D_KS))
        for d in D_KS:
            q, k = prefix(q_all, d), prefix(k_all, d)
            raw = [s for row in matmul(q, transpose(k)) for s in row]
            raw_var = variance(raw)
            for name, scale_of in VARIANTS.items():
                scale = scale_of(d)
                ent, mw, jac, jac_med, sat = row_stats(attention_weights(q, k, scale))
                cell = per[name][d]
                cell["score_var"].append(raw_var * scale * scale)
                cell["entropy"].append(ent)
                cell["max_w"].append(mw)
                cell["jac"].append(jac)
                cell["jac_med"].append(jac_med)
                cell["sat"].append(sat)
    max_bits = math.log2(SEQ)
    out = {}
    for name in VARIANTS:
        out[name] = []
        for d in D_KS:
            cell = per[name][d]
            out[name].append({
                "d_k": d,
                "score_variance": summary(cell["score_var"]),
                "entropy_bits": summary(cell["entropy"]),
                "entropy_fraction_of_max": summary([e / max_bits for e in cell["entropy"]]),
                "max_weight": summary(cell["max_w"]),
                "softmax_jacobian_norm": summary(cell["jac"]),
                "softmax_jacobian_norm_median_row": summary(cell["jac_med"]),
                "saturated_row_share": summary(cell["sat"]),
            })
    return out


def figure_rows():
    """One query against 12 keys, seed 0: the rows drawn in the animated figure."""
    rng = random.Random(0)
    q_all, k_all = rand_matrix(rng, 1, max(D_KS)), rand_matrix(rng, FIG_KEYS, max(D_KS))
    rows = []
    for d in D_KS:
        q, k = prefix(q_all, d), prefix(k_all, d)
        for name in ("none", "sqrt"):
            w = attention_weights(q, k, VARIANTS[name](d))[0]
            rows.append({"d_k": d, "scale": name, "weights": w, "entropy_bits": entropy_bits(w), "max_weight": max(w)})
    return rows


def worked_examples():
    # Exact variance of q.k when every entry is +1 or -1 with equal chance:
    # enumerate every sign pattern of the product q_i * k_i.
    exact = []
    for d in (2, 4, 8):
        dots = [sum(signs) for signs in itertools.product((1, -1), repeat=d)]
        exact.append({"d_k": d, "possible_dot_products": sorted(set(dots)), "variance": variance(dots)})
    small, big = [2.0, 1.0, 0.0], [20.0, 10.0, 0.0]
    # Softmax Jacobian norm of a row split evenly over m of the 64 keys (the rest 0).
    even = {m: jacobian_norm([1.0 / m] * m + [0.0] * (SEQ - m)) for m in range(1, SEQ + 1)}
    return {
        "jacobian_norm_even_split_over_m_keys": {
            "values": {str(m): v for m, v in even.items()},
            "largest_at_m": max(even, key=even.get),
        },
        "plus_minus_one_dot_products": exact,
        "softmax_small_vs_10x": {"scores": [small, big], "weights": [softmax(small), softmax(big)],
                                  "entropy_bits": [entropy_bits(softmax(small)), entropy_bits(softmax(big))]},
    }


def main():
    uptime_before = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    result = {
        "script": "measure_scaling.py",
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "uptime_before": uptime_before,
        "setup": {
            "entries": "N(0, 1), random.Random(seed).gauss",
            "d_ks": D_KS, "seeds": SEEDS, "queries_per_seed": SEQ, "keys_per_query": SEQ,
            "max_entropy_bits": math.log2(SEQ),
            "per_seed_statistic": "mean over the 64 query rows (softmax_jacobian_norm_median_row: median over rows); "
                                  "score_variance over all 64x64 scores; saturated_row_share: rows with max weight > 0.99",
            "summary": "median, min and max over seeds",
            "variants": {"none": "scale 1", "sqrt": "scale 1/sqrt(d_k)", "linear": "scale 1/d_k"},
        },
        "sweep": sweep(),
        "figure": {"keys": FIG_KEYS, "seed": 0, "max_entropy_bits": math.log2(FIG_KEYS), "rows": figure_rows()},
        "worked_examples": worked_examples(),
    }
    result["uptime_after"] = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    out = Path(__file__).resolve().parent / "results" / f"scaling-{datetime.now():%Y-%m-%d}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out.relative_to(Path(__file__).resolve().parent))
    for name in ("none", "sqrt", "linear"):
        print(f"\nscale={name}")
        print("  d_k  score_var    entropy(bits)  frac   max_w   jac_mean  jac_medrow  saturated")
        for r in result["sweep"][name]:
            print(f"  {r['d_k']:>3}  {r['score_variance']['median']:>9.3f}  "
                  f"{r['entropy_bits']['median']:>6.3f} [{r['entropy_bits']['min']:.2f}-{r['entropy_bits']['max']:.2f}]  "
                  f"{r['entropy_fraction_of_max']['median']:.3f}  {r['max_weight']['median']:.3f}  "
                  f"{r['softmax_jacobian_norm']['median']:.4f}    {r['softmax_jacobian_norm_median_row']['median']:.4f}     "
                  f"{r['saturated_row_share']['median']:.3f}")


if __name__ == "__main__":
    main()
