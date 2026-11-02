"""Time the two decode paths in kvcache/cache.py at growing sequence lengths.

    uv run python bench_pure.py            # writes results/pure-<date>.json

Each length runs both paths REPEATS times. The no-cache path is quadratic in
projections, so the longest length is capped to keep the run under a minute
or so on a laptop.
"""

import datetime
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from kvcache.cache import KVCache, TinyAttention, decode_with_cache, decode_without_cache

LENGTHS = [64, 128, 256, 512, 1024]
DIM = 16
VOCAB_SIZE = 256
REPEATS = 3
SEED = 2026


class Counting(TinyAttention):
    def __init__(self, *args):
        super().__init__(*args)
        self.projections = 0

    def project(self, token):
        self.projections += 1
        return super().project(token)


def build(rng):
    def row():
        return [rng.uniform(-1, 1) for _ in range(DIM)]

    embeddings = {t: row() for t in range(VOCAB_SIZE)}
    wq, wk, wv = ([row() for _ in range(DIM)] for _ in range(3))
    return embeddings, wq, wk, wv


def run(path, weights, tokens):
    model = Counting(*weights)
    start = time.perf_counter()
    if path == "with_cache":
        out = decode_with_cache(model, tokens, KVCache())
    else:
        out = decode_without_cache(model, tokens)
    return time.perf_counter() - start, model.projections, out


def main():
    rng = random.Random(SEED)
    weights = build(rng)
    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    print(uptime)
    rows = []
    for n in LENGTHS:
        tokens = [rng.randrange(VOCAB_SIZE) for _ in range(n)]
        row = {"length": n}
        outputs = {}
        for path in ("with_cache", "without_cache"):
            times = []
            for _ in range(REPEATS):
                seconds, projections, out = run(path, weights, tokens)
                times.append(seconds)
            outputs[path] = out
            row[path] = {
                "projections": projections,
                "seconds": times,
                "median_s": statistics.median(times),
            }
        # Same numbers either way; record the worst gap to show it.
        row["max_abs_diff"] = max(
            abs(a - b)
            for va, vb in zip(outputs["with_cache"], outputs["without_cache"])
            for a, b in zip(va, vb)
        )
        rows.append(row)
        print(
            f"n={n:5d}  proj {row['with_cache']['projections']:>7,} vs "
            f"{row['without_cache']['projections']:>9,}  "
            f"time {row['with_cache']['median_s']:.3f}s vs "
            f"{row['without_cache']['median_s']:.3f}s  "
            f"max|diff| {row['max_abs_diff']:.1e}"
        )
    result = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "machine": platform.platform(),
        "uptime_before": uptime,
        "uptime_after": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        "dim": DIM,
        "repeats": REPEATS,
        "rows": rows,
    }
    out = Path("results") / f"pure-{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
