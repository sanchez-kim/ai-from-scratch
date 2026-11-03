"""Wall-clock time of one training epoch of the plain-Python MLP.

    uv run python bench_epoch.py                     # full: 4,000 names, 3 repeats
    uv run python bench_epoch.py --names 50 --out -  # smoke test, prints only

One epoch is one pass over every training example in shuffled batches of 32,
with the 1/sqrt(fan_in) init and lr 0.1. Run it on an idle machine: the
numbers depend on everything else the CPU is doing. The JSON records
`uptime` before and after so a busy run is visible.
"""

import argparse
import datetime
import json
import platform
import random
import statistics
import time
from pathlib import Path

from mlp import init_params, train_step
from mlp.setup import BLOCK, EMB, HIDDEN, load_split, uptime

BATCH, LR = 32, 0.1


def one_epoch(train, vocab, seed):
    p = init_params(vocab, EMB, BLOCK, HIDDEN, "fan_in", random.Random(seed))
    order = list(train)
    random.Random(seed + 1).shuffle(order)
    t0 = time.perf_counter()
    for i in range(0, len(order), BATCH):
        train_step(p, order[i:i + BATCH], LR)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=None, help="use only the first N training names")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="'-' prints instead of writing results/")
    args = ap.parse_args()

    train, _, itos = load_split()
    if args.names is not None:
        # Examples are in name order, so cut at the end of the N-th name.
        ends = [i + 1 for i, (_, t) in enumerate(train) if t == 0]
        train = train[:ends[args.names - 1]]
    before = uptime()
    seconds = [one_epoch(train, len(itos), seed) for seed in range(args.repeats)]
    out = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
           "python": platform.python_version(), "machine": platform.machine(),
           "examples": len(train), "batch": BATCH, "repeats": args.repeats,
           "seconds": seconds, "median_seconds": statistics.median(seconds),
           "examples_per_second": len(train) / statistics.median(seconds),
           "uptime_before": before, "uptime_after": uptime()}
    if args.out == "-":
        print(json.dumps(out, indent=1))
        return
    path = Path(args.out) if args.out else Path(__file__).parent / "results" / f"epoch-{out['date'][:10]}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
