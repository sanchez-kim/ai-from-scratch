"""Check that two training runs with the same settings gave the same numbers.

    uv run python compare_runs.py results/train-causal-cpu-main.json results/train-causal-cpu-repeat.json ...

Arguments come in pairs. Writes results/repeat-check.json.
"""

import json
import sys
from pathlib import Path

FIELDS = ("curve", "final", "by_position", "attention_final", "grids", "samples")


def max_diff(a, b):
    ca, cb = a["curve"], b["curve"]
    return max(abs(x[k] - y[k]) for x, y in zip(ca, cb) for k in ("train", "val_full", "val_last"))


def main():
    paths = sys.argv[1:]
    pairs = []
    for pa, pb in zip(paths[::2], paths[1::2]):
        a, b = json.loads(Path(pa).read_text()), json.loads(Path(pb).read_text())
        same = {f: a[f] == b[f] for f in FIELDS}
        pairs.append({"a": Path(pa).name, "b": Path(pb).name, "identical": same,
                      "max_abs_curve_diff": max_diff(a, b),
                      "final_a": a["final"], "final_b": b["final"]})
        print(Path(pa).name, Path(pb).name, same, f"max curve diff {pairs[-1]['max_abs_curve_diff']:.3g}")
    out = Path(__file__).parent / "results" / "repeat-check.json"
    out.write_text(json.dumps(pairs, indent=1))
    print("wrote", out.name)


if __name__ == "__main__":
    main()
