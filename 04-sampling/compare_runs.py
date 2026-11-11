"""Check that two collect_probs.py runs saw the same probabilities.

    uv run python compare_runs.py results/probs-<stamp>.json results/repeat-probs-<...>.json

Writes results/repeat-check.json: for every position in the second file, the
largest log-probability difference among the top 100 and whether the token
order and every llama.cpp survivor set are identical.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())
    first = {(p["passage"], p["prefix_len"]): p for p in a["positions"]}
    max_diff, same_order, same_surv = 0.0, 0, 0
    for p in b["positions"]:
        q = first[(p["passage"], p["prefix_len"])]
        same_order += [t for t, _ in p["top"]] == [t for t, _ in q["top"]]
        same_surv += p["survivors"] == q["survivors"]
        lq = dict(q["top"])
        max_diff = max([max_diff] + [abs(lp - lq[t]) for t, lp in p["top"] if t in lq])
    out = {"first": Path(sys.argv[1]).name, "second": Path(sys.argv[2]).name, "positions": len(b["positions"]),
           "same_top100_order": same_order, "same_survivor_sets_all_chains": same_surv,
           "max_abs_logprob_diff": max_diff,
           "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}
    Path("results/repeat-check.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
