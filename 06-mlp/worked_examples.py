"""The small numbers the post explains with, computed rather than typed.

    uv run python worked_examples.py      # writes results/worked-examples.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from mlp import cross_entropy, log_sum_exp
from mlp.setup import BLOCK, EMB, HIDDEN, uptime


def try_exp(x):
    try:
        return math.exp(x)
    except OverflowError as err:
        return f"OverflowError: {err}"


def main():
    vocab = 27
    out = {"uptime": uptime(), "ln_27": math.log(vocab),
           "loss_for_p": {str(p): -math.log(p) for p in (1.0, 0.5, 0.1, 1 / vocab, 0.01)},
           "float64_max": sys.float_info.max,
           "exp_limit": math.log(sys.float_info.max),
           "math_exp": {str(x): try_exp(x) for x in (1, 10, 100, 709, 710)}}
    with np.errstate(over="ignore", invalid="ignore"):
        big = np.exp(np.float64(710.0))
        out["numpy_exp_710"] = repr(float(big))
        out["numpy_inf_over_inf"] = repr(float(big / big))
    # Worked log-sum-exp example: three logits too large for exp().
    z, target = [1000.0, 999.0, 998.0], 1
    m = max(z)
    shifted = [v - m for v in z]
    exps = [math.exp(v) for v in shifted]
    try:
        cross_entropy(z, target, stable=False)
        naive = "no error"
    except (OverflowError, ValueError) as err:
        naive = f"{type(err).__name__}: {err}"
    loss, probs = cross_entropy(z, target)
    out["lse_example"] = {"logits": z, "target_index": target, "naive": naive, "max": m,
                          "shifted": shifted, "exp_shifted": exps, "sum": sum(exps),
                          "log_sum": math.log(sum(exps)), "lse": log_sum_exp(z),
                          "loss": loss, "probs": probs}
    small, _ = cross_entropy([2.0, 1.0, 0.0], 1)
    out["small_example"] = {"logits": [2.0, 1.0, 0.0], "target_index": 1,
                            "stable": small, "naive": cross_entropy([2.0, 1.0, 0.0], 1, stable=False)[0]}
    out["model"] = {"block": BLOCK, "emb": EMB, "hidden": HIDDEN, "vocab": vocab,
                    "params": vocab * EMB + BLOCK * EMB * HIDDEN + HIDDEN + HIDDEN * vocab + vocab}
    path = Path(__file__).parent / "results" / "worked-examples.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
