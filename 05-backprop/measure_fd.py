"""Finite-difference error against epsilon: forward vs central difference.

    uv run python measure_fd.py        # writes results/fd-<date>.json

Error = |finite difference - gradient from backward()|. Both sides use the
same Value forward pass, in float64. eps = 10^(-k/2) for k = 2..28, so
1e-1 down to 1e-14. Deterministic: no randomness, no timing.
"""

import datetime
import json
import math
from pathlib import Path

from backprop import Value, central_diff, forward_diff
from backprop.runinfo import header, uptime

FUNCS = {
    "x*x at x=3": (lambda v: v * v, 3.0, lambda x: 2 * x),
    "tanh at x=0.5": (lambda v: v.tanh(), 0.5, lambda x: 1 - math.tanh(x) ** 2),
    "exp at x=1": (lambda v: v.exp(), 1.0, math.exp),
    "neuron loss (tanh(1.5w + 0.3) - 0.5)^2 at w=0.8": (
        lambda v: ((v * 1.5 + 0.3).tanh() - 0.5) ** 2, 0.8,
        lambda w: 2 * (math.tanh(1.5 * w + 0.3) - 0.5) * (1 - math.tanh(1.5 * w + 0.3) ** 2) * 1.5),
}
KS = range(2, 29)


def main():
    result = header("measure_fd.py")
    result["setup"] = {"eps": "10^(-k/2), k = 2..28", "float": "float64",
                       "error": "|finite difference - backward() gradient|"}
    funcs = {}
    for name, (expr, x0, closed) in FUNCS.items():
        v = Value(x0)
        expr(v).backward()
        g = v.grad
        f = lambda t: expr(Value(t)).data
        rows = []
        for k in KS:
            eps = 10 ** (-k / 2)
            rows.append({"k": k, "eps": eps,
                         "forward_error": abs(forward_diff(f, x0, eps) - g),
                         "central_error": abs(central_diff(f, x0, eps) - g)})
        best = {m: min(rows, key=lambda r: r[f"{m}_error"]) for m in ("forward", "central")}
        funcs[name] = {"x": x0, "backward_grad": g,
                       "closed_form_grad": closed(x0),
                       "backward_vs_closed_form": abs(g - closed(x0)),
                       "best": {m: {"eps": r["eps"], "error": r[f"{m}_error"]} for m, r in best.items()},
                       "rows": rows}
        print(name, g, {m: (b["eps"], b[f"{m}_error"]) for m, b in best.items()})
    result["functions"] = funcs
    result["uptime_after"] = uptime()
    out = Path("results") / f"fd-{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
