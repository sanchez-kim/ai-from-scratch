"""Naive vs stable cross-entropy on the same training run, at four learning rates.

    nice -n 10 uv run python measure_overflow.py   # writes results/overflow-<date>.json

For each learning rate, three runs start from the same weights and see the
same batches:
  stable        the log-sum-exp loss in plain Python
  naive_python  the textbook formula in plain Python: it raises when it breaks
  naive_numpy   the textbook formula in numpy float64: it returns inf or nan
                and the run goes on, the way array code behaves
"""

import datetime
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from mlp import cross_entropy, forward, init_params, train_step
from mlp.setup import BLOCK, EMB, HIDDEN, load_split, uptime

LRS, STEPS, BATCH, SEED = [0.1, 1.0, 10.0, 100.0], 1000, 32, 0


def numpy_naive(batch_logits, targets):
    """The textbook formula in numpy float64: exp, divide, log, mean."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = np.array(batch_logits, dtype=np.float64)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        per = -np.log(p[np.arange(len(targets)), targets])
    return {"mean_loss": repr(float(per.mean())), "nan_rows": int(np.isnan(per).sum()),
            "inf_rows": int(np.isinf(per).sum()), "rows": len(targets)}


def naive_numpy(z, target):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(np.array(z, dtype=np.float64))
        probs = e / e.sum()
        return float(-np.log(probs[target])), probs.tolist()


def naive_python(z, target):
    return cross_entropy(z, target, stable=False)


MODES = {"stable": cross_entropy, "naive_python": naive_python, "naive_numpy": naive_numpy}


def nan_share(p):
    vals = [v for k in p for row in (p[k] if isinstance(p[k][0], list) else [p[k]]) for v in row]
    return sum(math.isnan(v) for v in vals) / len(vals)


def enc(v):
    return v if math.isfinite(v) else repr(v)  # strict JSON has no nan/inf


def boundary(ok, lo, hi):
    """Bisect between lo (ok) and hi (not ok); return the last ok x and the first not-ok x."""
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, hi


def exp_boundaries():
    """Where exp() in each float type stops being finite and stops being above 0.

    ln(smallest subnormal) is where exp() reaches the smallest positive float.
    exp() rounds to exactly 0 only somewhat below that, so both are recorded.
    """
    out = {}
    with np.errstate(over="ignore", under="ignore"):
        for name, dtype in (("float64", np.float64), ("float32", np.float32), ("float16", np.float16)):
            last_pos, first_zero = boundary(lambda x: np.exp(dtype(x)) > 0, -1.0, -2000.0)
            last_fin, first_inf = boundary(lambda x: np.isfinite(np.exp(dtype(x))), 0.0, 2000.0)
            out[name] = {"exp_finite_up_to": last_fin, "exp_inf_from": first_inf,
                         "exp_above_zero_down_to": last_pos, "exp_zero_from": first_zero,
                         "exp_at_last_positive": repr(float(np.exp(dtype(last_pos))))}
    return out


def run(lr, mode, train, vocab):
    p = init_params(vocab, EMB, BLOCK, HIDDEN, "fan_in", random.Random(SEED))
    rng = random.Random(SEED + 1)
    losses, tops, first = [], [], {}
    for step in range(1, STEPS + 1):
        batch = rng.sample(train, BATCH)
        logits = [forward(p, c)[0] for c, _ in batch]
        top = max(max(z) for z in logits)
        try:
            loss, _ = train_step(p, batch, lr, MODES[mode])
        except (OverflowError, ValueError) as err:
            targets = [t for _, t in batch]
            gaps = [cross_entropy(z, t)[0] for z, t in zip(logits, targets)]
            first["error"] = {"step": step, "error": f"{type(err).__name__}: {err}",
                              "max_logit": top, "min_logit": min(min(z) for z in logits),
                              "largest_stable_row_loss": max(gaps),
                              "numpy_float64": numpy_naive(logits, targets)}
            break
        losses.append(enc(loss))
        tops.append(enc(top))
        for key, hit in (("inf_loss", math.isinf(loss)), ("nan_loss", math.isnan(loss))):
            if hit and key not in first:
                first[key] = {"step": step, "max_logit": enc(top)}
        if mode == "naive_numpy" and "all_weights_nan" not in first:
            share = nan_share(p)
            if share > 0 and "any_weight_nan" not in first:
                first["any_weight_nan"] = {"step": step, "share": share}
            if share == 1.0:
                first["all_weights_nan"] = {"step": step}
    return {"lr": lr, "mode": mode, "loss": losses, "max_logit": tops, "first": first}


def main():
    train, _, itos = load_split()
    out = {"date": datetime.date.today().isoformat(), "uptime_before": uptime(),
           "setup": {"init": "fan_in", "steps": STEPS, "batch": BATCH, "seed": SEED, "lrs": LRS},
           "limits": {"ln_float64_max": math.log(sys.float_info.max),
                      "ln_float32_max": float(np.log(np.finfo(np.float32).max.astype(np.float64))),
                      "ln_float16_max": float(np.log(np.float64(np.finfo(np.float16).max))),
                      # exp(x) rounds to 0 below these: ln of the smallest subnormal
                      "ln_float64_tiny": math.log(sys.float_info.min * sys.float_info.epsilon),
                      "ln_float32_tiny": float(np.log(np.float64(np.finfo(np.float32).smallest_subnormal))),
                      "ln_float16_tiny": float(np.log(np.float64(np.finfo(np.float16).smallest_subnormal)))},
           "exp_boundaries": exp_boundaries(),
           "ln_vocab": math.log(len(itos)), "runs": []}
    for lr in LRS:
        for mode in MODES:
            r = run(lr, mode, train, len(itos))
            print(f"lr {lr} {mode}: steps {len(r['loss'])}, first {r['first']}", flush=True)
            out["runs"].append(r)
    for lr in LRS:
        s, n = [r for r in out["runs"] if r["lr"] == lr][:2]  # stable vs naive_python
        diffs = [abs(a - b) for a, b in zip(s["loss"], n["loss"])]
        out.setdefault("twins", {})[str(lr)] = {
            "compared_steps": len(diffs), "max_abs_diff": max(diffs),
            "first_step_diff_gt_1e-9": next((i + 1 for i, d in enumerate(diffs) if d > 1e-9), None)}
    out["uptime_after"] = uptime()
    path = Path(__file__).parent / "results" / f"overflow-{out['date']}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(Path(__file__).parent))


if __name__ == "__main__":
    main()
