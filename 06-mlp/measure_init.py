"""Three init scales, same data, same batches: loss curves and tanh saturation.

    nice -n 10 uv run python measure_init.py      # writes results/init-<date>.json

std 1.0 is "plain randn", std 0.01 is "make it small", and "fan_in" is
1/sqrt(number of inputs) per weight matrix. Embeddings are N(0, 1) in all
three, and every run uses the same seeds, so only the scale differs.
"""

import datetime
import json
import math
import random
from pathlib import Path

import numpy as np

from mlp import cross_entropy, forward, init_params, train_step
from mlp.setup import BLOCK, EMB, HIDDEN, N_DEV, N_TRAIN, load_split, mean_loss, uptime

SCALES = [1.0, 0.01, "fan_in"]
STEPS, BATCH, LR, EVAL_EVERY, SEED = 4000, 32, 0.1, 100, 0
BINS = 20  # histogram of tanh outputs over [-1, 1]


def naive_failures(logits, targets):
    """Rows where the textbook loss gives inf or nan in numpy, per float type."""
    out = {}
    for name, dtype in (("float16", np.float16), ("float32", np.float32), ("float64", np.float64)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            e = np.exp(np.array(logits, dtype=dtype))
            per = -np.log((e / e.sum(axis=1, keepdims=True))[np.arange(len(targets)), targets])
        out[name] = {"inf_rows": int(np.isinf(per).sum()), "nan_rows": int(np.isnan(per).sum())}
    return out


def activations(p, examples):
    """Histogram of every hidden tanh value, share with |h| > 0.99, largest logit."""
    counts, sat, n, top, rows = [0] * BINS, 0, 0, -math.inf, []
    for ctx, _ in examples:
        z, (_, _, _, h) = forward(p, ctx)
        rows.append(z)
        top = max(top, max(z))
        for v in h:
            counts[min(int((v + 1) / 2 * BINS), BINS - 1)] += 1
            sat += abs(v) > 0.99
            n += 1
    return {"hist": counts, "saturated_share": sat / n, "values": n, "max_logit": top,
            "rows": len(rows), "naive_failures": naive_failures(rows, [t for _, t in examples])}


def sample_names(p, itos, n, seed):
    """Draw names letter by letter from the trained model's probabilities."""
    rng, names = random.Random(seed), []
    for _ in range(n):
        ctx, out = [0] * BLOCK, []
        while len(out) < 20:
            probs = cross_entropy(forward(p, ctx)[0], 0)[1]
            k = rng.choices(range(len(probs)), weights=probs)[0]
            if k == 0:
                break
            out.append(itos[k])
            ctx = ctx[1:] + [k]
        names.append("".join(out))
    return names


def run(scale, train, dev, itos):
    vocab = len(itos)
    p = init_params(vocab, EMB, BLOCK, HIDDEN, scale, random.Random(SEED))
    at_init = activations(p, dev) | {"dev_loss": mean_loss(p, dev)}
    rng = random.Random(SEED + 1)  # same batch order for every scale
    curve, train_loss, top = [(0, at_init["dev_loss"])], [], -math.inf
    for step in range(1, STEPS + 1):
        loss, m = train_step(p, rng.sample(train, BATCH), LR)
        train_loss.append(round(loss, 6))
        top = max(top, m)
        if step % EVAL_EVERY == 0:
            curve.append((step, mean_loss(p, dev)))
            print(f"  scale={scale} step {step} dev {curve[-1][1]:.4f}", flush=True)
    return {"scale": scale, "init": at_init, "final": activations(p, dev),
            "dev_curve": curve, "train_batch_loss": train_loss, "max_train_logit": top,
            "samples": sample_names(p, itos, 8, seed=123)}


def main():
    train, dev, itos = load_split()
    out = {"date": datetime.date.today().isoformat(), "uptime_before": uptime(),
           "setup": {"n_train_names": N_TRAIN, "n_dev_names": N_DEV, "train_examples": len(train),
                     "dev_examples": len(dev), "vocab": len(itos), "block": BLOCK, "emb": EMB,
                     "hidden": HIDDEN, "steps": STEPS, "batch": BATCH, "lr": LR, "seed": SEED,
                     "bins": BINS, "fan_in_std": {"W1": 1 / math.sqrt(BLOCK * EMB), "W2": 1 / math.sqrt(HIDDEN)}},
           "ln_vocab": math.log(len(itos)), "runs": []}
    for scale in SCALES:
        out["runs"].append(run(scale, train, dev, itos))
    out["uptime_after"] = uptime()
    path = Path(__file__).parent / "results" / f"init-{out['date']}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(Path(__file__).parent))


if __name__ == "__main__":
    main()
