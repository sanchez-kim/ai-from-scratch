"""A character-level MLP in plain Python: three letters in, the next letter out.

Parameters are plain lists:
  C   [vocab][emb]          one embedding row per character
  W1  [block*emb][hidden]   b1 [hidden]
  W2  [hidden][vocab]       b2 [vocab]
"""

import math

DOT = "."  # marks the start and the end of a name


def make_examples(words, block=3):
    """Every (previous `block` characters, next character) pair, as indices."""
    chars = sorted(set("".join(words)))
    itos = [DOT] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    examples = []
    for w in words:
        ctx = [0] * block
        for ch in w + DOT:
            examples.append((tuple(ctx), stoi[ch]))
            ctx = ctx[1:] + [stoi[ch]]
    return examples, itos


def init_params(vocab, emb, block, hidden, scale, rng):
    """scale is a number (the std of every weight) or "fan_in" (1/sqrt(inputs))."""
    def matrix(rows, cols):
        std = 1.0 / math.sqrt(rows) if scale == "fan_in" else scale
        return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]
    return {
        "C": [[rng.gauss(0.0, 1.0) for _ in range(emb)] for _ in range(vocab)],
        "W1": matrix(block * emb, hidden), "b1": [0.0] * hidden,
        "W2": matrix(hidden, vocab), "b2": [0.0] * vocab,
    }


def forward(p, ctx):
    """Logits for one context, plus what backward needs."""
    x = [v for i in ctx for v in p["C"][i]]
    pre = list(p["b1"])
    for xi, row in zip(x, p["W1"]):
        for j, w in enumerate(row):
            pre[j] += xi * w
    h = [math.tanh(v) for v in pre]
    z = list(p["b2"])
    for hj, row in zip(h, p["W2"]):
        for k, w in enumerate(row):
            z[k] += hj * w
    return z, (ctx, x, pre, h)


def log_sum_exp(z):
    """log(sum(exp(z))) without overflow: pull the largest value out first."""
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def cross_entropy(z, target, stable=True):
    """Loss -log p[target] and the probabilities p.

    stable=False is the textbook formula. math.exp raises OverflowError above
    log(sys.float_info.max), and math.log raises ValueError when p[target]
    underflows to 0. A float library such as numpy returns inf or nan instead.
    """
    if not stable:
        exps = [math.exp(v) for v in z]
        total = sum(exps)
        probs = [e / total for e in exps]
        return -math.log(probs[target]), probs
    lse = log_sum_exp(z)
    return lse - z[target], [math.exp(v - lse) for v in z]


def backward(p, cache, probs, target, g, weight=1.0):
    """Add d(loss)/d(param) * weight into g, a dict shaped like p."""
    ctx, x, pre, h = cache
    dz = [weight * (q - (k == target)) for k, q in enumerate(probs)]
    dh = [0.0] * len(h)
    for j, (hj, row, grow) in enumerate(zip(h, p["W2"], g["W2"])):
        for k, dzk in enumerate(dz):
            grow[k] += hj * dzk
            dh[j] += row[k] * dzk
    for k, dzk in enumerate(dz):
        g["b2"][k] += dzk
    dpre = [d * (1.0 - hj * hj) for d, hj in zip(dh, h)]  # tanh' = 1 - tanh^2
    emb = len(p["C"][0])
    for i, (xi, row, grow) in enumerate(zip(x, p["W1"], g["W1"])):
        dx = 0.0
        for j, dj in enumerate(dpre):
            grow[j] += xi * dj
            dx += row[j] * dj
        g["C"][ctx[i // emb]][i % emb] += dx
    for j, dj in enumerate(dpre):
        g["b1"][j] += dj


def zeros_like(p):
    return {k: [[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v)
            for k, v in p.items()}


def train_step(p, batch, lr, loss_fn=cross_entropy):
    """One SGD step on the mean loss of a batch. Returns (loss, largest logit).

    loss_fn(z, target) returns (loss, probs); the gradient uses those probs.
    """
    g = zeros_like(p)
    total, top = 0.0, -math.inf
    for ctx, target in batch:
        z, cache = forward(p, ctx)
        top = max(top, max(z))
        loss, probs = loss_fn(z, target)
        total += loss
        backward(p, cache, probs, target, g, 1.0 / len(batch))
    for k, v in p.items():
        if isinstance(v[0], list):
            for row, grow in zip(v, g[k]):
                for j, d in enumerate(grow):
                    row[j] -= lr * d
        else:
            for j, d in enumerate(g[k]):
                v[j] -= lr * d
    return total / len(batch), top
