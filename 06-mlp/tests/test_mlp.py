import math
import random
import sys

import numpy as np
import pytest

from mlp import (backward, cross_entropy, forward, init_params, log_sum_exp,
                 make_examples, train_step, zeros_like)


def test_examples_from_one_name():
    examples, itos = make_examples(["emma"], block=3)
    words = [("".join(itos[i] for i in ctx), itos[t]) for ctx, t in examples]
    assert words == [("...", "e"), ("..e", "m"), (".em", "m"), ("emm", "a"), ("mma", ".")]


def test_random_guess_loss_is_ln_vocab():
    # Equal logits mean equal probabilities: the loss of a uniform guess.
    loss, probs = cross_entropy([0.0] * 27, target=5)
    assert loss == pytest.approx(math.log(27), abs=1e-12)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


def test_stable_equals_naive_on_small_logits():
    rng = random.Random(0)
    for _ in range(200):
        z = [rng.uniform(-10, 10) for _ in range(27)]
        t = rng.randrange(27)
        a, pa = cross_entropy(z, t, stable=True)
        b, pb = cross_entropy(z, t, stable=False)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
        assert max(abs(x - y) for x, y in zip(pa, pb)) < 1e-12


def test_naive_breaks_where_stable_stays_finite():
    limit = math.log(sys.float_info.max)  # about 709.78
    z = [limit + 1.0, 0.0, -5.0]
    with pytest.raises(OverflowError):        # plain Python is loud
        cross_entropy(z, 0, stable=False)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(np.array(z))               # numpy is quiet: inf, then nan
        assert np.isnan(-np.log(e[0] / e.sum()))
    loss, probs = cross_entropy(z, 0)
    assert math.isfinite(loss) and loss == pytest.approx(math.exp(-limit - 1.0), abs=1e-300)
    assert probs[0] == pytest.approx(1.0)


def test_naive_breaks_on_underflow_too():
    # exp(-800) is 0.0 in float64, so p[target] is 0 and log(0) fails.
    with pytest.raises(ValueError):
        cross_entropy([-800.0, 0.0], 0, stable=False)
    loss, _ = cross_entropy([-800.0, 0.0], 0)
    assert loss == pytest.approx(800.0)


def test_log_sum_exp_worked_example():
    # log(e^1000 + e^1000) = 1000 + log 2, though e^1000 itself overflows.
    assert log_sum_exp([1000.0, 1000.0]) == pytest.approx(1000.0 + math.log(2), abs=1e-12)
    assert log_sum_exp([1.0, 2.0, 3.0]) == pytest.approx(math.log(sum(math.exp(v) for v in [1, 2, 3])))


def test_gradients_match_finite_differences():
    rng = random.Random(1)
    p = init_params(vocab=5, emb=2, block=2, hidden=4, scale="fan_in", rng=rng)
    batch = [((1, 2), 3), ((0, 4), 0), ((2, 2), 1)]

    def loss(params):
        return sum(cross_entropy(forward(params, c)[0], t)[0] for c, t in batch) / len(batch)

    g = zeros_like(p)
    for c, t in batch:
        z, cache = forward(p, c)
        backward(p, cache, cross_entropy(z, t)[1], t, g, 1.0 / len(batch))
    eps, worst = 1e-6, 0.0
    for name, v in p.items():
        cells = [(r, j) for r in range(len(v)) for j in range(len(v[r]))] if isinstance(v[0], list) \
            else [(None, j) for j in range(len(v))]
        for r, j in cells:
            row, grow = (v[r], g[name][r]) if r is not None else (v, g[name])
            old = row[j]
            row[j] = old + eps
            up = loss(p)
            row[j] = old - eps
            down = loss(p)
            row[j] = old
            worst = max(worst, abs((up - down) / (2 * eps) - grow[j]))
    assert worst < 1e-8


def test_training_lowers_the_loss():
    examples, itos = make_examples(["emma", "olivia", "ava", "isabella", "sophia"])
    p = init_params(len(itos), emb=4, block=3, hidden=16, scale="fan_in", rng=random.Random(0))
    first, _ = train_step(p, examples, lr=0.5)
    for _ in range(50):
        last, _ = train_step(p, examples, lr=0.5)
    assert last < first - 1.0
