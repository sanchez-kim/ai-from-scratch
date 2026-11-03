import math
import random

from sampling import run_chain, sample, softmax, temperature, top_p


def from_probs(probs):
    # log(p) works as a logit: softmax() gives the same p back.
    return [(tok, math.log(p)) for tok, p in enumerate(probs)]


def test_temperature_to_zero_becomes_argmax():
    cands = [(0, 1.0), (1, 2.0), (2, 0.5), (3, 1.9)]
    for t in (1.0, 0.1, 0.01):
        top = dict(softmax(temperature(cands, t)))[1]
        print(f"T={t}: p(argmax)={top:.6f}")
    # Gap between the top two logits is 0.1, so at T=0.01 the ratio is e^10.
    assert dict(softmax(temperature(cands, 0.01)))[1] > 0.9999
    # T=0 is the limit itself: only the argmax survives, every draw picks it.
    assert temperature(cands, 0) == [(1, 2.0)]
    rng = random.Random(0)
    assert {sample(cands, [("temperature", 0)], rng) for _ in range(200)} == {1}


def test_top_p_keeps_the_smallest_set_that_reaches_p():
    cands = from_probs([0.5, 0.3, 0.15, 0.05])
    assert [tok for tok, _ in top_p(cands, 0.79)] == [0, 1]
    assert [tok for tok, _ in top_p(cands, 0.81)] == [0, 1, 2]
    assert [tok for tok, _ in top_p(cands, 1.0)] == [0, 1, 2, 3]
    # The token that crosses p stays in: 0.5 alone is below 0.75, so the 0.3
    # token is kept. A loop that stopped before crossing would keep only [0].
    assert [tok for tok, _ in top_p(cands, 0.75)] == [0, 1]
    probs = dict(softmax(cands))
    for p in (0.1, 0.5, 0.6, 0.9, 0.97):
        kept = top_p(cands, p)
        mass = sum(probs[tok] for tok, _ in kept)
        assert mass >= p                                   # reaches p
        assert mass - probs[kept[-1][0]] < p               # one fewer would not
    # top-p renormalizes over what it gets: after top-k it only sees those tokens.
    assert len(top_p(cands[:2], 0.6)) == 1                 # 0.5/0.8 = 0.625 >= 0.6


def test_temperature_before_or_after_top_p_changes_the_survivors():
    cands = from_probs([0.6, 0.25, 0.1, 0.05])
    first = run_chain(cands, [("temperature", 0.5), ("top_p", 0.9)])
    last = run_chain(cands, [("top_p", 0.9), ("temperature", 0.5)])
    # T=0.5 squares the probabilities: 0.83, 0.14, ... -> 0.9 is reached in 2.
    # Without it: 0.6 + 0.25 = 0.85 < 0.9, so a third token is needed.
    assert [tok for tok, _ in first] == [0, 1]
    assert [tok for tok, _ in last] == [0, 1, 2]
    # At T=1 the order cannot matter.
    assert run_chain(cands, [("temperature", 1.0), ("top_p", 0.9)]) == \
        run_chain(cands, [("top_p", 0.9), ("temperature", 1.0)])


def test_same_survivors_give_the_same_final_distribution():
    # Order only matters through the survivor set: with equal sets, temperature
    # before or after the cut gives identical probabilities.
    cands = from_probs([0.4, 0.3, 0.2, 0.1])
    a = softmax(run_chain(cands, [("temperature", 0.7), ("top_k", 3)]))
    b = softmax(run_chain(cands, [("top_k", 3), ("temperature", 0.7)]))
    assert [t for t, _ in a] == [t for t, _ in b]
    assert all(math.isclose(pa, pb) for (_, pa), (_, pb) in zip(a, b))
