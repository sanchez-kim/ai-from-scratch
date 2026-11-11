import math
import random

from sampling import min_p, ranked, run_chain, sample, softmax, temperature, top_k, top_p


def from_probs(probs):
    # log(p) works as a logit: softmax() gives the same p back.
    return [(tok, math.log(p)) for tok, p in enumerate(probs)]


def test_temperature_to_zero_becomes_argmax():
    cands = [(0, 1.0), (1, 2.0), (2, 0.5), (3, 1.9)]
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


def test_top_k_keeps_the_best_k_with_logits_unchanged():
    cands = from_probs([0.4, 0.3, 0.2, 0.1])
    # Exactly k survivors, the highest ones, in rank order, logits not rescaled.
    assert top_k(cands, 2) == [cands[0], cands[1]]
    assert top_k(cands, 1) == [cands[0]]
    assert top_k(cands, 4) == cands
    # k = 0 means "off": everything stays.
    assert top_k(cands, 0) == cands
    # Input order does not matter: the best two are still found.
    shuffled = [cands[2], cands[0], cands[3], cands[1]]
    assert top_k(shuffled, 2) == [cands[0], cands[1]]


def test_min_p_cutoff_is_a_fraction_of_the_top_probability():
    # Top probability 0.5, p = 0.2: cutoff 0.2 * 0.5 = 0.1.
    # Keep 0.5, 0.3, 0.15. Drop 0.05 (< 0.1).
    cands = from_probs([0.5, 0.3, 0.15, 0.05])
    assert [tok for tok, _ in min_p(cands, 0.2)] == [0, 1, 2]
    # A larger p raises the cutoff: p = 0.5 -> 0.25, keeps 0.5 and 0.3 only.
    assert [tok for tok, _ in min_p(cands, 0.5)] == [0, 1]
    # Boundary: top 0.5, p = 0.5 -> cutoff 0.25. A token at exactly 0.25 stays.
    tie = from_probs([0.5, 0.25, 0.25])
    assert [tok for tok, _ in min_p(tie, 0.5)] == [0, 1, 2]
    # The top token always survives, even at p = 1.
    assert [tok for tok, _ in min_p(cands, 1.0)] == [0]
    # p = 0 turns the filter off.
    assert min_p(cands, 0) == cands


def test_sample_follows_the_survivor_probabilities():
    rng = random.Random(0)
    cands = from_probs([0.3, 0.7])
    draws = 20000
    hits = sum(sample(cands, [], rng) == 1 for _ in range(draws))
    # Binomial(20000, 0.7): mean 14000, sd = sqrt(20000 * 0.7 * 0.3) = 64.8.
    # Tolerance 4 sd = 260. Always-first-token gives 0, uniform gives 10000.
    assert abs(hits - 14000) < 260
    # Top-k = 1 leaves one survivor, so every draw is the argmax.
    three = from_probs([0.2, 0.5, 0.3])
    assert {sample(three, [("top_k", 1)], rng) for _ in range(300)} == {1}
    # A token that was cut is never drawn.
    cut = {sample(three, [("top_k", 2)], rng) for _ in range(2000)}
    assert cut == {1, 2}


def test_equal_logits_rank_by_lower_token_id():
    tied = [(5, 1.0), (2, 1.0), (9, 1.0), (1, 0.0)]
    assert [tok for tok, _ in ranked(tied)] == [2, 5, 9, 1]
    assert [tok for tok, _ in top_k(tied, 2)] == [2, 5]


def test_softmax_does_not_overflow_on_large_logits():
    # exp(1000) overflows a float. Only the gaps between logits may matter.
    assert softmax([(0, 1000.0), (1, 1000.0)]) == [(0, 0.5), (1, 0.5)]
    # Gap of 1: e / (e + 1) = 0.7310586, 1 / (e + 1) = 0.2689414.
    (_, a), (_, b) = softmax([(0, 1001.0), (1, 1000.0)])
    assert abs(a - 0.7310586) < 1e-6 and abs(b - 0.2689414) < 1e-6


def test_softmax_handles_a_logit_gap_wider_than_exp_can_hold():
    # Gap 1000: exp(1000) overflows, exp(-1000) underflows to exactly 0.0.
    # Shifting by the largest logit gives exp(0) = 1 and exp(-1000) = 0.
    # Shifting by the smallest would need exp(1000) and raise OverflowError.
    assert softmax([(0, 1000.0), (1, 0.0)]) == [(0, 1.0), (1, 0.0)]
