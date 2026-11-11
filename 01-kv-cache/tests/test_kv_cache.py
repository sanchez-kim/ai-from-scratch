import random

from kvcache.cache import (
    KVCache,
    TinyAttention,
    decode_with_cache,
    decode_without_cache,
    extend_cached,
    reusable_prefix_length,
    softmax,
    weighted_sum,
)

DIM = 4
VOCAB = ["the", "cat", "dog", "sat", "on", "mat"]


class CountingAttention(TinyAttention):
    """TinyAttention that also counts how many times it projected a token."""

    def __init__(self, *args):
        super().__init__(*args)
        self.projections = 0

    def project(self, token):
        self.projections += 1
        return super().project(token)


def build_model(seed=7):
    # A private Random instance, so every call builds the same weights and the
    # two decode paths can be compared value by value.
    rng = random.Random(seed)

    def row():
        return [round(rng.uniform(-1, 1), 4) for _ in range(DIM)]

    embeddings = {token: row() for token in VOCAB}
    wq, wk, wv = ([row() for _ in range(DIM)] for _ in range(3))
    return CountingAttention(embeddings, wq, wk, wv)


def close(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def test_cache_gives_identical_outputs_with_fewer_projections():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    slow = build_model()
    full = decode_without_cache(slow, tokens)

    fast = build_model()
    cache = KVCache()
    cached = decode_with_cache(fast, tokens, cache)

    # Every position must match, not just the last one. A cache that only
    # gets the final token right is a coincidence, not a cache.
    assert len(cached) == len(full) == len(cache) == len(tokens)
    for i, (a, b) in enumerate(zip(full, cached)):
        assert close(a, b), f"position {i} differs"

    # Positions see different prefixes, so their outputs differ. If they were
    # all equal, attend() would be looking at a single key.
    assert not close(full[0], full[1])
    assert not close(full[1], full[2])

    # Feeding one token per call, as real decoding does, must agree too.
    stepper = build_model()
    step_cache = KVCache()
    stepwise = []
    for token in tokens:
        stepwise += decode_with_cache(stepper, [token], step_cache)
    for a, b in zip(full, stepwise):
        assert close(a, b)

    # Same values, less work: 6 projections against 1 + 2 + ... + 6 = 21.
    assert fast.projections == stepper.projections == 6
    assert slow.projections == 21


def test_changing_one_token_invalidates_everything_after_it():
    old = ["the", "cat", "sat", "on", "the", "mat"]
    new = ["the", "dog", "sat", "on", "the", "mat"]

    assert reusable_prefix_length(old, new) == 1
    assert reusable_prefix_length(old, old) == 6
    assert reusable_prefix_length(old, ["the", "cat"]) == 2
    assert reusable_prefix_length(old, ["a", "cat"]) == 0

    model = build_model()
    cache = KVCache()
    first = decode_with_cache(model, old, cache)

    model.projections = 0
    tail, kept = extend_cached(model, cache, old, new)
    assert kept == 1
    # Positions 1..5 all had to be recomputed, even though 4 of the 5 tokens
    # are the same words as before.
    assert model.projections == 5
    assert len(cache.keys) == len(cache.values) == len(new)

    combined = first[:kept] + tail
    expected = decode_without_cache(build_model(), new)
    for i, (a, b) in enumerate(zip(expected, combined)):
        assert close(a, b), f"position {i} differs"

    # Before the change: untouched. From the change to the end: every output
    # moved, because each later token attended to the changed key and value.
    assert close(first[0], combined[0])
    for i in range(kept, len(new)):
        assert not close(first[i], combined[i]), f"position {i} did not change"

    # Changing only the last token keeps everything before it.
    last = old[:-1] + ["dog"]
    cache2 = KVCache()
    decode_with_cache(model, old, cache2)
    model.projections = 0
    _, kept2 = extend_cached(model, cache2, old, last)
    assert kept2 == 5 and model.projections == 1


class KeysOnlyCache(KVCache):
    """The bug: truncate() forgets the values."""

    def truncate(self, length):
        del self.keys[length:]


def test_broken_invalidation_is_silent_and_wrong():
    old = ["the", "cat", "sat", "on", "the", "mat"]
    new = ["the", "dog", "sat", "on", "the", "mat"]
    expected = decode_without_cache(build_model(), new)

    # Bug 1: cut keys but not values. No exception: zip() in weighted_sum
    # quietly pairs the new keys with stale values from the old prompt.
    bad = KeysOnlyCache()
    decode_with_cache(build_model(), old, bad)
    tail, kept = extend_cached(build_model(), bad, old, new)
    assert len(bad.keys) == 6 and len(bad.values) == 11
    assert not close(tail[-1], expected[-1])

    # Bug 2: skip truncation and append. The last token now attends over 11
    # entries, 6 of them from a prompt that no longer exists.
    stale = KVCache()
    decode_with_cache(build_model(), old, stale)
    wrong = decode_with_cache(build_model(), new[kept:], stale)
    assert len(stale) == 11
    assert not close(wrong[-1], expected[-1])

    # The correct cache passes the same check, so the two failures above come
    # from the bugs and not from the comparison.
    good = KVCache()
    decode_with_cache(build_model(), old, good)
    right, _ = extend_cached(build_model(), good, old, new)
    assert close(right[-1], expected[-1])


def test_softmax_is_stable_for_huge_scores():
    # exp(1000) overflows a float. Equal scores must still give 0.5 each.
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    # Shift invariance: softmax([1001, 1000]) = softmax([1, 0]) = [e/(e+1), 1/(e+1)]
    # = [0.7310586, 0.2689414] (hand value, 7 digits).
    got = softmax([1001.0, 1000.0])
    assert abs(got[0] - 0.7310586) < 1e-6
    assert abs(got[1] - 0.2689414) < 1e-6


def test_weighted_sum_hand_value():
    # 0.25 * [4, 8] + 0.75 * [0, 4] = [1, 2] + [0, 3] = [1, 5]
    assert weighted_sum([0.25, 0.75], [[4.0, 8.0], [0.0, 4.0]]) == [1.0, 5.0]


def test_attend_scales_scores_by_sqrt_of_dimension():
    # query has d = 2, keys [2, 0] and [0, 0]: raw scores 2 and 0.
    # Scaled by 1/sqrt(2): scores sqrt(2) and 0.
    # Weight of the first value = e^1.41421 / (e^1.41421 + 1) = 4.1133 / 5.1133 = 0.80443.
    # Without the scale it would be e^2 / (e^2 + 1) = 0.88080.
    model = TinyAttention({}, None, None, None)
    out = model.attend([1.0, 1.0], [[2.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert abs(out[0] - 0.80443) < 1e-4
    assert abs(out[1] - 0.19557) < 1e-4
