"""A KV cache in plain Python: what it reuses, and where reuse must stop.

One attention head, no batching, no numpy. The point is to make the
bookkeeping visible, not to be fast.
"""

import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def softmax(scores):
    # Subtract the max so exp() never overflows; the result is unchanged.
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def weighted_sum(weights, vectors):
    width = len(vectors[0])
    return [sum(w * v[i] for w, v in zip(weights, vectors)) for i in range(width)]


class TinyAttention:
    """One causal attention head over a fixed embedding table."""

    def __init__(self, embeddings, wq, wk, wv):
        self.embeddings = embeddings
        self.wq, self.wk, self.wv = wq, wk, wv

    def project(self, token):
        # The expensive part we want to do once per token, not once per step.
        x = self.embeddings[token]
        return matvec(self.wq, x), matvec(self.wk, x), matvec(self.wv, x)

    def attend(self, query, keys, values):
        scale = math.sqrt(len(query))
        scores = [dot(query, k) / scale for k in keys]
        return weighted_sum(softmax(scores), values)


class KVCache:
    """Keys and values for every token seen so far, in order."""

    def __init__(self):
        self.keys = []
        self.values = []

    def __len__(self):
        return len(self.keys)

    def append(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def truncate(self, length):
        # Keys and values must always be cut together. If only the keys are
        # cut, key i gets paired with a stale value and nothing raises.
        del self.keys[length:]
        del self.values[length:]


def decode_without_cache(model, tokens):
    """Reference path: at every step, re-project the whole prefix."""
    outputs = []
    for step in range(len(tokens)):
        keys, values = [], []
        for token in tokens[: step + 1]:
            query, key, value = model.project(token)
            keys.append(key)
            values.append(value)
        # `query` is the last token's: each step asks "what does the newest
        # token see?", which is all a decoder needs.
        outputs.append(model.attend(query, keys, values))
    return outputs


def decode_with_cache(model, tokens, cache):
    """Project each new token once and attend over everything cached."""
    outputs = []
    for token in tokens:
        query, key, value = model.project(token)
        cache.append(key, value)
        # Safe only because attention is causal: a later token can never
        # change an earlier token's key or value.
        outputs.append(model.attend(query, cache.keys, cache.values))
    return outputs


def reusable_prefix_length(old_tokens, new_tokens):
    """How many leading tokens two prompts share. Reuse stops at the first
    difference, even if everything after it matches again."""
    shared = 0
    for a, b in zip(old_tokens, new_tokens):
        if a != b:
            break
        shared += 1
    return shared


def extend_cached(model, cache, old_tokens, new_tokens):
    """Serve `new_tokens` from a cache that holds `old_tokens`.

    Returns the outputs for the recomputed tail and how many tokens were
    reused. This is what a prompt cache does between two requests.
    """
    keep = reusable_prefix_length(old_tokens, new_tokens)
    cache.truncate(keep)
    return decode_with_cache(model, new_tokens[keep:], cache), keep
