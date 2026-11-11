"""Temperature, top-k, top-p and min-p as small functions you can chain.

A candidate list is [(token, logit), ...]. Every step takes one and returns
one, so the order of the steps is just the order of a Python list.
"""

import math
import random


def softmax(cands):
    # Subtract the largest logit first: same result, and exp() cannot overflow.
    top = max(logit for _, logit in cands)
    weights = [(tok, math.exp(logit - top)) for tok, logit in cands]
    total = sum(w for _, w in weights)
    return [(tok, w / total) for tok, w in weights]


def ranked(cands):
    # Highest logit first; ties broken by token id so the cut never wobbles.
    return sorted(cands, key=lambda c: (-c[1], c[0]))


def temperature(cands, t):
    if t <= 0:
        # The limit as t -> 0: all probability goes to the top token.
        return ranked(cands)[:1]
    return [(tok, logit / t) for tok, logit in cands]


def top_k(cands, k):
    return ranked(cands)[:k] if k > 0 else list(cands)


def top_p(cands, p):
    # Keep the smallest top set whose probability reaches p. The token that
    # crosses p stays in: stopping before it would keep less than p.
    if p >= 1.0:
        return list(cands)
    probs = dict(softmax(cands))
    kept, mass = [], 0.0
    for tok, logit in ranked(cands):
        kept.append((tok, logit))
        mass += probs[tok]
        if mass >= p:
            break
    return kept


def min_p(cands, p):
    # Keep tokens whose probability is at least p times the top probability.
    if p <= 0:
        return list(cands)
    cutoff = max(logit for _, logit in cands) + math.log(p)
    return [c for c in ranked(cands) if c[1] >= cutoff]


STEPS = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p}


def run_chain(cands, chain):
    """chain: [("temperature", 0.7), ("top_k", 40), ("top_p", 0.9)], applied in order."""
    for name, value in chain:
        cands = STEPS[name](cands, value)
    return cands


def sample(cands, chain, rng=random):
    survivors = softmax(run_chain(cands, chain))
    tokens, probs = zip(*survivors)
    return rng.choices(tokens, weights=probs)[0]
