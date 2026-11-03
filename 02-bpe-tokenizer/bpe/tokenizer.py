"""Byte-level BPE in plain Python: train merges, encode, decode.

Tokens start as the 256 possible byte values, so any UTF-8 text can be encoded
with no "unknown" token. Training adds one new token per merge: the pair of
neighbouring tokens that appears most often in the training text.
"""

import re
from collections import Counter

# Cut text into chunks first, so a merge never crosses a space or joins a word
# with punctuation. A simpler cousin of GPT-2's pattern: Python's \w already
# matches Hangul, Latin and digits. Each chunk keeps one leading space.
CHUNK = re.compile(r" ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+")


def chunks(text):
    return CHUNK.findall(text)


def merge(ids, pair, new_id):
    """Replace every non-overlapping `pair` in `ids`, left to right, with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def add_pairs(ids, weight, counts):
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight


def train(text, num_merges):
    """Learn up to `num_merges` merges. Returns {(id_a, id_b): new_id}, in learning order."""
    words = Counter(chunks(text))  # each distinct chunk once, with how often it occurs
    seqs = {w: list(w.encode("utf-8")) for w in words}
    counts = Counter()
    for w, n in words.items():
        add_pairs(seqs[w], n, counts)
    merges = {}
    for new_id in range(256, 256 + num_merges):
        if not counts:
            break
        # Most frequent pair. Ties go to the smallest pair, so training is deterministic.
        best = min(counts, key=lambda p: (-counts[p], p))
        merges[best] = new_id
        for w, n in words.items():
            old = seqs[w]
            if best[0] not in old:
                continue
            new = merge(old, best, new_id)
            if len(new) < len(old):
                # Only chunks that contained the pair change their pair counts.
                add_pairs(old, -n, counts)
                add_pairs(new, n, counts)
                seqs[w] = new
        counts = +counts  # drop pairs whose count fell to zero
    return merges


def encode(text, merges):
    ids = []
    for chunk in chunks(text):
        seq = list(chunk.encode("utf-8"))
        while len(seq) > 1:
            # Apply the earliest-learned merge first, the order training used.
            pair = min(zip(seq, seq[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            seq = merge(seq, pair, merges[pair])
        ids.extend(seq)
    return ids


def vocab(merges):
    """The bytes each token id stands for."""
    table = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in sorted(merges.items(), key=lambda item: item[1]):
        table[new_id] = table[a] + table[b]
    return table


def decode(ids, merges):
    table = vocab(merges)
    return b"".join(table[i] for i in ids).decode("utf-8")
