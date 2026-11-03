"""Character vocabulary and batches. All randomness comes from a torch.Generator."""

from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent.parent / "data" / "pride-and-prejudice.txt"


class Chars:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.index[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load(path=DATA, val_fraction=0.1):
    """The last 10% of the book is validation: text the model never trains on."""
    text = path.read_text(encoding="utf-8")
    chars = Chars(text)
    ids = torch.tensor(chars.encode(text), dtype=torch.long)
    cut = int(len(ids) * (1 - val_fraction))
    return chars, ids[:cut], ids[cut:]


def batch(ids, batch_size, context, generator):
    """Random windows. y is x shifted left by one: y[t] is the character after x[t]."""
    starts = torch.randint(len(ids) - context - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[s:s + context] for s in starts])
    y = torch.stack([ids[s + 1:s + context + 1] for s in starts])
    return x, y
