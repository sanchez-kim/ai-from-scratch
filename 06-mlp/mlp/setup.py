"""The data split and model size every script in this folder shares."""

import math
import random
import subprocess
from pathlib import Path

from .core import cross_entropy, forward, make_examples

NAMES = Path(__file__).resolve().parent.parent / "data" / "names.txt"
SEED_SPLIT, N_TRAIN, N_DEV = 42, 4000, 1000
BLOCK, EMB, HIDDEN = 3, 8, 64


def load_split():
    words = NAMES.read_text().split()
    random.Random(SEED_SPLIT).shuffle(words)
    train_words, dev_words = words[:N_TRAIN], words[N_TRAIN:N_TRAIN + N_DEV]
    # Build the vocabulary from all names, so train and dev share indices.
    _, itos = make_examples(words, BLOCK)
    stoi = {c: i for i, c in enumerate(itos)}

    def encode(ws):
        out = []
        for w in ws:
            ctx = [0] * BLOCK
            for ch in w + ".":
                out.append((tuple(ctx), stoi[ch]))
                ctx = ctx[1:] + [stoi[ch]]
        return out
    return encode(train_words), encode(dev_words), itos


def mean_loss(p, examples):
    return sum(cross_entropy(forward(p, c)[0], t)[0] for c, t in examples) / len(examples)


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
