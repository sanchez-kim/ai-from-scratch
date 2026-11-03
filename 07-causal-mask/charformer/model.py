"""A small character-level transformer with a switchable attention mask.

mask="causal": the future is filled with -inf before softmax (correct).
mask="none":   no mask; every position reads every other position.
mask="after":  softmax first, then the future is set to 0 (a common bug).

Attention is written out by hand instead of calling
F.scaled_dot_product_attention, so the mask step is visible.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKS = ("causal", "none", "after")


def causal_mask(t, device=None):
    """True above the diagonal: key j is in the future of query i when j > i."""
    return torch.ones(t, t, dtype=torch.bool, device=device).triu(1)


def attention_weights(scores, mask):
    """scores: (..., T, T) query-by-key. Returns weights of the same shape."""
    future = causal_mask(scores.shape[-1], scores.device)
    if mask == "causal":
        # exp(-inf) is exactly 0, so the future gets zero weight and the
        # remaining weights still add up to 1.
        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    weights = torch.softmax(scores, dim=-1)
    if mask == "after":
        # The bug: the future already took its share of the total in softmax.
        # Zeroing it afterwards leaves rows that add up to less than 1.
        return weights.masked_fill(future, 0.0)
    if mask == "none":
        return weights
    raise ValueError(f"unknown mask {mask!r}")


@dataclass
class Config:
    vocab_size: int
    context: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mask: str = "causal"


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.last_weights = None  # kept for the measurements, not for training

    def forward(self, x):
        b, t, d = x.shape
        h = self.cfg.n_heads
        # (b, t, 3d) -> three tensors of (b, h, t, d/h): heads own columns.
        q, k, v = self.qkv(x).view(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // h)
        weights = attention_weights(scores, self.cfg.mask)
        self.last_weights = weights.detach()
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx, targets=None):
        """idx: (b, t) character ids. Returns logits (b, t, vocab) and the loss."""
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits, None
        # Cross-entropy averages -log p(correct next character) over every
        # position. F.cross_entropy takes raw logits and does the softmax itself.
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def attention_maps(self):
        return [block.attn.last_weights for block in self.blocks]


@torch.no_grad()
def generate(model, idx, n_new, temperature=1.0, generator=None):
    """Append n_new characters one at a time. temperature=0 means greedy.

    At each step the model sees only the text so far. Nothing after the last
    character exists yet, which is exactly the situation a model trained
    without a mask never saw.
    """
    model.eval()
    for _ in range(n_new):
        logits, _ = model(idx[:, -model.cfg.context:])
        last = logits[:, -1, :]
        if temperature == 0:
            nxt = last.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(last / temperature, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(idx.device)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
