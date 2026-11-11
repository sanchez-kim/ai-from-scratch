"""The 4-token example in the post: scores, the mask, and the after-softmax bug.

    uv run python worked_example.py

Writes results/worked-example.json. The scores are made-up small integers so
the arithmetic is easy to follow; the functions are the ones the model uses.
"""

import json
from pathlib import Path

import torch

from charformer.model import attention_weights, causal_mask

TOKENS = ["c", "a", "t", "s"]
SCORES = [[2.0, 1.0, 0.0, 1.0],
          [1.0, 2.0, 1.0, 0.0],
          [0.0, 1.0, 2.0, 1.0],
          [1.0, 0.0, 1.0, 2.0]]


def main():
    s = torch.tensor(SCORES, dtype=torch.float64)
    masked = s.masked_fill(causal_mask(4), float("-inf"))
    r = lambda t: [[round(v, 4) if v != float("-inf") else "-inf" for v in row] for row in t.tolist()]
    out = {
        "tokens": TOKENS,
        "scores": SCORES,
        "mask_true_means_future": causal_mask(4).tolist(),
        "masked_scores": r(masked),
        "causal_weights": r(attention_weights(s, "causal")),
        "causal_row_sums": [round(v, 6) for v in attention_weights(s, "causal").sum(-1).tolist()],
        "no_mask_weights": r(attention_weights(s, "none")),
        "after_softmax_weights": r(attention_weights(s, "after")),
        "after_softmax_row_sums": [round(v, 4) for v in attention_weights(s, "after").sum(-1).tolist()],
    }
    path = Path(__file__).parent / "results" / "worked-example.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
