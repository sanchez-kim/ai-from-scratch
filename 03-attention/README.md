# 03 · Attention from scratch: why 1/√d_k

Scaled dot-product attention and multi-head attention in 65 lines of plain
Python (`attention/core.py`), nine tests, and one deterministic sweep that
measures what the 1/√d_k scaling does to the attention weights.

| File | What it does |
|---|---|
| `attention/core.py` | `softmax`, `attention_weights`, `attention`, `split_heads`, `merge_heads`, `multi_head_attention` |
| `attention/stats.py` | row entropy in bits, the softmax Jacobian and its norm, variance |
| `tests/test_attention.py` | numpy reference within 1e-9, rows sum to 1, scaling off vs on, a wrong head split that runs but gives wrong output, Jacobian vs finite differences, entropy endpoints, softmax with scores past `exp` overflow, `d_model` not divisible by the head count raises, population variance |
| `measure_scaling.py` | for d_k = 8 to 512 and 10 seeds: variance of q·k, row entropy, largest weight, softmax Jacobian norm, share of rows with one weight above 0.99, with no scaling, 1/√d_k and 1/d_k |
| `figures/make_attention_row.py` | the animated 12-bar attention row (`attention-row-v2.svg`) |
| `figures/make_entropy_chart.py` | entropy against d_k for the three scales (`entropy-by-dk-v2.svg`) |

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python measure_scaling.py          # writes results/scaling-<date>.json
uv run python figures/make_attention_row.py results/scaling-2026-11-02.json
uv run python figures/make_entropy_chart.py results/scaling-2026-11-02.json
```

The package has no runtime dependencies. numpy is a dev dependency, used only
as the reference implementation in the tests.

Queries and keys are random vectors with entries from N(0, 1), drawn with
`random.Random(seed)`, so every number is the same on every run. Nothing is
timed. `results/hypothesis.md` holds the expectations written before the
first run, and an addendum written after it.

No data is downloaded.
