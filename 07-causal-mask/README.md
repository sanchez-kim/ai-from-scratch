# 07 · Train without a causal mask

A small character-level transformer in PyTorch (`charformer/model.py`), trained
three times on the same text with the same starting weights and the same
batches. Only the attention mask changes:

- `causal`: future positions get -inf before softmax (correct).
- `none`: no mask. Each position can read the character it has to predict.
- `after`: softmax first, then the future is set to 0 (a common bug).

| File | What it does |
|---|---|
| `charformer/model.py` | `causal_mask`, `attention_weights` (the three mask settings), the transformer, `generate` |
| `charformer/data.py` | character vocabulary, train/validation split, random windows |
| `tests/test_mask.py` | -inf above the diagonal, rows sum to 1 with zero future weight, the after-softmax bug gives rows that do not sum to 1, output at position t does not change when later tokens change (only with the mask), targets are inputs shifted by one, attention matches `F.scaled_dot_product_attention`, positions change the output, residual connections, loss equals cross-entropy by hand, greedy `generate` takes the last position's argmax, batches stay inside the data, the last 10% is validation |
| `prepare_data.py` | downloads the book and strips the Project Gutenberg header and footer |
| `worked_example.py` | the 4-token example from the post (`results/worked-example.json`) |
| `train.py` | one training run: loss curves, the loss when each position sees only its past, attention statistics, samples (`results/train-<mask>-<device>-<tag>.json`) |
| `sample_check.py` | 20 samples per model and the share of real words before and after the context window fills (`results/samples.json`) |
| `compare_runs.py` | checks that repeated runs give the same numbers (`results/repeat-check.json`) |
| `bench_step.py` | milliseconds per training step on CPU and MPS (run it on an idle machine; `results/step-time-idle.json`) |
| `figures/make_mask_grid.py` | the animated 6×6 attention grid and loss curves (`mask-grid-v2.svg`) |
| `figures/make_loss_chart.py` | loss by position when the input stops at that position, for the three masks (`loss-by-position-v2.svg`) |

## Run it

```bash
uv venv && uv sync
uv run python prepare_data.py     # downloads data/pg1342.txt, writes data/pride-and-prejudice.txt
uv run pytest
uv run python worked_example.py
for m in causal none after; do
  nice -n 10 uv run python train.py --mask $m --device cpu --steps 5000 --tag main
done
uv run python sample_check.py causal-cpu-main none-cpu-main after-cpu-main
uv run python figures/make_mask_grid.py results/train-causal-cpu-main.json results/train-none-cpu-main.json
uv run python figures/make_loss_chart.py results/train-*-cpu-main.json
nice -n 10 uv run python bench_step.py --steps 200 --repeats 5 --tag idle
```

Each training run uses 2 CPU threads (`torch.set_num_threads(2)`) and saves its
weights to `checkpoints/` (not committed), which `sample_check.py` reads. On CPU
a repeated run gives the same numbers bit for bit. On MPS it does not; see
`results/repeat-check.json`. numpy is
installed only to silence a warning torch prints when numpy is missing.

## Data

*Pride and Prejudice* by Jane Austen, Project Gutenberg eBook #1342
(<https://www.gutenberg.org/ebooks/1342>). The book is in the public domain in
the United States. `prepare_data.py` removes the Project Gutenberg header,
footer and licence text and the front matter, and keeps the novel from its
first sentence. The text is not committed: `data/` is in `.gitignore`.
The `pg1342.txt` I used (downloaded 2026-11-02, "Most recently updated:
September 1, 2026") has SHA-256
`3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`.

`results/hypothesis.md` holds the expectations I wrote before the first
training run, and an addendum written before `sample_check.py`.

The samples in `results/samples.json` come from the checkpoints of the repeat
runs (`causal-cpu-repeat`, `none-cpu-rerun`, `after-cpu-rerun`). The first runs
did not save weights. `repeat-check.json` shows the repeat runs matched the
first runs in every recorded number.
