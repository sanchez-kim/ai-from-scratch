# 06 · An MLP in plain Python: init scale, log-sum-exp, NaN

A character-level MLP that reads three letters of a name and predicts the
next one, in about 120 lines of plain Python (`mlp/core.py`), with a
hand-written backward pass. Eight tests, and three deterministic measurements:

1. three init scales (std 1.0, std 0.01, 1/sqrt(fan_in)): loss curves and
   how many hidden tanh units start out stuck at +-1;
2. the textbook cross-entropy against the log-sum-exp version on the same
   training run, at learning rates 0.1 to 100: where the textbook one breaks;
3. which rows the textbook formula breaks in float16, float32 and float64.

| File | What it does |
|---|---|
| `mlp/core.py` | `make_examples`, `init_params`, `forward`, `log_sum_exp`, `cross_entropy` (stable or naive), `backward`, `train_step` |
| `mlp/setup.py` | the data split (4,000 train names, 1,000 dev names, seed 42) and model size every script shares |
| `tests/test_mlp.py` | examples from one name, ln 27 for a uniform guess, stable = naive on small logits, naive raises (Python) or gives nan (numpy) where stable stays finite, the underflow case, a log-sum-exp worked example, gradients vs finite differences, training lowers the loss |
| `download_data.py` | fetches `names.txt` into `data/` and checks its sha256 |
| `worked_examples.py` | the small numbers the post explains with (`results/worked-examples.json`) |
| `measure_init.py` | three init scales, 4,000 SGD steps each (`results/init-<date>.json`) |
| `measure_overflow.py` | stable vs naive (plain Python) vs naive (numpy float64) at lr 0.1, 1, 10, 100 (`results/overflow-<date>.json`) |
| `bench_epoch.py` | wall-clock time of one epoch; run it on an idle machine (`results/epoch-2026-11-03.json`) |
| `figures/make_init_and_nan.py` | the animated figure (`init-and-nan-v2.svg`) |
| `figures/make_loss_curves.py` | the loss-curve chart (`init-loss-curves-v2.svg`) |

## Run it

```bash
uv venv && uv sync
uv run python download_data.py          # 228 KB into data/ (gitignored)
uv run pytest
uv run python worked_examples.py
nice -n 10 uv run python measure_init.py       # a few minutes
nice -n 10 uv run python measure_overflow.py   # a few minutes
uv run python figures/make_init_and_nan.py results/init-2026-11-02.json results/overflow-2026-11-02.json
uv run python figures/make_loss_curves.py results/init-2026-11-02.json
uv run python bench_epoch.py            # timing; idle machine only
```

The model code has no dependencies. numpy is a dev dependency, used only as a
reference: the tests and `measure_overflow.py` give it the same logits to show
what a float library returns where plain Python raises an error.

Every random choice uses `random.Random(seed)`, so the measurements give the
same numbers on every run. `results/hypothesis.md` holds the expectations
written before the first run, and an addendum written after it.

## Data and licence

`names.txt` comes from [karpathy/makemore](https://github.com/karpathy/makemore),
pinned to commit `988aa59e4d8fefa526d06f3b453ad116258398d4`. The makemore
repository is MIT licensed (Copyright (c) 2022 Andrej Karpathy). Its README
says the file holds the most common 32K names from ssa.gov for the year 2018.
The file is not committed here; `download_data.py` fetches it and stops if
the sha256 differs from `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d`.
