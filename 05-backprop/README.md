# 05 · Backprop from scratch: the two bugs that never throw

A scalar autograd `Value` class in 66 non-blank lines of plain Python
(`backprop/value.py`), written in the style of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) (MIT). The code here is my
own. Next to it are two silent backprop bugs, tests that catch them, and two
deterministic measurements.

| File | What it does |
|---|---|
| `backprop/value.py` | `Value` with `+`, `*`, `**`, `tanh`, `exp` and `backward()` (reverse topological order, gradients accumulate with `+=`) |
| `backprop/bugs.py` | bug 1 `AssignValue` (`grad =` instead of `grad +=`), bug 2a `backward_forgot_reversed`, bug 2b `backward_recursive` (no topological sort) |
| `backprop/check.py` | forward and central differences, `numeric_grads`, `grad_check` |
| `backprop/graphs.py` | random small expression graphs, with or without shared nodes |
| `tests/test_value.py` | 8 tests: x*x slope, the diamond `a*3 + a*5`, finite-difference checks, both bugs, backward twice, numpy reference |
| `measure_fd.py` | finite-difference error for eps = 1e-1 to 1e-14, forward vs central, four functions |
| `measure_bugs.py` | share of 1,000 random graphs per size where each bug changes a leaf gradient, plus the worked examples |
| `bench_numpy.py` | pure-Python `Value` vs numpy for the gradient of sum(tanh(W x)); timing harness |
| `figures/make_diamond.py` | the animated diamond graph (`diamond-backward-v2.svg`) |
| `figures/make_fd_chart.py` | error against eps for tanh (`fd-error-by-eps-v2.svg`) |

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python measure_fd.py               # writes results/fd-<date>.json
uv run python measure_bugs.py             # writes results/bugs-<date>.json
uv run python bench_numpy.py --nodes-only # graph sizes and the numpy check, no timing
uv run python bench_numpy.py --sizes 16 32 64 128 --repeats 7   # timing: idle machine only
uv run python figures/make_diamond.py results/bugs-2026-11-02.json
uv run python figures/make_fd_chart.py results/fd-2026-11-02.json
```

The package has no runtime dependencies. numpy is a dev dependency, used
only as the reference in `bench_numpy.py` and its test.

`measure_fd.py` and `measure_bugs.py` use no wall clock and no unseeded
randomness, so they give the same numbers on every run. The timing run of
`bench_numpy.py` (sizes 16 to 128, 7 repeats) is in
`results/bench-2026-11-03.json`, with the load average before and after.
Wall-clock times change with machine load. `results/smoke-*.json` is a smoke
test at sizes 4 and 8, not a measurement. `results/hypothesis.md` holds the
expectations written before the first measurement run.

No data is downloaded.
