# 04 · Temperature, top-k, top-p, and the order they run in

Temperature, top-k, top-p and min-p in 71 lines of plain Python, comments included
(`sampling/sampler.py`), four tests, and a measurement on real next-token
probabilities from `qwen3.5:4b-q4_K_M`:

| Script | What it does |
|---|---|
| `collect_probs.py` | reads the ten passages in `passages.py` token by token through a `llama-server` you start, and saves the top-100 next-token log-probabilities at every position, plus the tokens that survive llama.cpp's own sampler for 14 settings |
| `analyze.py` | runs `sampling/sampler.py` on the saved probabilities in two orders (temperature first, temperature last) and counts survivors; compares with llama.cpp's survivors |
| `compare_runs.py` | checks that a second collection run returned the same probabilities and survivor sets |
| `figures/make_order_bars.py` | the animated figure: one real position, two orders |
| `figures/make_survivors_chart.py` | the static chart: survivor counts by temperature and order |

Nothing here measures time. Every number is a count or a probability, so a
busy machine does not change the results.

## Run it

```bash
uv venv && uv sync
uv run pytest
```

The probabilities file is about 1.7 MB: 541 positions x 100 candidates, plus
the llama.cpp survivor sets for 14 chains.

The measurement needs a `llama-server` (Homebrew `llama.cpp` 0.4.1, build
10964 was used) and the `qwen3.5:4b-q4_K_M` GGUF. If you use Ollama, the
blob path is the first `FROM` line of
`ollama show --modelfile qwen3.5:4b-q4_K_M`.

```bash
llama-server --model <blob> --port 18182 --host 127.0.0.1 --no-webui --offline \
  -c 4096 -np 1 --flash-attn auto
uv run python collect_probs.py --port 18182            # 541 positions x 15 requests
uv run python analyze.py results/probs-<model>-<stamp>.json --example 0:35
uv run python collect_probs.py --port 18182 --passages 0,4,8 --out results/repeat-probs-passages-0-4-8.json
uv run python compare_runs.py results/probs-<model>-<stamp>.json results/repeat-probs-passages-0-4-8.json
uv run python figures/make_survivors_chart.py results/survivors-<model>-<stamp>.json
uv run python figures/make_order_bars.py results/probs-<model>-<stamp>.json \
  --passage 0 --prefix 35 --t 0.7 --p 0.9 --k 15
```

Stop the server when you are done.

`collect_probs.py` sends the prompt as token ids, `n_predict: 1`, and reads:

- `n_probs: 100` with `post_sampling_probs: false`: the top 100 of a softmax
  over the full vocabulary at temperature 1, before any sampler runs;
- `n_probs: 100` with `post_sampling_probs: true` and a `samplers` list: the
  tokens left after llama.cpp's chain, in the order given.

Every chain in `analyze.py` includes top-k 40 before top-p, so top-p only
renormalizes over the top 40, and the top 100 are enough for exact counts.

## Passages

`passages.py` holds ten short texts I wrote for this test (story, recipe,
Python function, support reply, definition, travel note, packing list,
dialogue, news item, poem). They are MIT like the rest of the repo.

## Results

- `results/hypothesis.md`: the expected outcomes, written before the first request.
- `results/probs-*.json`: raw probabilities and llama.cpp survivor sets per position.
- `results/survivors-*.json`: `analyze.py` output; every number in the post comes from here
  (`example` holds the five-token worked example at passage 0, prefix 35).
- `results/repeat-probs-passages-0-4-8.json` and `results/repeat-check.json`: a second run of
  three passages, identical to the first (largest log-probability difference 0.0).
- `results/model-meta.json`: vocabulary size and model metadata, saved from `GET /v1/models`
  on the same server after the run (newer `collect_probs.py` runs store it as `model_meta`).
- `results/pytest.log`: the test run.

## License

MIT, see [../LICENSE](../LICENSE).
