# 01 · KV cache reuse and prefix invalidation

A KV cache in about 100 lines of plain Python (`kvcache/cache.py`), six
tests (three on what it reuses and where reuse must stop, three on the attention
math), and three
benchmarks:

| Script | What it measures |
|---|---|
| `bench_pure.py` | projections and time, with and without the cache, at 64 to 1,024 tokens |
| `bench_ollama.py` | time to first token on Ollama when one line of a ~3,050-token prompt changes at 0/25/50/75/100 % |
| `bench_llama_server.py` | the same sweep against a `llama-server` you start yourself, to separate llama.cpp from Ollama and to try other flags |

Raw results are in `results/`. Figures and the scripts that draw them are in
`figures/`.

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python bench_pure.py                    # ~1 minute
uv run python bench_ollama.py                  # needs Ollama and qwen3.5:4b-q4_K_M
uv run python bench_ollama.py --prime          # same, with a checkpoint before the line
```

`bench_ollama.py` sets only per-request options (`num_ctx` 8192,
temperature 0, seed 42, 4 output tokens, `think: false`, `keep_alive` 10m).
It never changes server settings. It waits while the 1-minute load average is
above 4.0 and stores the `uptime` line with every block. On macOS it also
copies the llama-server lines each request wrote to
`~/.ollama/logs/server.log`, which state how many tokens came from the cache.

For `bench_llama_server.py`, start a server on a spare port with the model
blob Ollama already downloaded (the path is in
the first `FROM` line of `ollama show --modelfile qwen3.5:4b-q4_K_M`):

These are the exact commands behind the three `results/llama-server-*.json`
files (Homebrew `llama-server` 0.4.1, build 10964). Stop the server between runs.

```bash
# 1) ollama-like: the flags Ollama 0.40.8 uses for its own runner
llama-server --model <blob> --port 18181 --host 127.0.0.1 --no-webui --offline \
  -c 8192 -np 1 --no-jinja --chat-template chatml --flash-attn auto \
  -b 1024 -ub 1024 --keep 4
uv run python bench_llama_server.py --port 18181 --label ollama-like

# 2) dense-256: 256-token micro-batch and checkpoint spacing
llama-server --model <blob> --port 18181 --host 127.0.0.1 --no-webui --offline \
  -c 8192 -np 1 --no-jinja --chat-template chatml --flash-attn auto \
  -b 256 -ub 256 --keep 4 --ctx-checkpoints 32 --checkpoint-min-step 256 \
  --log-verbosity 4
uv run python bench_llama_server.py --port 18181 --label dense-256

# 3) ollama-like-primed: same flags as 1), verbose log to see each checkpoint
llama-server --model <blob> --port 18181 --host 127.0.0.1 --no-webui --offline \
  -c 8192 -np 1 --no-jinja --chat-template chatml --flash-attn auto \
  -b 1024 -ub 1024 --keep 4 --log-verbosity 4
uv run python bench_llama_server.py --port 18181 --label ollama-like-primed --prime
```

## Test log

`.python-version` pins 3.12, the oldest version `pyproject.toml` allows.
The tests also passed on 3.14.6, which is what the benchmarks in `results/` ran on.

```
$ uv run pytest -v
platform darwin -- Python 3.12.13, pytest-9.2.0, pluggy-1.7.0 -- .venv/bin/python3
tests/test_kv_cache.py::test_cache_gives_identical_outputs_with_fewer_projections PASSED [ 16%]
tests/test_kv_cache.py::test_changing_one_token_invalidates_everything_after_it PASSED [ 33%]
tests/test_kv_cache.py::test_broken_invalidation_is_silent_and_wrong PASSED [ 50%]
tests/test_kv_cache.py::test_softmax_is_stable_for_huge_scores PASSED    [ 66%]
tests/test_kv_cache.py::test_weighted_sum_hand_value PASSED              [ 83%]
tests/test_kv_cache.py::test_attend_scales_scores_by_sqrt_of_dimension PASSED [100%]
============================== 6 passed in 0.01s ===============================
```

## Evidence files

- `results/hypothesis.md`: the expected outcomes, written before the first Ollama run.
- `results/ollama-runner-log-excerpt.txt`: the server settings and the runner's
  load lines (KV cache layers, recurrent state, checkpoint settings).
- `results/ollama-qwen3.5_4b-q4_K_M-2026-11-01-2052.json` (run 1) was written
  before `--prime` existed, so it has no `prime` key.

## License

MIT, see [../LICENSE](../LICENSE).
