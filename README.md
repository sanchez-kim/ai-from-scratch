# AI from Scratch, Measured

Small, tested, plain-Python builds of the pieces inside LLM systems, each timed
on one machine and checked against a real tool. Companion code for the series
on [automatone.win](https://automatone.win).

| Part | Folder | Topic |
|---|---|---|
| 1 | [`01-kv-cache/`](01-kv-cache/) | KV cache reuse and prefix invalidation, plus a prompt-cache sweep on Ollama |

Every folder is self-contained:

```bash
cd 01-kv-cache
uv venv && uv sync
uv run pytest
```

Raw measurement output lives in each folder's `results/` and is committed on
purpose: every number in the posts comes from those files.

MIT licensed. See [LICENSE](LICENSE).
