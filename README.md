# AI from Scratch, Measured

Small, tested, plain-Python builds of the pieces inside LLM systems, each timed
on one machine and checked against a real tool. Companion code for the series
on [automatone.win](https://automatone.win).

| Part | Folder | Topic |
|---|---|---|
| 1 | [`01-kv-cache/`](01-kv-cache/) | KV cache reuse and prefix invalidation, plus a prompt-cache sweep on Ollama |
| 2 | [`02-bpe-tokenizer/`](02-bpe-tokenizer/) | BPE tokenizer and what Korean costs |
| 3 | [`03-attention/`](03-attention/) | Attention from scratch: why 1/√d_k |
| 4 | [`04-sampling/`](04-sampling/) | Temperature, top-k, top-p, and the order they run in |
| 5 | [`05-backprop/`](05-backprop/) | Backprop from scratch: the two bugs that never throw |
| 6 | [`06-mlp/`](06-mlp/) | An MLP in plain Python: init scale, log-sum-exp, NaN |
| 7 | [`07-causal-mask/`](07-causal-mask/) | Train without a causal mask |
| 8 | [`08-bm25/`](08-bm25/) | BM25 from scratch, checked against bm25s and rank_bm25 |
| 9 | [`09-hybrid-search/`](09-hybrid-search/) | Hybrid search: BM25 + embeddings, fused with RRF |
| 10 | [`10-ann-recall/`](10-ann-recall/) | ANN recall: brute force, LSH and HNSW on 100K vectors |

Every folder is self-contained, except `09-hybrid-search/`, which reads the corpus and queries of `08-bm25/` (see its README):

```bash
cd 01-kv-cache
uv venv && uv sync
uv run pytest
```

Raw measurement output lives in each folder's `results/` and is committed on
purpose: every number in the posts comes from those files.

MIT licensed. See [LICENSE](LICENSE).
