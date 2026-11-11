# 09 · Hybrid search: BM25 + embeddings, fused with RRF

Two retrievers over the same 10,000 Simple English Wikipedia paragraphs as
Part 8 (`08-bm25`), and two ways to merge their ranked lists:

- `hybrid/fusion.py` (~60 lines, standard library only): `cosine`, `dot`,
  `normalise`, `rrf(rankings, k=60)` and `sum_scores` (min-max each list,
  then add).
- `hybrid/metrics.py`: recall@10, reciprocal rank (MRR@10 when averaged),
  Kendall's tau-b. Same recall and MRR definitions as
  `08-bm25/measure_retrieval.py`.
- `hybrid/bm25.py`: a copy of `08-bm25/bm25/bm25.py`, identical except for a one-line header comment.
- `hybrid/embed.py`: the only file that turns text into vectors
  (sentence-transformers, `BAAI/bge-small-en-v1.5` pinned to commit
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, CPU, 2 torch threads).

| Script | What it does |
|---|---|
| `embed_corpus.py` | embeds the 10,000 paragraphs and 1,000 queries once, caches `.npy` files in `data/` (model revision in the file name) |
| `measure_fusion.py` | recall@10 and MRR@10 for BM25, embeddings, score sum, RRF k=60, RRF k=1; a k sweep; a depth sweep; Kendall tau; the worked example |
| `bench_embed.py` | embedding throughput (paragraphs/s, queries/s). Wall clock: run on an idle machine |
| `figures/make_rrf_merge.py` | `figures/rrf-merge-v2.svg`, the animated merge of one real query |
| `figures/make_method_chart.py` | `figures/recall-by-method-v2.svg`, recall@10 by method and query set |

## Run it

```bash
# 1) The corpus and queries come from Part 8. Build the corpus once there:
cd ../08-bm25 && uv sync && uv run python build_corpus.py && cd ../09-hybrid-search
# 2) This folder
uv sync
uv run pytest
nice -n 10 uv run python embed_corpus.py       # downloads the model into the Hugging Face cache
nice -n 10 uv run python measure_fusion.py
uv run python figures/make_rrf_merge.py
uv run python figures/make_method_chart.py
# Timing, on an idle machine only:
nice -n 10 uv run python bench_embed.py --docs 2000 --queries 500 --repeats 3
```

`data/` is not committed. `results/embeddings.json` records the sha256 of
each cached `.npy`; `results/fusion.json` records the sha256 of the corpus
and query files it read.

## Choices fixed before measuring

- BM25 k1 = 1.5, b = 0.75, as in Part 8. Indexed text = title + paragraph,
  for both retrievers.
- Embeddings are normalised, so cosine = dot product.
- Queries get the prefix `Represent this sentence for searching relevant
  passages: `, from the model list in the
  [bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
  ("query instruction for retrieval"). Passages get no prefix. The card says
  v1.5 also works without it; `measure_fusion.py` reports recall without the
  prefix as a side check.
- Fusion depth: the top 100 of each list. A document missing from a list
  gets nothing from it (RRF) or 0 (score sum). Ties go to the lower document
  number.
- Kendall tau: tau-b between BM25 scores and cosine scores over the union
  of the two top-10 lists. After the first run showed this is negative for
  unrelated scores too (see `null_check_union_tau_random_scores`), two more
  were added: tau over BM25's own top 10 and over the embedding's own top 10.

## Swapping the embedding backend

Only `hybrid/embed.py` knows about the model. An Ollama embedding model
(for example `nomic-embed-text`, through `POST /api/embed`) could replace it.
The function must return one unit-length vector per input, in order. Use a
new file name tag in `common.py` so cached vectors from different models
never mix.

## Results

- `results/hypothesis.md`: expectations, written before any measurement.
- `results/embeddings.json`: versions, shapes, hashes, a batch-size re-run check.
- `results/fusion.json`: every number in the post.
- `results/per-query.jsonl`: gold rank per method, tau and overlap for each of the 1,000 queries.
- `results/worked-example.json`: the query shown in the post, chosen by the rule written in `measure_fusion.py`.
- `results/embed-throughput.json`: paragraphs and queries per second (2,000 and 500, median of 3), with `uptime` before and after.
- `results/embed-throughput-smoke.json`: a smoke test of the timing harness only. Not a measurement.
- `results/pytest.log`: the test run.

## Data licence

The paragraphs are from Simple English Wikipedia, built by
`08-bm25/build_corpus.py`; see `08-bm25/results/corpus-manifest.json`.
Text licence: the [pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)
lists CC BY-SA 3.0 and GFDL; [Simple English Wikipedia](https://simple.wikipedia.org)
now publishes its text under CC BY-SA 4.0. I treat the paragraphs and the
questions made from them as CC BY-SA.

The queries were written by `qwen3.5:4b` (see `08-bm25/results/queries-meta.json`).
`08-bm25/README.md` lists the paraphrase queries judged bad in a hand check
(10 of 20 read are vague or copy a name; 2 questions occur twice). All 500
paraphrase queries are kept here, the same set as Part 8, so the numbers stay
comparable.

The model `BAAI/bge-small-en-v1.5` is MIT-licensed.

## License

Code: MIT, see [../LICENSE](../LICENSE).
