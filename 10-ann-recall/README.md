# 10 · ANN recall: brute force, LSH and HNSW on 100K vectors

Exact nearest-neighbour search and random-hyperplane LSH in about 70 lines
(`ann/search.py`), twelve tests, and a recall sweep of
[hnswlib](https://github.com/nmslib/hnswlib) 0.8.0 on 100,000 real text
embeddings and on 100,000 random vectors of the same size.

| Script | What it does |
|---|---|
| `download_data.py` | downloads two parquet shards (120,000 rows) at a pinned revision and checks their sha256; writes `results/data-manifest.json` |
| `prepare_vectors.py` | seeded split into 100,000 corpus vectors and 1,000 held-out queries, scaled to length 1; the random control set; exact top-10 ground truth; writes `results/vectors.json` |
| `build_index.py` | builds one hnswlib index (M 16, ef_construction 200, random_seed 100, 1 thread) into `data/` |
| `measure_recall.py` | recall@10 for ef_search 10 to 320 on both sets; checks that two builds are identical; writes `results/recall-sweep.json` |
| `measure_filter.py` | filtered search at 1 %, 10 % and 50 % of the corpus: post-filtering vs hnswlib's filter callback; writes `results/filter-sweep.json` |
| `check_rebuild.py` | rebuilds one index as tag c in its own process, records start and end time, and compares the sha256 of the a, b and c files; writes `results/rebuild-<set>.json` |
| `check_high_ef.py` | sanity check: recall@10 at ef_search 1,000, 3,000 and 10,000 on both sets; writes `results/high-ef-check.json` |
| `measure_lsh.py` | LSH with 16 planes per table and 1 to 32 tables; writes `results/lsh-sweep.json` |
| `bench_qps.py` | wall-clock build time and queries per second; run it on an idle machine |
| `figures/make_greedy_walk.py` | the animated figure: a beam search on a 2-D neighbour graph, ef_search 1 vs 4 |
| `figures/make_recall_chart.py` | the static chart: recall@10 by ef_search |

Every number except `bench_qps.py` is a count or a recall, so a busy machine
does not change it. `results/qps-smoke.json` is a smoke test of the timing
script on 2,000 vectors and is not a measurement. The timing runs are
`results/qps-real-2026-11-03-1422.json` and `results/qps-random-2026-11-03-1417.json`.

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python download_data.py         # about 775 MB into data/ (data/ grows to about 2.5 GB after the builds)
uv run python prepare_vectors.py       # about 3 GB of RAM at peak
for s in real random; do for t in a b; do uv run python build_index.py --set $s --tag $t; done; done
uv run python check_rebuild.py --set real
uv run python check_rebuild.py --set random
uv run python measure_recall.py
uv run python measure_filter.py
uv run python measure_lsh.py
uv run python check_high_ef.py
uv run python figures/make_recall_chart.py results/recall-sweep.json
uv run python figures/make_greedy_walk.py
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run python bench_qps.py --set real
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run python bench_qps.py --set random
```

Each index build uses one thread, so it is repeatable. I ran the a and b builds of a set as two processes at the same time, so their files have the same timestamp; `check_rebuild.py` makes a third build later and compares all three. A random-set build
takes several minutes.

## Data and licences

- Vectors: [timescale/wikipedia-22-12-simple-embeddings](https://huggingface.co/datasets/timescale/wikipedia-22-12-simple-embeddings),
  Apache-2.0, shards `0000` and `0001` of the Hugging Face parquet conversion
  (`refs/convert/parquet`, commit `2f97d23433103634cff31a8c519f0a6f7445aadb`).
  Its card and NOTICE say it is a re-export of Cohere's
  `wikipedia-22-12-simple-embeddings` (Apache-2.0), with synthetic columns
  added. The vectors are 768-number embeddings of Simple English Wikipedia
  paragraphs; Cohere's [Embedding Archives post](https://cohere.com/blog/embedding-archives-wikipedia) names the model as `multilingual-22-12`.
- Text: Simple English Wikipedia, CC BY-SA. This folder does not store or
  redistribute any text or vectors; `data/` is in `.gitignore`.
- Code: MIT, like the rest of the repository.
