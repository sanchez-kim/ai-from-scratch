# 08 · BM25 from scratch, checked against bm25s and rank_bm25

Okapi BM25 in plain Python (`bm25/bm25.py`, 55 non-blank lines: tokenizer,
inverted index, one scoring formula), eight tests, and measurements on 10,000 Simple English
Wikipedia paragraphs with two sets of synthetic questions.

This folder also owns the **shared retrieval corpus** for Parts 9 and 10 of
the series. See [Corpus and queries for later parts](#corpus-and-queries-for-later-parts).

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python build_corpus.py          # 157 MB parquet into data/ (gitignored), writes data/corpus.jsonl, data/extra.jsonl
uv run python make_queries.py          # needs Ollama on localhost:11434 with qwen3.5:4b-q4_K_M; results/queries.jsonl is committed, so skip this
uv run python measure_agreement.py     # ours vs bm25s vs rank_bm25 -> results/agreement.json
uv run python measure_retrieval.py     # recall@10, MRR@10, k1/b grid, +10% documents -> results/retrieval.json
uv run python bench_latency.py         # timing; run on an idle machine -> results/latency-<date>.json
uv run python figures/make_saturation.py
uv run python figures/make_b_chart.py
```

Everything except `bench_latency.py` is deterministic. A second run gives the
same numbers.

## The formula and the idf variants

```
score(q, d) = sum over query words w of
    idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
```

Defaults: k1 = 1.5, b = 0.75 (the defaults of both libraries below).
Tokenizer: lowercase, `[a-z0-9]+`, no stemming, no stopword list. Both
libraries get our token lists, so only the scoring differs.

| | idf | tf part | notes |
|---|---|---|---|
| ours | ln(1 + (N − df + 0.5) / (df + 0.5)) | tf·(k1+1) / (tf + k1·norm) | idf is never negative |
| bm25s 0.3.16, `method="lucene"` (default) | same as ours ([`scoring.py` `_score_idf_lucene`](https://github.com/xhluca/bm25s/blob/0.3.16/bm25s/scoring.py)) | tf / (tf + k1·norm), no (k1+1) factor ([`_score_tfc_robertson`, used by `_score_tfc_lucene`](https://github.com/xhluca/bm25s/blob/0.3.16/bm25s/scoring.py)) | scores = ours / (k1+1); default dtype float32 |
| rank_bm25 0.2.2, `BM25Okapi` | ln((N − df + 0.5) / (df + 0.5)); if negative, replaced by 0.25 × average idf of the whole vocabulary ([`rank_bm25.py` `_calc_idf`](https://github.com/dorianbrown/rank_bm25/blob/0.2.2/rank_bm25.py)) | tf·(k1+1) / (tf + k1·norm) | same tf part as ours |

norm = 1 − b + b · len(d) / avgdl. Source read from the installed packages
(bm25s 0.3.16 from PyPI, rank_bm25 0.2.2 from PyPI).

## Corpus and queries for later parts

| File | Committed | What |
|---|---|---|
| `data/corpus.jsonl` | no (run `build_corpus.py`) | 10,000 paragraphs. Fields: `id`, `title`, `url`, `text` |
| `data/extra.jsonl` | no | 1,000 more paragraphs, same rule, only for the "+10%" experiment |
| `results/corpus-manifest.json` | yes | source, dataset revision, sha256 of the parquet and both JSONL files, counts |
| `results/queries.jsonl` | yes | 1,000 questions. Fields: `query_id`, `kind` (`lexical` or `paraphrase`), `query`, `gold_id` |
| `results/queries-meta.json` | yes | model, digest, options, prompts, Ollama version |

Use it from another part:

```python
import sys
sys.path.insert(0, "../08-bm25")
from bm25 import BM25, tokenize
from bm25.data import load_corpus, load_queries, searchable_text

docs = load_corpus()                  # list of dicts, in a fixed order
lexical = load_queries("lexical")     # 500 rows
paraphrase = load_queries("paraphrase")
index = BM25([tokenize(searchable_text(d)) for d in docs])
```

`searchable_text(doc)` is `title + "\n" + text`. Index that, so every part
searches the same text. `gold_id` is the `id` of the one paragraph the
question was written from. Other paragraphs may also answer it; the
metrics count only the gold one.

### Source

[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia),
config `20231101.simple` (Simple English Wikipedia, 1 November 2023 dump),
revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, file
`20231101.simple/train-00000-of-00001.parquet`, sha256
`31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0`.
Text licence: the [pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)
lists [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html);
[Simple English Wikipedia](https://simple.wikipedia.org) now publishes its
text under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
I treat the paragraphs and the questions made from them as CC BY-SA. The
paragraph text is not committed.

Selection rule (in `build_corpus.py`): order articles by sha256 of their ID,
split on blank lines, keep prose blocks (no line starting with a space, ends
with ".", 25 to 250 words), at most 2 per article, first 10,000 into the
corpus and the next 1,000 into `extra.jsonl`.

### How the queries were made

`make_queries.py` picks 500 paragraphs with `random.Random(8).sample` and
asks `qwen3.5:4b-q4_K_M` through Ollama's `/api/generate` for one question
per paragraph and per kind. Options: temperature 0, seed 42, num_predict 80,
`think: false`. One request at a time. The answer is cleaned to its first
line. Prompts (`{title}` and `{text}` are the paragraph's):

**lexical**

```
Here is a paragraph from a Simple English Wikipedia article titled "{title}".

{text}

Write one question that a person might type into a search engine, which this paragraph answers. Reply with the question only.
```

**paraphrase**

```
Here is a paragraph from a Simple English Wikipedia article titled "{title}".

{text}

Write one question that this paragraph answers. Do not use any noun, verb, adjective or name that appears in the paragraph or the title. Use synonyms and descriptions instead. Reply with the question only.
```

### Known problems in the queries (read before reusing them)

I read every 25th query of each set (ids `l000, l025, ... l475` and
`p000, p025, ... p475`, 20 per set) next to its gold paragraph.

- **lexical:** all 20 are clear and answered by the gold paragraph.
- **paraphrase:** 10 of 20 have a problem. 4 copy a name from the paragraph
  despite the prompt (`p225`, `p275`, `p300`, `p450`). 6 are too vague to
  have one answer (`p000`, `p050`, `p150`, `p200`, `p350`, `p425`).
  `p050` and `p150` are the same question, word for word; over the whole set,
  2 paraphrase questions occur twice each (4 rows).
- 13 paraphrase questions share no word with their gold paragraph
  (`results/retrieval.json`, `query_word_overlap`).

None were removed or edited. Treat paraphrase recall as a lower bound.

## Results

- `results/hypothesis.md`: expectations, written before the first measurement.
- `results/agreement.json`: score differences and top-10 agreement with both libraries, idf of the most common words, recall with rank_bm25's idf.
- `results/retrieval.json`: recall@10 and MRR@10 for a k1 × b grid, b = 0 vs 0.75 by paragraph length, the +10% documents experiment.
- `results/latency-2026-11-03.json`: query time and index build time for ours, bm25s and rank_bm25, with `uptime` before and after.
- `results/latency-smoke.json`: a smoke test of the timing harness only. Not a measurement.
- `results/pytest.log`: the test run.

## License

Code: MIT, see [../LICENSE](../LICENSE). Corpus text and queries: CC BY-SA
(see [Source](#source) for the exact versions).
