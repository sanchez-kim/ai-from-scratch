# Hypotheses, written before any measurement

Written: 2026-11-02 21:57:34 KST (`date` output). The corpus (10,000 paragraphs) exists. No query
has been generated yet and no measurement script has run.

## Agreement with the libraries (measure_agreement.py)

- bm25s, method "lucene", dtype float64: our scores divided by (k1 + 1) = 2.5
  match to about 1e-12 on every document. Top-10 lists identical.
- bm25s, default dtype float32: max absolute difference (after the 2.5 factor)
  around 1e-6. Top-10 lists identical except where two scores tie within
  float32 rounding; I expect at least 98% of queries to have the same top-10
  in the same order.
- rank_bm25 BM25Okapi: different idf (no "1 +" inside the log, and words in
  more than half the documents get epsilon * average idf). Absolute scores
  differ by about 1 or more for common words. Top-10 sets: about 80-90%
  overlap on average, top-1 the same for about 90% of queries.

## Retrieval quality (measure_retrieval.py)

- Lexical query set (the generator may copy words): recall@10 about 0.85-0.95,
  MRR about 0.7.
- Paraphrase query set (asked to avoid the paragraph's words): recall@10 much
  lower, about 0.4-0.6, MRR about 0.3.

## b = 0 vs b = 0.75

- With b = 0 (no length normalization), long paragraphs win more often: the
  mean length of top-10 results goes up by 20-40% compared to b = 0.75.
- Recall@10 on the lexical set drops by a few points with b = 0.

## Adding 10% more documents (1,000 extra paragraphs)

- avgdl moves by less than 2%. idf of a word with fixed df rises by about
  ln(1.1) ~ 0.095 at most.
- Scores of the original documents for the same query change by a few
  percent; the top-10 order of original documents (ignoring new ones) stays
  the same for at least 90% of queries.

## Timing (bench_latency.py)

- Pure-Python query on 10k paragraphs: 1-5 ms per query; bm25s faster. Not
  measured in this session (machine busy); harness smoke test only.
