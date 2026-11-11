# Hypotheses, written before any measurement

Written: 2026-11-02 23:47:27 KST (`date` output). No embedding has been computed and no
measurement script for this folder exists yet. The corpus (10,000 paragraphs)
and the two query sets (500 lexical, 500 paraphrase) come from 08-bm25.

What I had already seen: 08-bm25/results/retrieval.json from a partial run
(359 lexical queries) gave our BM25 recall@10 0.99 and MRR@10 0.90 on the
lexical set. No paraphrase number and no embedding number yet.

Setup decided now: BM25 k1 = 1.5, b = 0.75 (Part 8 default). Embeddings
BAAI/bge-small-en-v1.5, normalised, query instruction on. Fusion takes the
top 100 of each list. Metrics: recall@10 and MRR@10, as in Part 8.

## Single methods

- BM25, lexical: recall@10 about 0.98, MRR@10 about 0.88.
- BM25, paraphrase: recall@10 about 0.55-0.7, MRR@10 about 0.4.
- Embeddings, lexical: a little below BM25, recall@10 about 0.95.
- Embeddings, paraphrase: clearly above BM25, recall@10 about 0.8-0.85.

## Fusion

- RRF k = 60, lexical: about equal to BM25 (within 1 point of recall).
- RRF k = 60, paraphrase: beats the better single method (embeddings) by
  1-3 points of recall@10.
- Score sum (min-max normalised): within 1-2 points of RRF k = 60 on both
  sets, slightly lower.
- RRF k = 1: recall@10 a little below k = 60 (one list's top ranks dominate),
  MRR@10 about the same.

## Agreement (Kendall tau over the union of the two top-10 lists)

- Mean tau is low on both sets: about 0.2-0.3 on lexical, about 0.1 on
  paraphrase.
- Fusion gains most on queries where the two methods disagree (low tau) and
  gains nothing where they agree.

## Query instruction

- Without the "Represent this sentence for searching relevant passages: "
  prefix, embedding recall@10 drops by about 1 point (the model card says
  "slight degradation").
