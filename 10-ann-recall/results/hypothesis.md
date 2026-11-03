# Hypotheses, written before any search measurement

Written: 2026-11-02 22:12:57 KST (`date` output). At this point I had run only `download_data.py`
and `prepare_vectors.py` (the split, the exact top-10 ground truth and the
neighbour-cosine profile in `vectors.json`). No hnswlib index, no LSH table and
no recall number existed yet.

Setup I plan:

- Real set: 100,000 Simple English Wikipedia paragraph embeddings (Cohere
  multilingual-22-12, 768 dims, scaled to length 1), 1,000 held-out queries.
- Random set: 100,000 + 1,000 Gaussian vectors, 768 dims, length 1.
- hnswlib 0.8.0, space "ip", M = 16, ef_construction = 200, random_seed = 100,
  built with 1 thread. ef_search swept over 10, 20, 40, 80, 160, 320. k = 10.
- LSH: 16 random hyperplanes per table, 1 to 32 tables.
- Filter: keep a vector if (Wikipedia article id mod 100) < 1, 10 or 50.

## H1. Real vectors: recall@10 climbs fast with ef_search.
Guess: about 0.80 at ef 10, 0.95 at ef 40, at least 0.99 at ef 160.

## H2. Random vectors: recall is far lower at every ef_search.
Guess: about 0.30 at ef 10 and 0.80-0.90 at ef 320. Reason I expect: random
points have no clusters, so "close to my neighbour" does not mean "close to me",
and a greedy walk loses its way. I will check this reason directly by counting
how many of a point's 10 nearest neighbours are also among its nearest
neighbour's 10 nearest. Guess: about half for real vectors, under 10 % for random.

(The neighbour-cosine profile in vectors.json was already computed when I wrote
this. In standard-deviation units, the 10th neighbour stands out by about the same
amount in both sets, so that profile alone does not explain a recall gap.)

## H3. One thread + fixed seed = the same index twice.
Guess: two builds give identical results for every query, and identical saved
index files.

## H4. Filtering.
- Post-filter (search top 10, then drop vectors the filter rejects): recall about
  0.5 at 50 %, about 0.1-0.2 at 10 %, under 0.05 at 1 %.
- Post-filter with a top 100 search: about 0.99 at 50 %, 0.7 at 10 %, 0.1 at 1 %.
- hnswlib's filter callback: at least 0.95 at every selectivity for ef_search >= 40,
  because the source keeps walking until it has ef_search allowed results.

## H5. LSH needs many tables, and scores a large share of the corpus, to reach high recall.
Guess: 1 table under 0.3 on real vectors; 32 tables above 0.9 but scoring more
than 10 % of the corpus. On random vectors, much lower at the same number of tables.
