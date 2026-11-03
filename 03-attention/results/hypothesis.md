# Hypotheses, written before running measure_scaling.py
Written: 2026-11-02 21:22:05 KST

Setup: query and key entries drawn from N(0, 1), 64 queries x 64 keys per seed,
seeds 0-9, d_k = 8, 16, 32, 64, 128, 256, 512. Max entropy = log2(64) = 6 bits.

## H1 variance of q.k
Unscaled: variance close to d_k (8, 16, ..., 512), within about 10% per seed.
With 1/sqrt(d_k): close to 1 at every d_k. With 1/d_k: close to 1/d_k.

## H2 entropy with 1/sqrt(d_k)
Roughly flat across d_k. For n Gaussian scores with variance 1, entropy is about
ln(n) - 1/2 nats, so about 5.3 bits, around 0.88 of the 6-bit maximum.

## H3 entropy without scaling
Falls as d_k grows. Guess: about 2.5-3 bits at d_k = 8, under 0.3 bits at
d_k = 512. Largest weight per row above 0.9 from d_k = 128 on.

## H4 entropy with 1/d_k (over-scaling)
Rises toward the 6-bit maximum as d_k grows: above 0.99 of max at d_k = 512.
The rows go flat, so attention cannot pick out any key.

## H5 gradient through softmax (Frobenius norm of the softmax Jacobian)
Unscaled: shrinks toward 0 as rows go one-hot, by more than 10x from d_k = 8
to 512. With 1/sqrt(d_k): roughly flat. With 1/d_k: also shrinks, because a
flat row over 64 keys has a small Jacobian (about 1/sqrt(64) = 0.125).

## Addendum after the first run (2026-11-02 21:22:35 KST)
The first run (same seeds, same numbers) showed the MEAN Jacobian norm without
scaling fell only about 3x from d_k = 8 to 512 (0.296 -> 0.098), so H5's ">10x"
was wrong for the mean. Unscaled d_k = 8 even had a larger mean norm (0.296)
than any scaled row set (about 0.18). I added two per-row statistics to see
whether a few rows with a close runner-up carry that mean: the median row's
Jacobian norm, and the share of rows whose largest weight is above 0.99.
Expected before re-running: unscaled median-row norm falls much more than the
mean; saturated share near 0 with 1/sqrt(d_k) at every d_k.
