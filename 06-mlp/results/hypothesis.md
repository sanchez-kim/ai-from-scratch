# Hypotheses, written before running measure_init.py and measure_overflow.py
Written: 2026-11-02 21:41:59 KST

Setup: names.txt from makemore (32,033 names), shuffled with random.Random(42).
Train on the first 4,000 names, evaluate on the next 1,000. Context 3
characters, embedding 8, hidden 64 tanh units, 27 output characters.
SGD, batch 32, fixed seeds. Nothing timed except bench_epoch.py, which is a
smoke test only in this session.

## H1 loss of a random guess
ln 27 = 3.2958 nats. Any model that knows nothing should start there.

## H2 loss at init, by init scale (std of W1 and W2; embeddings always N(0,1))
- std 1.0 ("plain randn", too large): pre-activation std about sqrt(24) = 4.9,
  hidden units mostly at +-1, logits std about 8. Initial dev loss 12-20.
- std 0.01 (too small): logits near 0, initial loss within 0.01 of 3.296.
- std 1/sqrt(fan_in): logits std about 0.6, initial loss about 3.4-3.6.

## H3 tanh saturation at init (share of hidden values with |h| > 0.99)
- std 1.0: about 55-65%.
- std 0.01: 0%.
- 1/sqrt(fan_in): under 2%.

## H4 dev loss after 4,000 steps at lr 0.1
- 1/sqrt(fan_in): lowest, about 2.4-2.6.
- std 0.01: starts slow (flat for the first few hundred steps), ends within
  0.1 of fan_in.
- std 1.0: highest, at least 0.15 above fan_in, still saturated (> 30%).

## H5 naive vs stable cross-entropy, 1/sqrt(fan_in) init, lr 0.1 / 1 / 10 / 100
- The naive formula fails only when a logit passes log(float64 max) = 709.78
  (OverflowError in plain Python) or p[target] underflows to 0 (ValueError).
- lr 0.1 and lr 1: no failure in 1,000 steps; largest logit stays under 50.
- lr 10: unsure. Guess: no failure, logits under 709.
- lr 100: naive fails within the first 100 steps; numpy on the same logits
  gives nan or inf. Stable keeps going with a large but finite loss.
- Until the step where naive fails, the two runs give the same loss to
  about 1e-12.

## Addendum, written after the first run of both scripts
Written: 2026-11-02 21:46:52 KST

What run 1 showed, before any change:
- lr 100: the plain-Python naive loss failed at step 4, but with ValueError
  (log of 0, an underflow), not OverflowError. The largest logit in that batch
  was 654.9, below 709.78. I expected overflow first. numpy float64 on the same
  logits gave inf (2 of 32 rows), not nan.
- lr 10: no failure, but the naive and stable twins drifted apart (largest
  loss difference 57.5), although they agree to 1e-15 at lr 0.1 and 1.

Changes for run 2 (same seeds, same data):
1. measure_overflow.py adds a third mode: the naive formula computed by numpy
   float64 per example, the way most array code writes it, so the run keeps
   going through inf and nan instead of raising. It records the first step with
   an inf loss, the first step with a nan loss, the largest logit at those
   steps, and the first step where any weight and all weights are nan.
   train_step takes a loss_fn argument for this; the math is unchanged.
2. It records the first step where the twins differ by more than 1e-9.
3. For each failing row, it records the stable loss (the gap between
   log-sum-exp and the target logit), to compare with -log(smallest float).
4. measure_init.py also counts dev rows where the naive formula in numpy
   float16, float32 and float64 gives inf or nan, at init and after training.

Expectations for run 2:
- numpy mode at lr 100: inf at step 4 (as above), weights still finite then.
  nan at step 5, where the stable run's largest logit was 838 (above 709.78).
  All weights nan one step later.
- float16 rows at init: std 1.0 init has logits up to 26.4, above
  ln(float16 max) = 11.09, so some rows fail; under 20%. float32 and float64:
  no failures in any run.
