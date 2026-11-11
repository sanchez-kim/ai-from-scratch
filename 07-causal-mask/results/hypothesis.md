# Hypotheses, written before any training run

Written: 2026-11-02 21:42:45 KST (`date` output). No model in this folder had been trained yet.
Only the unit tests had run.

Setup I plan: a 4-layer, 4-head, d_model 128 character transformer, context 64,
on Pride and Prejudice (first 90% train, last 10% validation), AdamW, fixed
seed, the same initial weights and the same batches for every mask setting.
Three settings: `causal` (mask before softmax), `none` (no mask), `after`
(softmax, then zero the future).

## H1. Training loss
- `none` falls much faster and much lower than `causal`. Position t can read
  the character it must predict (x[t+1] = y[t]), so the model can learn to copy.
  I expect the `none` training loss to go under 0.3 nats per character, while
  `causal` ends around 1.4 to 1.7 nats.
- `after` also leaks: future keys sit in the softmax denominator. I expect it
  to fall below `causal`, but less than `none`, because the leak is indirect.

## H2. Validation loss, as trained (full sequence, same mask)
- Same picture as H1. The `none` model "looks" much better on validation
  too, because validation is scored the same way it was trained.

## H3. Honest loss (each position sees only its past, as in generation)
- `causal`: equal to its full-sequence loss, to float precision.
- `none`: far worse than `causal`, above 2.5 nats, maybe above the
  log(vocab) = ~4.45 nats of a uniform guess at some positions.
- `after`: worse than `causal`, by less than `none`.

## H4. Copying
- In the `none` model at least one head puts most of its weight (> 0.5 on
  average) on the key at position t+1.

## H5. Samples
- `causal` produces English-looking words and spacing.
- `none` produces repeated characters or junk, since at the last position the
  thing it learned to read is missing.

## H6. Row sums of the after-softmax bug
- In row t of T, the sum is the weight left on keys 0..t. At initialisation
  the scores are small, so I expect the sum to be close to (t+1)/T.

## H7. Determinism
- CPU runs with a fixed seed repeat bit for bit. MPS may not; I will check by
  running the same setting twice.

## Addendum, written after the causal and no-mask runs, before sample_check.py

Written: 2026-11-02 21:54:11 KST. I had seen train-causal-cpu-main.json and train-none-cpu-main.json.

What surprised me: the no-mask model's loss at the last position was 1.57
nats, not above 2.5. Reason: in a 64-character training window, the target of
position 63 is never in the input, even without a mask. So the no-mask model
learned honest prediction at position 63 only, and copying everywhere else.
Its loss at positions 0-62 when the input stops there is 6.5-11 nats, worse
than a uniform guess (4.45 nats). Its samples were junk for the first ~55 new
characters and word-like after that, which is when generation reaches
position 63.

New measurement, sample_check.py: 20 samples at temperature 1.0 per model,
split at new character 55 (where prediction moves to position 63).

- H8: the no-mask model's share of real words is under 30% in the first part
  and above 60% in the second. The causal model is above 70% in both.
- H9: the after-softmax model sits in between: a weak leak, so its first part
  is worse than causal's, but not as bad as no-mask.

Note added 2026-11-02 22:26:56 KST: when I wrote the addendum above, I had also seen one log line
of the after-softmax run, step 200 (val_full 1.269, val_last 2.514). I had not
seen its final numbers or samples.
