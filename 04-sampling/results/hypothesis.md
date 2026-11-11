# Hypotheses, written before any measurement

Written: 2026-11-02 21:24:04 KST (`date` output), before starting llama-server or collecting a
single probability.

Setup I plan: qwen3.5:4b-q4_K_M (the GGUF blob Ollama stores), Homebrew
llama-server 0.4.1. Ten passages I wrote, read token by token; at each of 500
positions, the top-100 next-token probabilities. Two orders:

- temperature first: temperature -> top-k 40 -> top-p (the order in HF transformers)
- temperature last: top-k 40 -> top-p -> temperature (the default order in llama.cpp)

Settings: T 0.7 / 1.0 / 1.3, top-p 0.9 / 0.95.

## H1. At T = 1.0 the two orders agree at every position.
Dividing by 1 changes nothing. Expected: 0 of 500 positions differ. If any
differ, my chain or my reading of the data is wrong.

## H2. Many positions are nearly decided.
Expected: at 30-40 % of positions the top token has probability >= 0.9, and
there top-p 0.9 keeps 1 token in both orders, so the order cannot matter there.

## H3. With temperature last, T does not change the survivor set at all.
That follows from the order itself: the cut happens before temperature.
Expected median survivors with top-p 0.95: about 5-8 tokens (a guess).

## H4. With temperature first, T = 0.7 shrinks the set and T = 1.3 grows it.
Guess: at T = 0.7 / top-p 0.95 the set differs from the temperature-last set
at about half of the positions, and the median drops to about half.
At T = 1.3 the set differs at about half of the positions too, and at some
positions it hits the top-k 40 cap.

## H5. llama.cpp's own sampler gives the same survivor counts as my 60 lines.
Checked with post_sampling_probs on the same positions, for both orders
(llama-server accepts a custom "samplers" order). Expected: equal at all
positions, except maybe a few where the cumulative sum lands within float32
rounding of p.

## Note added 2026-11-02, after the run

H5 says "my 60 lines": that was the plan. The finished `sampling/sampler.py` is 71 lines with comments. The guesses above are unchanged.
