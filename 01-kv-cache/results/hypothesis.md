# Hypotheses, written before any Ollama/llama-server measurement
Written: 2026-11-01 20:41:33 KST

## H1 (series plan, pure-attention reasoning)
TTFT after changing one line grows with the number of prompt tokens AFTER that line.
Line at 0% -> about cold TTFT; at 100% -> close to a full cache hit.
prompt_eval_count: unknown whether it counts only recomputed tokens.

## H2 (revised before measuring, after reading model metadata and the server log)
qwen3.5:4b-q4_K_M is a hybrid model: qwen35.full_attention_interval = 4,
recurrent_layers = 24 of 32 (Gated DeltaNet / SSM state). Ollama 0.40.8 runs it in the
bundled llama-server with "context checkpoints enabled, max = 32, min spacing = 8192".
Old log lines show one checkpoint at n_tokens-4 and "erased invalidated context checkpoint"
when a prompt diverged early. A recurrent state cannot be rolled back to an arbitrary
position, so llama-server can only resume from a saved checkpoint.
Expected: positions 0/25/50/75% all near cold TTFT (full re-prefill). 100% also near cold,
because the chat template adds more than 4 tokens after the changed line.
Only a byte-identical prompt should be fast.

## H3 (llama-server cross-check with --checkpoint-min-step small)
With denser checkpoints, reuse should come back in steps: recompute = tokens after the
last checkpoint before the changed line.
