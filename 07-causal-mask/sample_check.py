"""Generate from each trained model and count how much of the output is real words.

    uv run python sample_check.py causal-cpu-main none-cpu-main after-cpu-main

For each checkpoint: 20 samples of 150 new characters at temperature 1.0
(seeds 0-19) and one greedy sample, all from the prompt "Elizabeth".

The prompt has 9 characters and the context is 64. New character k is
predicted at input position min(9 + k, 64) - 1, so characters 0-54 come from
positions 8-62 and characters 55 and later from position 63, the last slot.
In training, position 63 is the only one whose answer is never in the input,
even without a mask. The two parts are counted separately.

"Real word": a run of letters that appears as a word in the training text.
The first and last run of each part are dropped, since they may be cut.
"""

import json
import re
import sys
from pathlib import Path

import torch

from charformer.data import load
from charformer.model import Config, CharTransformer, generate

PROMPT, N_NEW, SEEDS = "Elizabeth", 150, range(20)
WORD = re.compile(r"[A-Za-z]+")


def real_word_share(text, vocab):
    words = WORD.findall(text)[1:-1]
    return sum(w.lower() in vocab for w in words), len(words)


def main():
    torch.set_num_threads(2)
    chars, train_ids, _ = load()
    vocab = {w.lower() for w in WORD.findall(chars.decode(train_ids.tolist()))}
    root = Path(__file__).parent
    out = {"prompt": PROMPT, "new_chars": N_NEW, "seeds": list(SEEDS), "temperature": 1.0,
           "train_vocab_words": len(vocab), "models": {}}
    for name in sys.argv[1:]:
        ck = torch.load(root / "checkpoints" / f"{name}.pt")
        assert ck["chars"] == chars.chars
        model = CharTransformer(Config(**ck["config"]))
        model.load_state_dict(ck["state"])
        ctx = model.cfg.context
        split = ctx - len(PROMPT)  # first new character predicted at position 63
        prompt = torch.tensor([chars.encode(PROMPT)])
        parts = {"positions_8_to_62": [0, 0], "position_63": [0, 0]}
        samples = []
        for seed in SEEDS:
            text = chars.decode(generate(model, prompt, N_NEW, 1.0, torch.Generator().manual_seed(seed))[0].tolist())
            new = text[len(PROMPT):]
            for key, seg in (("positions_8_to_62", new[:split]), ("position_63", new[split:])):
                hit, total = real_word_share(seg, vocab)
                parts[key][0] += hit
                parts[key][1] += total
            samples.append(text)
        greedy = chars.decode(generate(model, prompt, N_NEW, 0)[0].tolist())
        out["models"][name] = {
            "split_at_new_char": split,
            "real_words": {k: {"real": h, "total": t, "share": h / t if t else None} for k, (h, t) in parts.items()},
            "samples": samples, "greedy": greedy,
        }
        print(name, {k: f"{h}/{t}" for k, (h, t) in parts.items()})
    path = root / "results" / "samples.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path.name)


if __name__ == "__main__":
    main()
