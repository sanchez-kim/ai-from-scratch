"""Train two BPE tokenizers and count tokens on English, Korean and Python code.

    uv run python download_data.py      # once
    uv run python measure_tokens.py

Everything here is a count, so the output is the same on every run and every
machine (given the same corpus files; their sha256 is stored in the result).
No timing is recorded.

Tokenizers:
  ours-en     our BPE trained on English Wikipedia text only
  ours-mixed  our BPE trained on the same number of bytes, half English, half Korean
  cl100k_base, o200k_base   OpenAI's tokenizers via tiktoken

Writes results/token-counts.json and results/merges-{en,mixed}.txt.
"""

import hashlib
import json
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import tiktoken

from bpe.tokenizer import chunks, encode, train, vocab

HERE = Path(__file__).parent
DATA, RESULTS = HERE / "data", HERE / "results"
MAX_MERGES = 8000
SWEEP = [0, 250, 500, 1000, 2000, 4000, 8000]
EVAL = {"english": "wiki-en-test.txt", "korean": "wiki-ko-test.txt", "code": "python-code.txt"}
EXAMPLE_WORD = "한국어"  # "Korean language"
HANGUL = re.compile("[가-힣]")


def read(name):
    return (DATA / name).read_text(encoding="utf-8")


def head_bytes(text, limit):
    """The longest run of whole lines from the start whose UTF-8 size is <= limit."""
    out, size = [], 0
    for line in text.splitlines(keepends=True):
        n = len(line.encode("utf-8"))
        if size + n > limit:
            break
        out.append(line)
        size += n
    return "".join(out)


def save_merges(merges, path):
    lines = [f"{a} {b} {new}" for (a, b), new in merges.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def first(merges, k):
    """The tokenizer after only its first k merges."""
    return {pair: new for pair, new in merges.items() if new < 256 + k}


class Ours:
    def __init__(self, merges):
        self.merges, self.cache = merges, {}

    def count(self, text):
        # encode() works chunk by chunk, so counting each distinct chunk once is
        # exact and much faster. main() checks this against encode() on a sample.
        total = 0
        for chunk in chunks(text):
            if chunk not in self.cache:
                self.cache[chunk] = len(encode(chunk, self.merges))
            total += self.cache[chunk]
        return total

    def pieces(self, text):
        table = vocab(self.merges)
        return [table[i] for i in encode(text, self.merges)]


class Tik:
    def __init__(self, name):
        self.enc = tiktoken.get_encoding(name)

    def count(self, text):
        return len(self.enc.encode_ordinary(text))

    def pieces(self, text):
        return [self.enc.decode_single_token_bytes(t) for t in self.enc.encode_ordinary(text)]


def per_1000(tokens, chars):
    return round(1000 * tokens / chars, 1)


def show(pieces):
    # Bytes as hex, plus the text when the piece is valid UTF-8 on its own.
    out = []
    for p in pieces:
        try:
            text = p.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        out.append({"hex": p.hex(" "), "text": text})
    return out


def hangul_tokens(table, partial_ok=False):
    """How many vocabulary entries contain a whole Hangul syllable.

    Strict: the entry must decode as UTF-8 on its own. partial_ok: stray partial
    bytes at the edges are dropped first, so an entry such as 한 plus the first
    byte of the next syllable also counts.
    """
    n = 0
    for b in table:
        try:
            text = b.decode("utf-8", errors="ignore" if partial_ok else "strict")
        except UnicodeDecodeError:
            continue
        if HANGUL.search(text):
            n += 1
    return n


def main():
    uptime_before = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    en_train, ko_train = read("wiki-en-train.txt"), read("wiki-ko-train.txt")
    budget = len(en_train.encode("utf-8"))
    mixed_en = head_bytes(en_train, budget // 2)
    mixed_ko = head_bytes(ko_train, budget - len(mixed_en.encode("utf-8")))
    training = {"ours-en": en_train, "ours-mixed": mixed_en + mixed_ko}

    merges = {}
    for name, text in training.items():
        merges[name] = train(text, MAX_MERGES)
        save_merges(merges[name], RESULTS / f"merges-{name.split('-')[1]}.txt")
    tokenizers = {"ours-en": Ours(merges["ours-en"]), "ours-mixed": Ours(merges["ours-mixed"]),
                  "cl100k_base": Tik("cl100k_base"), "o200k_base": Tik("o200k_base")}

    texts = {k: read(v) for k, v in EVAL.items()}
    for name in ("ours-en", "ours-mixed"):
        for t in texts.values():
            sample = t[:20000]
            assert tokenizers[name].count(sample) == len(encode(sample, merges[name]))
    sets = {k: {"file": EVAL[k], "chars": len(t), "bytes": len(t.encode("utf-8")),
                "bytes_per_char": round(len(t.encode("utf-8")) / len(t), 3),
                "hangul_share": round(len(HANGUL.findall(t)) / len(t), 3),
                "space_share": round(t.count(" ") / len(t), 3)}
            for k, t in texts.items()}

    counts = {}
    for name, tok in tokenizers.items():
        counts[name] = {}
        for k, t in texts.items():
            n = tok.count(t)
            counts[name][k] = {"tokens": n, "per_1000_chars": per_1000(n, len(t)),
                               "chars_per_token": round(len(t) / n, 2)}

    # Same meaning, two languages: FLORES-200 devtest, line i is a translation of line i.
    eng, kor = read("flores-eng.txt").splitlines(), read("flores-kor.txt").splitlines()
    assert len(eng) == len(kor)
    flores = {"sentences": len(eng), "chars": {"english": sum(map(len, eng)), "korean": sum(map(len, kor))}}
    for name, tok in tokenizers.items():
        e = [tok.count(s) for s in eng]
        k = [tok.count(s) for s in kor]
        ratios = sorted(b / a for a, b in zip(e, k))
        flores[name] = {"english_tokens": sum(e), "korean_tokens": sum(k),
                        "korean_over_english": round(sum(k) / sum(e), 3),
                        "english_per_1000_chars": per_1000(sum(e), flores["chars"]["english"]),
                        "korean_per_1000_chars": per_1000(sum(k), flores["chars"]["korean"]),
                        "median_sentence_ratio": round(ratios[len(ratios) // 2], 3)}

    sweep = {}
    for name in ("ours-en", "ours-mixed"):
        sweep[name] = []
        for k in SWEEP:
            tok = Ours(first(merges[name], k))
            row = {"merges": k, "vocab_size": 256 + k}
            for key, t in texts.items():
                row[key] = per_1000(tok.count(t), len(t))
            sweep[name].append(row)

    example = {"word": EXAMPLE_WORD, "utf8_hex": EXAMPLE_WORD.encode("utf-8").hex(" "),
               "tokens": {name: show(tok.pieces(EXAMPLE_WORD)) for name, tok in tokenizers.items()}}

    vocab_facts = {}
    for name in ("ours-en", "ours-mixed"):
        table = vocab(merges[name])
        learned = [table[i] for i in range(256, 256 + len(merges[name]))]
        vocab_facts[name] = {
            "merges_learned": len(merges[name]),
            "merges_with_non_ascii_bytes": sum(1 for b in learned if max(b) >= 0x80),
            "merges_containing_hangul": hangul_tokens(learned),
            "merges_containing_hangul_partial_ok": hangul_tokens(learned, partial_ok=True),
        }
    for name in ("cl100k_base", "o200k_base"):
        enc = tokenizers[name].enc
        table = []
        for i in range(enc.n_vocab):
            try:
                table.append(enc.decode_single_token_bytes(i))
            except KeyError:  # unused ids and special tokens have no bytes
                pass
        vocab_facts[name] = {"vocab_size": len(table), "tokens_containing_hangul": hangul_tokens(table),
                             "tokens_containing_hangul_partial_ok": hangul_tokens(table, partial_ok=True)}

    # Why does ours-en get much better at code between 4,000 and 8,000 merges?
    # Split the saving into whitespace-only chunks (indentation) and the rest.
    code_chunks = Counter(chunks(texts["code"]))
    half, full = first(merges["ours-en"], 4000), merges["ours-en"]
    saved = {"whitespace_chunks": 0, "other_chunks": 0,
             "whitespace_tokens_at_4000": 0, "whitespace_tokens_at_8000": 0}
    for chunk, n in code_chunks.items():
        before, after = n * len(encode(chunk, half)), n * len(encode(chunk, full))
        saved["whitespace_chunks" if chunk.isspace() else "other_chunks"] += before - after
        if chunk.isspace():
            saved["whitespace_tokens_at_4000"] += before
            saved["whitespace_tokens_at_8000"] += after
    en_table = vocab(merges["ours-en"])
    saved["whitespace_merges"] = [{"merge": i - 255, "bytes": en_table[i].hex(" ")}
                                  for i in sorted(en_table) if i >= 256 and en_table[i].isspace()]

    result = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": platform.python_version(),
        "tiktoken": tiktoken.__version__,
        "uptime_before": uptime_before,
        "uptime_after": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        "training": {name: {"bytes": len(t.encode("utf-8")), "chars": len(t),
                            "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(),
                            "korean_bytes": len(mixed_ko.encode("utf-8")) if name == "ours-mixed" else 0}
                     for name, t in training.items()},
        "max_merges": MAX_MERGES,
        "eval_sets": sets,
        "counts": counts,
        "flores_devtest": flores,
        "sweep_per_1000_chars": sweep,
        "example": example,
        "vocab": vocab_facts,
        "code_saving_ours_en_4000_to_8000": saved,
    }
    out = RESULTS / "token-counts.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in counts:
        print(f"{name:12s}", "  ".join(f"{k} {v['per_1000_chars']:7.1f}" for k, v in counts[name].items()),
              f"  flores ko/en {flores[name]['korean_over_english']}")


if __name__ == "__main__":
    main()
