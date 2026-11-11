"""Check decode(encode(x)) == x on every evaluation file, with both trained tokenizers.

    uv run python check_roundtrip.py      # after measure_tokens.py

Reads the merges measure_tokens.py saved and writes results/roundtrip.json.
"""

import json
import time
from pathlib import Path

from bpe.tokenizer import decode, encode

HERE = Path(__file__).parent
FILES = ["wiki-en-test.txt", "wiki-ko-test.txt", "python-code.txt", "flores-eng.txt", "flores-kor.txt"]


def load_merges(path):
    merges = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        a, b, new = map(int, line.split())
        merges[(a, b)] = new
    return merges


def main():
    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "checks": []}
    for name in ("en", "mixed"):
        merges = load_merges(HERE / "results" / f"merges-{name}.txt")
        for file in FILES:
            text = (HERE / "data" / file).read_text(encoding="utf-8")
            ids = encode(text, merges)
            ok = decode(ids, merges) == text
            result["checks"].append({"tokenizer": f"ours-{name}", "file": file,
                                     "chars": len(text), "tokens": len(ids), "round_trip": ok})
            print(f"ours-{name:6s} {file:18s} {len(ids):>8,} tokens  round trip {ok}")
    (HERE / "results" / "roundtrip.json").write_text(json.dumps(result, indent=1) + "\n")
    assert all(c["round_trip"] for c in result["checks"])


if __name__ == "__main__":
    main()
