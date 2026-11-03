"""Download names.txt from karpathy/makemore into data/ (gitignored).

    uv run python download_data.py

The file is pinned to one commit of the makemore repository, so a later
download gets the same bytes. The sha256 below was recorded on 2026-11-02;
the script stops if the download does not match it.

Licence: the makemore repository is MIT licensed (Copyright (c) 2022 Andrej
Karpathy). Its README says names.txt holds the most common 32K names from
ssa.gov for the year 2018.
"""

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "988aa59e4d8fefa526d06f3b453ad116258398d4"
URL = f"https://raw.githubusercontent.com/karpathy/makemore/{COMMIT}/names.txt"
SHA256 = "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"
OUT = Path(__file__).parent / "data" / "names.txt"


def main():
    with urllib.request.urlopen(URL, timeout=30) as resp:
        body = resp.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: got {digest}, expected {SHA256}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(body)
    print(f"wrote {OUT.relative_to(Path(__file__).parent)}: {len(body.split())} names, sha256 ok")


if __name__ == "__main__":
    main()
