"""Download Pride and Prejudice from Project Gutenberg and keep only the novel.

    uv run python prepare_data.py

Writes data/pride-and-prejudice.txt. The book is in the public domain in the
United States. The Project Gutenberg header, footer and licence text are
removed, as the Gutenberg licence asks when the text is redistributed without
the trademark. The front matter (illustrations, preface) is dropped too, so
the text starts at the first sentence of Chapter 1.
"""

import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
RAW = Path(__file__).parent / "data" / "pg1342.txt"
OUT = Path(__file__).parent / "data" / "pride-and-prejudice.txt"
FIRST = "It is a truth universally acknowledged"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def main():
    RAW.parent.mkdir(exist_ok=True)
    if not RAW.exists():
        urllib.request.urlretrieve(URL, RAW)
    raw = RAW.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    body = raw[raw.index(FIRST):raw.index(END)].rstrip() + "\n"
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(body):,} characters, {len(set(body))} distinct")


if __name__ == "__main__":
    main()
