"""Build the shared retrieval corpus: 10,000 Simple English Wikipedia paragraphs.

    uv run python build_corpus.py

Source: the Hugging Face dataset wikimedia/wikipedia, config 20231101.simple
(a dump of Simple English Wikipedia from 1 November 2023), pinned to one
dataset revision. Text licence: the dataset card lists CC BY-SA 3.0 and
GFDL; Simple English Wikipedia now uses CC BY-SA 4.0.

The script downloads one parquet file (157 MB) into data/, checks its sha256,
and picks paragraphs with a fixed rule, so every run gives the same corpus:

1. Order the articles by sha256 of their article ID. This is a shuffle that
   does not depend on a random seed or on the parquet row order.
2. Split each article on blank lines. Keep a block if it is prose: no line
   starts with a space (those are list items), it ends with ".", and it has
   25 to 250 words. Headings and lists drop out.
3. Take at most 2 paragraphs per article, the first ones that qualify.
4. The first 10,000 paragraphs are data/corpus.jsonl. The next 1,000 are
   data/extra.jsonl, used only to measure what adding 10% more documents does.

Writes results/corpus-manifest.json with the source, revision, sha256 of every
file and the counts. The paragraph text itself is not committed.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

REPO = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
FILE = "20231101.simple/train-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"
SHA256 = "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0"
PARQUET = DATA / "simplewiki-20231101.parquet"

N_CORPUS, N_EXTRA = 10_000, 1_000
MIN_WORDS, MAX_WORDS, PER_ARTICLE = 25, 250, 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    DATA.mkdir(exist_ok=True)
    if not PARQUET.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, PARQUET)
    got = sha256_file(PARQUET)
    if got != SHA256:
        raise SystemExit(f"sha256 mismatch: {got}")


def prose_paragraphs(text):
    for block in text.split("\n\n"):
        lines = block.strip("\n").split("\n")
        if any(line.startswith(" ") for line in lines):
            continue
        para = " ".join(line.strip() for line in lines).strip()
        if para.endswith(".") and MIN_WORDS <= len(para.split()) <= MAX_WORDS:
            yield para


def select(table):
    ids = table.column("id").to_pylist()
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    titles, urls, texts = (table.column(c) for c in ("title", "url", "text"))
    picked = []
    for i in order:
        for k, para in enumerate(prose_paragraphs(texts[i].as_py())):
            if k == PER_ARTICLE:
                break
            picked.append({"id": f"{ids[i]}-p{k}", "title": titles[i].as_py(),
                           "url": urls[i].as_py(), "text": para})
            if len(picked) == N_CORPUS + N_EXTRA:
                return picked
    raise SystemExit("not enough paragraphs")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    download()
    table = pq.read_table(PARQUET, columns=["id", "url", "title", "text"])
    picked = select(table)
    corpus, extra = picked[:N_CORPUS], picked[N_CORPUS:]
    write_jsonl(DATA / "corpus.jsonl", corpus)
    write_jsonl(DATA / "extra.jsonl", extra)
    words = sorted(len(d["text"].split()) for d in corpus)
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "config": "20231101.simple",
        "revision": REVISION,
        "file": FILE,
        "file_sha256": SHA256,
        "licence": "dataset card at this revision: cc-by-sa-3.0, gfdl; simple.wikipedia.org now: CC BY-SA 4.0",
        "articles_in_dump": table.num_rows,
        "rule": {"order": "sha256(article id)", "split": "blank lines",
                 "keep": "no line starts with a space, ends with '.'",
                 "min_words": MIN_WORDS, "max_words": MAX_WORDS, "per_article": PER_ARTICLE},
        "files": {
            "data/corpus.jsonl": {"rows": len(corpus), "articles": len({d["title"] for d in corpus}),
                                  "sha256": sha256_file(DATA / "corpus.jsonl")},
            "data/extra.jsonl": {"rows": len(extra), "sha256": sha256_file(DATA / "extra.jsonl")},
        },
        "corpus_words_per_paragraph": {"min": words[0], "median": words[len(words) // 2],
                                       "max": words[-1], "mean": round(sum(words) / len(words), 2)},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["files"], indent=2))
    print(manifest["corpus_words_per_paragraph"])


if __name__ == "__main__":
    main()
