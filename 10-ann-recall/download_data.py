"""Download two parquet shards of Simple English Wikipedia embeddings into data/.

Source: timescale/wikipedia-22-12-simple-embeddings (Apache-2.0), a re-export of
Cohere's wikipedia-22-12-simple-embeddings. The Hugging Face parquet conversion
splits it into shards of 60,000 rows; shards 0000 and 0001 give 120,000 rows,
which covers the 100,000 corpus vectors and 1,000 queries this folder uses.

The revision is pinned, and each file is checked against its sha256.
"""
import hashlib
import json
import pathlib
import time
import urllib.request

REPO = "timescale/wikipedia-22-12-simple-embeddings"
REVISION = "2f97d23433103634cff31a8c519f0a6f7445aadb"  # refs/convert/parquet
SHARDS = {
    "default/train/0000.parquet": "0197589f68b57e5595feed0ec96b1b8b94a4bc1b6c73bb7264060e321d5dcab8",
    "default/train/0001.parquet": "6983bfde4f7e54006d3eda2fa59001e05490fdfce4ed6469bdac27cd7802a549",
}
HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    manifest = {"repo": REPO, "revision": REVISION, "files": []}
    for name, expected in SHARDS.items():
        out = DATA / name.split("/")[-1]
        if not out.exists() or sha256(out) != expected:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            print("downloading", url)
            tmp = out.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(out)
        got = sha256(out)
        if got != expected:
            raise SystemExit(f"sha256 mismatch for {out.name}: {got}")
        manifest["files"].append({"path": f"data/{out.name}", "bytes": out.stat().st_size, "sha256": got})
        print("ok", out.name, got)
    manifest["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (HERE / "results" / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
