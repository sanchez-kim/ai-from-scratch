"""Download the small corpora this folder uses into data/ (gitignored).

    uv run python download_data.py

Three sources, all with clear licences:

1. Wikipedia prose: the same topics from Simple English Wikipedia and Korean
   Wikipedia (text CC BY-SA 4.0). Each Simple English article is fetched with
   its Korean language link, so both languages cover the same subjects.
   Topics with no Korean article are skipped. Every 4th usable topic goes to
   the held-out test set; the rest is training text.
2. Python code: a few modules from CPython's standard library at tag v3.12.7
   (PSF License 2.0). Used only for evaluation, never for training.
3. FLORES-200 devtest (CC BY-SA 4.0): 1,012 sentences translated into many
   languages. Only English and Korean are kept. Same meaning on both sides.

Wikipedia pages change. The script records each page's revision ID and the
sha256 of the text it saved in results/corpus-manifest.json, so a later run
can tell whether it got the same text.
"""

import hashlib
import io
import json
import re
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
UA = "ai-from-scratch-bpe/0.1 (https://github.com/sanchez-kim/ai-from-scratch)"
MAX_CHARS = 6000  # cap per article, so a few long pages cannot dominate a corpus

TOPICS = """
Water; Sun; Moon; Earth; Mars; Jupiter; Star; Galaxy; Gravity; Light; Sound;
Electricity; Magnet; Atom; Molecule; Oxygen; Carbon; Iron; Gold; Salt;
Volcano; Earthquake; Ocean; River; Mountain; Desert; Forest; Island; Glacier; Rain;
Snow; Wind; Cloud; Weather; Climate; Season; Tree; Flower; Grass; Leaf;
Seed; Fruit; Apple; Banana; Orange (fruit); Grape; Rice; Wheat; Bread; Cheese;
Milk; Egg; Coffee; Tea; Sugar; Chocolate; Honey; Soup; Pizza; Noodle;
Dog; Cat; Horse; Cow; Pig; Sheep; Chicken; Fish; Bird; Bee;
Ant; Butterfly; Spider; Snake; Frog; Lion; Tiger; Elephant; Whale; Shark;
Human; Brain; Heart; Blood; Bone; Skin; Eye; Tooth; Sleep; Dream;
Language; Alphabet; Book; Library; School; University; Teacher; Mathematics; Number; Geometry;
Music; Piano; Guitar; Song; Painting; Photography; Film; Theatre; Dance; Poetry;
Computer; Internet; Telephone; Television; Radio; Camera; Robot; Software; Video game; Email;
Car; Bicycle; Train; Airplane; Ship; Bridge; Road; City; Village; House;
Money; Bank; Market; Shop; Factory; Farm; Hospital; Doctor; Medicine; Vaccine;
Football; Basketball; Tennis; Swimming; Chess; Olympic Games; Running; Baseball; Golf; Boxing;
History; Science; Physics; Chemistry; Biology; Philosophy; Religion; Law; Government; Democracy;
Family; Friendship; Love; Fear; Happiness; Time; Clock; Calendar; Year; Week
"""


def get_json(url, params):
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_page(host, title, want_ko_link):
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "titles": title, "prop": "extracts|revisions", "explaintext": 1, "rvprop": "ids",
    }
    if want_ko_link:
        params["prop"] += "|langlinks"
        params["lllang"] = "ko"
    page = get_json(f"https://{host}/w/api.php", params)["query"]["pages"][0]
    if page.get("missing"):
        return None
    ko = page.get("langlinks", [{}])[0].get("title") if want_ko_link else None
    return {"title": page["title"], "revid": page["revisions"][0]["revid"],
            "text": clean(page.get("extract", "")), "ko_title": ko}


def clean(text):
    # Drop section headings ("== History ==") and blank lines; keep prose lines.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"=+.*=+", ln)]
    out, size = [], 0
    for ln in lines:
        if size + len(ln) > MAX_CHARS:
            break
        out.append(ln)
        size += len(ln) + 1
    return "\n".join(out)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download_wiki(manifest):
    topics = [t.strip() for t in TOPICS.replace("\n", " ").split(";") if t.strip()]
    pairs = []
    for title in topics:
        en = fetch_page("simple.wikipedia.org", title, want_ko_link=True)
        time.sleep(0.2)
        if not en or not en["ko_title"] or len(en["text"]) < 500:
            continue
        ko = fetch_page("ko.wikipedia.org", en["ko_title"], want_ko_link=False)
        time.sleep(0.2)
        if not ko or len(ko["text"]) < 300:
            continue
        pairs.append((en, ko))
    for lang in ("en", "ko"):
        for split in ("train", "test"):
            docs = [p[0 if lang == "en" else 1] for i, p in enumerate(pairs)
                    if (i % 4 == 3) == (split == "test")]
            text = "\n\n".join(d["text"] for d in docs) + "\n"
            path = DATA / f"wiki-{lang}-{split}.txt"
            path.write_text(text, encoding="utf-8")
            manifest["files"][path.name] = {
                "sha256": sha(text), "chars": len(text), "bytes": len(text.encode("utf-8")),
                "articles": [{"title": d["title"], "revid": d["revid"], "sha256": sha(d["text"])}
                             for d in docs],
            }
    manifest["topics_requested"] = len(topics)
    manifest["topics_used"] = len(pairs)


CPYTHON_TAG = "v3.12.7"
CPYTHON_FILES = ["textwrap.py", "heapq.py", "bisect.py", "fractions.py", "shlex.py",
                 "string.py", "colorsys.py", "statistics.py", "calendar.py", "difflib.py"]


def download_code(manifest):
    parts = []
    for name in CPYTHON_FILES:
        url = f"https://raw.githubusercontent.com/python/cpython/{CPYTHON_TAG}/Lib/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            parts.append(resp.read().decode("utf-8"))
    text = "\n".join(parts)
    path = DATA / "python-code.txt"
    path.write_text(text, encoding="utf-8")
    manifest["files"][path.name] = {
        "sha256": sha(text), "chars": len(text), "bytes": len(text.encode("utf-8")),
        "source": f"CPython {CPYTHON_TAG} Lib/: " + ", ".join(CPYTHON_FILES),
    }


FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_KEEP = {"devtest/eng_Latn.devtest": "flores-eng.txt",
               "devtest/kor_Hang.devtest": "flores-kor.txt"}


def download_flores(manifest):
    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()  # about 25 MB, kept in memory only
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            for suffix, out in FLORES_KEEP.items():
                if member.name.endswith(suffix):
                    text = tar.extractfile(member).read().decode("utf-8")
                    (DATA / out).write_text(text, encoding="utf-8")
                    manifest["files"][out] = {
                        "sha256": sha(text), "chars": len(text),
                        "bytes": len(text.encode("utf-8")), "lines": text.count("\n"),
                        "source": f"{FLORES_URL} :: {member.name}",
                    }
    manifest["flores_archive_sha256"] = hashlib.sha256(blob).hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    manifest = {
        "downloaded": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "licences": {
            "wiki-*": "Wikipedia text, CC BY-SA 4.0 (Simple English and Korean Wikipedia)",
            "python-code.txt": "CPython standard library, PSF License 2.0",
            "flores-*": "FLORES-200, CC BY-SA 4.0",
        },
        "files": {},
    }
    download_wiki(manifest)
    download_code(manifest)
    download_flores(manifest)
    out = HERE / "results" / "corpus-manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, info in manifest["files"].items():
        print(f"{name:22s} {info['chars']:>8,} chars {info['bytes']:>8,} bytes")


if __name__ == "__main__":
    main()
