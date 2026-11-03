"""Generate two synthetic query sets with a local model through Ollama.

    uv run python make_queries.py            # needs Ollama on localhost:11434

For 500 paragraphs, picked with random.Random(8).sample, the model writes:

- a "lexical" question: one the paragraph answers, words may be copied;
- a "paraphrase" question: one the paragraph answers, written without the
  paragraph's own words, so that word-matching search has a harder time.

Each row of results/queries.jsonl: query_id, kind, query, gold_id. The
paragraph with gold_id is the one known answer. Requests go one at a time.
The script resumes: rows already in the file are kept.
results/queries-meta.json records the prompts, model digest and options.
"""

import json
import random
import subprocess
import time
import urllib.request
from pathlib import Path

from bm25.data import load_corpus

HERE = Path(__file__).parent
OUT = HERE / "results" / "queries.jsonl"
META = HERE / "results" / "queries-meta.json"
OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:4b-q4_K_M"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 80}
N_QUERIES, SAMPLE_SEED = 500, 8

HEADER = 'Here is a paragraph from a Simple English Wikipedia article titled "{title}".\n\n{text}\n\n'
PROMPTS = {
    "lexical": HEADER + (
        "Write one question that a person might type into a search engine, "
        "which this paragraph answers. Reply with the question only."),
    "paraphrase": HEADER + (
        "Write one question that this paragraph answers. Do not use any noun, verb, "
        "adjective or name that appears in the paragraph or the title. Use synonyms "
        "and descriptions instead. Reply with the question only."),
}


def post(path, body):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def clean(text):
    """First non-empty line, without surrounding quotes or a 'Question:' label."""
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    if line.lower().startswith("question:"):
        line = line[9:].strip()
    return line.strip('"').strip("*").strip()


def model_digest():
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=30) as resp:
        for m in json.load(resp)["models"]:
            if m["name"] == MODEL:
                return m["digest"]


def main():
    corpus = load_corpus()
    sample = sorted(random.Random(SAMPLE_SEED).sample(range(len(corpus)), N_QUERIES))
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            row = json.loads(line)
            done[row["query_id"]] = row
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for kind, template in PROMPTS.items():
        for n, idx in enumerate(sample):
            qid = f"{kind[0]}{n:03d}"
            if qid in done:
                continue
            doc = corpus[idx]
            r = post("/api/generate", {"model": MODEL, "stream": False, "think": False,
                                       "prompt": template.format(title=doc["title"], text=doc["text"]),
                                       "options": OPTIONS})
            done[qid] = {"query_id": qid, "kind": kind, "query": clean(r["response"]),
                         "gold_id": doc["id"]}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(done[qid], ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(kind, n, done[qid]["query"], flush=True)
    rows = sorted(done.values(), key=lambda r: r["query_id"])
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with urllib.request.urlopen(OLLAMA + "/api/version", timeout=30) as resp:
        version = json.load(resp)["version"]
    META.write_text(json.dumps({
        "model": MODEL, "model_digest": model_digest(), "ollama_version": version,
        "options": OPTIONS, "think": False, "sample": {"n": N_QUERIES, "seed": SAMPLE_SEED},
        "prompts": PROMPTS, "cleaning": "first non-empty line, strip 'Question:' label, quotes and *",
        "last_run_started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        "rows": {k: sum(r["kind"] == k for r in rows) for k in PROMPTS},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
