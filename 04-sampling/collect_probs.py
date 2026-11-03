"""Collect real next-token probabilities from a llama-server you start yourself.

    uv run python collect_probs.py --port 18182

For every token of every passage in passages.py (except the first), the
model reads the text up to that token and the server returns:

1. the top-100 next-token log-probabilities before any sampler runs
   (`n_probs`, `post_sampling_probs: false`): a softmax over the full
   vocabulary at temperature 1;
2. for each sampler setting in CHAINS, the tokens that survive llama.cpp's
   own sampler chain (`post_sampling_probs: true`, with a custom `samplers`
   order). analyze.py compares these with sampling/sampler.py.

The prompt is sent as token ids, so the model always reads my text, not its
own sampled output. Nothing here depends on timing.
"""

import argparse
import datetime
import json
import subprocess
import urllib.request
from pathlib import Path

from passages import PASSAGES

N_PROBS = 100
FIRST_ORDER = ["temperature", "top_k", "top_p"]   # HF transformers order
LAST_ORDER = ["top_k", "top_p", "temperature"]    # llama.cpp default order


def chains():
    out = {}
    for t in (0.7, 1.0, 1.3):
        for p in (0.9, 0.95):
            for label, order in (("first", FIRST_ORDER), ("last", LAST_ORDER)):
                out[f"T{t}-p{p}-{label}"] = {
                    "samplers": order, "temperature": t, "top_k": 40, "top_p": p, "min_p": 0.0}
    # llama.cpp's full default chain (top_k 40, top_p 0.95, min_p 0.05, temp 0.8),
    # and the same numbers with temperature moved to the front.
    out["llamacpp-default"] = {"samplers": ["top_k", "top_p", "min_p", "temperature"],
                               "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}
    out["llamacpp-default-temp-first"] = {"samplers": ["temperature", "top_k", "top_p", "min_p"],
                                          "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}
    return out


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def get(base, path):
    return json.loads(urllib.request.urlopen(base + path, timeout=60).read())


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18182)
    ap.add_argument("--out", default=None)
    ap.add_argument("--passages", default=None, help="comma-separated indices, for a repeat check")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    props = get(base, "/props")
    settings = chains()
    result = {
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "uptime_start": uptime(),
        "server_build": props.get("build_info"),
        "server_default_samplers": props["default_generation_settings"]["params"]["samplers"],
        "model_file": Path(props.get("model_path", "")).name,
        "model_meta": get(base, "/v1/models")["data"][0].get("meta"),
        "n_probs": N_PROBS,
        "chains": settings,
        "positions": [],
    }

    chosen = [int(x) for x in args.passages.split(",")] if args.passages else range(len(PASSAGES))
    for pi in chosen:
        text = PASSAGES[pi]
        tokens = post(base, "/tokenize", {"content": text, "add_special": True})["tokens"]
        for i in range(1, len(tokens)):
            prompt = tokens[:i]
            # temperature 0: greedy pick, but the returned probabilities are a
            # plain softmax of the logits (server README, n_probs).
            r = post(base, "/completion", {
                "prompt": prompt, "n_predict": 1, "n_probs": N_PROBS,
                "temperature": 0, "cache_prompt": True})
            top = r["completion_probabilities"][0]["top_logprobs"]
            pos = {
                "passage": pi, "prefix_len": i, "actual_next": tokens[i],
                # Text for the figures and the worked example.
                "context_tail": post(base, "/detokenize", {"tokens": prompt[-12:]})["content"],
                "actual_next_text": post(base, "/detokenize", {"tokens": [tokens[i]]})["content"],
                "top": [[c["id"], round(c["logprob"], 6)] for c in top],
                "top_text": [c["token"] for c in top[:20]],
                "survivors": {},
            }
            for name, chain in settings.items():
                r = post(base, "/completion", {
                    "prompt": prompt, "n_predict": 1, "n_probs": N_PROBS,
                    "post_sampling_probs": True, "seed": 42, "cache_prompt": True, **chain})
                pos["survivors"][name] = [c["id"] for c in r["completion_probabilities"][0]["top_probs"]]
            result["positions"].append(pos)
        print(f"passage {pi}: {len(tokens) - 1} positions", flush=True)

    result["uptime_end"] = uptime()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out = Path(args.out or f"results/probs-qwen3.5_4b-q4_K_M-{stamp}.json")
    out.write_text(json.dumps(result, separators=(",", ":")))
    print("wrote", out, len(result["positions"]), "positions")


if __name__ == "__main__":
    main()
