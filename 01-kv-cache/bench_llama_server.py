"""Same sweep as bench_ollama.py, against a llama-server you start yourself.

This separates llama.cpp's prompt-cache behaviour from Ollama's, and lets you
change the checkpoint spacing, which Ollama does not expose. Start a server
on a spare port first (see README), then:

    uv run python bench_llama_server.py --port 18181 --label ollama-like
    uv run python bench_llama_server.py --port 18181 --label dense-256
    uv run python bench_llama_server.py --port 18181 --label ollama-like-primed --prime

The prompt is the same text as bench_ollama.py, wrapped in the ChatML template
by hand (Ollama 0.40.8 also starts its runner with --chat-template chatml).
The server's `timings.cache_n` and `timings.prompt_n` say how many prompt
tokens came from the cache and how many were computed.
"""

import argparse
import datetime
import json
import statistics
import time
import urllib.request
from pathlib import Path

from bench_ollama import (
    POSITIONS,
    REPEATS,
    build_messages,
    chatml,
    context_lines,
    instruction_lines,
    load_average,
    marker,
    stats,
    wait_for_quiet,
)


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def tokens_before_marker(base, messages):
    # Where the changed line starts, in tokens, so the reuse can be checked
    # against the line position and not just against the line index.
    text = chatml(messages)
    cut = text.index("request-id: ")
    count = lambda s: len(json.loads(post(f"{base}/tokenize", {"content": s}).read())["tokens"])
    return count(text[:cut]), count(text)


def complete(base, prompt):
    start = time.perf_counter()
    first = None
    final = None
    with post(f"{base}/completion", {
        "prompt": prompt, "n_predict": 4, "temperature": 0, "seed": 42,
        "cache_prompt": True, "stream": True,
    }) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if first is None:
                first = time.perf_counter() - start
            chunk = json.loads(line[6:])
            if chunk.get("stop"):
                final = chunk
    t = final["timings"]
    return {
        "ttft_ms": round(first * 1000, 1),
        "cache_n": t.get("cache_n"),
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": round(t.get("prompt_ms", 0), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18181)
    ap.add_argument("--label", required=True)
    ap.add_argument("--system-lines", type=int, default=74)
    ap.add_argument("--context-lines", type=int, default=40)
    ap.add_argument("--prime", action="store_true",
                    help="before each block, send the prompt cut just before the changed line")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    system = instruction_lines(args.system_lines)
    user = context_lines(args.context_lines)
    props = json.loads(urllib.request.urlopen(f"{base}/props").read())
    result = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "prime": args.prime,
        "server_build": props.get("build_info"),
        "model_path": props.get("model_path"),
        "repeats": REPEATS,
        "load_log": [],
        "blocks": [],
    }
    wait_for_quiet(result["load_log"])
    complete(base, chatml(build_messages(system, user, 0.0, marker())))

    for position in POSITIONS + [None]:
        uptime = wait_for_quiet(result["load_log"])
        p = 1.0 if position is None else position
        first = build_messages(system, user, p, marker())
        before, total = tokens_before_marker(base, first)
        block = {"position": position, "kind": "identical" if position is None else "position",
                 "marker_token_start": before, "prompt_tokens": total, "uptime": uptime}
        if args.prime and position is not None:
            # A prompt that ends where the changed line starts leaves a checkpoint
            # there; later prompts that diverge at the line can resume from it.
            text = chatml(first)
            block["prime"] = complete(base, text[: text.index("request-id: ")])
        block["warmup"] = complete(base, chatml(first))
        runs = []
        for _ in range(REPEATS):
            msgs = first if position is None else build_messages(system, user, p, marker())
            runs.append(complete(base, chatml(msgs)))
        block["runs"] = runs
        block["uptime_after"] = load_average()[1]
        block["ttft_ms"] = stats([r["ttft_ms"] for r in runs])
        block["prompt_n"] = statistics.median(r["prompt_n"] for r in runs)
        block["cache_n"] = statistics.median(r["cache_n"] for r in runs)
        result["blocks"].append(block)
        label = "identical" if position is None else f"{position:.0%}"
        print(f"{label:>9}  line at token {before:5d}/{total}  cache_n {block['cache_n']:6.0f}  "
              f"prompt_n {block['prompt_n']:6.0f}  TTFT median {block['ttft_ms']['median']:7.1f} ms "
              f"(p10 {block['ttft_ms']['p10']:.0f}, p90 {block['ttft_ms']['p90']:.0f})  | {uptime}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out = Path("results") / f"llama-server-{args.label}-{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
