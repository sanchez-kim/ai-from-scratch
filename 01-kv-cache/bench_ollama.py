"""Where does one changed line sit, and what does it cost? Measured on Ollama.

    uv run python bench_ollama.py                  # writes results/ollama-<date>-<time>.json
    uv run python bench_ollama.py --prime         # checkpoint before the line first

A ~3,000-token chat prompt (long system prompt + a context document) carries
one line, `request-id: <uuid>`. For each position of that line (0, 25, 50, 75,
100 % of the prompt's lines) the script sends one warm-up request, then
REPEATS requests that change only the uuid. Each of those differs from the
request before it in that one line, so whatever the server cached before the
line is reusable and everything after it is not.

Two controls: `identical` resends the exact previous prompt (best case), and
`cold` puts a fresh line before everything (nothing reusable).

It never touches the server's configuration. Per-request options only:
num_ctx, temperature, seed, num_predict, think=false, keep_alive.
If the server is Ollama's bundled llama-server, the script also copies the
llama-server lines that each request wrote to ~/.ollama/logs/server.log,
which state how many tokens were taken from the cache.
"""

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

HOST = "http://127.0.0.1:11434"
SERVER_LOG = Path.home() / ".ollama" / "logs" / "server.log"
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
REPEATS = 5
MAX_LOAD = 4.0
OPTIONS = {"num_ctx": 8192, "temperature": 0, "seed": 42, "num_predict": 4}

TOPICS = [
    "refunds", "shipping delays", "account recovery", "invoices", "data export",
    "API rate limits", "password resets", "team seats", "outages", "webhooks",
    "billing currency", "tax receipts", "plan downgrades", "SSO setup", "audit logs",
]
ACTIONS = [
    "confirm the order number before you promise anything",
    "link the matching help-center article instead of pasting it",
    "ask for the workspace ID if the user has more than one",
    "never quote a delivery date that the carrier has not confirmed",
    "escalate to a human if the amount is above the self-serve limit",
    "summarize the user's request in one sentence before answering",
    "say plainly when a feature does not exist rather than suggesting a workaround",
    "keep answers under 120 words unless the user asks for detail",
]


def instruction_lines(n):
    # Deterministic filler that reads like a real system prompt, so the
    # tokenizer sees ordinary English and not one repeated string.
    return [
        f"Rule {i + 1}: when the user asks about {TOPICS[i % len(TOPICS)]}, "
        f"{ACTIONS[(i * 7) % len(ACTIONS)]}."
        for i in range(n)
    ]


def context_lines(n):
    return [
        f"Doc {i + 1}: the {TOPICS[(i * 4) % len(TOPICS)]} page was last reviewed in week "
        f"{(i * 13) % 52 + 1} and applies to plan tier {(i % 3) + 1}."
        for i in range(n)
    ]


def build_messages(system, user, position, marker):
    lines = system + user
    at = round(position * len(lines))
    lines = lines[:at] + [marker] + lines[at:]
    # A line inserted exactly at the boundary goes to the system message.
    cut = len(system) + (1 if at <= len(system) else 0)
    return [
        {"role": "system", "content": "\n".join(lines[:cut])},
        {"role": "user", "content": "\n".join(lines[cut:]) + "\n\nReply with OK."},
    ]


def chatml(messages):
    # The ChatML text Ollama 0.40.8 hands its llama.cpp runner for this model
    # (the runner is started with --chat-template chatml). An empty think
    # block is what think=false renders.
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prime(model, messages):
    """Send the prompt cut just before the changed line, as a raw prompt.

    llama.cpp checkpoints a hybrid model's recurrent state near the end of
    each prompt. A prompt that ends where the changed line starts leaves a
    checkpoint there, and later prompts that diverge at that line can
    resume from it instead of from token 0.
    """
    text = chatml(messages)
    # Same OPTIONS (above all num_ctx) as the real requests: with a different
    # num_ctx Ollama starts a new runner process, and the checkpoint stays
    # behind in the old one.
    body = json.dumps({
        "model": model, "prompt": text[: text.index("request-id: ")], "raw": True,
        "stream": False, "keep_alive": "10m", "options": {**OPTIONS, "num_predict": 1},
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    offset = server_log_size()
    reply = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return {"prompt_eval_count": reply.get("prompt_eval_count"),
            "prompt_eval_ms": round(reply.get("prompt_eval_duration", 0) / 1e6, 1),
            "server": summarize_server_lines(server_log_since(offset))}


def load_average():
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    one_min = float(re.search(r"load averages?: ([\d.]+)", out).group(1))
    return one_min, out


def wait_for_quiet(log):
    waited = 0
    while True:
        one_min, line = load_average()
        if one_min <= MAX_LOAD or waited >= 1200:
            if waited:
                log.append({"waited_s": waited, "uptime": line})
            return line
        log.append({"waiting": True, "uptime": line})
        print("load high, waiting:", line)
        time.sleep(120)
        waited += 120


def server_log_size():
    try:
        return SERVER_LOG.stat().st_size
    except FileNotFoundError:
        return None


def server_log_since(offset):
    if offset is None:
        return []
    time.sleep(0.3)  # llama-server writes its timing lines just after the reply
    with open(SERVER_LOG, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    keep = ("slot ", "srv ", "[GIN]", "level=")
    return [ln for ln in text.splitlines() if ln.startswith(keep)][-80:]


def summarize_server_lines(lines):
    """Pull the cache facts out of llama-server's own log lines."""
    facts = {"cached_n_tokens": [], "restored": [], "erased": 0, "prompt_eval": None}
    for ln in lines:
        m = re.search(r"cached n_tokens = (\d+)", ln)
        if m:
            facts["cached_n_tokens"].append(int(m.group(1)))
        m = re.search(r"restored context checkpoint .*?n_tokens = (\d+)", ln)
        if m:
            facts["restored"].append(int(m.group(1)))
        if "erased invalidated context checkpoint" in ln:
            facts["erased"] += 1
        m = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", ln)
        if m:
            facts["prompt_eval"] = {"ms": float(m.group(1)), "tokens": int(m.group(2))}
    return facts


def chat(model, messages):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": OPTIONS,
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    offset = server_log_size()
    start = time.perf_counter()
    first = None
    final = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            if first is None:
                first = time.perf_counter() - start
            chunk = json.loads(raw)
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - start
    server = server_log_since(offset)
    return {
        "ttft_ms": round(first * 1000, 1),
        "wall_ms": round(total * 1000, 1),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_ms": round(final.get("prompt_eval_duration", 0) / 1e6, 1),
        "load_ms": round(final.get("load_duration", 0) / 1e6, 1),
        "eval_count": final.get("eval_count"),
        "server": summarize_server_lines(server),
        "server_lines": server,
    }


def stats(values):
    values = sorted(values)
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return {"median": statistics.median(values), "p10": deciles[0], "p90": deciles[-1]}


def marker():
    return f"request-id: {uuid.uuid4()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    ap.add_argument("--system-lines", type=int, default=74)
    ap.add_argument("--context-lines", type=int, default=40)
    ap.add_argument("--prime", action="store_true",
                    help="before each position block, send the prompt cut before the line")
    args = ap.parse_args()

    system = instruction_lines(args.system_lines)
    user = context_lines(args.context_lines)
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    show = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{HOST}/api/show", data=json.dumps({"model": args.model}).encode())).read())
    digest = next((m["digest"] for m in json.loads(urllib.request.urlopen(
        f"{HOST}/api/tags").read())["models"] if m["name"] == args.model), None)

    result = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ollama_version": version,
        "model": args.model,
        "digest": digest,
        "details": show.get("details"),
        "options": OPTIONS,
        "repeats": REPEATS,
        "prime": args.prime,
        "system_lines": args.system_lines,
        "context_lines": args.context_lines,
        "load_log": [],
        "blocks": [],
    }

    # Load the model with this num_ctx once, so no block pays for a reload.
    wait_for_quiet(result["load_log"])
    result["load_request"] = chat(args.model, build_messages(system, user, 0.0, marker()))

    plan = [("position", p) for p in POSITIONS] + [("identical", None), ("cold", None)]
    for kind, position in plan:
        uptime = wait_for_quiet(result["load_log"])
        block = {"kind": kind, "position": position, "uptime": uptime, "runs": []}
        if kind == "position":
            if args.prime:
                block["prime"] = prime(args.model, build_messages(system, user, position, marker()))
            block["warmup"] = chat(args.model, build_messages(system, user, position, marker()))
            for _ in range(REPEATS):
                block["runs"].append(chat(args.model, build_messages(system, user, position, marker())))
        elif kind == "identical":
            fixed = build_messages(system, user, 1.0, marker())
            block["warmup"] = chat(args.model, fixed)
            for _ in range(REPEATS):
                block["runs"].append(chat(args.model, fixed))
        else:
            for _ in range(REPEATS):
                fresh = build_messages([marker()] + system, user, 1.0, marker())
                block["runs"].append(chat(args.model, fresh))
        block["uptime_after"] = load_average()[1]
        for key in ("ttft_ms", "prompt_eval_ms"):
            block[key] = stats([r[key] for r in block["runs"]])
        block["prompt_eval_count"] = sorted({r["prompt_eval_count"] for r in block["runs"]})
        block["server_prompt_tokens"] = sorted({
            r["server"]["prompt_eval"]["tokens"] for r in block["runs"] if r["server"]["prompt_eval"]})
        result["blocks"].append(block)
        label = f"{position:.0%}" if position is not None else kind
        print(f"{label:>9}  TTFT median {block['ttft_ms']['median']:7.1f} ms "
              f"(p10 {block['ttft_ms']['p10']:.0f}, p90 {block['ttft_ms']['p90']:.0f})  "
              f"prompt_eval_count {block['prompt_eval_count']}  "
              f"server recomputed {block['server_prompt_tokens']}  | {uptime}")

    Path("results").mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    safe = args.model.replace(":", "_").replace("/", "_")
    tag = "-primed" if args.prime else ""
    out = Path("results") / f"ollama-{safe}{tag}-{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
