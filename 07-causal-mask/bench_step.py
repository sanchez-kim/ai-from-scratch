"""Time one training step on CPU and on MPS, for each mask setting.

    nice -n 10 uv run python bench_step.py --steps 200 --repeats 5

Run this on an idle machine. Wall-clock numbers depend on everything else the
machine is doing. Writes results/step-time-<tag>.json with uptime before/after.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from charformer.data import batch, load
from charformer.model import MASKS, Config, CharTransformer


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip().split("load")[-1]


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def time_steps(device, mask, steps, warmup, train_ids, vocab):
    torch.manual_seed(0)
    model = CharTransformer(Config(vocab_size=vocab, mask=mask)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    batches = [batch(train_ids, 32, model.cfg.context, g) for _ in range(warmup + steps)]
    batches = [(x.to(device), y.to(device)) for x, y in batches]  # copy outside the timer

    def step(x, y):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for x, y in batches[:warmup]:
        step(x, y)
    sync(device)
    t0 = time.perf_counter()
    for x, y in batches[warmup:]:
        step(x, y)
    sync(device)
    return (time.perf_counter() - t0) / steps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    chars, train_ids, _ = load()
    devices = [torch.device("cpu")] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    before = uptime()
    rows = []
    for device in devices:
        for mask in MASKS:
            ms = [time_steps(device, mask, args.steps, args.warmup, train_ids, len(chars.chars))
                  for _ in range(args.repeats)]
            rows.append({"device": device.type, "mask": mask, "ms_per_step": ms, "median_ms": statistics.median(ms)})
            print(f"{device.type:4} {mask:7} median {rows[-1]['median_ms']:.1f} ms/step", flush=True)
    out = Path(__file__).parent / "results" / f"step-time-{args.tag}.json"
    out.write_text(json.dumps({
        "steps": args.steps, "warmup": args.warmup, "repeats": args.repeats, "cpu_threads": args.threads,
        "torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
        "rows": rows, "load_before": before, "load_after": uptime(),
    }, indent=1))
    print("wrote", out.name)


if __name__ == "__main__":
    main()
