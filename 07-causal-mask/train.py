"""Train the character transformer with one mask setting and record everything.

    nice -n 10 uv run python train.py --mask causal --device cpu --steps 3000

Every mask setting starts from the same weights and sees the same batches, so
the mask is the only difference. Writes results/train-<mask>-<device>-<tag>.json.
Nothing here is timed; the speed comparison is bench_step.py.
"""

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from charformer.data import batch, load
from charformer.model import Config, CharTransformer, causal_mask, generate

PROMPT = "Elizabeth"
SNIPPET = "Bennet"  # six characters for the 6x6 attention grid in the figure


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip().split("load")[-1]


@torch.no_grad()
def evaluate(model, x, y):
    """Three numbers on fixed validation windows.

    full: the loss scored the way the model trained (all positions, its own mask).
    last: the loss at the last position only. That position has no future in
          the input, which is the situation in generation.
    """
    model.eval()
    logits, full = model(x, y)
    last = F.cross_entropy(logits[:, -1, :], y[:, -1])
    model.train()
    return full.item(), last.item()


@torch.no_grad()
def honest_by_position(model, x, y):
    """Loss at position t when the input stops at t, for every t.

    This is how generation sees the text: x[0..t] and nothing after it.
    Also returns the as-trained loss at each position for comparison.
    """
    model.eval()
    logits, _ = model(x)
    trained = F.cross_entropy(logits.transpose(1, 2), y, reduction="none").mean(0)
    honest = []
    for t in range(x.shape[1]):
        lt, _ = model(x[:, :t + 1])
        honest.append(F.cross_entropy(lt[:, -1, :], y[:, t]).item())
    return trained.tolist(), honest


@torch.no_grad()
def attention_stats(model, x):
    """Per layer and head: mean weight on key t+1 (the answer) and row sums."""
    model.eval()
    model(x)
    t = x.shape[1]
    out = []
    for layer, w in enumerate(model.attention_maps()):  # w: (b, h, t, t)
        nxt = torch.diagonal(w, offset=1, dim1=-2, dim2=-1)  # weight of row i on key i+1
        sums = w.sum(-1)  # (b, h, t)
        for head in range(w.shape[1]):
            out.append({
                "layer": layer, "head": head,
                "mean_weight_on_next": nxt[:, head].mean().item(),
                "mean_future_weight_per_row": w[:, head].masked_fill(~causal_mask(t, w.device), 0).sum(-1)[:, :-1].mean().item(),
                "row_sum_by_position": sums[:, head].mean(0).tolist(),
            })
    return out


@torch.no_grad()
def grids(model, chars, device):
    """The 6x6 weights of every layer and head on SNIPPET, for the figure."""
    idx = torch.tensor([chars.encode(SNIPPET)], device=device)
    model.eval()
    model(idx)
    return [[w[0, h].cpu().tolist() for h in range(w.shape[1])] for w in model.attention_maps()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", choices=["causal", "none", "after"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    torch.set_num_threads(2)
    load_before = uptime()
    chars, train_ids, val_ids = load()
    cfg = Config(vocab_size=len(chars.chars), mask=args.mask)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)  # same initial weights for every mask
    model = CharTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data_gen = torch.Generator().manual_seed(args.seed)  # same batches for every mask
    vx, vy = batch(val_ids, 256, cfg.context, torch.Generator().manual_seed(1234))
    vx, vy = vx.to(device), vy.to(device)

    init_attn = attention_stats(model, vx)
    curve, running = [], []
    for step in range(1, args.steps + 1):
        x, y = batch(train_ids, args.batch, cfg.context, data_gen)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % args.eval_every == 0 or step == 1:
            full, last = evaluate(model, vx, vy)
            curve.append({"step": step, "train": sum(running) / len(running), "val_full": full, "val_last": last})
            running = []
            print(f"{args.mask} step {step}: train {curve[-1]['train']:.4f} val_full {full:.4f} val_last {last:.4f}", flush=True)

    trained_pos, honest_pos = honest_by_position(model, vx, vy)
    attn = attention_stats(model, vx)
    copy_head = max(attn, key=lambda a: a["mean_weight_on_next"])
    g = torch.Generator().manual_seed(0)
    prompt = torch.tensor([chars.encode(PROMPT)], device=device)
    sampled = chars.decode(generate(model, prompt, 300, temperature=1.0, generator=g)[0].tolist())
    greedy = chars.decode(generate(model, prompt, 200, temperature=0)[0].tolist())

    result = {
        "mask": args.mask, "device": args.device, "tag": args.tag,
        "torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "config": vars(cfg), "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "data": {"file": "data/pride-and-prejudice.txt", "train_chars": len(train_ids), "val_chars": len(val_ids),
                 "vocab": len(chars.chars), "uniform_loss_nats": math.log(len(chars.chars))},
        "curve": curve,
        "final": {"val_full": curve[-1]["val_full"], "val_last": curve[-1]["val_last"],
                  "honest_mean": sum(honest_pos) / len(honest_pos),
                  "trained_mean": sum(trained_pos) / len(trained_pos)},
        "by_position": {"trained": trained_pos, "honest": honest_pos},
        "attention_init": init_attn, "attention_final": attn,
        "copy_head": {"layer": copy_head["layer"], "head": copy_head["head"],
                      "mean_weight_on_next": copy_head["mean_weight_on_next"]},
        "grids": {"snippet": SNIPPET, "weights_by_layer_head": grids(model, chars, device)},
        "samples": {"prompt": PROMPT, "temperature_1": sampled, "greedy": greedy},
        "load_before": load_before, "load_after": uptime(),
    }
    name = f"{args.mask}-{args.device}-{args.tag}"
    ckpt = Path(__file__).parent / "checkpoints"
    ckpt.mkdir(exist_ok=True)  # weights are not committed; sample_check.py reads them
    torch.save({"config": vars(cfg), "chars": chars.chars, "state": model.state_dict()}, ckpt / f"{name}.pt")
    out = Path(__file__).parent / "results" / f"train-{name}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("wrote", out.name)


if __name__ == "__main__":
    main()
