"""Write figures/mask-grid-v2.svg: the animated attention grids and loss curves.

    uv run python figures/make_mask_grid.py results/train-causal-cpu-main.json results/train-none-cpu-main.json

Left: one head of the trained causal model reading the six characters
"Bennet". The upper triangle (the future) fades to -inf. Right: the same layer
and head of the model trained without a mask. That head is the one with the
most weight on the next character, and the cells (t, t+1) are outlined.
Below: validation loss as trained for both models, then the no-mask model's
loss when each position sees only its past.

CSS animation only, so it plays inside an <img> tag. The static attributes are
the final frame, so prefers-reduced-motion turns the animation off and shows it.
"""

import json
import sys
from pathlib import Path

W, H = 460, 576
CELL = 28
LEFT, RIGHT, TOP = 46, 270, 128
DUR = 12.0
PX0, PX1, PY0, PY1 = 46, 400, 388, 528   # loss panel: x range and y range in pixels
YMAX = 4.5
# Sequential blue steps for attention weight (palette v2). Dark mode reverses
# the order so large weights stay the most visible cells.
STEPS_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
BINS = [0.05, 0.15, 0.3, 0.5, 0.75]


def step(w):
    return sum(w >= b for b in BINS)


def kf_show(name, t_in, fade=0.5):
    a, b = t_in / DUR * 100, (t_in + fade) / DUR * 100
    return f"@keyframes {name} {{ 0% {{opacity:0}} {a:.2f}% {{opacity:0}} {b:.2f}% {{opacity:1}} 100% {{opacity:1}} }}"


def kf_draw(name, t_in, length, span=2.0):
    a, b = t_in / DUR * 100, (t_in + span) / DUR * 100
    return (f"@keyframes {name} {{ 0% {{stroke-dashoffset:{length:.0f}}} {a:.2f}% {{stroke-dashoffset:{length:.0f}}} "
            f"{b:.2f}% {{stroke-dashoffset:0}} 100% {{stroke-dashoffset:0}} }}")


def grid_svg(x0, weights, chars, masked, outline_next, ids):
    out = []
    for i, c in enumerate(chars):
        out.append(f'<text class="ink c" x="{x0 - 12}" y="{TOP + i * CELL + CELL / 2 + 5}">{c}</text>')
        out.append(f'<text class="ink c" x="{x0 + i * CELL + CELL / 2}" y="{TOP - 8}">{c}</text>')
    for i in range(6):
        for j in range(6):
            x, y = x0 + j * CELL, TOP + i * CELL
            w = weights[i][j]
            out.append(f'<rect class="w{step(w)}" x="{x + 1}" y="{y + 1}" width="{CELL - 2}" height="{CELL - 2}"/>')
            if masked and j > i:
                out.append(f'<g id="{ids}m{i}{j}" class="a"><rect class="blk" x="{x + 1}" y="{y + 1}" '
                           f'width="{CELL - 2}" height="{CELL - 2}"/>'
                           f'<text class="sec c sm" x="{x + CELL / 2}" y="{y + CELL / 2 + 4}">-∞</text></g>')
            if outline_next and j == i + 1:
                out.append(f'<rect id="{ids}o{i}" class="a ol" x="{x + 1.5}" y="{y + 1.5}" width="{CELL - 3}" height="{CELL - 3}"/>')
    return "".join(out)


def path(curve, key):
    pts = []
    for p in curve:
        x = PX0 + (p["step"] / curve[-1]["step"]) * (PX1 - PX0)
        y = PY1 - min(p[key], YMAX) / YMAX * (PY1 - PY0)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts), pts[-1]


def main():
    causal = json.loads(Path(sys.argv[1]).read_text())
    nomask = json.loads(Path(sys.argv[2]).read_text())
    assert causal["mask"] == "causal" and nomask["mask"] == "none"
    layer, head = nomask["copy_head"]["layer"], nomask["copy_head"]["head"]
    chars = nomask["grids"]["snippet"]
    gc = causal["grids"]["weights_by_layer_head"][layer][head]
    gn = nomask["grids"]["weights_by_layer_head"][layer][head]
    next_w = [gn[i][i + 1] for i in range(5)]

    kfs, rules = [], []
    for i in range(6):
        for j in range(i + 1, 6):
            kfs.append(kf_show(f"m{i}{j}", 0.8 + 0.12 * (i + j)))
            rules.append(f"#cm{i}{j}{{animation-name:m{i}{j}}}")
    for i in range(5):
        kfs.append(kf_show(f"o{i}", 5.0 + 0.25 * i, fade=0.3))
        rules.append(f"#no{i}{{animation-name:o{i}}}")
    kfs += [kf_show("gr", 3.6), kf_show("lc", 1.5), kf_show("ln", 3.8), kf_show("lh", 7.6)]
    rules += ["#gr{animation-name:gr}", "#lc{animation-name:lc}", "#ln{animation-name:ln}", "#lh{animation-name:lh}"]
    lines = []
    for name, data, key, cls, t_in, label in [
        ("pc", causal, "val_full", "l1", 1.5, f'causal {causal["final"]["val_full"]:.3f}'),
        ("pn", nomask, "val_full", "l2", 3.8, f'no mask {nomask["final"]["val_full"]:.3f}'),
        ("ph", nomask, "val_last", "l2 dash", 7.6, f'no mask, past only {nomask["final"]["val_last"]:.3f}'),
    ]:
        pts, end = path(data["curve"], key)
        kfs.append(kf_draw(name, t_in, 600))
        rules.append(f"#{name}{{animation-name:{name}}}")
        lines.append((name, pts, end, cls, label))

    css = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sec{{fill:#52514e}} .mut{{fill:#898781}}
  .grid{{stroke:#e1e0d9;stroke-width:1}} .base{{stroke:#c3c2b7;stroke-width:1}} .blk{{fill:#c3c2b7}}
  .w0{{fill:#cde2fb}} .w1{{fill:#9ec5f4}} .w2{{fill:#6da7ec}} .w3{{fill:#3987e5}} .w4{{fill:#256abf}} .w5{{fill:#184f95}}
  .s1{{fill:#2a78d6}} .s2{{fill:#eb6834}}
  .ol{{fill:none;stroke:#eb6834;stroke-width:2.5}}
  .l1{{fill:none;stroke:#2a78d6;stroke-width:2}} .l2{{fill:none;stroke:#eb6834;stroke-width:2}}
  .dash{{stroke-dasharray:5 4}} .draw{{stroke-dasharray:600 600}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .h{{font-size:15px;font-weight:600}} .c{{text-anchor:middle}} .sm{{font-size:13px}}
  .a{{animation-duration:{DUR}s;animation-iteration-count:infinite;animation-timing-function:ease-in-out}}
  {' '.join(rules)}
  {' '.join(kfs)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sec{{fill:#c3c2b7}} .mut{{fill:#898781}}
    .grid{{stroke:#2c2c2a}} .base{{stroke:#383835}} .blk{{fill:#383835}}
    .w0{{fill:#184f95}} .w1{{fill:#256abf}} .w2{{fill:#3987e5}} .w3{{fill:#6da7ec}} .w4{{fill:#9ec5f4}} .w5{{fill:#cde2fb}}
    .s1{{fill:#3987e5}} .s2{{fill:#d95926}} .ol{{stroke:#d95926}}
    .l1{{stroke:#3987e5}} .l2{{stroke:#d95926}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    yticks = "".join(
        f'<line class="grid" x1="{PX0}" x2="{PX1}" y1="{PY1 - v / YMAX * (PY1 - PY0):.1f}" y2="{PY1 - v / YMAX * (PY1 - PY0):.1f}"/>'
        f'<text class="mut" x="{PX0 - 6}" y="{PY1 - v / YMAX * (PY1 - PY0) + 4:.1f}" text-anchor="end">{v}</text>'
        for v in (0, 1, 2, 3, 4))
    curve_svg = ""
    for name, pts, end, cls, label in lines:
        ex, ey = map(float, end.split(","))
        group = {"pc": "lc", "pn": "ln", "ph": "lh"}[name]
        lab_y = {"pc": 18, "pn": -8, "ph": -8}[name]
        curve_svg += (f'<polyline id="{name}" class="a draw {cls}" points="{pts}"/>'
                      f'<text id="{group}" class="a sec" x="{PX1}" y="{ey + lab_y:.1f}" text-anchor="end">{label}</text>')
    desc = (f"Two 6 by 6 attention grids for the characters {chars}, layer {layer + 1}, head {head + 1}. "
            f"In the causal model every cell above the diagonal is -inf and gets weight 0. "
            f"In the model trained without a mask the same head puts weight "
            + ", ".join(f"{w:.2f}" for w in next_w)
            + " on the next character. Validation loss as trained after 5000 steps: causal "
            f'{causal["final"]["val_full"]:.2f}, no mask {nomask["final"]["val_full"]:.3f} nats. '
            f'No-mask loss at a position that sees only its past: {nomask["final"]["val_last"]:.2f} nats.')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Attention with and without a causal mask</title>
<desc>{desc}</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="{LEFT - 24}" y="26">"{chars}", layer {layer + 1}, head {head + 1}</text>
<text class="sec" x="{LEFT - 24}" y="46">each row reads the columns</text>
<text class="sec" x="{LEFT - 24}" y="70">weight 0</text>
{"".join(f'<rect class="w{i}" x="{LEFT + 46 + i * 15}" y="60" width="13" height="13"/>' for i in range(6))}
<text class="sec" x="{LEFT + 46 + 6 * 15 + 4}" y="70">1</text>
<rect class="blk" x="{LEFT + 166}" y="60" width="13" height="13"/>
<text class="sec" x="{LEFT + 185}" y="70">blocked (-∞)</text>
<rect class="s1" x="{LEFT}" y="{TOP - 40}" width="12" height="12"/>
<text class="ink" x="{LEFT + 18}" y="{TOP - 30}">causal mask</text>
{grid_svg(LEFT, gc, chars, True, False, "c")}
<g id="gr" class="a"><rect class="s2" x="{RIGHT}" y="{TOP - 40}" width="12" height="12"/><text class="ink" x="{RIGHT + 18}" y="{TOP - 30}">no mask</text>{grid_svg(RIGHT, gn, chars, False, True, "n")}</g>
<text class="sec" x="{LEFT - 24}" y="{TOP + 6 * CELL + 26}">orange outline: the next character, the</text>
<text class="sec" x="{LEFT - 24}" y="{TOP + 6 * CELL + 44}">one to predict. The no-mask head reads it.</text>
<text class="ink" x="{LEFT - 24}" y="{PY0 - 20}">validation loss (nats per character)</text>
{yticks}
<line class="base" x1="{PX0}" x2="{PX1}" y1="{PY1 + 0.5}" y2="{PY1 + 0.5}"/>
{curve_svg}
<text class="mut" x="{PX0}" y="{PY1 + 20}">0</text>
<text class="mut" x="{PX1}" y="{PY1 + 20}" text-anchor="end">5,000 steps</text>
<text class="mut" x="{LEFT - 24}" y="{H - 10}">Pride and Prejudice, same seed and batches.</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "mask-grid-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
