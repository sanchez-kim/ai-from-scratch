"""Write figures/init-loss-curves-v2.svg from a measure_init.py result file.

    uv run python figures/make_loss_curves.py results/init-<date>.json

Static chart, one y-axis (dev loss in nats) against training step, for the
three init scales. The y-axis is cut at YMAX so the end of training is
readable; the std 1.0 run starts above it, and its starting value is written
next to the line. A dashed line marks ln 27, the loss of a uniform guess.

Colours mark the init scale, the same as in init-and-nan-v2.svg:
1/sqrt(fan_in) blue (slot 1), std 1.0 orange (slot 2), std 0.01 aqua (slot 3).
Every line is labelled directly; the labels stay in ink colours.
"""

import json
import sys
from pathlib import Path

W, H = 460, 320
LEFT, RIGHT, TOP, BOTTOM = 44, 312, 60, 270
YMIN, YMAX = 2.2, 4.2
NAMES = {"1.0": "std 1.0", "0.01": "std 0.01", "fan_in": "1/√fan_in"}
CLS = {"1.0": "big", "0.01": "small", "fan_in": "fan"}
SW = {"1.0": "s2", "0.01": "s3", "fan_in": "s1"}


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    steps = data["setup"]["steps"]
    x = lambda s: LEFT + (RIGHT - LEFT) * s / steps
    y = lambda v: BOTTOM - (BOTTOM - TOP) * (min(v, YMAX) - YMIN) / (YMAX - YMIN)

    parts = []
    for v in (2.2, 2.6, 3.0, 3.4, 3.8, 4.2):
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="mut" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for s in (0, 1000, 2000, 3000, 4000):
        parts.append(f'<text class="mut" x="{x(s):.1f}" y="{BOTTOM + 18}" text-anchor="middle">{s}</text>')
    ln27 = data["ln_vocab"]
    parts.append(f'<line class="ref" x1="{LEFT}" x2="{RIGHT}" y1="{y(ln27):.1f}" y2="{y(ln27):.1f}"/>'
                 f'<text class="mut" x="{RIGHT + 8}" y="{y(ln27) + 4:.1f}">ln 27 = {ln27:.2f}</text>')
    label_dy = {"1.0": -2, "0.01": 2, "fan_in": 14}
    for r in data["runs"]:
        key = str(r["scale"])
        pts = [(s, v) for s, v in r["dev_curve"]]
        # Clip the part above YMAX: keep the first point inside, plus the crossing.
        vis = [(s, v) for s, v in pts if v <= YMAX]
        if vis[0][0] != pts[0][0]:
            i = pts.index(vis[0])
            (s0, v0), (s1, v1) = pts[i - 1], pts[i]
            sc = s0 + (s1 - s0) * (v0 - YMAX) / (v0 - v1)
            vis = [(sc, YMAX)] + vis
            parts.append(f'<text class="ink" x="{x(sc) + 6:.1f}" y="{TOP - 8}">{NAMES[key]} starts at {pts[0][1]:.2f}</text>')
        pl = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in vis)
        parts.append(f'<polyline class="ln {CLS[key]}" points="{pl}"/>')
        end = pts[-1][1]
        ly = y(end) + label_dy[key]
        parts.append(f'<rect class="{SW[key]}" x="{RIGHT + 6}" y="{ly - 6:.1f}" width="10" height="4" rx="1"/>'
                     f'<text class="ink" x="{RIGHT + 20}" y="{ly:.1f}">{NAMES[key]} {end:.3f}</text>')

    desc = "; ".join(f'{NAMES[str(r["scale"])]}: {r["dev_curve"][0][1]:.2f} at step 0, '
                     f'{r["dev_curve"][-1][1]:.3f} at step {steps}' for r in data["runs"])
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .ref{stroke:#c3c2b7;stroke-width:1.5;stroke-dasharray:5 4}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round} .fan{stroke:#2a78d6;stroke-width:2.5} .big{stroke:#eb6834} .small{stroke:#1baf7a}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s3{fill:#1baf7a}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .ref{stroke:#383835} .fan{stroke:#3987e5} .big{stroke:#d95926} .small{stroke:#199e70}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s3{fill:#199e70}
  }
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Dev loss by training step for three init scales</title>
<desc>Dev loss in nats, measured every 100 steps, SGD lr 0.1, batch 32, same seeds. Line colours: 1/sqrt(fan_in) blue, std 1.0 orange, std 0.01 aqua. {desc}. A dashed line marks ln 27 = {ln27:.2f}, the loss of a uniform guess.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="{LEFT - 30}" y="24">dev loss (nats) by training step</text>
{''.join(parts)}
<text class="mut" x="{RIGHT}" y="{BOTTOM + 36}" text-anchor="end">step</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "init-loss-curves-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
