"""Write figures/loss-by-position-v2.svg: the loss when the input stops at t.

    uv run python figures/make_loss_chart.py results/train-causal-cpu-main.json \
        results/train-none-cpu-main.json results/train-after-cpu-main.json

For each position t, the model reads characters 0..t only and predicts
character t+1. That is the situation in generation. One line per mask setting,
a dashed line for a uniform guess over the vocabulary. Static, one y-axis.
"""

import json
import sys
from pathlib import Path

W, H = 460, 340
X0, X1, Y0, Y1 = 46, 330, 62, 290
YMAX = 13
NAMES = {"causal": "causal mask", "none": "no mask", "after": "mask after softmax"}
SLOT = {"causal": "l1", "none": "l2", "after": "l3"}  # palette v2 slots 1, 2, 3


def xy(t, v, n):
    return X0 + t / (n - 1) * (X1 - X0), Y1 - min(v, YMAX) / YMAX * (Y1 - Y0)


def label(out, mask, tx, ty, px=None, py=None):
    """Swatch + ink text at (tx, ty), with an optional leader line to the curve point (px, py)."""
    if px is not None:
        end = ty + 3 if py > ty else ty - 14  # leave from below or above the label
        out.append(f'<line class="lead" x1="{px:.1f}" y1="{py:.1f}" x2="{tx + 7:.1f}" y2="{end:.1f}"/>')
    out.append(f'<line class="{SLOT[mask]}" x1="{tx}" x2="{tx + 14}" y1="{ty - 4}" y2="{ty - 4}"/>')
    out.append(f'<text class="ink" x="{tx + 19}" y="{ty}">{NAMES[mask]}</text>')


def main():
    runs = {json.loads(Path(p).read_text())["mask"]: json.loads(Path(p).read_text()) for p in sys.argv[1:]}
    uniform = runs["causal"]["data"]["uniform_loss_nats"]
    out = []
    for v in range(0, 13, 2):
        y = Y1 - v / YMAX * (Y1 - Y0)
        out.append(f'<line class="grid" x1="{X0}" x2="{X1}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="mut" x="{X0 - 6}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    yu = Y1 - uniform / YMAX * (Y1 - Y0)
    out.append(f'<line class="ref" x1="{X0}" x2="{X1}" y1="{yu:.1f}" y2="{yu:.1f}"/>')
    # Left end of the dashed line: the band between the causal line and the others is empty there.
    out.append(f'<text class="sec" x="{X0 + 4}" y="{yu - 5:.1f}">uniform guess {uniform:.2f}</text>')
    h = {m: r["by_position"]["honest"] for m, r in runs.items()}
    for mask in ("after", "none", "causal"):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(t, v, 64) for t, v in enumerate(h[mask])))
        out.append(f'<polyline class="{SLOT[mask]}" points="{pts}"/>')
    # Direct labels sit above the curves, each with a short leader to its line.
    px, py = xy(3, h["none"][3], 64)
    label(out, "none", X0 + 40, Y0 - 16, px, py - 3)
    px, py = xy(52, h["after"][52], 64)
    label(out, "after", X0 + 170, Y0 - 16, px, py - 3)
    label(out, "causal", X0 + 30, Y1 - 2.0 / YMAX * (Y1 - Y0))
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .base{stroke:#c3c2b7;stroke-width:1}
  .ref{stroke:#c3c2b7;stroke-width:1.5;stroke-dasharray:5 4} .lead{stroke:#898781;stroke-width:1}
  .l1{fill:none;stroke:#2a78d6;stroke-width:2} .l2{fill:none;stroke:#eb6834;stroke-width:2}
  .l3{fill:none;stroke:#1baf7a;stroke-width:2}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:15px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .base{stroke:#383835} .ref{stroke:#5a5a55}
    .l1{stroke:#3987e5} .l2{stroke:#d95926} .l3{stroke:#199e70}
  }
"""
    desc = ("Validation loss in nats when the input stops at position t, for t from 0 to 63, after 5000 steps. "
            + "; ".join(f'{NAMES[m]}: {r["by_position"]["honest"][0]:.2f} at position 0, '
                        f'{r["final"]["honest_mean"]:.2f} on average, {r["by_position"]["honest"][-1]:.2f} at position 63'
                        for m, r in runs.items())
            + f". A uniform guess over {runs['causal']['data']['vocab']} characters is {uniform:.2f}.")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Loss when each position sees only its past</title>
<desc>{desc}</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="{X0 - 30}" y="20">loss when the input stops at t (nats)</text>
{''.join(out)}
<line class="base" x1="{X0}" x2="{X1}" y1="{Y1 + 0.5}" y2="{Y1 + 0.5}"/>
<text class="mut" x="{X0}" y="{Y1 + 20}">position 0</text>
<text class="mut" x="{X1}" y="{Y1 + 20}" text-anchor="end">63</text>
<text class="mut" x="{X0 - 30}" y="{H - 8}">Same text, weights and batches; 5,000 steps.</text>
</svg>
"""
    out_path = Path(__file__).resolve().parent / "loss-by-position-v2.svg"
    out_path.write_text(svg)
    print("wrote", out_path.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
