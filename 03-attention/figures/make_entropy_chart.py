"""Write figures/entropy-by-dk-v2.svg from a measure_scaling.py result file.

    uv run python figures/make_entropy_chart.py results/scaling-<date>.json

Static chart, one y-axis (bits). Mean row entropy for 64 keys against d_k,
for three scales: none, 1/sqrt(d_k) and 1/d_k. Dots are the median over
seeds, whiskers the min-max over seeds.

v2 colours (series palette v2): blue = 1/sqrt(d_k), orange = no scaling,
aqua (dashed) = 1/d_k. Every line has a direct label in ink colour.
"""

import json
import math
import sys
from pathlib import Path

W, H = 460, 304
LEFT, RIGHT, TOP, BOTTOM = 48, 352, 80, 228
YMAX = 6.0

LABELS = {"none": "no scaling", "sqrt": "1/√d_k", "linear": "1/d_k"}


def y(bits):
    return BOTTOM - (BOTTOM - TOP) * bits / YMAX


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    sweep = data["sweep"]
    d_ks = data["setup"]["d_ks"]
    lo, hi = math.log2(d_ks[0]), math.log2(d_ks[-1])

    def x(d):
        return LEFT + (RIGHT - LEFT) * (math.log2(d) - lo) / (hi - lo)

    parts = []
    for b in range(0, int(YMAX) + 1, 2):
        parts.append(f'<line class="grid" x1="{LEFT - 6}" x2="{RIGHT + 6}" y1="{y(b):.1f}" y2="{y(b):.1f}"/>')
        parts.append(f'<text class="ax" x="{LEFT - 12}" y="{y(b) + 4:.1f}" text-anchor="end">{b}</text>')
    for d in d_ks:
        parts.append(f'<text class="ax" x="{x(d):.1f}" y="{BOTTOM + 20}" text-anchor="middle">{d}</text>')

    for name, cls in (("linear", "lin"), ("none", "non"), ("sqrt", "sq")):
        rows = sweep[name]
        pts = " ".join(f'{x(r["d_k"]):.1f},{y(r["entropy_bits"]["median"]):.1f}' for r in rows)
        parts.append(f'<polyline class="ln {cls}l" points="{pts}"/>')
        for r in rows:
            e = r["entropy_bits"]
            parts.append(f'<line class="wh" x1="{x(r["d_k"]):.1f}" x2="{x(r["d_k"]):.1f}" '
                         f'y1="{y(e["min"]):.1f}" y2="{y(e["max"]):.1f}"/>')
            parts.append(f'<circle class="pt {cls}" cx="{x(r["d_k"]):.1f}" cy="{y(e["median"]):.1f}" r="4"/>')
        last = rows[-1]["entropy_bits"]["median"]
        dy = {"linear": -4, "sqrt": 10, "none": 4}[name]
        parts.append(f'<text class="ink" x="{RIGHT + 14}" y="{y(last) + dy:.1f}">{LABELS[name]}</text>')
        # Value labels at both ends of the line.
        first = rows[0]["entropy_bits"]["median"]
        parts.append(f'<text class="mut" x="{x(d_ks[0]) + 8:.1f}" y="{y(first) + (14 if name == "sqrt" else -8):.1f}">'
                     f'{first:.2f}</text>')
        if name == "none":
            parts.append(f'<text class="mut" x="{x(d_ks[-1]) - 8:.1f}" y="{y(last) - 8:.1f}" text-anchor="end">{last:.2f}</text>')

    def desc_of(name):
        return ", ".join(f'{r["d_k"]}: {r["entropy_bits"]["median"]:.2f}' for r in sweep[name])

    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .mut{fill:#52514e} .ax{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1}
  .ln{fill:none;stroke-width:2} .nonl{stroke:#eb6834} .sql{stroke:#2a78d6} .linl{stroke:#1baf7a;stroke-dasharray:5 4}
  .wh{stroke:#52514e;stroke-width:1.5}
  .pt{stroke:#fcfcfb;stroke-width:1.5} .non{fill:#eb6834} .sq{fill:#2a78d6} .lin{fill:#1baf7a}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .mut{fill:#c3c2b7}
    .grid{stroke:#2c2c2a} .nonl{stroke:#d95926} .sql{stroke:#3987e5} .linl{stroke:#199e70}
    .wh{stroke:#c3c2b7} .pt{stroke:#1a1a19} .non{fill:#d95926} .sq{fill:#3987e5} .lin{fill:#199e70}
  }
"""
    n_seeds = len(data["setup"]["seeds"])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Attention row entropy by d_k, three scales</title>
<desc>Mean entropy of a softmax row over 64 keys, in bits (maximum 6), median of {n_seeds} seeds, random N(0, 1) queries and keys. No scaling (orange): {desc_of("none")}. 1/sqrt(d_k) (blue): {desc_of("sqrt")}. 1/d_k (aqua, dashed): {desc_of("linear")}.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Attention row entropy by d_k</text>
<text class="mut" x="16" y="42">bits, 64 keys (max 6) · median of {n_seeds} seeds</text>
{"".join(parts)}
<text class="mut" x="16" y="{H - 26}">x: d_k (log scale) · whiskers: min–max</text>
<text class="mut" x="16" y="{H - 9}">random N(0, 1) vectors, no training</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "entropy-by-dk-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
