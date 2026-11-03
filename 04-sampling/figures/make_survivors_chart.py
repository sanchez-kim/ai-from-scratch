"""Write figures/survivors-by-order-v2.svg from an analyze.py result file.

    python figures/make_survivors_chart.py results/survivors-<model>-<stamp>.json

Static chart, one y-axis (tokens that survive). For temperature 0.7, 1.0 and
1.3 with top-k 40 and top-p 0.95: the median survivor count with a p10-p90
whisker, temperature first vs temperature last, plus how often the two
survivor sets differ. Blue (series slot 1) = temperature first, the HF
order; orange (slot 2) = temperature last, the llama.cpp order.
"""

import json
import sys
from pathlib import Path

W, H = 460, 316
LEFT, RIGHT, TOP, BOTTOM = 96, 300, 60, 214
YMAX = 40
TS = ("0.7", "1.0", "1.3")
P = "0.95"


def y(v):
    return BOTTOM - (BOTTOM - TOP) * v / YMAX


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    step = (RIGHT - LEFT) / (len(TS) - 1)

    def x(i):
        return LEFT + step * i

    parts = []
    for v in range(0, YMAX + 1, 10):
        parts.append(f'<line class="{"base" if v == 0 else "grid"}" x1="{LEFT - 30}" x2="{RIGHT + 24}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="mut" x="{LEFT - 36}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')

    desc = []
    # Where both orders have the same median, two labels would sit on top of
    # each other, so they share one label up-left, clear of both lines.
    shared = [i for i, t in enumerate(TS)
              if data["settings"][f"T{t}-p{P}"]["survivors_a"]["median"]
              == data["settings"][f"T{t}-p{P}"]["survivors_b"]["median"]]
    for i in shared:
        m = data["settings"][f"T{TS[i]}-p{P}"]["survivors_a"]["median"]
        parts.append(f'<text class="ink" x="{x(i) - 18:.1f}" y="{y(m) - 9:.1f}" text-anchor="end">both {m:g}</text>')
    for key, cls, dx in (("survivors_b", "d", 7), ("survivors_a", "p", -7)):
        pts = []
        for i, t in enumerate(TS):
            s = data["settings"][f"T{t}-p{P}"][key]
            cx = x(i) + dx
            pts.append((cx, s))
            parts.append(f'<line class="wh {cls}w" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(s["p10"]):.1f}" y2="{y(s["p90"]):.1f}"/>')
        line = " ".join(f"{cx:.1f},{y(s['median']):.1f}" for cx, s in pts)
        parts.append(f'<polyline class="ln {cls}l" points="{line}"/>')
        for i, (cx, s) in enumerate(pts):
            parts.append(f'<circle class="pt {cls}" cx="{cx:.1f}" cy="{y(s["median"]):.1f}" r="4.5"/>')
            if i in shared:
                continue  # one "both" label instead, drawn below
            anchor = "start" if dx > 0 else "end"
            parts.append(f'<text class="ink" x="{cx + (8 if dx > 0 else -8):.1f}" y="{y(s["median"]) + 4:.1f}" '
                         f'text-anchor="{anchor}">{s["median"]:g}</text>')
        name = "T last" if cls == "d" else "T first"
        parts.append(f'<text class="ink" x="{RIGHT + 44}" y="{y(pts[-1][1]["median"]) + 4:.1f}">{name}</text>')
        full = "temperature last" if cls == "d" else "temperature first"
        desc.append(full + ": " + ", ".join(
            f'T {t} median {s["median"]:g} (p10 {s["p10"]:g}, p90 {s["p90"]:g})' for t, (_, s) in zip(TS, pts)))

    for i, t in enumerate(TS):
        st = data["settings"][f"T{t}-p{P}"]
        parts.append(f'<text class="ink" x="{x(i):.1f}" y="{BOTTOM + 20}" text-anchor="middle">T {t}</text>')
        parts.append(f'<text class="sec" x="{x(i):.1f}" y="{BOTTOM + 37}" text-anchor="middle">'
                     f'{st["differ_share"]:.0%} differ</text>')
        desc.append(f'T {t}: survivor sets differ at {st["differ"]} of {st["positions"]} positions')

    n = data["positions"]
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .base{stroke:#c3c2b7;stroke-width:1}
  .ln{fill:none;stroke-width:2} .dl{stroke:#eb6834} .pl{stroke:#2a78d6}
  .wh{stroke-width:2} .dw{stroke:#eb6834} .pw{stroke:#2a78d6}
  .pt{stroke:#fcfcfb;stroke-width:2} .d{fill:#eb6834} .p{fill:#2a78d6}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .base{stroke:#383835} .dl{stroke:#d95926} .pl{stroke:#3987e5}
    .dw{stroke:#d95926} .pw{stroke:#3987e5} .pt{stroke:#1a1a19} .d{fill:#d95926} .p{fill:#3987e5}
  }
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Tokens that survive top-k and top-p, by sampler order</title>
<desc>qwen3.5:4b-q4_K_M, {n} positions in ten passages, top-k 40 and top-p {P}. Blue: temperature first (Hugging Face order). Orange: temperature last (llama.cpp order). {". ".join(desc)}.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Tokens left after top-k 40, top-p {P}</text>
<text class="sec" x="16" y="42">median, p10–p90 · {n} positions, qwen3.5 4B</text>
{"".join(parts)}
<text class="sec" x="16" y="{H - 24}">T first: temperature before the cuts (HF)</text>
<text class="sec" x="16" y="{H - 8}">T last: after the cuts (llama.cpp default)</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "survivors-by-order-v2.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
