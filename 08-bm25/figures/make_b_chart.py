"""Write figures/top10-length-by-b-v2.svg from results/retrieval.json.

    uv run python figures/make_b_chart.py

Static chart, one y-axis: mean length (in tokens) of the paragraphs that
reach the top 10, against b, at k1 = 1.5, for both query sets (lexical blue, slot 1; paraphrase orange, slot 2).
A dashed neutral line
marks the average paragraph length of the whole corpus (avgdl).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 460, 330
LEFT, RIGHT, TOP, BOTTOM = 52, 340, 66, 256
BS = (0.0, 0.25, 0.5, 0.75, 1.0)
Y_LO, Y_HI = 40, 120


def x(b):
    return LEFT + (RIGHT - LEFT) * b


def y(v):
    return BOTTOM - (BOTTOM - TOP) * (v - Y_LO) / (Y_HI - Y_LO)


def main():
    data = json.loads((HERE.parent / "results" / "retrieval.json").read_text())
    avgdl = json.loads((HERE.parent / "results" / "agreement.json").read_text())["avgdl"]
    series = {kind: [data["grid"][f"k1=1.5,b={b}"][kind]["mean_len_of_top10_docs"] for b in BS]
              for kind in ("lexical", "paraphrase")}
    parts = []
    for v in range(Y_LO, Y_HI + 1, 20):
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="mut" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for b in BS:
        parts.append(f'<text class="mut" x="{x(b):.1f}" y="{BOTTOM + 18}" text-anchor="middle">{b:g}</text>')
    parts.append(f'<line class="neu" x1="{LEFT}" x2="{RIGHT}" y1="{y(avgdl):.1f}" y2="{y(avgdl):.1f}"/>')
    parts.append(f'<text class="mut" x="{LEFT + 6}" y="{y(avgdl) + 16:.1f}">corpus average {avgdl:.1f}</text>')
    for kind, cls in (("lexical", "s1"), ("paraphrase", "s2")):
        vals = series[kind]
        pts = " ".join(f"{x(b):.1f},{y(v):.1f}" for b, v in zip(BS, vals))
        parts.append(f'<polyline class="ln {cls}l" points="{pts}"/>')
        for b, v in zip(BS, vals):
            parts.append(f'<circle class="{cls}f" cx="{x(b):.1f}" cy="{y(v):.1f}" r="4"/>')
        # Direct labels: series name at the right end, values at b = 0 and b = 0.75.
        upper = vals[0] >= series["lexical" if kind == "paraphrase" else "paraphrase"][0]
        parts.append(f'<text class="sec" x="{x(0) + 8:.1f}" y="{y(vals[0]) + 4:.1f}">{vals[0]:.1f}</text>')
        parts.append(f'<text class="sec" x="{x(0.75):.1f}" y="{y(vals[3]) + (-10 if upper else 20):.1f}" '
                     f'text-anchor="middle">{vals[3]:.1f}</text>')
    lab_y = {k: y(series[k][-1]) for k in series}
    if abs(lab_y["lexical"] - lab_y["paraphrase"]) < 16:
        hi, lo = sorted(series, key=lambda k: lab_y[k])
        mid = (lab_y[hi] + lab_y[lo]) / 2
        lab_y[hi], lab_y[lo] = mid - 8, mid + 8
    for kind, cls in (("lexical", "s1"), ("paraphrase", "s2")):
        # Both lines end at almost the same point, so each label gets a colour swatch.
        parts.append(f'<line class="ln {cls}l" x1="{RIGHT + 6}" x2="{RIGHT + 18}" '
                     f'y1="{lab_y[kind]:.1f}" y2="{lab_y[kind]:.1f}"/>')
        parts.append(f'<text class="ink b" x="{RIGHT + 22}" y="{lab_y[kind] + 4:.1f}">{kind}</text>')
    desc = "; ".join(f"{k}: " + ", ".join(f"b {b:g} {v:.1f}" for b, v in zip(BS, s)) for k, s in series.items())
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .neu{stroke:#c3c2b7;stroke-width:1.5;stroke-dasharray:5 4}
  .s1l{stroke:#2a78d6} .s2l{stroke:#eb6834} .s3l{stroke:#1baf7a} .s1f{fill:#2a78d6} .s2f{fill:#eb6834}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600} .b{font-weight:700} .lab{font-weight:700}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .neu{stroke:#383835}
    .s1l{stroke:#3987e5} .s2l{stroke:#d95926} .s3l{stroke:#199e70} .s1f{fill:#3987e5} .s2f{fill:#d95926}
  }
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Length of top-10 paragraphs against b</title>
<desc>Mean length in tokens of the paragraphs in the top 10, k1 = 1.5, 10,000 paragraphs, 500 queries per set. {desc}. Corpus average {avgdl:.1f} tokens.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Lower b lets long paragraphs win</text>
<text class="sec" x="16" y="42">mean tokens in a top-10 paragraph · k1 = 1.5</text>
{"".join(parts)}
<text class="mut" x="16" y="{H - 26}">x: b (0 = no length normalization)</text>
<text class="mut" x="16" y="{H - 9}">y: tokens per paragraph</text>
</svg>
"""
    out = HERE / "top10-length-by-b-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
