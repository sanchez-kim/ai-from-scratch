"""Static SVG: recall@10 against ef_search, real embeddings vs random vectors.

    uv run python figures/make_recall_chart.py results/recall-sweep.json
"""
import json
import math
import pathlib
import sys

OUT = pathlib.Path(__file__).parent / "recall-vs-ef-v2.svg"
W, H = 460, 330
L, R, T, B = 52, 330, 70, 270  # plot box; the right margin holds the direct labels

STYLE = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .mut{fill:#898781} .lab{fill:#0b0b0b}
  .grid{stroke:#e1e0d9;stroke-width:1} .ln{fill:none;stroke-width:2;stroke:#eb6834}
  .dash{stroke-dasharray:6 4}
  .pt{stroke:#fcfcfb;stroke-width:2} .hn{fill:#eb6834} .hno{fill:#fcfcfb;stroke:#eb6834;stroke-width:2}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600} .sub{fill:#52514e}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink,.lab{fill:#f0efec} .grid{stroke:#2c2c2a} .sub{fill:#c3c2b7}
    .ln{stroke:#d95926} .pt{stroke:#1a1a19} .hn{fill:#d95926} .hno{fill:#1a1a19;stroke:#d95926}
  }
"""


def main(path):
    data = json.loads(pathlib.Path(path).read_text())
    sets = data["sets"]
    efs = [r["ef_search"] for r in sets["real"]["sweep"]]
    lo, hi = math.log2(efs[0]), math.log2(efs[-1])
    x = lambda ef: L + (R - L) * (math.log2(ef) - lo) / (hi - lo)
    y = lambda v: B - (B - T) * v

    parts = []
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="mut" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for e in efs:
        parts.append(f'<text class="mut" x="{x(e):.1f}" y="{B + 18}" text-anchor="middle">{e}</text>')
    parts.append(f'<text class="mut" x="{(L + R) / 2:.0f}" y="{B + 38}" text-anchor="middle">ef_search (log scale)</text>')

    desc = []
    # One method (HNSW, slot 2 orange) on two datasets: solid line + filled markers = Wikipedia,
    # dashed line + hollow markers = random vectors. Labels stay in ink.
    for name, line_cls, mark_cls, label in (("real", "ln", "pt hn", "Wikipedia"), ("random", "ln dash", "hno", "random")):
        rows = sets[name]["sweep"]
        pts = [(x(r["ef_search"]), y(r["recall_at_10"])) for r in rows]
        parts.append(f'<polyline class="{line_cls}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
        parts += [f'<circle class="{mark_cls}" cx="{a:.1f}" cy="{b:.1f}" r="4"/>' for a, b in pts]
        first, last = rows[0]["recall_at_10"], rows[-1]["recall_at_10"]
        parts.append(f'<text class="lab" x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 4:.1f}">{label}</text>')
        parts.append(f'<text class="lab" x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 21:.1f}">{last:.3f}</text>')
        dy = -10 if name == "random" else 20  # random sits on the axis, so label above; real rises, so below
        parts.append(f'<text class="lab" x="{pts[0][0] + 2:.1f}" y="{pts[0][1] + dy:.1f}">{first:.3f}</text>')
        desc.append(f"{label}: " + ", ".join(f"ef {r['ef_search']} {r['recall_at_10']:.3f}" for r in rows))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">\n'
        f"<title>recall@10 by ef_search, real embeddings vs random vectors</title>\n"
        f'<desc>hnswlib {data["hnswlib"]}, M {data["M"]}, ef_construction {data["ef_construction"]}, '
        f"100,000 vectors of 768 dimensions, 1,000 held-out queries. {'. '.join(desc)}.</desc>\n"
        f"<style>{STYLE}</style>\n"
        f'<rect class="bg" width="{W}" height="{H}"/>\n'
        f'<text class="ink h" x="16" y="24">recall@10 as ef_search grows</text>\n'
        f'<text class="sub" x="16" y="42">HNSW (hnswlib), M {data["M"]}, 100,000 vectors, 768 dims</text>\n'
        + "".join(parts) + "\n</svg>\n"
    )
    OUT.write_text(svg)
    print(OUT.name, f"{len(svg.encode()) / 1024:.1f} KB")


if __name__ == "__main__":
    main(sys.argv[1])
