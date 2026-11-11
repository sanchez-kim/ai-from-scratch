"""Write figures/recall-by-method-v2.svg from results/fusion.json.

    uv run python figures/make_method_chart.py

Static horizontal bar chart, one value axis (recall@10, 0 to 100%). Two
blocks, lexical and paraphrase questions, five bars each. Every bar has its
method name on the left and its value on the right (direct labels), so the
light aqua, which is below 3:1 on the light background, never carries
meaning alone.

v2 colours (series palette v2, same meaning in Parts 8-10):
BM25 = blue, embeddings = orange, fused results (score sum, RRF) = aqua.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = 460
LEFT, RIGHT = 100, 392          # bar area: 0% at LEFT, 100% at RIGHT
TOP = 58                        # first block header baseline
BAR_H, PITCH = 16, 22           # bar height, row pitch (6 px gap between bars)
METHODS = [("bm25", "BM25", "bm25"), ("embedding", "embed", "emb"), ("sum", "sum", "fus"),
           ("rrf60", "RRF 60", "fus"), ("rrf1", "RRF 1", "fus")]
KINDS = (("lexical", "Lexical questions"), ("paraphrase", "Paraphrase questions"))


def x(v):
    return LEFT + (RIGHT - LEFT) * v


def main():
    data = json.loads((HERE.parent / "results" / "fusion.json").read_text())["results"]
    block_h = 26 + PITCH * len(METHODS) + 14
    axis_y = TOP + block_h * len(KINDS) - 6
    H = axis_y + 64
    parts = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP + 8}" y2="{axis_y}"/>')
        parts.append(f'<text class="mut" x="{x(v):.1f}" y="{axis_y + 18}" text-anchor="middle">{v * 100:.0f}</text>')
    for b, (kind, title) in enumerate(KINDS):
        y0 = TOP + b * block_h
        parts.append(f'<text class="ink b" x="16" y="{y0}">{title}</text>')
        for i, (key, label, cls) in enumerate(METHODS):
            v = data[kind]["methods"][key]["recall@10"]
            yb = y0 + 14 + i * PITCH
            parts.append(f'<text class="ink" x="{LEFT - 8}" y="{yb + 12}" text-anchor="end">{label}</text>')
            parts.append(f'<rect class="{cls}" x="{LEFT}" y="{yb}" width="{x(v) - LEFT:.1f}" height="{BAR_H}"/>')
            parts.append(f'<text class="ink" x="{x(v) + 6:.1f}" y="{yb + 12}">{v * 100:.1f}</text>')
    parts.append(f'<line class="axis" x1="{LEFT}" x2="{LEFT}" y1="{TOP + 8}" y2="{axis_y}"/>')

    def desc(kind):
        return ", ".join(f'{label} {data[kind]["methods"][k]["recall@10"] * 100:.1f}%' for k, label, _ in METHODS)

    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .axis{stroke:#c3c2b7;stroke-width:1}
  .bm25{fill:#2a78d6} .emb{fill:#eb6834} .fus{fill:#1baf7a}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .b{font-weight:600} .h{font-size:14px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .axis{stroke:#383835}
    .bm25{fill:#3987e5} .emb{fill:#d95926} .fus{fill:#199e70}
  }
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Recall at 10 by method and query set</title>
<desc>Share of queries whose one gold paragraph is in the top 10, in percent, 500 queries per set, 10,000 Simple English Wikipedia paragraphs. BM25 blue, embeddings orange, fused results aqua. Lexical: {desc("lexical")}. Paraphrase: {desc("paraphrase")}.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Recall@10 by method (%, 500 questions each)</text>
{"".join(parts)}
<text class="sec" x="{LEFT}" y="{axis_y + 36}">recall@10, %</text>
<text class="sec" x="16" y="{H - 8}">fusion of each top 100 · RRF k = 60 and k = 1</text>
</svg>
"""
    out = HERE / "recall-by-method-v2.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
