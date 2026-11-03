"""Write figures/tf-saturation-v2.svg: how much each extra copy of a word adds.

    uv run python figures/make_saturation.py

Animated SVG (CSS only). The y values come from BM25.term_score in
bm25/bm25.py with idf = 1 and a paragraph of average length, so the curve is
tf * (k1 + 1) / (tf + k1). Three k1 values are drawn one after another, then
the k1 = 1.5 curve (blue) gets dots at tf = 1, 2, 4 and 8. The plain count (no
saturation) is a dashed line. With reduced motion, the final frame shows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bm25 import BM25  # noqa: E402

W, H = 460, 340
LEFT, RIGHT, TOP, BOTTOM = 48, 392, 62, 270
TF_MAX, Y_MAX = 10, 4.0
K1S = (0.5, 1.5, 3.0)
DOTS = (1, 2, 4, 8)
# Colour marks the k1 setting: the default 1.5 is slot 1 (blue), the others
# slots 2 and 3, in palette order. Every curve has a direct label.
SLOT = {1.5: "s1", 0.5: "s2", 3.0: "s3"}


def weight(k1, tf):
    index = BM25([["w"], ["x"]], k1=k1, b=0.75)  # avgdl = 1, so dl = avgdl
    return index.term_score(tf, dl=1, idf=1.0)


def x(tf):
    return LEFT + (RIGHT - LEFT) * tf / TF_MAX


def y(v):
    return BOTTOM - (BOTTOM - TOP) * min(v, Y_MAX) / Y_MAX


def path(k1):
    steps = [i / 10 for i in range(0, TF_MAX * 10 + 1)]
    return "M" + " L".join(f"{x(t):.1f},{y(weight(k1, t)):.1f}" for t in steps)


def main():
    parts = []
    for v in range(0, int(Y_MAX) + 1):
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="mut" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for t in (0, 2, 4, 6, 8, 10):
        parts.append(f'<text class="mut" x="{x(t):.1f}" y="{BOTTOM + 18}" text-anchor="middle">{t}</text>')
    # Plain count: weight = tf. Clipped at the top of the chart.
    parts.append(f'<line class="neu" x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(Y_MAX):.1f}" y2="{y(Y_MAX):.1f}"/>')
    parts.append(f'<text class="mut" x="{x(Y_MAX) + 6:.1f}" y="{y(Y_MAX) + 12:.1f}">plain count</text>')
    for n, k1 in enumerate(K1S):
        cls = SLOT[k1]
        parts.append(f'<path class="curve {cls}l c{n}" pathLength="1" d="{path(k1)}"/>')
        end = weight(k1, TF_MAX)
        parts.append(f'<text class="ink lab c{n}l" x="{RIGHT + 6}" y="{y(end) + 4:.1f}">k1={k1}</text>')
    for n, tf in enumerate(DOTS):
        v = weight(1.5, tf)
        parts.append(f'<g class="dot d{n}"><circle class="s1f" cx="{x(tf):.1f}" cy="{y(v):.1f}" r="4.5"/>'
                     f'<text class="sec" x="{x(tf) + 6:.1f}" y="{y(v) + 20:.1f}">{v:.2f}</text></g>')
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .neu{stroke:#c3c2b7;stroke-width:1.5;stroke-dasharray:5 4}
  .s1l{stroke:#2a78d6} .s2l{stroke:#eb6834} .s3l{stroke:#1baf7a} .s1f{fill:#2a78d6} .s2f{fill:#eb6834}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600} .b{font-weight:700} .lab{font-weight:700}
  .curve{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .curve{stroke-dasharray:1;animation:draw 12s linear infinite both}
  .c1{animation-name:draw1} .c2{animation-name:draw2}
  .c0l{animation:show0 12s linear infinite both} .c1l{animation:show1 12s linear infinite both}
  .c2l{animation:show2 12s linear infinite both}
  .d0{animation:dot0 12s linear infinite both} .d1{animation:dot1 12s linear infinite both}
  .d2{animation:dot2 12s linear infinite both} .d3{animation:dot3 12s linear infinite both}
  @keyframes draw{0%{stroke-dashoffset:1}15%,100%{stroke-dashoffset:0}}
  @keyframes draw1{0%,15%{stroke-dashoffset:1}30%,100%{stroke-dashoffset:0}}
  @keyframes draw2{0%,30%{stroke-dashoffset:1}45%,100%{stroke-dashoffset:0}}
  @keyframes show0{0%,14%{opacity:0}15%,100%{opacity:1}}
  @keyframes show1{0%,29%{opacity:0}30%,100%{opacity:1}}
  @keyframes show2{0%,44%{opacity:0}45%,100%{opacity:1}}
  @keyframes dot0{0%,49%{opacity:0}50%,100%{opacity:1}}
  @keyframes dot1{0%,54%{opacity:0}55%,100%{opacity:1}}
  @keyframes dot2{0%,59%{opacity:0}60%,100%{opacity:1}}
  @keyframes dot3{0%,64%{opacity:0}65%,100%{opacity:1}}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781}
    .grid{stroke:#2c2c2a} .neu{stroke:#383835}
    .s1l{stroke:#3987e5} .s2l{stroke:#d95926} .s3l{stroke:#199e70} .s1f{fill:#3987e5} .s2f{fill:#d95926}
  }
  @media (prefers-reduced-motion: reduce) {
    .curve,.c0l,.c1l,.c2l,.d0,.d1,.d2,.d3{animation:none}
  }
"""
    dots = ", ".join(f"tf {t}: {weight(1.5, t):.2f}" for t in DOTS)
    ends = ", ".join(f"k1 {k}: {weight(k, TF_MAX):.2f} at tf 10" for k in K1S)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>BM25 term-frequency saturation for three k1 values</title>
<desc>Weight of one word, tf * (k1 + 1) / (tf + k1), with idf 1 and a paragraph of average length. {ends}. Dots on k1 = 1.5: {dots}. The dashed line is the plain count.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Each extra copy of a word adds less</text>
<text class="sec" x="16" y="42">BM25 weight of one word · idf 1 · average length</text>
{"".join(parts)}
<text class="mut" x="16" y="{H - 26}">x: times the word occurs in the paragraph (tf)</text>
<text class="mut" x="16" y="{H - 9}">y: weight; it never passes k1 + 1</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "tf-saturation-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
