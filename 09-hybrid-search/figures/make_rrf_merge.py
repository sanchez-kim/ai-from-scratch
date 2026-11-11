"""Write figures/rrf-merge-v2.svg: two ranked lists merging into one with RRF.

    uv run python figures/make_rrf_merge.py

Data: results/worked-example.json (a real query, chosen by a fixed rule in
measure_fusion.py). CSS animation only (no script, no SMIL), so it plays
inside an <img> tag. Static attributes are the final frame, so
prefers-reduced-motion switches the animations off and shows that frame.

v2 colours (series palette v2, same meaning in Parts 8-10):
BM25 = blue, embeddings = orange, fused result = aqua. Each merged row shows
its two RRF terms behind a blue and an orange chip, so the reader sees which
list each part of the score came from. The answer paragraph has a yellow
outline AND an asterisk. Text is always in ink colours, never in a series colour.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 480, 340
COLS = (16, 150, 284)       # x of the three columns
BAND = (121, 121, 188)      # width of the answer outline per column
ROW0, ROW_H = 96, 34        # y of the first row baseline, row pitch (title + sum line)
DUR = 10                    # seconds per loop
CHARS = 12                  # title characters per row (13px monospace, ~118 px)
CW = 7.9                    # approximate width of one 13px monospace character
SOURCES = ("bm25", "emb")   # chip classes for the two lists


def short(title):
    return title if len(title) <= CHARS else title[: CHARS - 1] + "…"


def kf(name, frames):
    return f"@keyframes {name} {{ " + " ".join(f"{p}% {{ {c} }}" for p, c in frames) + " }"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    ex = json.loads((HERE.parent / "results" / "worked-example.json").read_text())
    bm, em, fu = ex["bm25_top5"], ex["embedding_top5"], ex["rrf60_top5"]
    parts, keyframes, rules = [], [], []
    tick = ("bm25", "emb", "fus")

    def row(col, i, item, cls_id=None):
        y = ROW0 + ROW_H * i
        x = COLS[col]
        idattr = f' id="{cls_id}" class="a"' if cls_id else ""
        out = [f"<g{idattr}>"]
        if item["gold"]:
            out.append(f'<rect class="gold" x="{x - 1}" y="{y - 16}" width="{BAND[col]}" height="22" rx="3"/>')
        # A short bar in the list's colour ties each row to its column.
        out.append(f'<rect class="{tick[col]}" x="{x - 8}" y="{y - 13}" width="4" height="16"/>')
        star = "*" if item["gold"] else ""
        out.append(f'<text class="ink" x="{x + 4}" y="{y}">{i + 1} {esc(short(item["title"]))}{star}</text>')
        if col == 2:
            # The sum line: [blue] 1/(60+r_bm25) + [orange] 1/(60+r_emb) = total
            sx, sy = x + 4, y + 16
            for n, (src, r) in enumerate(zip(SOURCES, (item["bm25_rank"], item["embedding_rank"]))):
                if n:
                    out.append(f'<text class="sec" x="{sx:.1f}" y="{sy}">+</text>')
                    sx += 1.5 * CW
                if r is None:  # not in that list's top 100: hollow chip, no term
                    out.append(f'<rect class="none" x="{sx:.1f}" y="{sy - 10}" width="10" height="10"/>')
                    sx += 14
                    continue
                out.append(f'<rect class="{src}" x="{sx:.1f}" y="{sy - 10}" width="10" height="10"/>')
                term = f"1/{60 + r}"
                out.append(f'<text class="sec" x="{sx + 13:.1f}" y="{sy}">{term}</text>')
                sx += 13 + len(term) * CW + 4
            total = "= " + f"{item['rrf']:.4f}".lstrip("0")
            out.append(f'<text class="sec" x="{sx:.1f}" y="{sy}">{total}</text>')
            assert sx + len(total) * CW <= W - 16, (item["title"], sx)  # 16 px margin for wider fonts
        out.append("</g>")
        return "".join(out)

    heads = (("bm25", "BM25"), ("emb", "embedding"), ("fus", "RRF, k = 60"))
    for (cls, label), x in zip(heads, COLS):
        parts.append(f'<rect class="{cls}" x="{x - 8}" y="{ROW0 - 41}" width="12" height="12"/>')
        parts.append(f'<text class="ink b" x="{x + 10}" y="{ROW0 - 30}">{label}</text>')
    for i, item in enumerate(bm):
        parts.append(row(0, i, item))
    for i, item in enumerate(em):
        parts.append(row(1, i, item))

    # Fused rows arrive one by one (22%..62% of the loop), each with its sum.
    for i, item in enumerate(fu):
        name = f"f{i}"
        parts.append(row(2, i, item, cls_id=name))
        start = 22 + i * 8
        keyframes.append(kf(name, [(0, "opacity:0;transform:translateX(-24px)"),
                                   (start, "opacity:0;transform:translateX(-24px)"),
                                   (start + 6, "opacity:1;transform:translateX(0)"),
                                   (96, "opacity:1;transform:translateX(0)"),
                                   (100, "opacity:0;transform:translateX(0)")]))
        rules.append(f"#{name}{{animation-name:{name}}}")

    g = next(x for x in fu if x["gold"])
    gb, ge = g["bm25_rank"], g["embedding_rank"]
    foot1 = f"* answer: BM25 #{gb}, embedding #{ge}"
    foot2 = f"RRF: 1/{60 + gb} + 1/{60 + ge} = {g['rrf']:.4f} → #{fu.index(g) + 1}"
    parts.append(f'<g id="ft" class="a">'
                 f'<rect class="gold" x="16" y="{H - 43}" width="12" height="12" rx="2"/>'
                 f'<text class="ink" x="34" y="{H - 32}">{esc(foot1)}</text>'
                 f'<text class="ink b" x="16" y="{H - 12}">{esc(foot2)}</text></g>')
    keyframes.append(kf("ft", [(0, "opacity:0"), (66, "opacity:0"), (72, "opacity:1"),
                               (96, "opacity:1"), (100, "opacity:0")]))
    rules.append("#ft{animation-name:ft}")

    css = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sec{{fill:#52514e}}
  .bm25{{fill:#2a78d6}} .emb{{fill:#eb6834}} .fus{{fill:#1baf7a}}
  .none{{fill:none;stroke:#c3c2b7;stroke-width:1.5}}
  .gold{{fill:none;stroke:#eda100;stroke-width:2}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .b{{font-weight:600}} .h{{font-size:14px;font-weight:600}}
  .a{{animation-duration:{DUR}s;animation-iteration-count:infinite;animation-timing-function:ease-out}}
  {' '.join(rules)}
  {' '.join(keyframes)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sec{{fill:#c3c2b7}}
    .bm25{{fill:#3987e5}} .emb{{fill:#d95926}} .fus{{fill:#199e70}}
    .none{{stroke:#383835}} .gold{{stroke:#c98500}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    q = ex["query"]
    desc = (f"Real query from the paraphrase set: {q} BM25 top 5 (blue): "
            + ", ".join(f"{i + 1} {x['title']}" for i, x in enumerate(bm))
            + ". Embedding top 5 (orange): " + ", ".join(f"{i + 1} {x['title']}" for i, x in enumerate(em))
            + ". RRF k = 60 top 5 (aqua), each with a blue BM25 term and an orange embedding term: "
            + ", ".join(f"{i + 1} {x['title']} {x['rrf']:.4f}" for i, x in enumerate(fu))
            + f". The answer paragraph ({g['title']}) is #{gb} in BM25 and #{ge} in embeddings, and #1 after RRF.")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Two ranked lists merged by reciprocal rank fusion</title>
<desc>{esc(desc)}</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">One real query, two top-5 lists, one merge</text>
<text class="sec" x="16" y="42">each list gives a paragraph 1/(60 + rank)</text>
{"".join(parts)}
</svg>
"""
    out = HERE / "rrf-merge-v2.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
