"""Write figures/tokens-per-1000-chars-v2.svg from results/token-counts.json.

    uv run python figures/make_counts_chart.py

Static horizontal bar chart, one x-axis: tokens per 1,000 characters for
English prose, Korean prose and Python code, four tokenizers each.
v2 colours mark the kind of text (series palette v2): English = blue (slot 1),
Korean = orange (slot 2), Python code = aqua (slot 3). Each group header has a
swatch and every bar its tokenizer name and value, so colour is never the only cue.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
W, LEFT, RIGHT = 460, 176, 396
XMAX = 2500
BAR, GAP, GROUP_GAP = 16, 6, 18
NAMES = {"ours-en": "ours, English-only", "ours-mixed": "ours, Eng+Korean",
         "cl100k_base": "cl100k_base", "o200k_base": "o200k_base"}
GROUPS = {"english": "English prose", "korean": "Korean prose", "code": "Python code"}
SLOT = {"english": "s1", "korean": "s2", "code": "s3"}


def x(v):
    return LEFT + (RIGHT - LEFT) * v / XMAX


def main():
    data = json.loads((HERE / "results" / "token-counts.json").read_text(encoding="utf-8"))
    counts = data["counts"]
    parts, y = [], 56
    for key, title in GROUPS.items():
        parts.append(f'<rect class="{SLOT[key]}" x="16" y="{y + 1}" width="13" height="13" rx="2"/>')
        parts.append(f'<text class="ink" x="36" y="{y + 12}" font-size="14" font-weight="600">{title}</text>')
        y += 20
        for name, label in NAMES.items():
            v = counts[name][key]["per_1000_chars"]
            cls = SLOT[key]
            parts.append(f'<text class="ink" x="{LEFT - 8}" y="{y + 12.5}" text-anchor="end">{label}</text>')
            parts.append(f'<rect class="{cls}" x="{LEFT}" y="{y}" width="{x(v) - LEFT:.1f}" height="{BAR}"/>')
            parts.append(f'<text class="ink" x="{x(v) + 5:.1f}" y="{y + 12.5}">{v:,.0f}</text>')
            y += BAR + GAP
        y += GROUP_GAP
    top, bottom = 56, y - GROUP_GAP
    grid = []
    for v in range(0, XMAX + 1, 1000):
        grid.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{bottom}"/>')
        grid.append(f'<text class="tick" x="{x(v):.1f}" y="{bottom + 16}" text-anchor="middle">{v:,}</text>')
    grid.append(f'<line class="base" x1="{LEFT}" x2="{LEFT}" y1="{top}" y2="{bottom}"/>')
    h = bottom + 58
    foot = [f'<text class="mut" x="16" y="{bottom + 38}">Held-out Wikipedia text, CPython stdlib.</text>',
            f'<text class="mut" x="16" y="{bottom + 54}">Ours: byte-level BPE, 8,000 merges each.</text>']
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .mut{fill:#52514e} .tick{fill:#898781}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s3{fill:#1baf7a} .grid{stroke:#e1e0d9;stroke-width:1}
  .base{stroke:#c3c2b7;stroke-width:1}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .mut{fill:#c3c2b7}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s3{fill:#199e70} .grid{stroke:#2c2c2a} .base{stroke:#383835}
  }
"""
    rows = "; ".join(f"{GROUPS[k]}: " + ", ".join(f"{NAMES[n]} {counts[n][k]['per_1000_chars']:,.0f}"
                                                   for n in NAMES) for k in GROUPS)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img">'
           f'<title>Tokens per 1,000 characters</title><desc>{rows}. Bars are blue for English prose, orange for Korean prose, aqua for Python code.</desc><style>{css}</style>'
           f'<rect class="bg" width="{W}" height="{h}"/>'
           f'<text class="ink" x="16" y="26" font-size="15" font-weight="600">Tokens per 1,000 characters</text>'
           f'<text class="mut" x="16" y="44">Fewer is cheaper</text>'
           + "".join(grid) + "".join(parts) + "".join(foot) + "</svg>\n")
    out = HERE / "figures" / "tokens-per-1000-chars-v2.svg"
    out.write_text(svg, encoding="utf-8")
    print(out.name, len(svg.encode("utf-8")), "bytes")


if __name__ == "__main__":
    main()
