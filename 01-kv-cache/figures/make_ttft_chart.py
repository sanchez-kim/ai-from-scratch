"""Write figures/ttft-by-position-v2.svg from two bench_ollama.py result files.

    python figures/make_ttft_chart.py results/ollama-<model>-<stamp>.json \
        results/ollama-<model>-primed-<stamp>.json

Static chart, one y-axis (ms). For each position of the changed line: the
median time to first token with a p10-p90 whisker, for the default run and
the primed run, plus the identical-prompt median as a reference line.
"""

import json
import sys
from pathlib import Path

# Text is 13px or larger so it stays legible at 370 px wide. Grey marks are
# #8A8A93 (3.3:1 on #FAFAFA) and #71717A (3.7:1 on #18181B).
# Footer lines stay <= 46 characters: at 358 px wide and a fallback monospace
# font ~10% wider than SF Mono they still end >= 24 px before the right edge.
W, H = 460, 260
LEFT, RIGHT, TOP, BOTTOM = 64, 360, 58, 194
YMAX = 4000


def y(ms):
    return BOTTOM - (BOTTOM - TOP) * ms / YMAX


def series(data):
    return [b for b in data["blocks"] if b["kind"] == "position"]


def main():
    default = json.loads(Path(sys.argv[1]).read_text())
    primed = json.loads(Path(sys.argv[2]).read_text())
    assert not default.get("prime") and primed.get("prime")
    d_blocks, p_blocks = series(default), series(primed)
    step = (RIGHT - LEFT) / (len(d_blocks) - 1)

    def x(i):
        return LEFT + step * i

    parts = []
    for ms in range(0, YMAX + 1, 1000):
        parts.append(f'<line class="grid" x1="{LEFT - 8}" x2="{RIGHT + 8}" y1="{y(ms):.1f}" y2="{y(ms):.1f}"/>')
        parts.append(f'<text class="mut" x="{LEFT - 14}" y="{y(ms) + 4:.1f}" text-anchor="end">{ms:,}</text>')

    same = next(b for b in default["blocks"] if b["kind"] == "identical")["ttft_ms"]["median"]
    parts.append(f'<line class="ref" x1="{LEFT - 8}" x2="{RIGHT + 8}" y1="{y(same):.1f}" y2="{y(same):.1f}"/>')
    parts.append(f'<text class="mut" x="{LEFT - 8}" y="{y(same) - 5:.1f}">'
                 f'identical prompt: {same:.0f} ms</text>')

    spread = 0
    for blocks, cls, above in ((d_blocks, "d", True), (p_blocks, "p", False)):
        pts = " ".join(f'{x(i):.1f},{y(b["ttft_ms"]["median"]):.1f}' for i, b in enumerate(blocks))
        parts.append(f'<polyline class="ln {cls}l" points="{pts}"/>')
        for i, b in enumerate(blocks):
            t = b["ttft_ms"]
            spread = max(spread, t["p90"] - t["p10"])
            parts.append(f'<line class="wh" x1="{x(i):.1f}" x2="{x(i):.1f}" '
                         f'y1="{y(t["p10"]):.1f}" y2="{y(t["p90"]):.1f}"/>')
            parts.append(f'<circle class="pt {cls}" cx="{x(i):.1f}" cy="{y(t["median"]):.1f}" r="4.5"/>')
            # Label every point except where the two series sit on top of each other.
            # The primed 0% value has no free spot next to its dot (the default label,
            # the y-axis label and both lines crowd it), so the footer carries it.
            if not above and i == 0:
                continue
            last_point = i == len(blocks) - 1
            if above and i == len(blocks) - 2:
                # Above-right: the steep segment from the previous point runs above-left.
                lx, ly, anchor = x(i) + 7, y(t["median"]) - 8, "start"
            elif last_point and not above:
                # Left of the point, so it clears the series name and the x labels.
                lx, ly, anchor = x(i) - 9, y(t["median"]) + 4, "end"
            else:
                lx, ly, anchor = x(i), y(t["median"]) + (-10 if above else 17), "middle"
            parts.append(f'<text class="ink" x="{lx:.1f}" y="{ly:.1f}" '
                         f'text-anchor="{anchor}">{t["median"]:,.0f}</text>')
        last = blocks[-1]["ttft_ms"]["median"]
        name = "default" if cls == "d" else "primed"
        parts.append(f'<text class="ink" x="{RIGHT + 14}" y="{y(last) + 4:.1f}">{name}</text>')

    for i, b in enumerate(d_blocks):
        parts.append(f'<text class="ink" x="{x(i):.1f}" y="{BOTTOM + 20}" text-anchor="middle">'
                     f'{b["position"]:.0%}</text>')

    def desc_of(blocks):
        return ", ".join(f'{b["position"]:.0%} {b["ttft_ms"]["median"]:.0f} ms' for b in blocks)

    css = """
  .bg{fill:#FAFAFA} .ink{fill:#17171A} .mut{fill:#71717A}
  .grid{stroke:#E4E4E7;stroke-width:1} .ref{stroke:#8A8A93;stroke-width:1;stroke-dasharray:4 3}
  .ln{fill:none;stroke-width:2} .dl{stroke:#8A8A93} .pl{stroke:#7C3AED}
  .wh{stroke:#17171A;stroke-width:1.5}
  .pt{stroke:#FAFAFA;stroke-width:2} .d{fill:#71717A} .p{fill:#7C3AED}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#18181B} .ink{fill:#F4F4F5} .mut{fill:#A1A1AA}
    .grid{stroke:#3F3F46} .ref{stroke:#71717A} .dl{stroke:#71717A} .pl{stroke:#9061F9}
    .wh{stroke:#F4F4F5} .pt{stroke:#18181B} .d{fill:#A1A1AA} .p{fill:#9061F9}
  }
"""
    version = default["ollama_version"].split()[-1]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Time to first token by position of the changed line</title>
<desc>{default["model"]} on Ollama {version}, a prompt of about 3,050 tokens with one changed line, median of {default["repeats"]} runs. Default: {desc_of(d_blocks)}. Primed with a request that ends just before the line: {desc_of(p_blocks)}. Identical prompt: {same:.0f} ms.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Time to first token by changed-line position</text>
<text class="mut" x="16" y="42">milliseconds · {default["model"]}, Ollama {version}</text>
{"".join(parts)}
<text class="mut" x="16" y="{H - 24}">x: where the line changed · p10–p90 ≤ {spread:.0f} ms</text>
<text class="mut" x="16" y="{H - 8}">primed: prefix sent first · primed 0%: {p_blocks[0]["ttft_ms"]["median"]:,.0f}</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "ttft-by-position-v2.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
