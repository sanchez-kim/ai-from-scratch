"""Write figures/fd-error-by-eps-v2.svg from a measure_fd.py result file.

    uv run python figures/make_fd_chart.py results/fd-<date>.json

Static chart, one y-axis: |finite difference - backward() gradient| for
tanh at x = 0.5, forward and central difference, against eps. Both axes
are log scale. eps shrinks to the right.

Colour marks the method: central difference, the one the post recommends,
is slot 1 (blue); forward difference is slot 2 (orange). Labels are ink
text next to each line, with a short swatch of the line's colour.
"""

import json
import math
import sys
from pathlib import Path

W, H = 460, 330
LEFT, RIGHT, TOP, BOTTOM = 56, 440, 70, 262
FUNC = "tanh at x=0.5"
Y_HI, Y_LO = 0, -13          # log10 of the error range shown
X_HI, X_LO = -1, -14         # log10 of eps, left to right


def x(eps):
    return LEFT + (RIGHT - LEFT) * (X_HI - math.log10(eps)) / (X_HI - X_LO)


def y(err):
    e = max(err, 10 ** Y_LO)
    return TOP + (BOTTOM - TOP) * (Y_HI - math.log10(e)) / (Y_HI - Y_LO)


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m.removesuffix('.0')}e{int(e)}"


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    f = data["functions"][FUNC]
    rows = f["rows"]
    parts = []
    for p in range(Y_HI, Y_LO - 1, -3):
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y(10 ** p):.1f}" y2="{y(10 ** p):.1f}"/>')
        parts.append(f'<text class="mut" x="{LEFT - 8}" y="{y(10 ** p) + 4:.1f}" text-anchor="end">1e{p}</text>')
    for p in (-1, -4, -7, -10, -13):
        parts.append(f'<text class="mut" x="{x(10 ** p):.1f}" y="{BOTTOM + 18}" text-anchor="middle">1e{p}</text>')
    for m, cls in (("forward", "fw"), ("central", "ce")):
        pts = " ".join(f'{x(r["eps"]):.1f},{y(r[f"{m}_error"]):.1f}' for r in rows)
        parts.append(f'<polyline class="ln {cls}l" points="{pts}"/>')
        b = f["best"][m]
        parts.append(f'<circle class="{cls}" cx="{x(b["eps"]):.1f}" cy="{y(b["error"]):.1f}" r="5"/>')
        # Labels go below the lowest points, where no line runs.
        lx, ly, anchor = ((x(b["eps"]) - 4, y(b["error"]) + 20, "start") if m == "forward"
                          else (x(b["eps"]), y(b["error"]) + 20, "middle"))
        parts.append(f'<text class="ink" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">'
                     f'best {sci(b["error"])} at ε {sci(b["eps"])}</text>')
        lx, ly = (LEFT + 8, y(rows[0][f"{m}_error"]) - 8) if m == "forward" else (LEFT + 8, y(1e-9))
        parts.append(f'<line class="ln {cls}l" x1="{lx}" x2="{lx + 16}" y1="{ly - 4:.1f}" y2="{ly - 4:.1f}"/>'
                     f'<text class="ink b" x="{lx + 22}" y="{ly:.1f}">{m}</text>')
    desc_rows = "; ".join(f'eps {sci(r["eps"])}: forward {sci(r["forward_error"])}, central {sci(r["central_error"])}'
                          for r in rows if r["k"] % 2 == 0)
    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round} .fwl{stroke:#eb6834} .cel{stroke:#2a78d6}
  .fw{fill:#eb6834;stroke:#fcfcfb;stroke-width:2} .ce{fill:#2a78d6;stroke:#fcfcfb;stroke-width:2}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600} .b{font-weight:700}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .grid{stroke:#2c2c2a}
    .fwl{stroke:#d95926} .cel{stroke:#3987e5}
    .fw{fill:#d95926;stroke:#1a1a19} .ce{fill:#3987e5;stroke:#1a1a19}
  }
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Finite-difference error against epsilon, tanh at x = 0.5</title>
<desc>Absolute error of the forward difference (orange) and the central difference (blue) against the backward() gradient of tanh at x = 0.5, float64, log scales. {desc_rows}.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Finite-difference error vs ε</text>
<text class="sec" x="16" y="42">tanh at x = 0.5 · |estimate − backward()| · float64</text>
{"".join(parts)}
<text class="mut" x="16" y="{H - 26}">x: ε, log scale, smaller to the right</text>
<text class="mut" x="16" y="{H - 9}">y: absolute error, log scale</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "fd-error-by-eps-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
