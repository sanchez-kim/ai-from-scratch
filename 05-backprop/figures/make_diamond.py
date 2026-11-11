"""Write figures/diamond-backward-v2.svg: the animated diamond a*3 + a*5.

    uv run python figures/make_diamond.py results/bugs-<date>.json

Two copies of the same graph, a = 2. Top: correct code, grad +=. Bottom:
bug 1, grad =. Gradients flow right to left in the order Value.backward
runs them: L, then a*5, then a*3.

Colour marks the two paths from L back to a, so the reader can see that a
gets one part from each: blue (slot 1) through a*3, orange (slot 2) through
a*5. The top row adds both parts (5 + 3 = 8). In the bottom row the orange
part is overwritten, so its edge fades and a ends at 3. Text stays in ink
colours. The final a.grad values are read from the results file.

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Static attributes are the final frame, so prefers-reduced-motion switches
the animations off and shows that frame.
"""

import json
import sys
from pathlib import Path

W, H = 460, 424
NW, NH = 96, 46
XA, XM, XL = 16, 196, 348          # node x: a, the two products, L
DUR = 9.0
T_L, T_LB, T_C, T_B = 0.6, 1.4, 2.6, 4.4   # L.grad set, pass to products, a*5 runs, a*3 runs
PULSE = 0.8
END_FADE = 0.5
LOST = 0.3                          # opacity of an overwritten path

keyframes = []


def pct(t):
    return max(0.0, min(100.0, t / DUR * 100))


def anim(el_id, frames):
    body = " ".join(f"{p:.2f}% {{opacity:{o}}}" for p, o in frames)
    keyframes.append(f"@keyframes k{el_id} {{ {body} }} #{el_id}{{animation-name:k{el_id}}}")


def appear(el_id, t_on, t_off=None, settle=1):
    """Opacity 0 until t_on, then 1; from t_off on it drops to `settle`.
    Returns the static opacity, which is the final frame."""
    end = DUR - END_FADE
    frames = [(0, 0), (pct(t_on), 0), (pct(t_on + 0.25), 1)]
    if t_off is not None:
        frames += [(pct(t_off), 1), (pct(t_off + 0.4), settle)]
    frames += [(pct(end), settle if t_off is not None else 1), (pct(end + 0.25), 0), (100, 0)]
    anim(el_id, frames)
    return str(settle if t_off is not None else 1)


def visible_until(el_id, t_off):
    """Opacity 1 from the start of the loop until t_off; hidden in the final frame."""
    anim(el_id, [(0, 1), (pct(t_off), 1), (pct(t_off + 0.25), 0), (100, 0)])
    return "0"


def pulse(el_id, t_on):
    anim(el_id, [(0, 0), (pct(t_on), 0), (pct(t_on + 0.15), 1), (pct(t_on + PULSE), 1),
                 (pct(t_on + PULSE + 0.3), 0), (100, 0)])
    return "0"


def node(x, y, label):
    return (f'<rect class="nd" x="{x}" y="{y}" width="{NW}" height="{NH}" rx="6"/>'
            f'<text class="ink" x="{x + 10}" y="{y + 19}">{label}</text>')


def row(y0, tag, title, final_a, bug):
    ya, yb, yc = y0 + 66, y0 + 30, y0 + 102     # node tops: a, a*3, a*5
    yl = ya
    mid = lambda y: y + NH / 2
    out = [f'<text class="ink h" x="{XA}" y="{y0 + 12}">{title}</text>']
    # path "p3" runs L -> a*3 -> a (slot 1), path "p5" runs L -> a*5 -> a (slot 2)
    edges = {
        "lb": (XM + NW, mid(yb), XL, mid(yl), "p3", T_LB),
        "lc": (XM + NW, mid(yc), XL, mid(yl), "p5", T_LB),
        "ba": (XA + NW, mid(ya), XM, mid(yb), "p3", T_B),
        "ca": (XA + NW, mid(ya), XM, mid(yc), "p5", T_C),
    }
    for k, (x1, y1, x2, y2, path, t) in edges.items():
        geo = f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
        out.append(f'<line class="ed" {geo}/>')
        # The coloured edge stays after its pulse. With the bug, the orange
        # part into a is overwritten, so that edge fades once a*3 has run.
        lost = bug and k == "ca"
        i = f"{tag}s{k}"
        op = appear(i, t + 0.3, T_B + 0.5 if lost else None, LOST if lost else 1)
        out.append(f'<line id="{i}" class="a sg {path}" {geo} opacity="{op}"/>')
        i = f"{tag}p{k}"
        out.append(f'<line id="{i}" class="a pl {path}" {geo} opacity="{pulse(i, t)}"/>')
    # Labels outside the two edges into a: what each _backward writes.
    op = "=" if bug else "+="
    for k, t, val, dy in (("ca", T_C, 5, 20), ("ba", T_B, 3, -9)):
        x1, y1, x2, y2, _, _ = edges[k]
        i = f"{tag}e{k}"
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + dy
        out.append(f'<text id="{i}" class="a ink b" x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" '
                   f'opacity="{appear(i, t + 0.4)}">{op} {val}</text>')
    out.append(node(XM, yb, "a*3 = 6"))
    out.append(node(XM, yc, "a*5 = 10"))
    out.append(node(XL, yl, "L = 16"))
    out.append(node(XA, ya, "a = 2"))
    for x, y, t in ((XL, yl, T_L), (XM, yb, T_LB + 0.6), (XM, yc, T_LB + 0.6)):
        i = f"{tag}g{x}{y}"
        out.append(f'<text id="{i}" class="a sec" x="{x + 10}" y="{y + 37}" opacity="{appear(i, t)}">grad 1</text>')
    # a.grad: 0, then 5, then the final value.
    i = f"{tag}a0"
    out.append(f'<text id="{i}" class="a sec" x="{XA + 10}" y="{ya + 37}" opacity="{visible_until(i, T_C + 0.5)}">grad 0</text>')
    i = f"{tag}a1"
    out.append(f'<text id="{i}" class="a sec" x="{XA + 10}" y="{ya + 37}" opacity="{appear(i, T_C + 0.5, T_B + 0.5, 0)}">grad 5</text>')
    i = f"{tag}af"
    out.append(f'<text id="{i}" class="a ink b" x="{XA + 10}" y="{ya + 37}" opacity="{appear(i, T_B + 0.5)}">grad {final_a:g}</text>')
    i = f"{tag}an"
    note = "5 + 3 = 8" if not bug else "5 overwritten"
    out.append(f'<text id="{i}" class="a sec" x="{XA}" y="{ya + NH + 20}" opacity="{appear(i, T_B + 0.7)}">{note}</text>')
    return out


def main():
    ex = json.loads(Path(sys.argv[1]).read_text())["examples"]
    good, bad = ex["diamond_a*3+a*5_correct"], ex["diamond_a*3+a*5_bug1_assign"]
    parts = [f'<text class="ink h" x="{XA}" y="24">Backward pass on L = a*3 + a*5, a = 2</text>']
    parts += row(44, "c", "correct: grad += …", good, bug=False)
    parts += row(222, "w", "bug 1: grad = …", bad, bug=True)
    ly = H - 34
    parts.append(f'<line class="sg p3" x1="{XA}" x2="{XA + 22}" y1="{ly - 4}" y2="{ly - 4}"/>'
                 f'<text class="sec" x="{XA + 30}" y="{ly}">path via a*3 sends 3</text>'
                 f'<line class="sg p5" x1="{XA + 226}" x2="{XA + 248}" y1="{ly - 4}" y2="{ly - 4}"/>'
                 f'<text class="sec" x="{XA + 256}" y="{ly}">path via a*5 sends 5</text>')
    css = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sec{{fill:#52514e}}
  .nd{{fill:#fcfcfb;stroke:#898781;stroke-width:1}}
  .ed{{stroke:#c3c2b7;stroke-width:2}} .sg{{stroke-width:3;stroke-linecap:round}} .pl{{stroke-width:6;stroke-linecap:round}}
  .p3{{stroke:#2a78d6}} .p5{{stroke:#eb6834}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .h{{font-size:14px;font-weight:600}} .b{{font-weight:700}}
  .a{{animation-duration:{DUR}s;animation-iteration-count:infinite;animation-timing-function:linear;animation-fill-mode:both}}
  {' '.join(keyframes)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sec{{fill:#c3c2b7}}
    .nd{{fill:#1a1a19;stroke:#898781}} .ed{{stroke:#383835}}
    .p3{{stroke:#3987e5}} .p5{{stroke:#d95926}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Backward pass on a diamond graph, correct and with the assignment bug</title>
<desc>L = a*3 + a*5 with a = 2. Gradients flow from L back to a along two paths: blue through a*3, orange through a*5. L gets grad 1, both products get grad 1, then the orange path sends 5 to a and the blue path sends 3. With grad += a ends at {good:g}. With grad = the 3 overwrites the 5, the orange path fades, and a ends at {bad:g}.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
{"".join(parts)}
<text class="sec" x="{XA}" y="{H - 10}">order: L, then a*5, then a*3 (reverse topological)</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "diamond-backward-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
