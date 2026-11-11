"""Write figures/attention-row-v2.svg: the animated attention row.

    uv run python figures/make_attention_row.py results/scaling-<date>.json

Twelve bars are the softmax weights of one query against 12 keys (seed 0,
the "figure" section of the results file). Without scaling, the row is shown
at d_k = 8, 16, ..., 512 and collapses toward one bar. In the last phase
1/sqrt(d_k) is switched on at d_k = 512 and the row spreads out again.

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Every element's static attributes are the final frame, so
prefers-reduced-motion switches the animations off and shows that frame.

v2 colours (series palette v2): orange = no scaling, blue = 1/sqrt(d_k).
The bars, the entropy bar and a swatch next to each heading carry the
colour; all text stays in ink colours.
"""

import json
import sys
from pathlib import Path

W, H = 460, 300
N = 12
X0, STEP, BW = 22, 35, 26
BASE, BAR_MAX = 206, 120        # bar baseline y, height of a weight of 1.0
METER_Y, METER_W = 238, 416
HOLD, LAST, TRANS = 1.25, 3.25, 0.3   # seconds per unscaled phase, final phase, transition


def kf(name, frames):
    body = " ".join(f"{pct:.2f}% {{ {css} }}" for pct, css in frames)
    return f"@keyframes {name} {{ {body} }}"


def stepped(name, values, starts, dur, prop):
    # Hold values[i] from starts[i] + TRANS until starts[i + 1]; the first
    # phase also fades in from the last value so the loop restarts smoothly.
    frames = [(0.0, prop(values[-1]))]
    for i, v in enumerate(values):
        frames.append(((starts[i] + TRANS) / dur * 100, prop(v)))
        end = starts[i + 1] if i + 1 < len(values) else dur
        frames.append((end / dur * 100, prop(v)))
    return kf(name, frames)


def window(name, i, starts, dur, fade=0.2):
    start = starts[i]
    end = starts[i + 1] if i + 1 < len(starts) else dur
    p = lambda t: max(0.0, min(100.0, t / dur * 100))
    if i == len(starts) - 1:
        frames = [(0, "opacity:0"), (p(start), "opacity:0"), (p(start + fade), "opacity:1"), (100, "opacity:1")]
    else:
        frames = [(0, "opacity:0" if start > 0 else "opacity:1"), (p(start), "opacity:0" if start > 0 else "opacity:1"),
                  (p(start + fade), "opacity:1"), (p(end - fade), "opacity:1"), (p(end), "opacity:0"), (100, "opacity:0")]
        if start == 0:
            frames = [(0, "opacity:1"), (p(end - fade), "opacity:1"), (p(end), "opacity:0"), (100, "opacity:0")]
    return kf(name, frames)


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    fig = data["figure"]
    assert fig["keys"] == N
    rows = fig["rows"]
    phases = [r for r in rows if r["scale"] == "none"] + [
        next(r for r in rows if r["scale"] == "sqrt" and r["d_k"] == 512)]
    max_bits = fig["max_entropy_bits"]
    starts = [i * HOLD for i in range(len(phases))]
    dur = starts[-1] + LAST
    final = phases[-1]

    css_kf, bars, overlay = [], [], []
    for b in range(N):
        vals = [p["weights"][b] for p in phases]
        css_kf.append(stepped(f"b{b}", vals, starts, dur, lambda v: f"transform:scaleY({v:.4f})"))
        x = X0 + b * STEP
        geo = f'x="{x}" y="{BASE - BAR_MAX}" width="{BW}" height="{BAR_MAX}"'
        bars.append(f'<rect id="g{b}" class="a s un" {geo} style="transform:scaleY({final["weights"][b]:.4f})"/>')
        overlay.append(f'<rect class="s sc" {geo} style="transform:scaleY({final["weights"][b]:.4f})"/>')
    fracs = [p["entropy_bits"] / max_bits for p in phases]
    css_kf.append(stepped("meter", fracs, starts, dur, lambda v: f"transform:scaleX({max(v, 0.002):.4f})"))
    css_kf.append(window("ov", len(phases) - 1, starts, dur, fade=0.3))
    css_kf.append(window("ovm", len(phases) - 1, starts, dur, fade=0.3))
    texts = []
    for i, p in enumerate(phases):
        css_kf.append(window(f"t{i}", i, starts, dur))
        scale = "1/√d_k scaling on" if p["scale"] == "sqrt" else "no scaling"
        last = i == len(phases) - 1
        texts.append(
            f'<g id="t{i}" class="a" opacity="{1 if last else 0}">'
            f'<rect class="{"sc" if p["scale"] == "sqrt" else "un"}" x="{X0}" y="19" width="13" height="13" rx="2"/>'
            f'<text class="ink h" x="{X0 + 20}" y="30">d_k = {p["d_k"]} · {scale}</text>'
            f'<text class="ink" x="{X0}" y="{METER_Y - 8}">entropy {p["entropy_bits"]:.2f} bits of {max_bits:.2f} max'
            f' · largest {p["max_weight"]:.2f}</text></g>')

    anim = " ".join(f"#g{b}{{animation-name:b{b}}}" for b in range(N))
    anim += " #meter{animation-name:meter} #ov{animation-name:ov} #ovm{animation-name:ovm} "
    anim += " ".join(f"#t{i}{{animation-name:t{i}}}" for i in range(len(phases)))
    css = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sec{{fill:#52514e}} .mut{{fill:#898781}}
  .un{{fill:#eb6834}} .sc{{fill:#2a78d6}} .trk{{fill:#e1e0d9}} .base{{stroke:#c3c2b7;stroke-width:1}} .ref{{stroke:#e1e0d9;stroke-width:1;stroke-dasharray:4 3}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .h{{font-size:15px;font-weight:600}}
  .a{{animation-duration:{dur}s;animation-iteration-count:infinite;animation-timing-function:ease-in-out}}
  .s{{transform-box:fill-box;transform-origin:center bottom}}
  .l{{transform-box:fill-box;transform-origin:left center}}
  {anim}
  {' '.join(css_kf)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sec{{fill:#c3c2b7}}
    .un{{fill:#d95926}} .sc{{fill:#3987e5}} .trk{{fill:#2c2c2a}} .base{{stroke:#383835}} .ref{{stroke:#2c2c2a}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    seq = ", ".join(f'd_k {p["d_k"]}: {p["entropy_bits"]:.2f} bits, largest weight {p["max_weight"]:.2f}' for p in phases[:-1])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>One attention row as d_k grows, without and with 1/sqrt(d_k)</title>
<desc>Twelve bars are the softmax weights of one random query against 12 random keys (seed 0). Orange bars, without scaling: {seq}. Blue bars, with 1/sqrt(d_k) at d_k 512: {final["entropy_bits"]:.2f} bits, largest weight {final["max_weight"]:.2f}. Maximum entropy for 12 keys is {max_bits:.2f} bits.</desc>
<style>{css}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="sec" x="{X0}" y="50">one query, 12 keys, seed 0: softmax weights</text>
{"".join(texts)}
{"".join(bars)}
<g id="ov" class="a">{"".join(overlay)}</g>
<line class="base" x1="{X0 - 4}" x2="{X0 + (N - 1) * STEP + BW + 4}" y1="{BASE + 0.5}" y2="{BASE + 0.5}"/>
<line class="ref" x1="{X0 - 4}" x2="{X0 + (N - 1) * STEP + BW + 4}" y1="{BASE - BAR_MAX + 0.5}" y2="{BASE - BAR_MAX + 0.5}"/>
<text class="mut" x="{X0}" y="{BASE - BAR_MAX - 6}">weight 1.0</text>
<rect class="trk" x="{X0}" y="{METER_Y}" width="{METER_W}" height="12" rx="3"/>
<rect id="meter" class="a l un" x="{X0}" y="{METER_Y}" width="{METER_W}" height="12" rx="3" style="transform:scaleX({max(fracs[-1], 0.002):.4f})"/>
<g id="ovm" class="a"><rect class="l sc" x="{X0}" y="{METER_Y}" width="{METER_W}" height="12" rx="3" style="transform:scaleX({max(fracs[-1], 0.002):.4f})"/></g>
<text class="sec" x="{X0}" y="{METER_Y + 30}">bar: entropy as a share of the maximum</text>
<rect class="un" x="{X0}" y="{H - 21}" width="12" height="12" rx="2"/><text class="sec" x="{X0 + 18}" y="{H - 10}">no scaling</text>
<rect class="sc" x="{X0 + 130}" y="{H - 21}" width="12" height="12" rx="2"/><text class="sec" x="{X0 + 148}" y="{H - 10}">1/√d_k scaling</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "attention-row-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
