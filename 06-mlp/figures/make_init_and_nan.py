"""Write figures/init-and-nan-v2.svg: the animated figure.

    uv run python figures/make_init_and_nan.py results/init-<date>.json results/overflow-<date>.json

Top panel: histogram of every hidden tanh value on the dev set at init,
cycling through the three init scales. Each scale has its own colour, the
same as in init-loss-curves-v2.svg: 1/sqrt(fan_in) blue, std 1.0 orange,
std 0.01 aqua. The shaded bands mark the two outer bins (beyond +-0.9).
Bottom panel: training loss per step at lr 100 for the first STEPS steps.
The stable loss (blue) keeps going; the naive loss in numpy float64 (orange)
stops when it turns into inf, and the steps where it becomes nan are marked.

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Static attributes are the final frame, so prefers-reduced-motion shows that.
"""

import json
import sys
from pathlib import Path

W, H = 460, 560
X0, PW = 22, 416                 # left edge and width of both panels
HB, HMAX = 238, 150              # histogram baseline y, height of the share YMAX_H
YMAX_H = 0.55                    # share axis top
BINS = 20
LB, LH, YMAX_L = 500, 160, 600   # loss panel baseline y, height, loss axis top
STEPS = 20
PHASE = 3.0                      # seconds per init scale
NAMES = {"1.0": "std 1.0", "0.01": "std 0.01", "fan_in": "std 1/√fan_in"}
SLOT = {"fan_in": "s1", "1.0": "s2", "0.01": "s3"}   # palette slot per init scale


def kf(name, frames):
    return f"@keyframes {name} {{ " + " ".join(f"{p:.2f}% {{ {c} }}" for p, c in frames) + " }"


def window(i, n, dur, fade=0.25):
    """Opacity keyframes: phase i of n is visible, with a short crossfade."""
    t0, t1 = i * PHASE / dur * 100, (i + 1) * PHASE / dur * 100
    f = fade * PHASE / dur * 100
    if i == 0:
        return [(0, "opacity:1"), (t1 - f, "opacity:1"), (t1, "opacity:0"), (100 - f, "opacity:0"), (100, "opacity:1")]
    if i == n - 1:
        return [(0, "opacity:0"), (t0 - f, "opacity:0"), (t0, "opacity:1"), (100 - f, "opacity:1"), (100, "opacity:0")]
    return [(0, "opacity:0"), (t0 - f, "opacity:0"), (t0, "opacity:1"), (t1 - f, "opacity:1"), (t1, "opacity:0"), (100, "opacity:0")]


def main():
    init = json.loads(Path(sys.argv[1]).read_text())
    over = json.loads(Path(sys.argv[2]).read_text())
    runs = init["runs"]
    dur = PHASE * len(runs)
    bw = PW / BINS

    css, groups = [], []
    # One group per init scale: its bars, a colour swatch and its caption.
    # The groups crossfade; the last one is the static final frame.
    for i, r in enumerate(runs):
        key = str(r["scale"])
        last = i == len(runs) - 1
        bars = []
        for b in range(BINS):
            share = r["init"]["hist"][b] / r["init"]["values"]
            hgt = min(share, YMAX_H) / YMAX_H * HMAX
            bars.append(f'<rect x="{X0 + b * bw + 1:.1f}" y="{HB - hgt:.1f}" width="{bw - 2:.1f}" height="{hgt:.1f}"/>')
        css.append(kf(f"p{i}", window(i, len(runs), dur)))
        groups.append(
            f'<g id="p{i}" class="a" opacity="{1 if last else 0}">'
            f'<rect class="{SLOT[key]}" x="{X0}" y="42" width="14" height="14" rx="2"/>'
            f'<text class="ink h" x="{X0 + 22}" y="54">init {NAMES[key]}</text>'
            f'<g class="{SLOT[key]}">{"".join(bars)}</g>'
            f'<text class="ink" x="{X0}" y="{HB + 40}">|h| &gt; 0.99: {r["init"]["saturated_share"] * 100:.1f}%'
            f' · loss at init {r["init"]["dev_loss"]:.2f}</text></g>')
    anim = " ".join(f"#p{i}{{animation-name:p{i}}}" for i in range(len(runs)))

    # Loss panel.
    lr100 = [r for r in over["runs"] if r["lr"] == 100.0]
    stable = next(r for r in lr100 if r["mode"] == "stable")["loss"][:STEPS]
    numpy_run = next(r for r in lr100 if r["mode"] == "naive_numpy")
    first = numpy_run["first"]
    naive = [v for v in numpy_run["loss"][:STEPS] if isinstance(v, float)]  # finite steps only
    lx, lw = X0 + 34, PW - 34  # room for the y labels on the left
    sx = lambda step: lx + (step - 1) / (STEPS - 1) * lw
    sy = lambda v: LB - min(v, YMAX_L) / YMAX_L * LH
    sp = " ".join(f"{sx(i + 1):.1f},{sy(v):.1f}" for i, v in enumerate(stable))
    npts = " ".join(f"{sx(i + 1):.1f},{sy(v):.1f}" for i, v in enumerate(naive))
    inf_step, nan_step = first["inf_loss"]["step"], first["nan_loss"]["step"]
    css.append(kf("draw", [(0, "stroke-dashoffset:1"), (70, "stroke-dashoffset:0"), (100, "stroke-dashoffset:0")]))
    reveal = lambda at: kf(f"r{at}", [(0, "opacity:0"), (at, "opacity:0"), (at + 3, "opacity:1"), (100, "opacity:1")])
    n_at = round(len(naive) / STEPS * 70)  # the naive line ends here in the drawing
    nan_at = round(nan_step / STEPS * 70)
    css += [reveal(n_at), reveal(nan_at)]
    anim += f" #sl,#nl{{animation-name:draw}} #mi,#nlab{{animation-name:r{n_at}}} #mn{{animation-name:r{nan_at}}}"
    grid = []
    for v in (0, 200, 400, 600):
        grid.append(f'<line class="grid" x1="{lx}" x2="{X0 + PW}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                    f'<text class="mut" x="{lx - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    xi, xn = sx(inf_step), sx(nan_step)
    end_x, end_y = sx(len(naive)), sy(naive[-1])
    loss_panel = f"""
<text class="ink h" x="{X0}" y="{LB - LH - 32}">lr 100: loss per training step</text>
{''.join(grid)}
<polyline id="sl" class="a ln l1" pathLength="1" points="{sp}"/>
<polyline id="nl" class="a ln l2" pathLength="1" points="{npts}"/>
<g id="mi" class="a"><line class="mark" x1="{xi:.1f}" x2="{xi:.1f}" y1="{LB - LH:.1f}" y2="{LB}"/>
<text class="sec" x="{xi + 4:.1f}" y="{LB - 8}">inf</text></g>
<g id="mn" class="a"><line class="mark" x1="{xn:.1f}" x2="{xn:.1f}" y1="{LB - LH:.1f}" y2="{LB}"/>
<text class="sec" x="{xn + 4:.1f}" y="{LB - 8}">NaN from step {nan_step}</text></g>
<g id="nlab" class="a"><circle class="s2 dot" cx="{end_x:.1f}" cy="{end_y:.1f}" r="4.5"/>
<text class="ink" x="{end_x - 6:.1f}" y="{end_y - 12:.1f}" text-anchor="end">naive</text></g>
<rect class="s1" x="{X0 + PW - 176}" y="{LB - LH - 22}" width="14" height="4" rx="1"/>
<text class="ink" x="{X0 + PW}" y="{LB - LH - 16}" text-anchor="end">stable (log-sum-exp)</text>
<text class="mut" x="{lx}" y="{LB + 20}">step 1</text>
<text class="mut" x="{X0 + PW}" y="{LB + 20}" text-anchor="end">step {STEPS}</text>
<rect class="s2" x="{X0}" y="{H - 24}" width="14" height="4" rx="1"/>
<text class="sec" x="{X0 + 22}" y="{H - 18}">naive: textbook loss in numpy float64</text>"""

    style = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sec{{fill:#52514e}} .mut{{fill:#898781}}
  .s1{{fill:#2a78d6}} .s2{{fill:#eb6834}} .s3{{fill:#1baf7a}}
  .band{{fill:#e1e0d9;opacity:.55}} .base{{stroke:#c3c2b7;stroke-width:1}} .grid{{stroke:#e1e0d9;stroke-width:1}}
  .ln{{fill:none;stroke-width:2.5;stroke-dasharray:1;stroke-linejoin:round}} .l1{{stroke:#2a78d6}} .l2{{stroke:#eb6834;stroke-width:3.5}}
  .mark{{stroke:#eb6834;stroke-width:1;stroke-dasharray:3 3}} .dot{{stroke:#fcfcfb;stroke-width:2}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .h{{font-size:15px;font-weight:600}}
  .a{{animation-duration:{dur}s;animation-iteration-count:infinite;animation-timing-function:ease-in-out}}
  {anim}
  {' '.join(css)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sec{{fill:#c3c2b7}} .mut{{fill:#898781}}
    .s1{{fill:#3987e5}} .s2{{fill:#d95926}} .s3{{fill:#199e70}}
    .band{{fill:#2c2c2a;opacity:1}} .base{{stroke:#383835}} .grid{{stroke:#2c2c2a}}
    .l1{{stroke:#3987e5}} .l2{{stroke:#d95926}} .mark{{stroke:#d95926}} .dot{{stroke:#1a1a19}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    hist_desc = "; ".join(f'{NAMES[str(r["scale"])]}: {r["init"]["saturated_share"] * 100:.1f}% of values above 0.99 in size, '
                          f'loss at init {r["init"]["dev_loss"]:.2f}' for r in runs)
    band_h = HMAX + 10
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Hidden activations at init by scale, and naive vs stable loss at lr 100</title>
<desc>Top: histogram of hidden tanh values at init over the dev set, 20 bins from -1 to 1, one colour per init scale (std 1/sqrt(fan_in) blue, std 1.0 orange, std 0.01 aqua). Shaded bands mark the outer bins beyond plus or minus 0.9. {hist_desc}. Bottom: training loss for steps 1 to {STEPS} at lr 100. The stable loss (blue) stays finite. The naive loss in numpy float64 (orange) is inf at step {inf_step} and NaN from step {nan_step}.</desc>
<style>{style}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="mut" x="{X0}" y="26">hidden tanh values at init (dev set)</text>
<rect class="band" x="{X0}" y="{HB - band_h}" width="{bw:.1f}" height="{band_h}"/>
<rect class="band" x="{X0 + PW - bw:.1f}" y="{HB - band_h}" width="{bw:.1f}" height="{band_h}"/>
<text class="sec" x="{X0 + PW}" y="{HB - band_h - 6}" text-anchor="end">shaded: beyond ±0.9</text>
{''.join(groups)}
<line class="base" x1="{X0}" x2="{X0 + PW}" y1="{HB + 0.5}" y2="{HB + 0.5}"/>
<text class="mut" x="{X0}" y="{HB + 18}">-1</text>
<text class="mut" x="{X0 + PW / 2}" y="{HB + 18}" text-anchor="middle">0</text>
<text class="mut" x="{X0 + PW}" y="{HB + 18}" text-anchor="end">+1</text>
{loss_panel}
</svg>
"""
    out = Path(__file__).resolve().parent / "init-and-nan-v2.svg"
    out.write_text(svg)
    print("wrote", out.name, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
